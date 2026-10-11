"""Tests for conversation routing in the gateway.

The routing decision has no network in it, so it is tested directly rather than
through a live proxy. Every payload here is shaped after traffic captured from
Codex and n3 against this deployment rather than invented -- the nested
`client_metadata.session_id` in particular is the reason a scan of top-level
body keys concludes that chat completions carries no conversation identity.
"""

import argparse
import asyncio
import json
import os
import shutil
import tempfile
import textwrap
import time
import types
import unittest

import gateway


def headers(**kwargs):
    return [(name.replace("_", "-"), value) for name, value in kwargs.items()]


def body(payload):
    return json.dumps(payload).encode()


def tally(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


class ConversationKey(unittest.TestCase):
    def test_responses_uses_prompt_cache_key(self):
        key = gateway.conversation_key(
            headers(content_type="application/json"),
            body(
                {
                    "model": "glm",
                    "prompt_cache_key": "01a074ca-1d86-7870-ba89-416001960f00",
                    "input": "hello",
                }
            ),
        )
        self.assertEqual(key, "prompt_cache_key:01a074ca-1d86-7870-ba89-416001960f00")

    def test_chat_completions_uses_nested_session_id(self):
        key = gateway.conversation_key(
            headers(content_type="application/json"),
            body(
                {
                    "model": "glm",
                    "messages": [{"role": "user", "content": "hi"}],
                    "client_metadata": {
                        "session_id": "01a06b54-b709-70e1-9139-7b14562f9792",
                        "thread_id": "01a06b54-b709-70e1-9139-7b14562f9792",
                        "turn_id": "01a06b54-b8ee-7132-a476-d2a6e2000000",
                    },
                }
            ),
        )
        self.assertEqual(key, "client_metadata.session_id:01a06b54-b709-70e1-9139-7b14562f9792")

    def test_turn_id_is_never_the_key(self):
        """turn_id changes every turn; pinning on it would defeat the point."""
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "client_metadata": {"session_id": "S", "turn_id": "T1"},
        }
        first = gateway.conversation_key(headers(), body(payload))
        payload["client_metadata"]["turn_id"] = "T2"
        payload["messages"].append({"role": "assistant", "content": "there"})
        self.assertEqual(first, gateway.conversation_key(headers(), body(payload)))

    def test_header_wins_over_body(self):
        key = gateway.conversation_key(
            headers(x_conversation_id="explicit"),
            body({"prompt_cache_key": "ignored"}),
        )
        self.assertEqual(key, "hdr:explicit")

    def test_prefix_hash_is_stable_as_the_conversation_grows(self):
        payload = {
            "messages": [
                {"role": "system", "content": "You are a kernel engineer."},
                {"role": "user", "content": "Optimise problem 90794."},
            ]
        }
        first = gateway.conversation_key(headers(), body(payload))
        self.assertTrue(first.startswith("prefix:"))
        for turn in range(5):
            payload["messages"].append({"role": "assistant", "content": "turn %d" % turn})
            payload["messages"].append({"role": "user", "content": "next %d" % turn})
            self.assertEqual(first, gateway.conversation_key(headers(), body(payload)))

    def test_prefix_hash_separates_different_openings(self):
        def key_for(opening):
            return gateway.conversation_key(
                headers(),
                body(
                    {
                        "messages": [
                            {"role": "system", "content": "same prompt"},
                            {"role": "user", "content": opening},
                        ]
                    }
                ),
            )

        self.assertNotEqual(key_for("problem 90794"), key_for("problem 90795"))

    def test_instructions_participate_in_the_prefix(self):
        """Responses puts the system prompt in `instructions`, not in the input."""

        def key_for(instructions):
            return gateway.conversation_key(
                headers(), body({"instructions": instructions, "input": "same opening"})
            )

        self.assertNotEqual(key_for("be terse"), key_for("be verbose"))

    def test_malformed_bodies_do_not_raise(self):
        for payload in (
            b"not json",
            b"",
            None,
            b"[1,2,3]",
            b'"a string"',
            body({"model": "glm"}),
            body({"messages": []}),
        ):
            self.assertIsNone(gateway.conversation_key(headers(), payload))

    def test_blank_header_falls_through_to_the_body(self):
        key = gateway.conversation_key(
            headers(x_conversation_id="   "), body({"prompt_cache_key": "real"})
        )
        self.assertEqual(key, "prompt_cache_key:real")


class LogRendering(unittest.TestCase):
    """The log line has to tell two conversations apart, which is the whole job."""

    def test_two_keys_from_one_client_do_not_render_identically(self):
        prefix = "client_metadata.session_id:"
        first = gateway.short_convo(prefix + "01a06b54-b709-70e1-9139-7b14562f9792")
        second = gateway.short_convo(prefix + "01a06b54-b709-70e1-9139-000000000000")
        self.assertNotEqual(first, second)

    def test_short_keys_are_left_alone(self):
        self.assertEqual("hdr:abc", gateway.short_convo("hdr:abc"))

    def test_none_renders_as_a_dash(self):
        self.assertEqual("-", gateway.short_convo(None))

    def test_the_source_survives_shortening(self):
        rendered = gateway.short_convo("client_metadata.session_id:" + "x" * 40)
        self.assertTrue(rendered.startswith("client_metadata."))


class Routing(unittest.TestCase):
    def setUp(self):
        self.router = gateway.Router(ttl=1800, capacity=100)

    def test_same_conversation_returns_to_the_same_backend(self):
        loads = {"a": (0, 0), "b": (0, 0)}
        first = self.router.route("convo:1", loads, {"a", "b"})
        for _ in range(10):
            self.assertEqual(first, self.router.route("convo:1", loads, {"a", "b"}))
        self.assertEqual(self.router.hits, 10)

    def test_new_conversations_spread_evenly(self):
        placed = []
        for index in range(6):
            counts = self.router.counts({"a": None, "b": None, "c": None})
            loads = {job: (counts[job], 0) for job in ("a", "b", "c")}
            placed.append(self.router.route("convo:%d" % index, loads, {"a", "b", "c"}))
        self.assertEqual(sorted(tally(placed).values()), [2, 2, 2])

    def test_a_dead_backend_rehomes_rather_than_failing(self):
        first = self.router.route("convo:1", {"a": (0, 0), "b": (0, 0)}, {"a", "b"})
        survivor = "b" if first == "a" else "a"
        self.assertEqual(survivor, self.router.route("convo:1", {survivor: (0, 0)}, {survivor}))
        self.assertEqual(self.router.rehomed, 1)

    def test_a_pin_survives_its_backend_leaving_the_accepting_set(self):
        """Draining must not evict conversations that are already running."""
        self.assertEqual("a", self.router.route("convo:1", {"a": (0, 0)}, {"a"}))
        # `a` now drains: still serving, no longer accepting.
        self.assertEqual("a", self.router.route("convo:1", {"b": (0, 0)}, {"a", "b"}))
        # A new conversation goes to the one that is accepting.
        self.assertEqual("b", self.router.route("convo:2", {"b": (0, 0)}, {"a", "b"}))

    def test_no_backend_returns_none(self):
        self.assertIsNone(self.router.route("convo:1", {}, set()))
        self.assertIsNone(self.router.route(None, {}, set()))

    def test_pinned_conversation_with_nothing_accepting_still_gets_its_backend(self):
        """Every backend draining: an in-flight conversation must not 503."""
        self.assertEqual("a", self.router.route("convo:1", {"a": (0, 0)}, {"a"}))
        self.assertEqual("a", self.router.route("convo:1", {}, {"a"}))

    def test_unidentified_requests_balance_without_pinning(self):
        self.assertEqual("b", self.router.route(None, {"a": (0, 5), "b": (0, 0)}, {"a", "b"}))
        self.assertEqual(0, len(self.router.pins))

    def test_conversation_count_outranks_inflight(self):
        """An idle agent still occupies a backend, and must still count."""
        self.assertEqual("b", self.router.route("new", {"a": (5, 0), "b": (1, 9)}, {"a", "b"}))

    def test_pins_expire(self):
        self.router.route("convo:1", {"a": (0, 0)}, {"a"}, now=1000)
        self.router.route("convo:2", {"a": (0, 0)}, {"a"}, now=1000 + 1801)
        self.assertNotIn("convo:1", self.router.pins)

    def test_activity_postpones_expiry(self):
        self.router.route("convo:1", {"a": (0, 0)}, {"a"}, now=1000)
        for step in range(1, 6):
            self.router.route("convo:1", {"a": (0, 0)}, {"a"}, now=1000 + step * 1000)
        self.assertIn("convo:1", self.router.pins)

    def test_capacity_drops_the_least_recently_used(self):
        router = gateway.Router(ttl=1800, capacity=3)
        for index in range(5):
            router.route("convo:%d" % index, {"a": (0, 0)}, {"a"})
        self.assertEqual(3, len(router.pins))
        self.assertNotIn("convo:0", router.pins)
        self.assertIn("convo:4", router.pins)


class AcceptingSet(unittest.TestCase):
    """Fleet.accepting() is what replaces the single-active election."""

    def fleet(self, margin=1800, **backends):
        args = argparse.Namespace(
            new_conversation_margin=margin,
            sticky_ttl=1800,
            sticky_capacity=100,
            users="/nonexistent",
            fleet_dir="/nonexistent",
            route_policy="least_conversations",
            router_state=None,
            key_sources=None,
        )
        fleet = gateway.Fleet(args)
        now = time.time()
        for job_id, (healthy, remaining) in backends.items():
            backend = gateway.Backend(
                {
                    "job_id": job_id,
                    "url": "http://host:8000",
                    "end_time": now + remaining,
                    "heartbeat": now,
                }
            )
            backend.healthy = healthy
            fleet.backends[job_id] = backend
        return fleet

    def test_healthy_backends_all_accept(self):
        fleet = self.fleet(a=(True, 7200), b=(True, 7200))
        self.assertEqual({"a", "b"}, set(fleet.accepting()))
        self.assertEqual({"a", "b"}, fleet.serving())

    def test_unhealthy_backends_neither_serve_nor_accept(self):
        fleet = self.fleet(a=(True, 7200), b=(False, 7200))
        self.assertEqual({"a"}, set(fleet.accepting()))
        self.assertEqual({"a"}, fleet.serving())

    def test_a_backend_near_its_end_stops_taking_new_conversations(self):
        fleet = self.fleet(a=(True, 7200), b=(True, 600))
        self.assertEqual({"a"}, set(fleet.accepting()))
        # ...but it keeps serving the ones it already has.
        self.assertEqual({"a", "b"}, fleet.serving())

    def test_draining_is_excluded(self):
        fleet = self.fleet(a=(True, 7200), b=(True, 7200))
        fleet.draining["b"] = 0
        self.assertEqual({"a"}, set(fleet.accepting()))
        self.assertEqual({"a", "b"}, fleet.serving())

    def test_superseded_backends_still_accept(self):
        """Every instance but the longest-lived one is superseded by definition.

        Excluding them collapsed a fleet of N healthy instances into one
        accepting instance the moment a successor was elected -- so the routing
        was multi-instance in name only. Reproduced against a live gateway:
        `accepting` went from ['OLD'] to ['NEW'] the instant NEW won the
        election, with OLD still healthy and still serving.
        """
        fleet = self.fleet(old=(True, 7200), new=(True, 93600))
        fleet.superseded.add("old")
        self.assertEqual({"old", "new"}, set(fleet.accepting()))

    def test_a_whole_fleet_keeps_accepting_after_an_election(self):
        fleet = self.fleet(a=(True, 7200), b=(True, 7200), c=(True, 90000))
        # c has the longest life, so an election supersedes both others.
        fleet.superseded.update({"a", "b"})
        self.assertEqual({"a", "b", "c"}, set(fleet.accepting()))

    def test_all_backends_ageing_out_still_offers_the_longest_lived(self):
        """Refusing every new conversation would be worse than a short one."""
        fleet = self.fleet(a=(True, 300), b=(True, 900))
        self.assertEqual({"b"}, set(fleet.accepting()))

    def test_no_healthy_backend_accepts_nothing(self):
        fleet = self.fleet(a=(False, 7200))
        self.assertEqual({}, fleet.accepting())
        self.assertEqual(set(), fleet.serving())


class Policies(unittest.TestCase):
    """Placement policy is selectable, and each one means what it says."""

    @staticmethod
    def router(policy):
        return gateway.Router(ttl=1800, capacity=100, policy=policy)

    # (conversations, inflight, remaining_seconds)
    LOAD = {"a": (5, 0, 7200), "b": (1, 9, 600)}

    def test_least_conversations_ignores_inflight(self):
        self.assertEqual("b", self.router("least_conversations").route("k", self.LOAD, {"a", "b"}))

    def test_least_inflight_ignores_conversations(self):
        self.assertEqual("a", self.router("least_inflight").route("k", self.LOAD, {"a", "b"}))

    def test_longest_lived_picks_the_most_remaining_time(self):
        self.assertEqual("a", self.router("longest_lived").route("k", self.LOAD, {"a", "b"}))

    def test_round_robin_cycles(self):
        router = self.router("round_robin")
        placed = [
            router.route("k%d" % i, {"a": (0, 0, 0), "b": (0, 0, 0)}, {"a", "b"}) for i in range(4)
        ]
        self.assertEqual(["a", "b", "a", "b"], placed)

    def test_policy_can_change_between_calls(self):
        router = self.router("least_conversations")
        self.assertEqual("b", router.route("k1", self.LOAD, {"a", "b"}))
        router.policy = "least_inflight"
        self.assertEqual("a", router.route("k2", self.LOAD, {"a", "b"}))

    def test_two_tuple_loads_still_work(self):
        """Callers that do not supply a lifetime must not crash the policy."""
        router = self.router("longest_lived")
        self.assertIn(router.route("k", {"a": (0, 0), "b": (0, 0)}, {"a", "b"}), ("a", "b"))


class ManualControl(unittest.TestCase):
    def setUp(self):
        self.router = gateway.Router(ttl=1800, capacity=100)

    def test_a_manual_pin_overrides_the_load_picture(self):
        self.router.pin("convo:1", "a")
        for _ in range(5):
            self.assertEqual("a", self.router.route("convo:1", {"b": (0, 0, 0)}, {"a", "b"}))

    def test_a_manual_pin_does_not_expire(self):
        self.router.pin("convo:1", "a")
        self.assertEqual(
            "a", self.router.route("convo:1", {"b": (0, 0, 0)}, {"a", "b"}, now=time.time() + 99999)
        )

    def test_a_manual_pin_yields_to_a_backend_that_is_gone(self):
        """Being emphatic about a dead backend would just mean refusing to serve."""
        self.router.pin("convo:1", "a")
        self.assertEqual("b", self.router.route("convo:1", {"b": (0, 0, 0)}, {"b"}))

    def test_unpin_restores_normal_placement(self):
        self.router.pin("convo:1", "a")
        self.assertEqual("a", self.router.unpin("convo:1"))
        self.assertEqual("b", self.router.route("convo:1", {"b": (0, 0, 0)}, {"a", "b"}))

    def test_unpinning_something_unpinned_is_harmless(self):
        self.assertIsNone(self.router.unpin("never-seen"))


class Persistence(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(self.path) and os.unlink(self.path))

    def make(self):
        return gateway.Router(ttl=1800, capacity=100, state_path=self.path)

    def test_pins_survive_a_restart(self):
        first = self.make()
        first.route("convo:1", {"a": (0, 0, 0)}, {"a"})
        first.pin("convo:2", "b")
        first.paused.add("c")
        first.policy = "round_robin"
        first.dirty = True
        first.save()

        second = self.make()
        second.load()
        self.assertEqual("a", second.route("convo:1", {"b": (0, 0, 0)}, {"a", "b"}))
        self.assertEqual({"convo:2": "b"}, second.manual)
        self.assertEqual({"c"}, second.paused)
        self.assertEqual("round_robin", second.policy)

    def test_expired_pins_are_not_restored(self):
        first = self.make()
        first.pins["old"] = ("a", time.time() - 9999)
        first.dirty = True
        first.save()
        second = self.make()
        second.load()
        self.assertNotIn("old", second.pins)

    def test_a_corrupt_state_file_is_ignored_not_fatal(self):
        with open(self.path, "w") as handle:
            handle.write("{ not json")
        router = self.make()
        router.load()  # must not raise
        self.assertEqual(0, len(router.pins))

    def test_a_state_file_that_is_not_an_object_is_ignored(self):
        with open(self.path, "w") as handle:
            handle.write("[1, 2, 3]")
        router = self.make()
        router.load()
        self.assertEqual(0, len(router.pins))

    def test_saving_is_skipped_when_nothing_changed(self):
        router = self.make()
        router.save()
        self.assertFalse(os.path.exists(self.path))


class KeySourceConfiguration(unittest.TestCase):
    def test_a_custom_chain_changes_precedence(self):
        payload = body({"prompt_cache_key": "P", "client_metadata": {"session_id": "S"}})
        default = gateway.conversation_key(headers(), payload)
        self.assertEqual("prompt_cache_key:P", default)
        reordered = gateway.conversation_key(
            headers(), payload, ["body:client_metadata.session_id", "body:prompt_cache_key"]
        )
        self.assertEqual("client_metadata.session_id:S", reordered)

    def test_a_chain_can_exclude_the_prefix_fallback(self):
        payload = body({"messages": [{"role": "user", "content": "hi"}]})
        self.assertIsNone(gateway.conversation_key(headers(), payload, ["body:prompt_cache_key"]))

    def test_dotted_paths_reach_arbitrary_depth(self):
        payload = body({"a": {"b": {"c": "deep"}}})
        self.assertEqual("a.b.c:deep", gateway.conversation_key(headers(), payload, ["body:a.b.c"]))

    def test_a_path_through_a_non_object_does_not_raise(self):
        payload = body({"a": "not an object"})
        self.assertIsNone(gateway.conversation_key(headers(), payload, ["body:a.b.c"]))


class StaleHeartbeats(unittest.TestCase):
    """A shared filesystem stall must not empty the fleet.

    Every serving job writes its registration to the same directory, so one
    stall there makes every heartbeat look stale in the same sweep. Under load
    all four backends went `gone: no heartbeat for 30s` together and came back
    five seconds later, having answered /health the whole time -- 43 requests
    got 503 from a fleet that was healthy.
    """

    def fleet(self, stale_after=30):
        args = argparse.Namespace(
            new_conversation_margin=1800,
            sticky_ttl=1800,
            sticky_capacity=100,
            users="/nonexistent",
            fleet_dir=tempfile.mkdtemp(),
            stale_after=stale_after,
            route_policy="least_conversations",
            router_state=None,
            key_sources=None,
        )
        return gateway.Fleet(args)

    def register(self, fleet, job_id, heartbeat_age, healthy):
        now = time.time()
        record = {
            "job_id": job_id,
            "url": "http://host:8000",
            "end_time": now + 7200,
            "heartbeat": now - heartbeat_age,
        }
        path = os.path.join(fleet.args.fleet_dir, "%s.json" % job_id)
        with open(path, "w") as handle:
            json.dump(record, handle)
        return path

    def test_a_healthy_backend_survives_a_stale_heartbeat(self):
        fleet = self.fleet()
        self.register(fleet, "a", 0, True)
        fleet.discover()
        fleet.backends["a"].healthy = True
        # The heartbeat now looks 300s old, as it would after a filesystem stall.
        self.register(fleet, "a", 300, True)
        fleet.discover()
        self.assertIn("a", fleet.backends)

    def test_an_unhealthy_backend_with_a_stale_heartbeat_is_retired(self):
        fleet = self.fleet()
        self.register(fleet, "a", 0, True)
        fleet.discover()
        fleet.backends["a"].healthy = False
        self.register(fleet, "a", 300, False)
        fleet.discover()
        self.assertNotIn("a", fleet.backends)

    def test_a_deregistered_backend_is_retired_even_while_healthy(self):
        """Removing the file is how a job says it is going away."""
        fleet = self.fleet()
        path = self.register(fleet, "a", 0, True)
        fleet.discover()
        fleet.backends["a"].healthy = True
        os.unlink(path)
        fleet.discover()
        self.assertNotIn("a", fleet.backends)

    def test_a_whole_fleet_survives_one_stalled_sweep(self):
        fleet = self.fleet()
        for job in ("a", "b", "c", "d"):
            self.register(fleet, job, 0, True)
        fleet.discover()
        for job in fleet.backends.values():
            job.healthy = True
        for job in ("a", "b", "c", "d"):
            self.register(fleet, job, 300, True)
        fleet.discover()
        self.assertEqual({"a", "b", "c", "d"}, set(fleet.backends))


class ReviveDeadBackends(unittest.TestCase):
    """A deployment that exited but kept its allocation must be restarted.

    Twice in one night an in-service instance had its server killed, wrote
    "attempt 1 exited with status 143; allocation retained" and then sat dead
    for tens of minutes holding eight idle nodes. The gateway already knew how
    to restart such a job -- but only for a successor it had just submitted,
    never for one that had been serving for hours.
    """

    def fleet(
        self,
        state,
        healthy=False,
        heartbeat_age=0.0,
        revive_limit=3,
        revive_cooldown=180,
        no_relay=True,
    ):
        args = argparse.Namespace(
            new_conversation_margin=1800,
            sticky_ttl=1800,
            sticky_capacity=100,
            users="/nonexistent",
            fleet_dir=tempfile.mkdtemp(),
            stale_after=30,
            route_policy="least_conversations",
            router_state=None,
            key_sources=None,
            revive_limit=revive_limit,
            revive_cooldown=revive_cooldown,
            # Defaults to the deployed configuration: gateway.sbatch passes
            # --no-relay unconditionally.
            no_relay=no_relay,
        )
        fleet = gateway.Fleet(args)
        now = time.time()
        backend = gateway.Backend(
            {
                "job_id": "500",
                "url": "http://node-a:8400",
                "run_dir": "/run/500",
                "state": state,
                "end_time": now + 3600,
                "heartbeat": now - heartbeat_age,
            }
        )
        backend.healthy = healthy
        fleet.backends["500"] = backend
        return fleet

    def revive(self, fleet, scontrol=(1, "slurm_load_jobs error: Invalid job id specified")):
        """One supervision sweep, capturing what it asked serve.sh to do.

        `scontrol` is what `scontrol show job` answers. The default is the
        scheduler not knowing the job, which leaves revive to act as it would
        without asking.
        """
        calls = []

        async def fake_run(_fleet, *args):
            calls.append(args)
            return 0, ""

        async def fake_slurm(*command):
            self.assertEqual(("scontrol", "show", "job", "500"), command)
            return scontrol

        originals = gateway.run_serve_sh, gateway.run_slurm_command
        gateway.run_serve_sh, gateway.run_slurm_command = fake_run, fake_slurm
        gateway._PREEMPTION_NOTED.clear()
        try:
            asyncio.run(gateway.revive_dead_backends(fleet, time.time()))
        finally:
            gateway.run_serve_sh, gateway.run_slurm_command = originals
        return calls

    def test_an_exited_backend_is_restarted_in_place(self):
        fleet = self.fleet("attempt 1 exited with status 143; allocation retained")
        self.assertEqual([("restart", "/run/500")], self.revive(fleet))

    def test_a_stopped_backend_is_restarted(self):
        fleet = self.fleet("stopped; allocation retained")
        self.assertEqual([("restart", "/run/500")], self.revive(fleet))

    def test_a_serving_backend_is_left_alone(self):
        fleet = self.fleet("running attempt 1", healthy=True)
        self.assertEqual([], self.revive(fleet))

    def test_an_unhealthy_backend_that_did_not_exit_is_left_alone(self):
        """A failed probe can be a blip.

        Only the deployment's own state is evidence that its server is gone.
        """
        fleet = self.fleet("running attempt 1", healthy=False)
        self.assertEqual([], self.revive(fleet))

    def test_a_draining_backend_is_left_alone(self):
        """A roll is in flight -- restarting would fight fleetctl for the job."""
        fleet = self.fleet("attempt 1 exited with status 143; allocation retained")
        fleet.draining["500"] = time.time() + 600
        self.assertEqual([], self.revive(fleet))

    def test_a_superseded_backend_is_left_alone_while_relay_can_roll(self):
        """With relay on, superseded really does mean a roll may be in flight."""
        fleet = self.fleet("attempt 1 exited with status 143; allocation retained", no_relay=False)
        fleet.superseded.add("500")
        self.assertEqual([], self.revive(fleet))

    def test_a_superseded_backend_is_revived_under_no_relay(self):
        """Under --no-relay nothing ever leaves `superseded`, so it cannot gate revive.

        The promotion to `draining` that discards entries sits behind the
        --no-relay return in supervise(), so the set only grows: on the running
        fleet 10 of 14 healthy backends carried superseded=true, and three died
        holding eight nodes each with a fresh heartbeat before anyone noticed.
        """
        fleet = self.fleet("attempt 1 exited with status 143; allocation retained")
        fleet.superseded.add("500")
        self.assertEqual([("restart", "/run/500")], self.revive(fleet))

    def test_a_stale_controller_is_not_asked(self):
        """Nobody is left to read the control file, so writing it is noise."""
        fleet = self.fleet(
            "attempt 1 exited with status 143; allocation retained", heartbeat_age=3600
        )
        self.assertEqual([], self.revive(fleet))

    def test_restarts_are_capped(self):
        fleet = self.fleet(
            "attempt 1 exited with status 143; allocation retained", revive_cooldown=0
        )
        attempts = 0
        for _ in range(fleet.args.revive_limit + 3):
            attempts += len(self.revive(fleet))
        self.assertEqual(fleet.args.revive_limit, attempts)

    def test_cooldown_spaces_attempts(self):
        fleet = self.fleet("attempt 1 exited with status 143; allocation retained")
        self.assertEqual(1, len(self.revive(fleet)))
        self.assertEqual([], self.revive(fleet), "second attempt inside the cooldown")

    def test_disabled_by_zero_limit(self):
        fleet = self.fleet("attempt 1 exited with status 143; allocation retained", revive_limit=0)
        self.assertEqual([], self.revive(fleet))

    def test_an_allocation_under_preemption_notice_is_not_revived(self):
        """Slurm's SIGTERM ended the attempt; the grace period ends the allocation.

        Reviving inside that window admitted about ten minutes of requests,
        all cut mid-stream together when Slurm killed the job.
        """
        fleet = self.fleet("attempt 1 exited with status 143; allocation retained")
        out = (
            "JobId=500 JobState=RUNNING StartTime=2026-10-04T14:00:00 "
            "PreemptEligibleTime=2026-10-04T18:05:00 PreemptTime=2026-10-04T21:44:23"
        )
        self.assertEqual([], self.revive(fleet, scontrol=(0, out)))
        self.assertNotIn("500", fleet.revived, "the revive budget is not spent on it")

    def test_a_notice_from_before_a_requeue_does_not_block_revive(self):
        """A requeued job's last run was preempted; this run was not."""
        fleet = self.fleet("attempt 1 exited with status 143; allocation retained")
        out = "JobId=500 JobState=RUNNING StartTime=2026-10-05T01:00:00 PreemptTime=2026-10-04T21:44:23"
        self.assertEqual([("restart", "/run/500")], self.revive(fleet, scontrol=(0, out)))

    def test_no_preemption_notice_revives_as_before(self):
        fleet = self.fleet("attempt 1 exited with status 143; allocation retained")
        out = "JobId=500 JobState=RUNNING StartTime=2026-10-04T14:00:00 PreemptTime=None"
        self.assertEqual([("restart", "/run/500")], self.revive(fleet, scontrol=(0, out)))

    def test_the_pending_successor_is_left_to_supervise_pending(self):
        """Two paths restarting one job would double its attempt budget."""
        fleet = self.fleet("attempt 1 exited with status 143; allocation retained")
        fleet.pending = ("500", time.time())
        self.assertEqual([], self.revive(fleet))


class RegistrationPath(unittest.TestCase):
    """The job id becomes a filename, so it decides where the gateway writes."""

    def path(self, job_id, root="/fleet"):
        return gateway.registration_path(root, job_id)

    def test_an_ordinary_job_id_lands_in_the_fleet_directory(self):
        self.assertEqual("/fleet/279495.json", self.path("279495"))
        self.assertEqual("/fleet/inst-a_2.json", self.path("inst-a_2"))

    def test_traversal_is_refused(self):
        for bad in ("../evil", "../../etc/passwd", "a/b", "..", ".", "/etc/passwd", "a\\b"):
            self.assertIsNone(self.path(bad), bad)

    def test_a_leading_dot_is_refused(self):
        """.gitignore.json is a file the directory owner did not ask for."""
        self.assertIsNone(self.path(".hidden"))

    def test_empty_absurd_and_non_string_ids_are_refused(self):
        for bad in ("", "x" * 65, None, 279495, [], {"job_id": "a"}):
            self.assertIsNone(self.path(bad), repr(bad))

    def test_a_long_but_legal_id_is_allowed(self):
        self.assertIsNotNone(self.path("j" * 64))


def _policy_dir(tmp, **files):
    for name, body in files.items():
        with open(os.path.join(tmp, name + ".py"), "w") as handle:
            handle.write(textwrap.dedent(body))
    registry = gateway.PolicyDir(tmp)
    registry.reload()
    return registry


GOOD = """
    def select(accepting):
        return sorted(accepting)[-1]
"""
RAISES = """
    def select(accepting):
        raise RuntimeError("nope")
"""
LIES = """
    def select(accepting):
        return "not-a-backend"
"""

ACCEPTING = {"a": (1, 1, 7200), "b": (5, 5, 7200)}


class PolicyLoading(unittest.TestCase):
    """Custom policies come from files, and a bad file costs only itself."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_policy_file_supplies_a_policy_named_after_it(self):
        registry = _policy_dir(self.tmp, pick_last=GOOD)
        self.assertEqual(("pick_last",), registry.names())
        self.assertEqual("b", registry.run("pick_last", ACCEPTING))

    def test_a_file_without_select_is_not_a_policy(self):
        registry = _policy_dir(self.tmp, empty="x = 1\n")
        self.assertEqual((), registry.names())

    def test_a_file_that_fails_to_import_costs_only_itself(self):
        registry = _policy_dir(self.tmp, broken="def select(  # unclosed\n", fine=GOOD)
        self.assertEqual(("fine",), registry.names())

    def test_a_policy_may_not_shadow_a_built_in(self):
        registry = _policy_dir(self.tmp, round_robin=GOOD)
        self.assertEqual((), registry.names())

    def test_underscore_files_are_helpers_not_policies(self):
        registry = _policy_dir(self.tmp, _shared=GOOD, real=GOOD)
        self.assertEqual(("real",), registry.names())

    def test_deleting_the_file_withdraws_the_policy(self):
        registry = _policy_dir(self.tmp, gone=GOOD)
        os.remove(os.path.join(self.tmp, "gone.py"))
        registry.reload()
        self.assertEqual((), registry.names())
        self.assertIsNone(registry.run("gone", ACCEPTING))


class PolicyFailure(unittest.TestCase):
    """A policy runs on the request path, so it is contained, not trusted."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_raising_policy_returns_none_rather_than_propagating(self):
        registry = _policy_dir(self.tmp, bad=RAISES)
        self.assertIsNone(registry.run("bad", ACCEPTING))

    def test_a_policy_naming_a_backend_that_is_not_on_offer_is_refused(self):
        registry = _policy_dir(self.tmp, liar=LIES)
        self.assertIsNone(registry.run("liar", ACCEPTING))

    def test_repeated_failure_disables_the_policy(self):
        registry = _policy_dir(self.tmp, bad=RAISES)
        for _ in range(gateway.PolicyDir.strikes):
            registry.run("bad", ACCEPTING)
        self.assertIn("bad", registry.disabled)
        self.assertEqual((), registry.names())

    def test_a_working_policy_is_never_struck(self):
        registry = _policy_dir(self.tmp, fine=GOOD)
        for _ in range(10):
            self.assertEqual("b", registry.run("fine", ACCEPTING))
        self.assertEqual(set(), registry.disabled)

    def test_editing_the_file_clears_the_strikes(self):
        """Fixing a policy is how you re-enable it; restarting is not."""
        registry = _policy_dir(self.tmp, bad=RAISES)
        for _ in range(gateway.PolicyDir.strikes):
            registry.run("bad", ACCEPTING)
        self.assertIn("bad", registry.disabled)
        path = os.path.join(self.tmp, "bad.py")
        with open(path, "w") as handle:
            handle.write(textwrap.dedent(GOOD))
        os.utime(path, (0, 0))  # any change, not a later one
        registry.reload()
        self.assertEqual(("bad",), registry.names())
        self.assertEqual("b", registry.run("bad", ACCEPTING))


class PolicyInRouter(unittest.TestCase):
    """Placement uses a custom policy, and survives one that misbehaves."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def router(self, policy, **files):
        router = gateway.Router(
            1800, 100, policy=policy, state_path=None, key_sources=list(gateway.DEFAULT_KEY_SOURCES)
        )
        router.policies = _policy_dir(self.tmp, **files)
        return router

    def test_a_custom_policy_places_the_conversation(self):
        router = self.router("pick_last", pick_last=GOOD)
        self.assertEqual("b", router.select(ACCEPTING))

    def test_known_policies_lists_built_ins_and_custom_together(self):
        router = self.router("pick_last", pick_last=GOOD)
        known = gateway.known_policies(router)
        self.assertIn("least_conversations", known)
        self.assertIn("pick_last", known)

    def test_a_failing_custom_policy_falls_back_to_the_default(self):
        """The conversation that arrives during a bad policy still gets placed."""
        router = self.router("bad", bad=RAISES)
        self.assertEqual("a", router.select(ACCEPTING))  # least_conversations

    def test_a_policy_that_is_not_loaded_falls_back_rather_than_raising(self):
        router = self.router("never_written")
        self.assertEqual("a", router.select(ACCEPTING))

    def test_built_ins_still_work_with_a_policy_dir_present(self):
        router = self.router("least_inflight", pick_last=GOOD)
        self.assertEqual("a", router.select(ACCEPTING))


class PreemptionRecovery(unittest.TestCase):
    """Losing a node to the scheduler must be survivable without an operator.

    `revive_dead_backends` cannot cover this. It acts on the deployment's own
    "exited" state, which only a live controller writes -- and preemption takes
    the controller with it. These tests pin both halves: that every way a node
    disappears is recorded, and that recovery refuses to act while SLURM still
    owns the job, because PreemptMode here is REQUEUE and resubmitting on top
    of a requeue both duplicates the instance and burns the exemption already
    earned.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.users = os.path.join(self.tmp, "users.txt")
        with open(self.users, "w") as handle:
            handle.write("tester\n")
        self.config = os.path.join(self.tmp, "fleet.yaml")
        with open(self.config, "w") as handle:
            handle.write("instances: []\n")
        self.fleet_dir = os.path.join(self.tmp, "fleet")
        os.makedirs(self.fleet_dir)

    def args(self, *extra):
        # Built through parse_args rather than by hand: the defaults are half
        # the behaviour under test, and a hand-made Namespace silently invents
        # its own.
        return gateway.parse_args(
            [
                "--fleet-dir",
                self.fleet_dir,
                "--users",
                self.users,
                "--no-relay",
                *extra,
            ]
        )

    def register(self, job_id="500", state="running attempt 1", heartbeat_age=0.0, run_dir=None):
        record = {
            "job_id": job_id,
            "url": "http://node-a:8400",
            "run_dir": run_dir or "/runs/2026-09/11/junyix_091100_%s_kffleet_i07" % job_id,
            "state": state,
            "end_time": time.time() + 3600,
            "heartbeat": time.time() - heartbeat_age,
        }
        path = os.path.join(self.fleet_dir, "%s.json" % job_id)
        with open(path, "w") as handle:
            json.dump(record, handle)
        return path

    def sweep(self, fleet, squeue=None, fleetctl_rc=0, now=None):
        """One supervise() pass with the scheduler and launcher stubbed out."""
        calls = {"squeue": [], "fleetctl": [], "serve_sh": []}

        async def fake_status(job_id):
            calls["squeue"].append(job_id)
            return squeue

        async def fake_fleetctl(_fleet, *argv):
            calls["fleetctl"].append(argv)
            return fleetctl_rc, "brought up i07"

        async def fake_serve_sh(_fleet, *argv):
            calls["serve_sh"].append(argv)
            return 0, ""

        saved = (gateway.slurm_job_status, gateway.run_fleetctl, gateway.run_serve_sh)
        gateway.slurm_job_status = fake_status
        gateway.run_fleetctl = fake_fleetctl
        gateway.run_serve_sh = fake_serve_sh
        try:
            if now is None:
                asyncio.run(gateway.supervise(fleet))
            else:
                asyncio.run(gateway.recover_lost_backends(fleet, now))
        finally:
            (gateway.slurm_job_status, gateway.run_fleetctl, gateway.run_serve_sh) = saved
        return calls

    # -- every way a node goes away lands in `lost` ------------------------
    def test_sigterm_preemption_is_recorded(self):
        """serve.sh traps TERM and clear_fleet deletes the registration."""
        path = self.register()
        fleet = gateway.Fleet(self.args())
        fleet.discover()
        self.assertIn("500", fleet.backends)
        os.unlink(path)
        fleet.discover()
        self.assertNotIn("500", fleet.backends)
        self.assertIn("500", fleet.lost)

    def test_sigkill_preemption_is_recorded(self):
        """No trap runs, so the file survives and only the heartbeat stops."""
        self.register(heartbeat_age=0)
        fleet = gateway.Fleet(self.args())
        fleet.discover()
        self.register(heartbeat_age=120)  # rewrite it with a stale heartbeat
        fleet.backends["500"].healthy = False
        fleet.discover()
        self.assertNotIn("500", fleet.backends)
        self.assertIn("500", fleet.lost)

    def test_allocation_gone_is_recorded(self):
        """The controller notices its own allocation went and clears the file."""
        path = self.register(state="allocation gone; controller exiting")
        fleet = gateway.Fleet(self.args())
        fleet.discover()
        os.unlink(path)
        fleet.discover()
        self.assertIn("500", fleet.lost)

    def test_a_backend_that_comes_back_clears_lost(self):
        path = self.register()
        fleet = gateway.Fleet(self.args())
        fleet.discover()
        os.unlink(path)
        fleet.discover()
        self.assertIn("500", fleet.lost)
        self.register()  # SLURM requeued it and it re-registered
        fleet.discover()
        self.assertNotIn("500", fleet.lost)

    # -- recovery refuses to race the scheduler ---------------------------
    def lost_fleet(self, *extra):
        path = self.register()
        fleet = gateway.Fleet(self.args("--fleet-config", self.config, *extra))
        fleet.discover()
        os.unlink(path)
        fleet.discover()
        return fleet

    def test_a_requeued_job_is_not_resubmitted(self):
        """The whole point: REQUEUE keeps the job id, so squeue still has it."""
        fleet = self.lost_fleet()
        calls = self.sweep(fleet, squeue=("PENDING", "Resources"), now=time.time() + 3600)
        self.assertEqual(["500"], calls["squeue"])
        self.assertEqual([], calls["fleetctl"])
        self.assertIn("500", fleet.lost)

    def test_an_orphaned_job_reconciles_the_fleet(self):
        fleet = self.lost_fleet()
        calls = self.sweep(fleet, squeue=("GONE", ""), now=time.time() + 3600)
        self.assertEqual([("up",)], calls["fleetctl"])
        self.assertNotIn("500", fleet.lost)

    def test_recovery_waits_out_the_grace_period(self):
        fleet = self.lost_fleet()
        calls = self.sweep(fleet, squeue=("GONE", ""), now=time.time())
        self.assertEqual([], calls["squeue"])
        self.assertEqual([], calls["fleetctl"])

    def test_an_unanswerable_squeue_defers(self):
        """Never guess: a query that failed is not evidence the job is gone."""
        fleet = self.lost_fleet()
        calls = self.sweep(fleet, squeue=None, now=time.time() + 3600)
        self.assertEqual([], calls["fleetctl"])
        self.assertIn("500", fleet.lost)

    def test_recovery_is_off_without_a_fleet_config(self):
        path = self.register()
        fleet = gateway.Fleet(self.args())
        fleet.discover()
        os.unlink(path)
        fleet.discover()
        calls = self.sweep(fleet, squeue=("GONE", ""), now=time.time() + 3600)
        self.assertEqual([], calls["squeue"])
        self.assertEqual([], calls["fleetctl"])

    def test_the_cooldown_gates_a_second_reconciliation(self):
        fleet = self.lost_fleet()
        base = time.time() + 3600
        self.sweep(fleet, squeue=("GONE", ""), now=base)
        fleet.lost["501"] = ("/runs/501_kffleet_i08", base)
        calls = self.sweep(fleet, squeue=("GONE", ""), now=base + 60)
        self.assertEqual([], calls["fleetctl"])

    def test_a_job_that_never_returns_is_given_up_on(self):
        fleet = self.lost_fleet("--recover-limit", "2", "--recover-cooldown", "0")
        base = time.time() + 3600
        for attempt in range(4):
            # A successful reconcile drops it; it stays missing, so the next
            # sweep sees it again. Retired well in the past so the grace period
            # is never what is being tested here.
            fleet.lost.setdefault("500", ("/runs/500_kffleet_i07", base - 1000))
            self.sweep(fleet, squeue=("GONE", ""), now=base + attempt * 10)
        self.assertEqual(3, fleet.recovered["500"][0])  # 2 attempts, then muted

    def test_a_failed_fleetctl_keeps_the_job_for_retry(self):
        fleet = self.lost_fleet()
        calls = self.sweep(fleet, squeue=("GONE", ""), fleetctl_rc=1, now=time.time() + 3600)
        self.assertEqual([("up",)], calls["fleetctl"])
        self.assertIn("500", fleet.lost)

    # -- what --no-relay does and does not switch off ----------------------
    def test_no_relay_still_recovers(self):
        """Recovery runs ahead of the --no-relay return, unlike relay itself."""
        fleet = self.lost_fleet()
        fleet.lost["500"] = (fleet.lost["500"][0], time.time() - 3600)
        calls = self.sweep(fleet, squeue=("GONE", ""))
        self.assertEqual([("up",)], calls["fleetctl"])

    def test_no_relay_still_submits_nothing_itself(self):
        """Recovery goes through fleetctl; the gateway never submits directly."""
        fleet = self.lost_fleet()
        fleet.lost["500"] = (fleet.lost["500"][0], time.time() - 3600)
        fleet.ever_active = True
        calls = self.sweep(fleet, squeue=("GONE", ""))
        self.assertEqual([], [c for c in calls["serve_sh"] if c[:1] == ("submit",)])

    # -- fleetctl may not live on this host ---------------------------------
    def test_a_remote_fleet_config_is_accepted(self):
        """Fleetctl runs where the scheduler is, which need not be here.

        Then --fleet-config names a path on *that* host, and checking it
        locally refuses a correct configuration at startup.
        """
        args = self.args("--fleet-config", "/on/the/login/node/fleet.yaml")
        self.assertEqual("/on/the/login/node/fleet.yaml", args.fleet_config)

    def test_a_missing_fleetctl_is_still_refused(self):
        """This one *is* executed here, so its absence is knowable now."""
        with self.assertRaises(SystemExit):
            self.args("--fleet-config", self.config, "--fleetctl", "/no/such/fleetctl")

    def startup_check(self, fleetctl_rc, squeue_rc=0, recovery=True):
        """Run check_recovery with both of its probes stubbed.

        Returns (fleetctl calls, slurm calls) so a test can assert not just
        that the check passed but which halves of it were exercised.
        """
        extra = ("--fleet-config", self.config) if recovery else ()
        fleet = gateway.Fleet(self.args(*extra))
        fleetctl_calls, slurm_calls = [], []

        async def fake_fleetctl(_fleet, *argv):
            fleetctl_calls.append(argv)
            return fleetctl_rc, "squeue: command not found" if fleetctl_rc else ""

        async def fake_slurm(*argv):
            slurm_calls.append(argv)
            return squeue_rc, "squeue: command not found" if squeue_rc else ""

        originals = gateway.run_fleetctl, gateway.run_slurm_command
        gateway.run_fleetctl, gateway.run_slurm_command = fake_fleetctl, fake_slurm
        try:
            asyncio.run(gateway.check_recovery(fleet))
        finally:
            gateway.run_fleetctl, gateway.run_slurm_command = originals
        return fleetctl_calls, slurm_calls

    def test_startup_proves_recovery_could_run(self):
        fleetctl_calls, slurm_calls = self.startup_check(0)
        self.assertEqual([("status",)], fleetctl_calls)
        self.assertEqual(1, len(slurm_calls))

    def test_startup_warns_rather_than_refusing_to_serve(self):
        """A gateway that cannot recover should still route.

        The scheduler being unreachable is not a reason to stop answering, and
        recovery retries on its own. The point of the check is that the failure
        is visible at startup instead of at the first preemption hours later.
        """
        self.assertEqual([("status",)], self.startup_check(127)[0])

    def test_startup_also_proves_the_scheduler_can_be_asked(self):
        """The half that was missing, and the reason the check was misleading.

        Recovery is a query and an action travelling by different routes:
        `fleetctl up` submits, but nothing is submitted until squeue says the
        job is gone, and squeue was run directly rather than through fleetctl.
        A wrapper on only the action leaves the gateway announcing recovery is
        ready and then never recovering, because the query it gates on can
        never answer. So a working fleetctl must not be enough to pass.
        """
        fleetctl_calls, slurm_calls = self.startup_check(0, squeue_rc=127)
        self.assertEqual([("status",)], fleetctl_calls)
        self.assertEqual(1, len(slurm_calls), "squeue must be probed, not assumed")

    def test_the_scheduler_is_not_probed_when_fleetctl_already_failed(self):
        """No point asking; the second warning would only bury the first."""
        self.assertEqual([], self.startup_check(127)[1])

    def test_the_check_is_skipped_when_recovery_is_off(self):
        self.assertEqual(([], []), self.startup_check(0, recovery=False))

    def test_slurm_commands_go_through_the_wrapper_when_set(self):
        """Off-cluster, `squeue` is not a thing this host has.

        The bug this covers was silent: exec of a missing squeue returns
        "cannot tell", every caller correctly declines to act on an answer it
        did not get, and recovery does nothing forever without logging that it
        is doing nothing.
        """
        seen = []

        async def fake_exec(*command, **_kwargs):
            seen.append(command)
            raise OSError("no such file")

        original_exec = asyncio.create_subprocess_exec
        original_wrapper = gateway.SLURM_WRAPPER
        asyncio.create_subprocess_exec = fake_exec
        try:
            gateway.SLURM_WRAPPER = ""
            asyncio.run(gateway.run_slurm_command("squeue", "-j", "1"))
            gateway.SLURM_WRAPPER = "/srv/slurm-remote"
            asyncio.run(gateway.run_slurm_command("squeue", "-j", "1"))
        finally:
            asyncio.create_subprocess_exec = original_exec
            gateway.SLURM_WRAPPER = original_wrapper

        self.assertEqual(("squeue", "-j", "1"), seen[0])
        self.assertEqual(("/srv/slurm-remote", "squeue", "-j", "1"), seen[1])

    def test_a_missing_slurm_wrapper_is_refused(self):
        """Unlike --fleet-config, this one is executed here, so check it."""
        with self.assertRaises(SystemExit):
            self.args("--slurm-wrapper", "/no/such/wrapper")

    def test_a_label_is_recovered_from_the_run_directory(self):
        self.assertEqual(
            "i07", gateway.instance_label("/runs/2026-09/11/junyix_091100_500_kffleet_i07")
        )
        self.assertEqual("", gateway.instance_label(""))


class ProbeTarget(unittest.TestCase):
    """What the gateway asks a backend before believing in it.

    The regression this pins cost half an hour of a fleet answering 404s: a
    container registry on one of the GPU nodes' ports returns 200 OK to
    /health, so the gateway elected it as a backend while the real worker had
    already died on "Address already in use".
    """

    def request_line(self, status=b"HTTP/1.1 200 OK\r\n"):
        """Run one probe against a stubbed connection; return (line, result)."""
        sent = []

        class Writer:
            def write(self, data):
                sent.append(data)

            async def drain(self):
                pass

            def close(self):
                pass

            async def wait_closed(self):
                pass

        class Reader:
            async def readline(self):
                return status

        async def fake_open_connection(host, port):
            return Reader(), Writer()

        backend = gateway.Backend(
            {
                "job_id": "job",
                "url": "http://h:1",
                "end_time": 0.0,
                "heartbeat": 0.0,
            }
        )
        original = asyncio.open_connection
        asyncio.open_connection = fake_open_connection
        try:
            result = asyncio.run(gateway.probe(backend, 1.0))
        finally:
            asyncio.open_connection = original
        return b"".join(sent), result

    def test_it_asks_for_the_openai_route_not_health(self):
        """/health is answerable by anything; /v1/models is not.

        Any process at that address can return 200 to /health, and one on this
        cluster does. /v1/models is 404 until the OpenAI routes are mounted, so
        it proves both that something is listening and that it is ours.
        """
        line, _ = self.request_line()
        self.assertIn(b"GET /v1/models ", line)
        self.assertNotIn(b"GET /health ", line)

    def test_a_404_is_not_healthy(self):
        """The squatter's shape: listening, answering, unable to serve."""
        _, result = self.request_line(b"HTTP/1.1 404 Not Found\r\n")
        self.assertEqual("dead", result)

    def test_a_200_is_healthy(self):
        _, result = self.request_line()
        self.assertEqual("ok", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class Mirroring(unittest.TestCase):
    """One backend's requests copied to several places at once.

    Mirroring began single-target: job id -> (host, port), a plain overwrite.
    Multi-target changes the meaning of three operations that used to be
    unambiguous -- POST (start vs add), stop (all vs one), and auto-disable
    (the whole job vs just the dead target) -- and each wrong reading is
    silent, so each is pinned here.
    """

    def fleet(self, healthy=True, max_misses=3, state_path=None):
        args = argparse.Namespace(
            new_conversation_margin=1800,
            sticky_ttl=1800,
            sticky_capacity=100,
            users="/nonexistent",
            fleet_dir=tempfile.mkdtemp(),
            stale_after=30,
            route_policy="least_conversations",
            router_state=state_path,
            key_sources=None,
            no_relay=True,
            probe_timeout=0.5,
            mirror_max_misses=max_misses,
        )
        fleet = gateway.Fleet(args)
        now = time.time()
        backend = gateway.Backend(
            {
                "job_id": "500",
                "url": "http://node-a:8400",
                "run_dir": "/run/500",
                "state": "running attempt 1",
                "end_time": now + 3600,
                "heartbeat": now,
            }
        )
        backend.healthy = healthy
        fleet.backends["500"] = backend
        return fleet

    def control(self, fleet, payload, probe="ok"):
        """One call to the mirror control endpoint, probe faked."""
        body = json.dumps(payload).encode()
        headers = [("content-length", str(len(body)))]
        handler = types.SimpleNamespace(fleet=fleet)

        async def fake_probe(_host, _port, _timeout):
            return probe

        async def call():
            # The reader is built inside the loop: StreamReader() outside one
            # raises on 3.12, and the endpoint reads the body through it.
            reader = asyncio.StreamReader()
            reader.feed_data(body)
            reader.feed_eof()
            return await gateway.Gateway.control(handler, "/_gateway/mirror", b"", reader, headers)

        original = gateway.probe_addr
        gateway.probe_addr = fake_probe
        try:
            return asyncio.run(call())
        finally:
            gateway.probe_addr = original

    def test_a_second_target_is_added_beside_the_first_not_in_place_of_it(self):
        fleet = self.fleet()
        self.control(fleet, {"job_id": "500", "target": "shadow-a:9000"})
        code, reply = self.control(fleet, {"job_id": "500", "target": "shadow-b:9001"})
        self.assertEqual(200, code)
        # The overwrite reading -- the single-target behaviour -- would leave
        # only shadow-b here, and would have ended the first experiment with a
        # 200 that looks exactly like success.
        self.assertEqual(["shadow-a:9000", "shadow-b:9001"], reply["mirroring"])
        self.assertEqual([("shadow-a", 9000), ("shadow-b", 9001)], fleet.mirrors["500"])

    def test_repeating_a_target_says_so_instead_of_doubling_the_copies(self):
        fleet = self.fleet()
        self.control(fleet, {"job_id": "500", "target": "shadow-a:9000"})
        code, reply = self.control(fleet, {"job_id": "500", "target": "shadow-a:9000"})
        self.assertEqual(200, code)
        self.assertTrue(reply["already"])
        # A duplicate entry would send two copies of every request to one
        # place, which reads on the far side as double the load it was sent.
        self.assertEqual([("shadow-a", 9000)], fleet.mirrors["500"])

    def test_stopping_one_target_leaves_the_other_running(self):
        fleet = self.fleet()
        self.control(fleet, {"job_id": "500", "target": "shadow-a:9000"})
        self.control(fleet, {"job_id": "500", "target": "shadow-b:9001"})
        code, reply = self.control(
            fleet, {"job_id": "500", "target": "shadow-a:9000", "stop": True}
        )
        self.assertEqual(200, code)
        self.assertEqual(["shadow-b:9001"], reply["mirroring"])
        self.assertEqual([("shadow-b", 9001)], fleet.mirrors["500"])

    def test_stopping_the_last_target_removes_the_job_entirely(self):
        fleet = self.fleet()
        self.control(fleet, {"job_id": "500", "target": "shadow-a:9000"})
        self.control(fleet, {"job_id": "500", "target": "shadow-a:9000", "stop": True})
        # An empty list left behind would read as "mirrored to nowhere" on
        # every fleet report from now on.
        self.assertNotIn("500", fleet.mirrors)

    def test_stopping_a_target_that_is_not_there_is_a_404_not_a_shrug(self):
        fleet = self.fleet()
        self.control(fleet, {"job_id": "500", "target": "shadow-a:9000"})
        code, reply = self.control(
            fleet, {"job_id": "500", "target": "shadow-x:9099", "stop": True}
        )
        self.assertEqual(404, code)
        self.assertEqual(["shadow-a:9000"], reply["mirroring"])
        self.assertEqual([("shadow-a", 9000)], fleet.mirrors["500"])

    def test_a_null_target_still_stops_everything(self):
        # The original stop, spoken by every existing caller. It has to keep
        # meaning "all of them" now that there can be more than one.
        fleet = self.fleet()
        self.control(fleet, {"job_id": "500", "target": "shadow-a:9000"})
        self.control(fleet, {"job_id": "500", "target": "shadow-b:9001"})
        code, reply = self.control(fleet, {"job_id": "500", "target": None})
        self.assertEqual(200, code)
        self.assertEqual([], reply["mirroring"])
        self.assertEqual(["shadow-a:9000", "shadow-b:9001"], reply["was"])
        self.assertNotIn("500", fleet.mirrors)

    def test_an_unreachable_target_is_refused_and_force_overrides(self):
        fleet = self.fleet()
        code, _ = self.control(fleet, {"job_id": "500", "target": "shadow-a:9000"}, probe="dead")
        self.assertEqual(502, code)
        self.assertNotIn("500", fleet.mirrors)
        code, _ = self.control(
            fleet,
            {"job_id": "500", "target": "shadow-a:9000", "force": True},
            probe="dead",
        )
        self.assertEqual(200, code)
        self.assertEqual([("shadow-a", 9000)], fleet.mirrors["500"])

    def test_auto_disable_removes_only_the_dead_target(self):
        fleet = self.fleet(max_misses=2)
        fleet.mirrors["500"] = [("shadow-a", 9000), ("shadow-b", 9001)]
        fleet.mirrors["501"] = [("shadow-a", 9000)]
        for _ in range(2):
            gateway._mirror_missed(fleet, ("shadow-a", 9000), "refused")
        # shadow-a is gone from both jobs; shadow-b keeps running -- the
        # healthy half of the experiment is the half still worth having. And
        # 501, whose only target died, is removed outright rather than left
        # as an empty entry.
        self.assertEqual([("shadow-b", 9001)], fleet.mirrors["500"])
        self.assertNotIn("501", fleet.mirrors)

    def test_a_success_on_one_target_does_not_reset_anothers_misses(self):
        fleet = self.fleet(max_misses=3)
        gateway._mirror_missed(fleet, ("shadow-a", 9000), "refused")
        gateway._mirror_missed(fleet, ("shadow-a", 9000), "refused")
        # What mirror_request does on a 2xx, for the other target.
        fleet.mirror_misses.pop(("shadow-b", 9001), None)
        self.assertEqual(2, fleet.mirror_misses[("shadow-a", 9000)])

    def test_per_target_counters_split_what_the_totals_blur(self):
        fleet = self.fleet()
        gateway._mirror_count(fleet, ("shadow-a", 9000), "sent")
        gateway._mirror_count(fleet, ("shadow-a", 9000), "sent")
        gateway._mirror_count(fleet, ("shadow-b", 9001), "failed")
        self.assertEqual(2, fleet.mirror_stats["sent"])
        self.assertEqual(1, fleet.mirror_stats["failed"])
        self.assertEqual(2, fleet.mirror_target_stats["shadow-a:9000"]["sent"])
        self.assertEqual(1, fleet.mirror_target_stats["shadow-b:9001"]["failed"])
        self.assertNotIn("failed", fleet.mirror_target_stats["shadow-a:9000"])

    def test_mirrors_survive_a_restart_in_either_format(self):
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        first = gateway.Router(ttl=1800, capacity=100, state_path=path)
        first.mirrors = {"500": [("shadow-a", 9000), ("shadow-b", 9001)]}
        first.dirty = True
        first.save()
        second = gateway.Router(ttl=1800, capacity=100, state_path=path)
        second.load()
        self.assertEqual({"500": [("shadow-a", 9000), ("shadow-b", 9001)]}, second.mirrors)

    def test_a_state_file_from_the_single_target_gateway_still_restores(self):
        # A live handover starts the successor on new code while the state was
        # written by old code, so the old shape arrives at every upgrade --
        # it is not a one-time migration.
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        state = {
            "version": 1,
            "saved_at": time.time(),
            "mirrors": {"500": ["shadow-a", 9000]},
            "pins": {},
        }
        with open(path, "w") as handle:
            json.dump(state, handle)
        router = gateway.Router(ttl=1800, capacity=100, state_path=path)
        router.load()
        self.assertEqual({"500": [("shadow-a", 9000)]}, router.mirrors)

    def test_a_garbled_mirror_entry_drops_that_target_not_the_restore(self):
        path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        state = {
            "version": 1,
            "mirrors": {
                "500": [["shadow-a", 9000], ["broken"], ["shadow-b", "no"]],
                "501": "nonsense",
            },
            "pins": {},
        }
        with open(path, "w") as handle:
            json.dump(state, handle)
        router = gateway.Router(ttl=1800, capacity=100, state_path=path)
        router.load()
        self.assertEqual({"500": [("shadow-a", 9000)]}, router.mirrors)

    def test_the_fleet_report_shape_is_a_list_per_job(self):
        fleet = self.fleet()
        fleet.mirrors["500"] = [("shadow-a", 9000)]
        self.assertEqual({"500": ["shadow-a:9000"]}, gateway.mirror_table(fleet.mirrors))


# ---------------------------------------------------------------------------
# Pools: more than one fleet directory
# ---------------------------------------------------------------------------
# A second model's fleet joins the gateway as an extra pool: a fleet directory
# of its own, supervised through a fleet config of its own. Routing is not
# per pool -- every backend from every directory is one routing pool -- so the
# properties below are two: routing treats every pool's backends alike, and
# lifecycle actions on a pool's jobs go through that pool's config alone.
K3 = "kimi-k3"


class _CapturingWriter:
    """Enough of an asyncio StreamWriter for the introspection endpoints."""

    def __init__(self):
        self.data = b""

    def write(self, data):
        self.data += data

    async def drain(self):
        pass

    def close(self):
        pass

    async def wait_closed(self):
        pass

    def json(self):
        return json.loads(self.data.partition(b"\r\n\r\n")[2])

    def status(self):
        return int(self.data.split(b" ", 2)[1])


class PoolFixture(unittest.TestCase):
    """A default fleet directory and a kimi-k3 one, side by side."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        gw_dir = os.path.join(self.tmp, "gw")
        os.makedirs(gw_dir)
        self.users = os.path.join(gw_dir, "users.txt")
        with open(self.users, "w") as handle:
            handle.write("tester\n")
        self.pools_file = os.path.join(gw_dir, "pools.conf")
        self.fleet_dir = os.path.join(self.tmp, "var", "_fleet", "kffleet_glm5.3")
        self.k3_dir = os.path.join(self.tmp, "var", "_fleet", "kffleet_kimi-k3")
        os.makedirs(self.fleet_dir)
        os.makedirs(self.k3_dir)
        self.glm_config = os.path.join(self.tmp, "fleet.yaml")
        self.k3_config = os.path.join(self.tmp, "fleet_k3.yaml")
        for path in (self.glm_config, self.k3_config):
            with open(path, "w") as handle:
                handle.write("defaults: {}\ninstances: []\n")

    def args(self, *extra, pool=True, k3_config=False):
        argv = ["--fleet-dir", self.fleet_dir, "--users", self.users]
        if "--yaml" not in extra:
            argv.append("--no-relay")
        if pool:
            spec = "%s=%s" % (K3, self.k3_dir)
            if k3_config:
                spec += "," + self.k3_config
            argv += ["--pool", spec]
        return gateway.parse_args(argv + list(extra))

    def fleet(self, *extra, **kwargs):
        args = self.args("--router-state", "", *extra, **kwargs)
        return gateway.Fleet(args)

    def backend(
        self,
        fleet,
        job_id,
        pool,
        label="i00",
        end_in=86400,
        healthy=True,
        state="running attempt 1",
    ):
        model = "glm5.3" if pool == gateway.DEFAULT_POOL else K3
        now = time.time()
        backend = gateway.Backend(
            {
                "job_id": job_id,
                "url": "http://127.0.0.1:%d" % (9000 + len(fleet.backends)),
                "run_dir": "/var/2026-10/06/junyix_100612_%s_kffleet_%s_%s"
                % (job_id, model, label),
                "state": state,
                "end_time": now + end_in,
                "heartbeat": now,
            },
            pool=pool,
        )
        backend.healthy = healthy
        backend.healthy_since = now - 3600 if healthy else 0.0
        fleet.backends[job_id] = backend
        fleet.inflight.setdefault(job_id, 0)
        return backend

    def register(self, directory, job_id, heartbeat_age=0.0):
        now = time.time()
        path = os.path.join(directory, "%s.json" % job_id)
        with open(path, "w") as handle:
            json.dump(
                {
                    "job_id": job_id,
                    "url": "http://127.0.0.1:%d" % (9000 + int(job_id) % 1000),
                    "run_dir": "/var/runs/%s" % job_id,
                    "state": "running attempt 1",
                    "end_time": now + 86400,
                    "heartbeat": now - heartbeat_age,
                },
                handle,
            )
        return path


class PoolConfiguration(PoolFixture):
    """--pool NAME=FLEET_DIR[,FLEET_CONFIG], or the same per line in a file."""

    def test_a_spec_names_a_directory_and_optionally_a_config(self):
        spec = gateway.parse_pool_spec("kimi-k3=/var/_fleet/kffleet_kimi-k3,/d/fleet_k3.yaml")
        self.assertEqual(K3, spec.name)
        self.assertEqual("/var/_fleet/kffleet_kimi-k3", spec.fleet_dir)
        self.assertEqual("/d/fleet_k3.yaml", spec.fleet_config)
        bare = gateway.parse_pool_spec("kimi-k3=/var/_fleet/kffleet_kimi-k3")
        self.assertEqual("", bare.fleet_config)

    def test_unusable_specs_are_refused(self):
        for bad in (
            "",
            "kimi-k3",
            "=/d",
            "kimi-k3=",
            "kimi k3=/d",
            "default=/d",
            "_gateway=/d",
            "a/b=/d",
            "k3=/d,",
            "k3=/d,/c1,/c2",
        ):
            with self.assertRaises(ValueError, msg=repr(bad)):
                gateway.parse_pool_spec(bad)

    def test_flags_become_pools(self):
        args = self.args(k3_config=True)
        self.assertEqual([K3], [pool.name for pool in args.pools])
        self.assertEqual(self.k3_config, args.pools[0].fleet_config)

    def test_a_pool_may_not_share_a_fleet_directory(self):
        self.assertEqual([K3], [pool.name for pool in self.args().pools])
        with self.assertRaises(SystemExit):
            gateway.parse_args(
                ["--fleet-dir", self.fleet_dir, "--users", self.users, "--no-relay"]
                + ["--pool", "%s=%s" % (K3, self.fleet_dir)]
            )
        with self.assertRaises(SystemExit):
            self.args("--pool", "other=%s" % self.k3_dir)

    def test_a_name_defined_twice_must_say_the_same_thing(self):
        other = os.path.join(self.tmp, "other")
        accepted = self.args("--pool", "other=%s" % other)
        self.assertEqual([K3, "other"], [pool.name for pool in accepted.pools])
        self.assertEqual(
            [K3], [pool.name for pool in self.args("--pool", "%s=%s" % (K3, self.k3_dir)).pools]
        )
        with self.assertRaises(SystemExit):
            self.args("--pool", "%s=%s" % (K3, other))

    def test_the_pools_file_beside_the_users_file_is_read(self):
        with open(self.pools_file, "w") as handle:
            handle.write("# extra pools\n\n%s=%s,%s\n" % (K3, self.k3_dir, self.k3_config))
        args = self.args(pool=False)
        self.assertEqual([K3], [pool.name for pool in args.pools])
        self.assertEqual(self.k3_config, args.pools[0].fleet_config)

    def test_a_pool_in_both_the_file_and_a_flag_must_agree(self):
        with open(self.pools_file, "w") as handle:
            handle.write("%s=%s\n" % (K3, self.k3_dir))
        self.assertEqual([K3], [pool.name for pool in self.args().pools])
        with self.assertRaises(SystemExit):
            self.args(k3_config=True)

    def test_a_broken_pools_file_refuses_to_start(self):
        """A handover successor that refuses to start is aborted; one that guesses is not."""
        with open(self.pools_file, "w") as handle:
            handle.write("kimi-k3\n")
        with self.assertRaises(SystemExit):
            self.args(pool=False)

    def test_the_pools_file_can_be_switched_off(self):
        with open(self.pools_file, "w") as handle:
            handle.write("%s=%s\n" % (K3, self.k3_dir))
        self.assertEqual([], self.args("--pools-file", "", pool=False).pools)

    def test_without_flags_or_a_file_there_are_no_pools(self):
        args = self.args(pool=False)
        self.assertEqual([], args.pools)
        self.assertEqual({}, gateway.Fleet(args).pools)

    def test_a_pool_config_needs_a_runnable_fleetctl(self):
        self.assertEqual(self.k3_config, self.args(k3_config=True).pools[0].fleet_config)
        with self.assertRaises(SystemExit):
            self.args("--fleetctl", "/no/such/fleetctl", k3_config=True)


class UnifiedRouting(PoolFixture):
    """New conversations are placed across every pool; see PoolAffinity for what follows."""

    def make(self):
        fleet = self.fleet()
        for job in ("101", "102"):
            self.backend(fleet, job, gateway.DEFAULT_POOL)
        for job in ("201", "202"):
            self.backend(fleet, job, K3)
        return fleet

    @staticmethod
    def route(fleet, key, exclude=()):
        return gateway.Gateway(fleet).route(key, exclude=exclude)

    def test_every_pools_backends_are_offered_alike(self):
        fleet = self.make()
        self.assertEqual({"101", "102", "201", "202"}, set(fleet.accepting()))
        self.assertEqual({"101", "102", "201", "202"}, fleet.serving())

    def test_new_conversations_spread_over_every_pool(self):
        fleet = self.make()
        placed = [self.route(fleet, "hdr:c%d" % n) for n in range(8)]
        self.assertEqual({"101": 2, "102": 2, "201": 2, "202": 2}, tally(placed))

    def test_a_pin_holds_whichever_pool_its_backend_is_in(self):
        fleet = self.make()
        homes = {key: self.route(fleet, key) for key in ("hdr:a", "hdr:b", "hdr:c", "hdr:d")}
        self.assertEqual({"101", "102", "201", "202"}, set(homes.values()))
        for _ in range(3):
            for key, home in homes.items():
                self.assertEqual(home, self.route(fleet, key))
        self.assertEqual(0, fleet.router.rehomed)
        fleet.router.pin("hdr:by-hand", "202")
        self.assertEqual("202", self.route(fleet, "hdr:by-hand"))

    def test_a_conversation_with_no_history_can_land_in_any_pool(self):
        """Neither a recorded pool nor a pin: nothing ties it to a model yet."""
        fleet = self.make()
        self.assertIn(self.route(fleet, "hdr:fresh", exclude=["201", "202"]), {"101", "102"})
        self.assertIsNone(self.route(fleet, "hdr:fresh2", exclude=["101", "102", "201", "202"]))

    def test_a_hand_pinned_conversation_is_retried_inside_its_pins_pool(self):
        fleet = self.make()
        fleet.router.pin("hdr:retry", "201")
        self.assertEqual("202", self.route(fleet, "hdr:retry", exclude=["201"]))
        self.assertIsNone(self.route(fleet, "hdr:retry", exclude=["201", "202"]))

    def test_one_pool_with_nothing_healthy_leaves_the_rest_serving(self):
        fleet = self.make()
        for job in ("101", "102"):
            fleet.backends[job].healthy = False
        self.assertIn(self.route(fleet, "hdr:x"), {"201", "202"})
        self.assertIn(self.route(fleet, None), {"201", "202"})

    def test_a_drain_in_one_pool_is_honoured_by_the_shared_router(self):
        """Each pool's supervisor keeps its own drain, and routing reads every one."""
        fleet = self.make()
        fleet.pools[K3].draining["201"] = time.time() + 600
        fleet.draining["101"] = time.time() + 600
        self.assertEqual({"102", "202"}, set(fleet.accepting()))
        self.assertEqual({"101", "102", "201", "202"}, fleet.serving())

    def test_the_overall_active_is_what_one_election_over_every_pool_picks(self):
        fleet = self.make()
        fleet.backends["202"].end_time += 5000
        fleet.elect()
        self.assertEqual("202", fleet.overall_active())
        for job in ("201", "202"):
            fleet.backends[job].healthy = False
        fleet.elect()
        self.assertIn(fleet.overall_active(), {"101", "102"})


class PoolAffinity(PoolFixture):
    """A conversation, once served, is re-placed only inside its pool.

    Pools are models here. A conversation that moves to another pool continues
    on a model that did not write its history: on 10-09, 40 Kimi-K3 sessions
    carried GLM-5.3 tool-call markup that K3 then copied, every one of them
    moved over after its client abandoned a slow first event.
    """

    def make(self, *extra):
        fleet = self.fleet(*extra)
        for job in ("101", "102"):
            self.backend(fleet, job, gateway.DEFAULT_POOL)
        for job in ("201", "202"):
            self.backend(fleet, job, K3)
        return fleet

    @staticmethod
    def route(fleet, key, exclude=()):
        return gateway.Gateway(fleet).route(key, exclude=exclude)

    def serve_on_k3(self, fleet, key):
        """Place `key` on a K3 backend the way traffic would: GLM was busy then."""
        for job in ("101", "102"):
            fleet.backends[job].healthy = False
        home = self.route(fleet, key)
        for job in ("101", "102"):
            fleet.backends[job].healthy = True
        self.assertIn(home, {"201", "202"})
        return home

    def test_a_lost_backend_is_replaced_inside_the_pool(self):
        fleet = self.make()
        home = self.serve_on_k3(fleet, "hdr:a")
        fleet.backends[home].healthy = False
        other = ({"201", "202"} - {home}).pop()
        for _ in range(3):
            self.assertEqual(other, self.route(fleet, "hdr:a"))
        self.assertEqual(1, fleet.router.rehomed)

    def test_an_abandoned_conversation_comes_back_to_its_pool(self):
        """Unpinned after the client gave up on its first event, as the gateway does."""
        fleet = self.make()
        self.serve_on_k3(fleet, "hdr:b")
        # Load the K3 backends so that least_conversations would pick GLM.
        for n in range(4):
            fleet.router.pin("hdr:k3-load-%d" % n, "201" if n % 2 else "202")
        fleet.router.unpin("hdr:b")
        self.assertIn(self.route(fleet, "hdr:b"), {"201", "202"})

    def test_with_no_backend_of_its_pool_it_gets_none(self):
        fleet = self.make()
        self.serve_on_k3(fleet, "hdr:c")
        for job in ("201", "202"):
            fleet.backends[job].healthy = False
        self.assertIsNone(self.route(fleet, "hdr:c"))
        self.assertEqual(1, fleet.router.family_waits)
        # Its pool coming back is all it takes.
        fleet.backends["202"].healthy = True
        self.assertEqual("202", self.route(fleet, "hdr:c"))

    def test_a_retry_stays_in_the_pool(self):
        fleet = self.make()
        home = self.serve_on_k3(fleet, "hdr:d")
        other = ({"201", "202"} - {home}).pop()
        self.assertEqual(other, self.route(fleet, "hdr:d", exclude=[home]))
        self.assertIsNone(self.route(fleet, "hdr:d", exclude=["201", "202"]))

    def test_a_pin_from_before_pools_were_recorded_keeps_its_pool(self):
        """A pin restored from a predecessor has no recorded pool; its backend has one."""
        fleet = self.make()
        fleet.router._set_pin("hdr:old", "201", time.time())
        self.assertNotIn("hdr:old", fleet.router.families)
        fleet.backends["201"].healthy = False
        self.assertEqual("202", self.route(fleet, "hdr:old"))
        fleet.router._set_pin("hdr:older", "202", time.time())
        fleet.backends["202"].healthy = False
        self.assertIsNone(self.route(fleet, "hdr:older"))

    def test_new_conversations_still_go_to_every_pool(self):
        fleet = self.make()
        placed = [self.route(fleet, "hdr:new%d" % n) for n in range(8)]
        self.assertEqual({"101": 2, "102": 2, "201": 2, "202": 2}, tally(placed))

    def test_without_pool_affinity_a_conversation_can_change_pools(self):
        fleet = self.make("--no-pool-affinity")
        self.assertFalse(fleet.router.pool_affinity)
        self.serve_on_k3(fleet, "hdr:e")
        for job in ("201", "202"):
            fleet.backends[job].healthy = False
        self.assertIn(self.route(fleet, "hdr:e"), {"101", "102"})

    def test_conversation_pools_survive_a_restart(self):
        path = os.path.join(self.tmp, "router_state.json")
        router = gateway.Router(1800.0, 100, state_path=path)
        pools = {"101": gateway.DEFAULT_POOL, "201": K3}
        loads = {"101": (5, 0, 9e9), "201": (0, 0, 9e9)}
        self.assertEqual("201", router.route("hdr:f", loads, {"101", "201"}, pool_of=pools))
        router.save()
        restored = gateway.Router(1800.0, 100, state_path=path)
        restored.load()
        self.assertEqual(K3, restored.families["hdr:f"][0])
        restored.unpin("hdr:f")
        self.assertIsNone(restored.route("hdr:f", {"101": (0, 0, 9e9)}, {"101"}, pool_of=pools))

    def test_a_state_file_from_before_pools_were_remembered_loads(self):
        path = os.path.join(self.tmp, "router_state.json")
        now = time.time()
        with open(path, "w") as handle:
            json.dump(
                {"version": 1, "pins": {"hdr:g": ["201", now]}, "policy": "least_inflight"}, handle
            )
        router = gateway.Router(1800.0, 100, state_path=path)
        router.load()
        self.assertEqual({}, dict(router.families))
        # The pin still holds, and its next request records the pool.
        pools = {"101": gateway.DEFAULT_POOL, "201": K3}
        loads = {"101": (0, 0, 9e9), "201": (0, 0, 9e9)}
        self.assertEqual("201", router.route("hdr:g", loads, {"101", "201"}, pool_of=pools))
        self.assertEqual(K3, router.families["hdr:g"][0])


class PoolDiscovery(PoolFixture):
    def test_backends_are_tagged_by_the_directory_they_registered_in(self):
        self.register(self.fleet_dir, "100")
        self.register(self.k3_dir, "200")
        fleet = self.fleet()
        fleet.discover()
        self.assertEqual(gateway.DEFAULT_POOL, fleet.backends["100"].pool)
        self.assertEqual(K3, fleet.backends["200"].pool)

    def test_a_job_id_registered_in_two_pools_stays_in_the_first(self):
        self.register(self.fleet_dir, "300")
        self.register(self.k3_dir, "300")
        fleet = self.fleet()
        with self.assertLogs(gateway.LOG, "WARNING"):
            fleet.discover()
        self.assertEqual(gateway.DEFAULT_POOL, fleet.backends["300"].pool)
        fleet.discover()  # and the sweep after that does not flap
        self.assertEqual(gateway.DEFAULT_POOL, fleet.backends["300"].pool)

    def test_a_lost_backend_is_remembered_by_its_own_pool(self):
        path = self.register(self.k3_dir, "200")
        fleet = self.fleet()
        fleet.discover()
        os.unlink(path)
        fleet.discover()
        self.assertIn("200", fleet.pools[K3].lost)
        self.assertNotIn("200", fleet.lost)

    def test_a_backend_registered_over_http_can_name_a_pool(self):
        fleet = self.fleet()

        def register(payload):
            body = json.dumps(payload).encode()

            async def call():
                reader = asyncio.StreamReader()
                reader.feed_data(body)
                reader.feed_eof()
                return await gateway.Gateway(fleet).control(
                    "/_gateway/backend", b"", reader, [("content-length", str(len(body)))]
                )

            return asyncio.run(call())

        code, reply = register(
            {"job_id": "250", "url": "http://127.0.0.1:9", "pool": K3, "probe": False}
        )
        self.assertEqual(200, code, reply)
        self.assertTrue(os.path.exists(os.path.join(self.k3_dir, "250.json")))
        self.assertFalse(os.path.exists(os.path.join(self.fleet_dir, "250.json")))
        self.assertEqual(K3, fleet.backends["250"].pool)
        code, reply = register(
            {"job_id": "251", "url": "http://127.0.0.1:9", "pool": "nope", "probe": False}
        )
        self.assertEqual(400, code, reply)


class PoolSupervision(PoolFixture):
    """Each pool is supervised against its own fleet config, or not at all."""

    def sweep(self, fleet, squeue=("GONE", "")):
        calls = {"fleetctl": [], "serve_sh": [], "squeue": []}

        async def fake_fleetctl(view, *argv):
            calls["fleetctl"].append((view.args.fleet_config,) + argv)
            return 0, ""

        async def fake_serve_sh(view, *argv):
            calls["serve_sh"].append(argv)
            return 0, ""

        async def fake_status(job_id):
            calls["squeue"].append(job_id)
            return squeue

        saved = (gateway.run_fleetctl, gateway.run_serve_sh, gateway.slurm_job_status)
        gateway.run_fleetctl, gateway.run_serve_sh = fake_fleetctl, fake_serve_sh
        gateway.slurm_job_status = fake_status
        try:
            asyncio.run(gateway.supervise(fleet))
        finally:
            (gateway.run_fleetctl, gateway.run_serve_sh, gateway.slurm_job_status) = saved
        return calls

    def relay_fleet(self, **kwargs):
        return self.fleet(
            "--yaml",
            self.glm_config,
            "--relay-per-instance",
            "--fleet-config",
            self.glm_config,
            "--lead-time",
            "3300",
            k3_config=True,
            **kwargs,
        )

    def test_the_election_is_per_pool(self):
        """A K3 backend outliving every GLM one is not GLM's successor.

        The election is lifecycle state -- it decides what gets superseded,
        drained and quit -- so it stays per pool even though routing does not.
        """
        fleet = self.fleet(k3_config=True)
        self.backend(fleet, "100", gateway.DEFAULT_POOL, end_in=3600)
        self.backend(fleet, "200", K3, end_in=90000)
        fleet.elect()
        self.assertEqual("100", fleet.active)
        self.assertEqual("200", fleet.pools[K3].active)
        self.assertEqual(set(), fleet.superseded)
        self.assertEqual(set(), fleet.pools[K3].superseded)

    def test_a_shared_instance_label_is_not_a_replacement_across_pools(self):
        """GLM i00 and K3 i00 are two instances, and draining one for the other is fatal.

        Relay groups jobs by the label at the end of the run directory, so with
        one table the longer-lived K3 i00 would supersede GLM i00 -- which the
        drain then reclaims with `serve.sh quit`.
        """
        fleet = self.relay_fleet()
        self.backend(fleet, "100", gateway.DEFAULT_POOL, label="i00", end_in=600)
        self.backend(fleet, "200", K3, label="i00", end_in=90000)
        calls = self.sweep(fleet)
        self.assertNotIn("100", fleet.superseded)
        self.assertNotIn("100", fleet.draining)
        self.assertNotIn("100", fleet.pools[K3].superseded)
        self.assertEqual([], [c for c in calls["serve_sh"] if c[:1] == ("quit",)], calls)
        # GLM i00 is due, and is rolled through GLM's own fleet file.
        self.assertIn((self.glm_config, "up", "--only", "i00", "--force"), calls["fleetctl"])
        self.assertNotIn((self.k3_config, "up", "--only", "i00", "--force"), calls["fleetctl"])

    def test_walltime_relay_rolls_each_pool_against_its_own_config(self):
        fleet = self.relay_fleet()
        self.backend(fleet, "100", gateway.DEFAULT_POOL, label="i00", end_in=600)
        self.backend(fleet, "200", K3, label="k00", end_in=600)
        calls = self.sweep(fleet)
        self.assertEqual(
            sorted(
                [
                    (self.glm_config, "up", "--only", "i00", "--force"),
                    (self.k3_config, "up", "--only", "k00", "--force"),
                ]
            ),
            sorted(calls["fleetctl"]),
        )

    def test_a_lost_backend_is_recovered_through_its_own_pools_config(self):
        fleet = self.fleet("--fleet-config", self.glm_config, k3_config=True)
        long_ago = time.time() - 3600
        fleet.lost["100"] = ("/var/runs/junyix_100612_100_kffleet_glm5.3_i00", long_ago)
        fleet.pools[K3].lost["200"] = ("/var/runs/junyix_100612_200_kffleet_kimi-k3_k00", long_ago)
        calls = self.sweep(fleet)
        self.assertEqual(
            sorted([(self.glm_config, "up"), (self.k3_config, "up")]), sorted(calls["fleetctl"])
        )

    def test_a_configured_pools_exited_backend_is_revived_in_place(self):
        fleet = self.fleet(k3_config=True)
        self.backend(fleet, "100", gateway.DEFAULT_POOL)
        dead = self.backend(
            fleet,
            "200",
            K3,
            healthy=False,
            state="attempt 1 exited with status 1; allocation retained",
        )
        calls = self.sweep(fleet)
        self.assertEqual([("restart", dead.run_dir)], calls["serve_sh"])

    def test_an_unconfigured_pool_is_left_alone(self):
        fleet = self.fleet("--fleet-config", self.glm_config, k3_config=False)
        glm_dead = self.backend(
            fleet, "100", gateway.DEFAULT_POOL, healthy=False, state="stopped; allocation retained"
        )
        self.backend(fleet, "200", K3, healthy=False, state="stopped; allocation retained")
        fleet.pools[K3].lost["201"] = ("/var/runs/x_201_kffleet_kimi-k3_k01", time.time() - 3600)
        calls = self.sweep(fleet)
        self.assertEqual([("restart", glm_dead.run_dir)], calls["serve_sh"])
        self.assertEqual([], calls["fleetctl"])
        self.assertEqual([], calls["squeue"])

    def test_startup_checks_each_pools_recovery(self):
        fleet = self.fleet("--fleet-config", self.glm_config, k3_config=True)
        seen = []

        async def fake_fleetctl(view, *argv):
            seen.append((view.args.fleet_config,) + argv)
            return 0, ""

        async def fake_slurm(*argv):
            return 0, ""

        saved = gateway.run_fleetctl, gateway.run_slurm_command
        gateway.run_fleetctl, gateway.run_slurm_command = fake_fleetctl, fake_slurm
        try:
            asyncio.run(gateway.check_recovery(fleet))
        finally:
            gateway.run_fleetctl, gateway.run_slurm_command = saved
        self.assertEqual(
            sorted([(self.glm_config, "status"), (self.k3_config, "status")]), sorted(seen)
        )

    def test_stop_server_releases_only_the_default_pool(self):
        """The default pool's lifecycle endpoints must not stop jobs another pool's config owns."""
        fleet = self.fleet(k3_config=True)
        glm = self.backend(fleet, "100", gateway.DEFAULT_POOL)
        self.backend(fleet, "200", K3)
        self.assertTrue(fleet.supervisor_lock.acquire())
        self.addCleanup(fleet.supervisor_lock.release)
        quits = []

        async def fake_serve_sh(view, *argv):
            quits.append(argv)
            return 0, ""

        writer = _CapturingWriter()
        saved = gateway.run_serve_sh
        gateway.run_serve_sh = fake_serve_sh
        try:
            asyncio.run(
                gateway.Gateway(fleet).serve_introspection(
                    "POST", "/_gateway/stop_server", [], b"", None, writer
                )
            )
        finally:
            gateway.run_serve_sh = saved
        self.assertEqual(200, writer.status(), writer.data)
        self.assertEqual([("quit", glm.run_dir)], quits)
        self.assertEqual(["100"], writer.json()["released"])


class DefaultOnlyBehaviour(PoolFixture):
    """With no extra pool configured, nothing a client or an operator reads may change.

    The production data-gen gateway is handed over to this code, so the shape
    of what it reports is pinned key for key, not just "still works".
    """

    FLEET_KEYS = {
        "active",
        "pending_successor",
        "mirroring",
        "mirror_stats",
        "mirror_targets",
        "backends",
        "routing",
    }
    BACKEND_KEYS = {
        "url",
        "healthy",
        "healthy_for_s",
        "probe_timeouts",
        "state",
        "ends_at",
        "ends_in_s",
        "last_beat_s",
        "inflight",
        "conversations",
        "accepting",
        "superseded",
        "revived",
        "draining",
    }
    ROUTING_KEYS = {
        "enabled",
        "policy",
        "key_sources",
        "manual_pins",
        "paused",
        "state_file",
        "pinned",
        "hits",
        "misses",
        "rehomed",
        "accepting",
        "serving",
    }
    HEALTH_KEYS = {"status", "deployment", "active", "pending", "uptime_s"}

    def plain_backend(self, fleet, job_id):
        now = time.time()
        backend = gateway.Backend(
            {
                "job_id": job_id,
                "url": "http://127.0.0.1:9100",
                "run_dir": "/var/runs/%s" % job_id,
                "state": "running attempt 1",
                "end_time": now + 86400,
                "heartbeat": now,
            }
        )
        backend.healthy = True
        backend.healthy_since = now - 60
        fleet.backends[job_id] = backend
        return backend

    def introspect(self, fleet, path):
        fleet.users = {"tester"}
        writer = _CapturingWriter()
        asyncio.run(
            gateway.Gateway(fleet).serve_introspection(
                "GET", path, [("x-api-key", "tester")], b"", None, writer
            )
        )
        return writer.json()

    def test_the_fleet_report_has_exactly_the_keys_it_always_had(self):
        fleet = gateway.Fleet(self.args("--router-state", "", pool=False))
        self.plain_backend(fleet, "100")
        payload = self.introspect(fleet, "/_gateway/fleet")
        self.assertEqual(self.FLEET_KEYS, set(payload))
        self.assertEqual(self.BACKEND_KEYS, set(payload["backends"]["100"]))
        self.assertEqual(self.ROUTING_KEYS, set(payload["routing"]))

    def test_health_has_exactly_the_keys_it_always_had(self):
        fleet = gateway.Fleet(self.args("--router-state", "", pool=False))
        self.plain_backend(fleet, "100")
        fleet.elect()
        payload = self.introspect(fleet, "/_gateway/health")
        self.assertEqual(self.HEALTH_KEYS, set(payload))
        self.assertEqual("ok", payload["status"])

    def test_every_backend_is_in_the_default_pool(self):
        self.register(self.fleet_dir, "100")
        self.register(self.k3_dir, "200")  # a directory nobody configured
        fleet = gateway.Fleet(self.args("--router-state", "", pool=False))
        fleet.discover()
        self.assertEqual(["100"], sorted(fleet.backends))
        self.assertEqual("default", getattr(fleet.backends["100"], "pool", "default"))
