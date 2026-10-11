---
name: trtllm-kf-campaigns
description: >-
  Run Kernel Factory campaigns against a self-hosted TensorRT-LLM endpoint,
  fed by the shared kf-queue and rate-controlled to keep the fleet busy without
  overloading it. Use when launching or draining campaigns, wiring a BYO LLM
  endpoint into Kernel Factory, choosing campaign concurrency, or working out
  why campaigns fail, stall, or leave the queue inconsistent. Triggers on: kf
  campaign, kfrun, kfq, kernel factory, llm-endpoint, BYOM, campaign
  concurrency, saturate the GPUs, drain the queue, campaign stuck in submitted,
  authentication_error from KF, agents-per-round, effort, max-duration.
tags: [infrastructure, serving, kernel-factory, campaigns]
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Running Kernel Factory campaigns

Two moving parts. `kfq` is the shared work queue over the campaign problems;
`kfrun` turns claimed problems into campaigns, holds a target number in flight,
and hands outcomes back. Both live in the kf-queue directory, which sits next
to whatever runs `kf` — that host has to reach both Kernel Factory and the
gateway, and not every host does.

## Settled, and not to be asked about

The campaign shape is fixed, by standing decision, for **every** campaign run
out of the kf-problem queue:

```
--gpu-spec b300  --language cuda_cpp  --agent codex:gpt-5.6-sol
--agents-per-round 6  --effort high  --max-duration 4h
```

**These are already `kfrun`'s defaults, so pass none of them.** Spelling them
out on the command line only creates a second place to get them wrong — and
`cuda_cpp`, which nobody says out loud, is the one that gets mistyped. What is
worth doing periodically is checking the two have not drifted apart: a skill
that documents one shape while the code defaults to another is worse than
either alone, because it reads as confirmation.

```bash
grep -A1 'add_argument("--\(agent\|agents-per-round\|gpu-spec\|language\|effort\|max-duration\)"' kfrun
```

The values land as `max_duration: 14400`, `count: 6`, `gpu_spec: b300`,
`language: cuda_cpp`, `runtime: codex` / `model: gpt-5.6-sol` in the generated
`campaign.yaml` — check there if a run looks wrong, since that file is what the
server actually received.

**`--max-duration 4h` is deliberate and is not a mistake to correct.** It does
interact with `--effort high`, whose per-agent deadline is also 4h, so a
campaign is cut off around the time round one ends and rounds two onward never
happen. That was chosen with the interaction understood: with ~10k campaigns to
drain, a bounded cost per campaign is worth more than the best result from any
one of them. Do not raise it, and do not propose lowering `--effort` to "make
the budget meaningful", unless the user reopens it.

`--max-duration` is a **wall clock**, not a spend cap — "Server-side wall-clock
safety net for the whole campaign", bare values in seconds, `s`/`m`/`h`
suffixes, 168h maximum. The spend cap is the separate `--max-cost-usd`, which
nothing here sets and which lands at its 10000.0 default. Reading the two out
of `kf campaign init --help` with one grep puts the USD sentence directly under
the `--max-duration` heading and invites exactly the wrong conclusion; read
each flag's own block.

## The Kernel-Trace set differs

`--problems Kernel-Trace` (1241 problems, mirrored from
`dl/flashinfer/kernel-trace` `traces/nemotron-cuda`) runs a different shape,
by standing decision for that set only:

    --agents-per-round 1        one agent per round, not six
    every problem run 6 times   six separate campaigns, not six rounds

`max_rounds` stays unset, so the effort tier's 10 applies within each campaign.

The repeat is `kfrun --runs-per-problem 6`. The queue itself cannot express it
(`kfq`'s lists are mutually exclusive by construction), so kfrun holds the
claim across runs: each finished campaign is counted in
`state.meta["runs"][problem]` keyed by **campaign id** — a bare counter would
double-count whenever a refused kfq report makes reconcile re-see the same
terminal campaign — and the next run is started from the same prepared
workspace via the recycle mechanism (`kf campaign start --from` on an
already-started workspace makes a fresh campaign from the same baseline, so a
repeat costs a start, not a prepare). Runs are sequential per problem, which is
deliberate: KF caps one user at 500 active campaigns, and six concurrent runs
of one problem would spend that budget without adding throughput. The problem
reports to kfq once, after its last run: `success` if any run succeeded.
Infra-killed no-speedup campaigns do not consume a run slot when
`--recycle-prepared` is on — recycle intercepts them first, bounded by
`--start-attempts`.

## Ask for these before starting

Only three things vary. Propose the default rather than interrogating:

| Ask | Default to propose |
|---|---|
| **Which endpoint** | `kf llm-endpoint list` and offer what is there. `glm53-jhb-direct` is GLM-5.3, not 5.2 — check before assuming |
| **How much of the queue** | a small `--max-total` for a first run; unbounded only once one campaign has been watched end to end |
| **Rate** | `--per-backend 35 --auto`. A fixed `--concurrency` only when the fleet is not behind the gateway |

Everything else — the queue location, the reconciliation rules, the tag — has a
default that does not need a decision.

## The CLI, and three things its docs will mislead you about

`kf` is a static Rust binary from SolSwarm, installed by
`curl -fsSL https://kernelfactory.nvidia.com/install | sh` into `~/.local/bin`.
The installer also writes `kernel-factory` skills into `~/.claude/skills` and
`~/.codex/skills` — worth knowing if either is a symlink onto shared storage.

**`kf campaign run` does not exist.** The lifecycle is three commands, and every
authoring flag belongs to `init`; `start` takes only `--from`, `--watch`,
`--tenant`, `--compliance-model` and `--llm-endpoint`.

```bash
kf campaign init "$name" \
  --definition <problem>/definition.json --workloads <problem>/workload.jsonl \
  --gpu-spec b300 --language cuda_cpp \
  --agent codex:gpt-5.6-sol --agents-per-round 6 --effort high --max-duration 4h
kf campaign prepare --from "$name"/campaign.yaml
kf campaign start   --from "$name"/campaign.yaml --llm-endpoint <name>
```

**`--max-duration` is not the per-agent deadline.** `--effort` sets that — low
1h, medium 2h, high/max 4h — along with the default 10-round budget.
`--max-duration` is a server-side wall clock for the whole campaign, defaults
to seven days, and will cut a healthy campaign off mid-round. So the 4h above
does cost something real: left alone, these campaigns run 8h58m and reach round
two, and in one batch of four the round-two run produced the best result. That
is the price of the cap, not an argument against it — see the settled section.

**The obvious parameter names are mostly wrong:**

| intent | flag | note |
|---|---|---|
| runtime | `--agent codex:gpt-5.6-sol` | `n3` is the AVO runtime; `claude`, `kimicode`, `opencode` also exist |
| GPU | `--gpu-spec b300` | lowercase kebab; `B300` is rejected. Not a "verifier" — it selects the CudaGym eval backend |
| language | `--language cuda_cpp` | already the default; restrictive unless `allowed_libraries` is set |
| agents | `--agents-per-round 6` | 1-10, only valid with `--agent` |

## Wiring the endpoint

```bash
kf auth login --manual          # device code; paste the URL into a browser
kf llm-endpoint create --name <name> --url http://<gateway>:8333
```

**Check the address before anything else.** `create` validates the endpoint's
resolved IP against an operator-managed allowlist
(`SOLSWARM_LLM_ENDPOINT_ALLOWED_CIDRS`) and rejects anything outside it with a
400, before it ever tries to reach the endpoint. So a perfectly healthy fleet
behind a perfectly healthy gateway can be simply unusable, and nothing in the
fleet or the gateway will say so.

**The grain is the cluster, not the host.** Seven addresses measured:

| address | what it is | |
|---|---|---|
| `10.109.53.68` | jhb 7-day CPU node | accepted |
| `10.109.64.13` | jhb login node | accepted |
| `10.49.64.90` | aga `cpu` partition | blocked |
| `10.49.78.231` | aga `cpu_datamover` | blocked |
| `10.49.71.38` | aga GPU compute | blocked |
| `10.49.80.8` | aga login node | blocked |
| `10.176.206.169` / `10.6.77.161` | ipp2-1730 / the dashboard VM | blocked |

Two subnets on jhb are in and four on aga are out, so what decides it is which
cluster the host belongs to. Node type is not the lever: an aga CPU node is
blocked for the same reason its login node is. Do not spend an allocation
looking for a host of the right *kind* — probe one address per cluster and take
the answer for the whole cluster.

Giving a hostname does not help: it is resolved and the address is checked. The
fix is an operator adding the range, or putting the gateway on a cluster that
is already in — it is not something to solve from this side.

Probing costs nothing and is reversible: a `create` that succeeds leaves an
endpoint, and `kf llm-endpoint delete <name>` removes it.

Two more constraints that shape everything around this:

- **`kf llm-endpoint` stores and sends no credential.** So the endpoint must
  answer an unauthenticated request. For this gateway that means `anonymous` in
  the users file; without it every agent call is a 401 and the symptom is an
  `authentication_error` that says nothing about the endpoint being otherwise
  healthy.
- **BYO LLM refuses service accounts** — `kf auth ssa` gets a 403 — so this runs
  on an interactive token that lasts about an hour. Unattended draining needs
  the token renewed by a human, and the launcher must stop rather than claim
  work it cannot start.

`--url` is an origin with no path. Plaintext http is accepted to a private
address.

## What the agents actually send

Not what you would guess from "OpenAI-compatible":

- **codex agents call `/v1/responses`**, not `/v1/chat/completions` — streaming
  SSE, ~65 KB bodies, `last_event=response.completed`, ~3.2 s each in one
  measured round. Both paths appear; the Responses API dominates.
- **Conversation affinity comes from `prompt_cache_key`** in the body, not from
  a header. The gateway already reads it, and it works: seven consecutive calls
  in one agent session all landed on the same backend.
- A handful of `400`s with `body.input` and `body.model` both missing is normal
  alongside the working traffic. Do not read a 400 rate as the endpoint being
  broken — check whether 200s are also flowing.

**`kf campaign list -f json` inverts the table's columns.** There is no `id`
field: `name` holds the id and `display_name` holds the name you passed to
`init`. Matching on `name` when you mean the friendly name finds nothing,
silently.

**kfq and kf put data on stdout and narration on stderr.** `... 9740 more`,
`N moved to list_submitted`, `[info] campaign.yaml omitted max_rounds` — all
stderr. Merge the streams and the decoration becomes indistinguishable from
data; it is also *first*, so the first "problem name" a merged reader gets is
`... 9740 more`, which it will then try to file back to the queue.

## The queue

`kfq claim N` pops N problems and marks them `submitted` in one atomic step, so
concurrent workers never receive the same problem. That atomicity is also the
hazard: a problem leaves `todo` the moment it is handed over, and if the launch
then fails with nobody recording that, it sits in `submitted` with no one
working on it. 1,202 problems ended up that way once.

```bash
./kfq status                  # counts, and whether anyone holds the queue
./kfq claim 8                 # atomic
./kfq success <name> ...      # submitted -> success
./kfq retry <name> ...        # submitted -> retry
./kfq verify                  # every name in exactly one list
```

Reading the lists directly is safe (`os.replace` guarantees a whole file);
writing them by hand is not.

`list_submitted` being non-empty is not necessarily wrong — it is also how
another person's in-flight work looks. Check the journal for who claimed what
before assuming anything is stranded.

## Feeding it: `kfrun`

```bash
./kfrun --gateway http://<gw>:8333 --per-backend 35 --auto --endpoint <name>
```

The loop is: reconcile what finished, top the campaigns back up to target,
sleep, repeat. Three things about the target are the point of the script.

**It is per healthy backend, not a fixed number.** The target is
`--per-backend` × the healthy backends the gateway currently reports. A fixed
number is wrong in both directions and, worse, ignores preemption: lose a
quarter of the fleet and it keeps the same load on what is left. Scaling by the
fleet means a preemption lowers the offered load by itself and a recovered
instance raises it, with nobody in the loop.

**`prepare` is the rate limit, and it is not ours to raise.** Each launch is
init, then `prepare` -- which runs a *baseline evaluation on real GPUs on
Kernel Factory's side* -- then start. Eight pipelines produce about two
campaigns a minute, and the fleet will sit far below its serving capacity as a
result. That is worth knowing before concluding the fleet is the problem: at
one point it was carrying 8% of its measured per-instance load and the launcher
was the entire reason.

Raising `--launch-workers` to 24 was tried and the fill rate appeared to jump
to 28 a minute. It was not throughput: the campaigns were being created and
cancelled during baseline, 437 of them in forty minutes, all at round 0.
**Lowering it back to 8 did not restore service** -- the cancellations
continued, at the same concurrency that had worked for the preceding hour. So
the raise is not established as the cause, and this is not a knob with a known
safe setting; something on the platform side was exhausted or had changed, and
it did not recover on our timescale. Treat mass `baseline evaluation was
cancelled` as a platform condition to wait out, not a parameter to tune.

The tell is `prepare failed (rc=1): Error: baseline evaluation was cancelled`
in the launcher log, and a `Cancelled` count climbing in `kf campaign list`.
Watch the *cancelled* count, not the created count -- a launcher that is
failing fast looks exactly like a launcher that is working fast. And when
nearly every outcome is a cancellation, stop the launcher: it is claiming
problems, burning API calls and churning the queue to produce nothing.

Nothing was corrupted, because a cancelled campaign maps to retry and those
problems went back to the queue. That mapping exists for this.

**`kf campaign list` is capped at 500 rows** whatever `--limit` says, so a burst
of failures pushes the healthy campaigns out of the window entirely. Read the
gateway's conversation count for ground truth on whether work is actually
happening; the campaign list will tell you everything is gone when it is not.

**`--auto` ramps it against measured time to first token.** This workload is
prefill-bound, so a fleet at its limit shows up as a first token taking tens of
seconds well before anything fails. Additive increase below `--ttft-target`,
multiplicative decrease above `--ttft-limit`, and the gap between them is a
deadband — without one it oscillates, because a campaign takes minutes to start
loading the fleet and the controller would act several times before its last
move showed up. The probe goes through the gateway, because that is the path a
campaign takes.

**Nothing is claimed that cannot be launched.** Auth is checked before claiming,
never after; claims are small batches; a failed launch goes straight back to
the queue; and anything claimed-but-not-launched is in the state file before the
attempt, so a restart finishes or returns it. `--dry-run` reads `todo` rather
than claiming — a dry run that moved problems would be the exact failure the
script exists to prevent.

Outcomes map `Completed` → success, `Failed` and `Cancelled` → retry. Cancelled
is a retry because it is what a fleet restart leaves behind, which says nothing
about the problem.

## Numbers to start from

Measured: 4 × 6P1D carried 200 concurrent campaigns at 11.4 req/s peak — 50
campaigns and 2.84 req/s per instance, with TTFT p50 swinging 0.6-38 s. That is
"survived", not "comfortable", so start `--per-backend` around 35 and let
`--auto` find the rest.

A campaign issues roughly 0.057 req/s (six agents, ~105 s between requests
each), so at 4h it makes ~820 requests. With the fleet's ~10% preemption error
rate a 5xx is a certainty rather than a risk — survivable only because KF
retries and the gateway re-homes a conversation whose backend went away. Do not
design around campaigns seeing clean runs.

Only about 10% of campaigns beat 1x speedup, and the median is exactly 1.00.
That is the yield to expect, not a sign anything is broken.

## When it goes wrong

| Symptom | Cause |
|---|---|
| `400 invalid base_url: blocked IP address` on `llm-endpoint create` | The gateway's address is outside KF's allowlist. Checked at create time against the resolved IP, so a hostname does not dodge it and the fleet's health is irrelevant. Needs an operator, or a gateway inside an allowed range |
| `authentication_error` from agents | The gateway users file has no `anonymous`; KF sends no credential by design |
| Campaigns start then all fail around the same time | Check the fleet, not the campaigns. A preemption takes every campaign pinned to that backend with it |
| Queue has `submitted` entries with nothing running | A launcher died between claim and start. `kfq retry` them; check the journal for whose they were |
| `kfrun` stops with "NOT AUTHENTICATED" | The hour is up. `kf auth login --manual` and start again with the same `--tag`; the state file resumes |
| Campaign dies mid-round at ~4h with work in progress | Expected — that is `--max-duration` doing its job. Not a bug, and not to be raised |
| Every campaign slow at once, no failures | Over-subscribed. Lower `--per-backend`, or let `--auto` do it |

## Keeping this current

The measured numbers here came from one fleet on one cluster. Recompute them
from the analysis rollups after any run large enough to be worth it, and
replace rather than average — if per-instance capacity changes, say what
changed it.
