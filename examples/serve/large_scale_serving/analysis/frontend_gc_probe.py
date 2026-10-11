#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""What does the disaggregated frontend keep per request, with the cyclic GC off or on?

Runs the real OpenAIDisaggServer in worker mode (coordinator_url set, as a fleet worker is) against a
real DisaggCoordinatorService, with mocked ctx/gen workers that replay real trace samples: the ctx
mock answers the context_only Responses hop, the gen mock streams the recorded SSE text. Request
tracing and perf-metrics output are on, as in production. CPU only; nothing is sent anywhere else.

Process layout, so only the frontend's own allocations are measured:
  backend subprocess: ctx mock + gen mock + coordinator (uvicorn threads)
  driver subprocess:  aiohttp client sending the samples as streaming /v1/responses
  this process:       the frontend; measures RSS and the cyclic garbage after the driver ends

Samples are lines of {"body": <request body>, "sse": <SSE text the client received>}, e.g. pairs of
completed streaming /v1/responses records from a request trace (requests-*.jsonl / responses-*.jsonl).
Run it inside the serving container, outside MPI: clear SLURM_*/PMIX_* first, or an
`srun --overlap` step aborts in MPI_Init.

Optional environment: PROBE_PHASES=N repeats the main phase N times and records RSS and live-object
growth after each; PROBE_FREEZE=1 calls gc.freeze() after start-up.

Measured on 10-09 (f8efc82403, 400 replayed GLM requests, concurrency 16): GC off, RSS +3.4 MB per
request and 589,987 cyclic objects at the end, mostly pydantic ValidatorIterators held by the item
dicts they validate; GC on, RSS levels off (1,800 requests) and no cyclic garbage is left.

usage: frontend_gc_probe.py run <samples.jsonl> <n_requests> <gc:on|off> <outdir> [concurrency]
"""

import asyncio
import base64
import collections
import ctypes
import gc
import json
import os
import random
import socket
import subprocess
import sys
import threading
import time


def free_port():
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def rss_mib():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return -1.0


def trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


class UvicornThread:
    def __init__(self, app, port):
        import uvicorn

        self.port = port
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self):
        self.thread.start()
        for _ in range(300):
            if self.server.started:
                return self
            time.sleep(0.1)
        raise RuntimeError(f"uvicorn on {self.port} did not start")


# ---------------------------------------------------------------- backend (mocks + coordinator)
def mock_app(role, samples):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    app = FastAPI()

    @app.get("/health")
    async def health():
        return Response(status_code=200)

    @app.get("/server_info")
    async def server_info():
        return JSONResponse({"kv_cache_hash_algo": "v1"})

    @app.post("/v1/responses")
    async def responses(raw: Request):
        body = json.loads(await raw.body())
        dp = body.get("disaggregated_params") or {}
        if dp.get("request_type") == "context_only":
            rid = dp.get("disagg_request_id")
            n_tok = min(200_000, max(16, len(json.dumps(body.get("input", ""))) // 4))
            ids = list(range(n_tok))
            out = {
                "id": f"resp_ctx_{rid}",
                "created_at": 0,
                "model": body.get("model", "m"),
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "temperature": 1.0,
                "tool_choice": "auto",
                "tools": [],
                "top_p": 1.0,
                "background": False,
                "service_tier": "auto",
                "status": "in_progress",
                "top_logprobs": 0,
                "truncation": "disabled",
                "finish_reason": "length",
                "usage": {
                    "input_tokens": n_tok,
                    "output_tokens": 0,
                    "total_tokens": n_tok,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
                "disaggregated_params": {
                    "request_type": "context_only",
                    "ctx_request_id": rid,
                    "disagg_request_id": rid,
                    "first_gen_tokens": [1],
                },
            }
            if dp.get("return_prompt_token_ids_b64"):
                import numpy as np

                out["prompt_token_ids_b64"] = base64.b64encode(
                    np.asarray(ids, dtype=np.int32).tobytes()
                ).decode()
            else:
                out["prompt_token_ids"] = ids
            return JSONResponse(out)
        sse = random.choice(samples)["sse"]
        frames = [f + "\n\n" for f in sse.split("\n\n") if f]

        async def stream():
            for f in frames:
                yield f.encode()
                await asyncio.sleep(0)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def backend_main(samples_path, ports_path):
    samples = [json.loads(line) for line in open(samples_path)]
    ctx = UvicornThread(mock_app("ctx", samples), free_port()).start()
    gen = UvicornThread(mock_app("gen", samples), free_port()).start()
    cfg = make_config(ctx.port, gen.port, 0, None)
    import aiohttp

    from tensorrt_llm.serve.coordinator_server import CoordinatorServer
    from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinatorService
    from tensorrt_llm.serve.openai_client import OpenAIHttpClient

    class ReadinessClient:
        def __init__(self, router):
            self._router = router
            self._session = None

        async def check_ready(self):
            if self._session is None:
                self._session = aiohttp.ClientSession()
            ready, unready = await OpenAIHttpClient.check_ready_for_servers(
                self._session, self._router.servers
            )
            if ready:
                await self._router.prepare_servers(ready)
            return ready, unready

        async def shutdown(self):
            if self._session is not None:
                await self._session.close()

    svc = DisaggCoordinatorService(
        cfg, client_factory=lambda router, role, mr=1: ReadinessClient(router)
    )
    coord = UvicornThread(CoordinatorServer(svc).app, free_port()).start()
    with open(ports_path + ".tmp", "w") as fh:
        json.dump({"ctx": ctx.port, "gen": gen.port, "coord": coord.port}, fh)
    os.rename(ports_path + ".tmp", ports_path)
    while True:
        time.sleep(3600)


def make_config(ctx_port, gen_port, public_port, perf_dir):
    from tensorrt_llm.llmapi.disagg_utils import (
        CtxGenServerConfig,
        DisaggServerConfig,
        RouterConfig,
        ServerRole,
    )

    kwargs = {}
    if perf_dir:
        kwargs["perf_metrics_output_dir"] = perf_dir
    # As deployed: conversation router on ctx (stateful, delegated to the coordinator), default on gen.
    return DisaggServerConfig(
        server_configs=[
            CtxGenServerConfig(type="ctx", hostname="127.0.0.1", port=ctx_port),
            CtxGenServerConfig(type="gen", hostname="127.0.0.1", port=gen_port),
        ],
        hostname="127.0.0.1",
        port=public_port,
        ctx_router_config=RouterConfig(type="conversation", server_role=ServerRole.CONTEXT),
        gen_router_config=RouterConfig(type="round_robin", server_role=ServerRole.GENERATION),
        **kwargs,
    )


# ---------------------------------------------------------------- driver
def driver_main(samples_path, url, n, conc, result_path):
    import aiohttp

    samples = [json.loads(line) for line in open(samples_path)]

    async def main():
        stats = collections.Counter()
        sem = asyncio.Semaphore(conc)
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as sess:

            async def one(i):
                s = samples[i % len(samples)]
                body = dict(s["body"])
                body["stream"] = True
                body.pop("previous_response_id", None)
                async with sem:
                    try:
                        async with sess.post(
                            f"{url}/v1/responses",
                            json=body,
                            headers={"session_id": f"probe-{i % 64}"},
                        ) as r:
                            stats[f"http_{r.status}"] += 1
                            data = await r.read()
                            stats["bytes"] += len(data)
                            if b"response.completed" in data:
                                stats["completed"] += 1
                    except Exception as e:  # recorded, the probe keeps going
                        stats["exc_" + type(e).__name__] += 1

            await asyncio.gather(*(one(i) for i in range(n)))
        return stats

    t0 = time.time()
    stats = asyncio.run(main())
    stats["seconds"] = round(time.time() - t0, 1)
    with open(result_path, "w") as fh:
        json.dump(stats, fh)


# ---------------------------------------------------------------- analysis of the cyclic garbage
def describe(o):
    t = type(o)
    name = f"{t.__module__}.{t.__qualname__}"
    if t.__name__ == "frame":
        return f"frame:{o.f_code.co_qualname if hasattr(o.f_code, 'co_qualname') else o.f_code.co_name}"
    if t.__name__ in ("function", "coroutine", "async_generator", "generator", "method"):
        return f"{t.__name__}:{getattr(o, '__qualname__', '?')}"
    if t.__name__ == "cell":
        return "cell"
    return name


def link_name(src, dst):
    """Best-effort name of the reference src -> dst."""
    try:
        if type(src).__name__ == "frame":
            for k, v in src.f_locals.items():
                if v is dst:
                    return f"local {k}"
        d = getattr(src, "__dict__", None)
        if isinstance(d, dict):
            if d is dst:
                return "__dict__"
            for k, v in d.items():
                if v is dst:
                    return f".{k}"
        if isinstance(src, dict):
            for k, v in src.items():
                if v is dst:
                    return f"[{k!r}]"[:60]
        if isinstance(src, (list, tuple)):
            for i, v in enumerate(src):
                if v is dst:
                    return f"[{i}]"
        for attr in (
            "__traceback__",
            "tb_frame",
            "tb_next",
            "f_back",
            "cr_frame",
            "ag_frame",
            "gi_frame",
            "__context__",
            "__cause__",
            "cell_contents",
            "__self__",
            "__func__",
            "__closure__",
        ):
            if getattr(src, attr, None) is dst:
                return attr
    except Exception:
        pass
    return "?"


def shortest_cycle(start, ids, limit=300_000):
    from collections import deque

    prev = {id(start): None}
    objs = {id(start): start}
    q = deque([start])
    seen = 0
    while q and seen < limit:
        x = q.popleft()
        seen += 1
        for y in gc.get_referents(x):
            if y is start:
                path = [y, x]
                k = id(x)
                while prev[k] is not None:
                    k = prev[k]
                    path.append(objs[k])
                path.reverse()
                return path
            iy = id(y)
            if iy in ids and iy not in prev:
                prev[iy] = id(x)
                objs[iy] = y
                q.append(y)
    return None


def analyse_garbage(out, top=40):
    gc.set_debug(gc.DEBUG_SAVEALL)
    t0 = time.time()
    n = gc.collect()
    garbage = list(gc.garbage)
    gc.garbage.clear()
    gc.set_debug(0)
    out["gc_collect_found"] = n
    out["gc_collect_seconds"] = round(time.time() - t0, 2)
    hist = collections.Counter(describe(o) for o in garbage)
    out["garbage_types_top"] = hist.most_common(top)
    ids = {id(o) for o in garbage}
    wanted = [
        "tensorrt_llm.serve.openai_protocol.ResponsesRequest",
        "starlette.requests.Request",
        "tensorrt_llm.serve.openai_disagg_server.RawRequestResponseHooks",
        "tensorrt_llm.serve.request_trace.RequestTraceHandle",
        "_asyncio.Task",
    ]
    out["key_type_counts"] = {w: hist.get(w, 0) for w in wanted}
    # Also trace a cycle through one object of each of the most common garbage types.
    wanted = wanted + [t for t, _ in hist.most_common(4) if t not in wanted]
    holders_by_type = {}
    for t, _ in hist.most_common(4):
        tid = {id(o) for o in garbage if describe(o) == t}
        h = collections.Counter()
        for o in garbage:
            for y in gc.get_referents(o):
                if id(y) in tid:
                    h[f"{describe(o)} {link_name(o, y)}"] += 1
        holders_by_type[t] = h.most_common(8)
    out["holders_by_type"] = holders_by_type
    sample_reprs = {}
    for t, _ in hist.most_common(4):
        o = next(o for o in garbage if describe(o) == t)
        try:
            sample_reprs[t] = repr(o)[:300]
        except Exception as e:
            sample_reprs[t] = f"<repr failed: {e!r}>"
    out["sample_reprs"] = sample_reprs
    cycles = {}
    for w in wanted:
        cand = [o for o in garbage if describe(o) == w][:3]
        for c in cand[:1]:
            path = shortest_cycle(c, ids)
            if path:
                cycles[w] = [
                    f"{describe(a)} --{link_name(a, b)}--> " for a, b in zip(path, path[1:])
                ] + [describe(path[-1])]
            else:
                cycles[w] = "no cycle through it (held by another cycle)"
    # Who holds the ResponsesRequest objects directly, inside the garbage.
    holders = collections.Counter()
    reqs = [o for o in garbage if describe(o) == wanted[0]][:50]
    req_ids = {id(r) for r in reqs}
    for o in garbage:
        for y in gc.get_referents(o):
            if id(y) in req_ids:
                holders[f"{describe(o)} {link_name(o, y)}"] += 1
    out["request_direct_holders"] = holders.most_common(15)
    out["cycles"] = cycles
    del garbage, reqs, cand
    gc.collect()


def server_main(args):
    samples_path, n, gc_mode, outdir = args[0], int(args[1]), args[2], args[3]
    conc = int(args[4]) if len(args) > 4 else 16
    os.makedirs(outdir, exist_ok=True)
    os.environ["TRTLLM_REQUEST_TRACE_DIR"] = os.path.join(outdir, "trace")
    # The routes register a multiprocess Prometheus collector, as the fleet does.
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = os.path.join(outdir, "prom")
    os.makedirs(os.environ["PROMETHEUS_MULTIPROC_DIR"], exist_ok=True)
    ports_path = os.path.join(outdir, "ports.json")
    env = dict(os.environ)
    backend = subprocess.Popen(
        [sys.executable, __file__, "backend", samples_path, ports_path], env=env
    )
    try:
        for _ in range(600):
            if os.path.exists(ports_path):
                break
            time.sleep(0.2)
        ports = json.load(open(ports_path))
        from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer

        public = free_port()
        cfg = make_config(ports["ctx"], ports["gen"], public, os.path.join(outdir, "perf"))
        server = OpenAIDisaggServer(
            config=cfg, coordinator_url=f"http://127.0.0.1:{ports['coord']}"
        )
        if gc_mode == "off":
            gc.disable()  # what _init_fleet_worker_process does when TRTLLM_DISAGG_SERVER_DISABLE_GC=1
        # Time every automatic collection (none run when the GC is off).
        pauses = collections.defaultdict(list)
        started = {}

        def on_gc(phase, info):
            if phase == "start":
                started["t"] = time.perf_counter()
            elif "t" in started:
                pauses[info["generation"]].append((time.perf_counter() - started.pop("t")) * 1000)

        gc.callbacks.append(on_gc)
        UvicornThread(server.app, public).start()
        url = f"http://127.0.0.1:{public}"
        import urllib.request

        for _ in range(300):
            try:
                if urllib.request.urlopen(f"{url}/health", timeout=2).status == 200:
                    break
            except Exception:
                time.sleep(0.5)
        out = {"gc_mode": gc_mode, "n_requests": n, "concurrency": conc}

        def drive(k, tag):
            res = os.path.join(outdir, f"driver-{tag}.json")
            subprocess.run(
                [sys.executable, __file__, "drive", samples_path, url, str(k), str(conc), res],
                env=env,
                check=True,
            )
            time.sleep(3)
            return json.load(open(res))

        if os.environ.get("PROBE_FREEZE") == "1":
            gc.collect()
            gc.freeze()  # startup objects move to the permanent generation
            out["frozen_objects"] = gc.get_freeze_count()
        out["warmup"] = drive(min(40, n), "warmup")
        trim()
        out["rss_after_warmup_mib"] = round(rss_mib(), 1)
        out["gc_count_after_warmup"] = gc.get_count()
        phases = int(os.environ.get("PROBE_PHASES", "1"))
        live0 = collections.Counter(describe(o) for o in gc.get_objects()) if phases > 1 else None
        out["phase_rss_mib"] = []
        for ph in range(phases):
            out["main"] = drive(n, f"main{ph}")
            trim()
            out["phase_rss_mib"].append(round(rss_mib(), 1))
        if live0 is not None:
            live1 = collections.Counter(describe(o) for o in gc.get_objects())
            grow = {t: live1[t] - live0.get(t, 0) for t in live1}
            out["live_object_growth_top"] = sorted(grow.items(), key=lambda kv: -kv[1])[:25]
        n = n * phases
        out["n_requests"] = n
        out["rss_after_main_mib"] = round(rss_mib(), 1)
        out["gc_count_after_main"] = gc.get_count()
        out["rss_growth_per_request_kib"] = round(
            (out["rss_after_main_mib"] - out["rss_after_warmup_mib"]) * 1024 / max(1, n), 1
        )
        gc.callbacks.remove(on_gc)

        def pct(xs, q):
            xs = sorted(xs)
            return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else None

        out["automatic_gc_pauses_ms"] = {
            str(g): {
                "count": len(v),
                "total": round(sum(v), 1),
                "p50": pct(v, 0.5),
                "p99": pct(v, 0.99),
                "max": round(max(v), 2),
            }
            for g, v in sorted(pauses.items())
        }
        analyse_garbage(out)
        trim()
        out["rss_after_collect_mib"] = round(rss_mib(), 1)
        with open(os.path.join(outdir, f"result-gc{gc_mode}.json"), "w") as fh:
            json.dump(out, fh, indent=1, default=str)
        print(
            json.dumps(
                {k: out[k] for k in out if k not in ("garbage_types_top",)}, indent=1, default=str
            )
        )
        print("garbage_types_top:")
        for t, c in out.get("garbage_types_top", [])[:25]:
            print(f"  {c:9d}  {t}")
    finally:
        backend.kill()


if __name__ == "__main__":
    role = sys.argv[1]
    if role == "backend":
        backend_main(*sys.argv[2:4])
    elif role == "drive":
        driver_main(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
    elif role == "run":
        server_main(sys.argv[2:])
    else:
        sys.exit(__doc__)
