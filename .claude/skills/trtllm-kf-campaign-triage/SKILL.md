---
name: trtllm-kf-campaign-triage
description: >-
  Triage a batch of Kernel Factory campaign results against a self-hosted
  TensorRT-LLM endpoint: census the phases, classify every failure from its
  full STOP_REASON, split the ambiguous ones with CudaGym telemetry, verify or
  clear the serving side from its own logs, and produce a report that assigns
  each failure class an owner. Use when campaigns fail, stall, or a run needs
  a health report. Triggers on: campaign failed, why did campaigns fail,
  STOP_REASON, no passing submission, triage the run, campaign report,
  Failed campaigns, backoff limit, cudagym telemetry, is trtllm at fault.
tags: [kernel-factory, campaigns, triage, serving]
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Triage a Kernel Factory campaign run

Distilled from the `ktrace-0922` sweep. Worked examples — complete reports
produced by exactly this method, with real numbers and mid-sweep
retractions kept — live in `references/` beside this file:
`worked-example-ktrace-0922.md` (first sweep) and
`worked-example-ktrace-0924-followup.md` (the follow-up, where the
"fixed" class failed to flatten and the shape-gap hunt is shown end to
end). Read them when unsure what a section's output should look like.
The method is five passes, each producing numbers the next pass consumes.
Every classifier in here produced a confidently wrong answer at least once
before its trap was found; the traps are inline.

## 0. Ground rules

- **A verdict string is a result, not a cause.** "optimizer exited without a
  passing submission" says the agent lost; telemetry says whether it lost
  thinking or waiting. Never book the ambiguous class as "model difficulty"
  without the telemetry split — the worked example flipped from
  difficulty-majority to starvation-majority (65/25) on measurement.
- **Sample size**: few failures (≤ ~30) → read all; many → 30% seeded sample
  (`random.seed(<date>)`, record the seed in the report).
- Fixes to serving code are **editable installs: they bind at import**. A
  deployed fix does nothing until the instances restart — check activation
  before expecting a failure class to flatten (the worked example paid 96
  campaigns for this).
- **Verify a fix against the client's full shape catalog, not the shapes
  you fixed.** A live probe that returns 200 on the fixed shapes proves
  those shapes — nothing else. The follow-up sweep found 114 campaigns/day
  still dying through an untested shape (image parts under a tool result's
  `output` key, not `content`) while both instances verifiably ran the
  "working" fix. When a fixed class fails to flatten, suspect a shape gap
  first and read the real bodies from the request trace.

## 1. Census

```bash
kf campaign list --tag <tag> --phase Failed --limit 2000   # per phase
./kfq status --commit <rev>
```

Traps:
- `kf campaign list` **pages at 500** regardless of --limit, and **defaults
  to --limit 50 when the flag is omitted** — two different lies. A phase
  count of exactly 50 is the tell; always pass an explicit --limit and use
  the local ledger for >500 phases.
- Phase semantics: `Cancelled` is usually the `max_duration` wall — check the
  speedup column before calling it a failure (110 of 152 carried ≥1x in the
  worked example). `Completed` with speedup <1x is a real result (agent lost
  honestly), booked as success by the queue's banked-work rule.
- Runs-per-problem progress and the early-settle audit come from kfrun's
  state file: `runs/<tag>/state.json` → `meta.runs` (histogram) and
  `reported` (settled). Every early-settled problem should be a prepare-phase
  death; anything with runs settled before N/N is a repeat-logic bug.

## 2. Classify every Failed from its FULL stop reason

STOP_REASON is three nested layers and the outer two are identical on every
BYOM failure:

```
K8s shell   : AgentRun <id> failed: Job has reached the specified backoff limit
BYOM shell  : BYOM responses: <the actual cause>
layer 3     : the discriminating text
```

Traps:
- **Never truncate before classifying.** A 200-char cut collapses the whole
  set into one "backoff limit" class (424-for-424 in the worked example).
  Fetch full: `kf campaign show <id>` and take everything after
  `BYOM responses:`; parallelize with `xargs -P8`.
- Layer-3 signatures and owners (modes 1–7 live in the cowork PROBLEM.md):

| layer-3 contains | class | owner |
|---|---|---|
| `optimizer exited without a passing submission` | ambiguous — go to §3 | split |
| `input_image` / `ResponseInputTextParam` / `Unknown part type` | multimodal 400 (mode 5) | serving (fixed; check activation) |
| `HTTP 200: ClientPayloadError ... TransferEncodingError` | mid-stream cut | infra lifecycle — go to §4 |
| `HTTP 200: Incomplete response: IncompleteRead(0 bytes read)` | premature close, NOT a timeout (http.client timeouts raise socket.timeout) | KF platform-api rollouts severing streams, once §4 zeros hold |
| `init container vault-agent-init terminated` | pod died pre-agent (mode 7) | KF k8s / vault outage |
| `HTTP 400` other | read it; new protocol gap is possible | serving until proven otherwise |

One keyword can span two failure layers: `input_image` matched proxy-pydantic
400s AND ctx-side "Unknown part type" 400s (same root cause, different
emitting layer — an image part WITH `detail` passes pydantic and dies
deeper). Classify by error text, then split by emitting layer before
assigning a fix. For any deterministic-400 class, skip STOP_REASON
archaeology: the worker request trace records rejected requests with
`status=rejected_400`, `validation_errors`, and the FULL body — grep the
trace for the keyword near the server.log 400 timestamps and read the real
item shape directly.

## 3. Split the ambiguous class with CudaGym telemetry

```bash
kf campaign cudagym <id> --window 2d --format json
```

**Trap: the server default window is 1 HOUR.** Without `--window`, every
historical campaign reads "no calls" (an 87/87 all-no-calls pass happened on
exactly this). **Trap 2: the JSON layout is `rows[]` (one per endpoint) plus
`totals`** — guessing `endpoints` as the key scored 43/43 as no-calls, the
same lie by a different door. Row fields: `endpoint`, `avg_duration_ms`,
`p95_duration_ms`, `cold_start_count`, `retryable_failure_count`, `count`,
`last_error_sample`. The worst endpoint is not always `compile` — eval
`gpu` starvation looks identical and has shown up as the dominant one.

Tiers (worst endpoint):

```
avg >= 1200s                      starved-hard   (deadline constants: 840 =
                                  eval_timeout(600)+240; 1440 = 2x600+240,
                                  the x2 is server-side; per-attempt compile
                                  ceiling is 420s — a wall above that is
                                  RETRIES looping, not one slow compile)
avg 300–1200s, rf == count        starved-degraded (every call cold + retried)
avg <= ~30s                       genuinely healthy → honest model difficulty
```

Starved-* is KF-infra territory (compile pool / eval_image digest silent
hang); healthy loops are the only defensible "model could not do it".

## 4. Verify or clear the serving side (do not assume either way)

For any class that could implicate trtllm-serve, check in this order — each
is a different component and they fail independently:

```bash
# workers (per attempt dir under var/<date>/<run>/attempt-N/)
grep -c "Traceback (most recent call last)" gen-0.log            # want 0
grep -oE "Responses stream terminated before completion \([a-z_]+\)" gen-0.log
grep -oE "Client error to [^ ]+/v1/responses: [A-Za-z ]+" server.log

# gateway (job StdOut via scontrol show job <gw> | grep StdOut)
grep -aE "sse relay stopped side=|truncated upstream framing|backend (appeared|gone)" <log>
```

- `TransferEncodingError` at the client is **never the cause** — it means an
  upstream closed a chunked body early. Look one hop upstream, and correlate
  cut timestamps against `backend gone/appeared` lines: a preemption requeue
  (same job id, new IP, ~3 min) kills every in-flight stream and is not a
  bug anywhere.
- **Garbled-generation check** (one confirmed case, 2026-10-02): a session
  whose raw `text` is token-salad (multilingual fragments, no coherent words)
  while every neighbor in the same raw file is fine. Frames-level garbled =
  the model generated it (not a parser artifact); single-session = suspect a
  KV prefix corrupted across an instance-death gap, not weights. Heuristic:
  fewer than ~8 four-letter ASCII words in the first 400 chars. Mark the
  campaign's trajectory as poisoned for SFT; do not ship it as a negative.
- Deeper tool-call verification when needed: dedupe by `call_id` (Responses
  clients resend full history — item counts inflate ~30x), classify results
  on the **status line only** (bodies quote words like "aborted"), and read
  the declared tool set from the request's `additional_tools`, never from
  memory. Raw model text joins via `disagg_request_id` to
  `<attempt>/raw_output/`.
- For wire-correctness questions (does the final snapshot agree with the
  stream? did a parser rewrite content? is a binding lost in conversion?)
  switch to the sibling skill `trtllm-serving-consistency-audit` — that is
  reconciliation/replay work, not failure triage.

## 5. KF-side questions: cowork first, source second

- If a cowork peer session exists (`tmp/cowork/`): `cw sync`, hand off with a
  finding + msg, and watch their dir by **file snapshot hash, never ssh** —
  ssh fails systematically from backgrounded watchers (agent socket absent)
  and reads as "job gone".
- No peer or no reply: search scratch for an existing checkout
  (`/tmp/explore/solswarm`, `find /home/scratch* -name solswarm`) before
  cloning `https://gitlab-master.nvidia.com/atlas/solswarm`. Same rule for
  harness sources (codex/avo/opencode): local search before clone.
- Escalations to KF humans go through the operator — sessions may be
  DM-constrained.

## 6. Report shape

One file under `examples/serve/large_scale_serving/analysis/`, sections:
fleet/run shape → census (with bookkeeping audit) → failure classes table
(n, %, owner) → per-class evidence with the telemetry split → serving
verdict (state the zeros, not just "no complaints") → prepare-phase failures
→ actions done / pending-human → tooling corrections learned this sweep.
Every number measured; every retraction kept (a corrected mid-sweep verdict
is part of the record, not an embarrassment to delete).
