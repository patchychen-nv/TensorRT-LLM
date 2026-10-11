# Kernel-Trace run `ktrace-0922` — follow-up sweep, 24h later

Snapshot 2026-09-24 ~02:30Z; covers the window since the previous report
(KTRACE_0922_CAMPAIGN_REPORT.md, snapshot 09-23 00:40Z). Method: the
trtllm-kf-campaign-triage skill, five passes. Every number measured; the
mid-sweep dead-ends are kept.

## Fleet and run shape (what changed)

    instances   i00: 558517 -> rolled -> 576461 (09-23 00:38Z), requeued once
                     after a 22-min heartbeat flap (09-23 18:39-19:01Z)
                i01: 558519, same attempt since its 09-22 14:18Z requeue
    activation  BOTH instances have run the 09-22 mode5 fix the whole window
                (attempt start times postdate both fix mtimes; live probes
                against each worker returned 200 on the fixed shapes)
    kfrun       alive on ipp2 (pid 3831950), --runs-per-problem 6,
                --max-running 200, tag ktrace-0922

## 1. Census (ledger + per-phase queries; list API pages at 500 AND defaults --limit 50)

Campaigns, cumulative for the tag (ledger, refreshed to 02:22Z):

    Cancelled 1147 · Failed 814 · Completed 671 · in-flight 173
    (Running 94 / Reviewing 55 / Summarizing 24)

    Speedup >= 1x when cut: Cancelled 494/681-known (4h-wall banking, normal),
    Failed 207/368-known (banked as success by the queue).

Queue (kfq @ b141c7a5): todo 828 · submitted 298 · success 79 · retry 36.
todo unchanged in 24h is CORRECT: repeats cycle through the kfrun pool, not
list_todo; the first-wave 413 problems x 6 runs are still being consumed.

Runs-per-problem (390 problems with >=1 finished run; was 246):

    2/6: 4 · 3/6: 52 · 4/6: 145 · 5/6: 97 · 6/6: 92
    finished-run outcomes: success 1372 / retry 409  (was 293/129 — ~1359
    runs finished in 24h)

Bookkeeping audit: clean. 157 settled; all 65 early-settles are the known
prepare-phase deaths; zero premature settles among run-bearing problems.

## 2. New-window Failed (created > 09-23 00:40Z): 348, all STOP_REASONs read in full

| class | n | owner |
|---|---|---|
| no-pass-submission | 144 | split — see §2d |
| mode5 input_image | 114 | **ours — real gap found, fixed this sweep (§2a)** |
| IncompleteRead(0 bytes) | 48 (45 confirmed) | **new class — KF/egress side (§2b)** |
| TransferEncoding stream cut | 21 | infra lifecycle (§2c) |
| vault-agent-init | 21 | KF k8s — now chronic (§2e) |

## 2a. mode5 did NOT flatten — the degrade walk had a blind spot

The watch-next tripwire from the last report fired. 114 new mode5 deaths
with both instances verifiably on the fixed code. Live probes with the
*fixed* shapes (message.content part, bare top-level part) returned 200 on
both workers — the fix works for the shapes it covers. The residual was
found via the request trace, not the STOP_REASON: trace rows with
status=rejected_400 carry `validation_errors` plus the full body.

**Root cause: Codex also ships screenshots inside
`custom_tool_call_output.output`** — an agent plots something, the tool
returns the PNG, and the parts list lives under `output`, not `content`.
The degrade walk covered top-level items and `item["content"]` only. Two
sub-paths, one cause:

    image WITHOUT "detail"  -> dies at proxy pydantic
                               (104/114; loc ResponseInputTextParam...)
    image WITH "detail"     -> passes pydantic, dies in ctx semantic layer:
                               "Unknown part type: input_image"
                               (8 i01 + 2 i00 proxy tracebacks — the only
                               tracebacks in the window, same root cause)

Fix: the validator now degrades `item["output"]` lists too (single edit,
upstream of both sub-paths). Regression test added with the traced shape
verbatim, including a plain-string `output` passthrough check. 19/19 tests
green **inside the live worker container** (deploy-then-test, so the tested
bytes are the served bytes). Activation: fleetctl roll i01 (successor
604888) then i00 — zero-gap, in flight at snapshot time.

Lesson recorded: **probe the client's full shape catalog, not the shapes
you fixed.** The 200-returning probe "confirmed" a fix that was still
losing ~114 campaigns/day through the untested shape.

## 2b. NEW: HTTP 200 + `IncompleteRead(0 bytes read)` — not ours, evidence attached

Status line arrives, body EOFs at zero bytes. 45 fail-times estimated
(created + DURATION) cluster at 01:48-05:35Z, 10:44-13:51Z, 21:05-23:42Z —
**~9-10h apart, periodic-looking, and NOT aligned with the single backend
requeue event of the window.**

Our side is clean at every hop, stated as zeros:
- gateway: **zero** 200-completions with bytes=0 in the entire log; no
  ERROR/5xx inside the failure windows; `client abandoned` events flat all
  day (11-43/h, no clusters) — the F1 watcher is not implicated
- proxy: zero stream terminations; the only tracebacks are the §2a 400s
- gen/ctx: zero tracebacks, zero terminated-before-completion
- worker trace: only `accepted` / `rejected_400` statuses exist

Conclusion, upgraded by the KF session's source read (their reply to
`b-ktrace-0924-followup`): `IncompleteRead` is Python http.client wording
from KF's own byom-http-relay.py — and http.client read-timeouts raise
`socket.timeout`, never IncompleteRead, so this is a **premature close**
(200 headers then FIN before the first body byte), not a timeout. The only
peer that can close there is KF's platform-api; of its two candidate
mechanisms, pod-termination severing detached streaming tasks (axum
graceful_shutdown) fits both the 0-bytes-at-header-boundary signature and
the periodic clustering — and the ~9-10h cadence matches KF's prod rollout
rhythm (multiple rc/day; each platform-api rollout severs all in-flight
BYOM streams). Confirmation ask for the KF team: rollout/pod-restart
timestamps vs our three windows. If they line up, the fix is the
stream-retry/backoffLimit item already in the escalation package — rollouts
become survivable instead of fatal.

## 2c. TransferEncoding cuts (21): fully explained, again

20 of 21 fail-times sit inside 18:11-20:05Z — the 576461 heartbeat flap +
requeue window (18:39-19:01Z, new IP). Preemption kills in-flight streams;
HTTP cannot resume them. Same verdict as last sweep, different day.

## 2d. no-pass (144) — 30% cudagym sample (43, seed 924): pressure easing

    genuinely healthy loops (avg <= 30s)      16   37%
    mid 30-300s (not all-retried)             15   35%
    starved-degraded (avg>=300s, rf==count)   11   26%
    no-calls                                   1    2%

Versus yesterday's 65% starved / 25% healthy: the KF pool has partially
recovered, and the worst endpoint is now **gpu** (eval execution), not
compile. Honest model difficulty is now the plurality of this class.

## 2e. vault-agent-init (21): from one-off to chronic

Yesterday: 13 in one cluster (read as a vault outage window). Today: 21
spread flat across the day (~1/h). This is a standing failure rate, which
strengthens the escalation ask (make the vault sidecar conditional on the
knowledge set actually referencing external Git).

## 3. TRT-LLM verdict

One real defect found and it is ours: the mode5 degrade blind spot (§2a) —
found, fixed, tested in-container, rolling. Everything else checked clean:
zero gen/ctx exceptions, zero stream terminations, gateway completion path
spotless (the IncompleteRead class specifically has zero server-side
correlates). call_id and tool-name delivery remain healthy.

## 4. Actions

Done this sweep: output-shape degrade fix + regression tests (19 green in
the live container); deployed with md5 verification; fleetctl roll started
(i01 successor 604888 up and loading, i00 next); cowork finding + msg filed
for §2b and §2e; skill updated with the new traps.

Pending human (carried + new): (1) compile-starvation escalation forward —
note §2d shows partial recovery, update the numbers before sending;
(2) problem-set regeneration for the 65 parked prepare-failures;
(3) solswarm modality MR push; (4) KF-side answer on IncompleteRead cadence.

Watch next: mode5 must flatten within a day of BOTH rolls completing — the
degrade now covers content/top-level/output; if a fourth shape exists the
trace (status=rejected_400 + validation_errors) will name it immediately.

## Tooling corrections recorded this sweep

- `kf campaign list` defaults to `--limit 50` (on top of the 500-page cap):
  Running/Reviewing counts of exactly 50 are the tell.
- `kf campaign cudagym --format json` fields live under `rows[]`
  (per-endpoint) + `totals` — not `endpoints`. A wrong key reads as
  43/43 no-calls, the same lie as the 1h-window trap.
- The request trace is the fastest 400-class debugger we have:
  status=rejected_400 rows carry `validation_errors` AND the full body —
  skip STOP_REASON archaeology and grep the trace first.
- One grep keyword can span two failure layers: `input_image` matched both
  proxy-pydantic 400s and ctx "Unknown part type" 400s. Classify by error
  *text*, then split by *emitting layer* before assigning a fix.
