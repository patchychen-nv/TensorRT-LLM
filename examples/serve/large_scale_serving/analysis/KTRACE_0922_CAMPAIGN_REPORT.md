# Kernel-Trace campaign run `ktrace-0922` — failure report

Snapshot 2026-09-23 ~00:40Z, run started 2026-09-22 07:50Z. Joint
investigation with the KF-source session; its findings live in
`tmp/cowork/findings/a-*.md`, mine in `b-*.md`, and every number below was
measured, not inferred.

## Fleet and run shape

    endpoint    glm53-jhb-datagen -> gateway cpu-0002:8333 (relay lead 3300s,
                abandonment watcher + unpin live)
    instances   2x 4P1D GLM-5.3-FP8, 10 nodes each, segment single-rack, 7d wall
    problems    Kernel-Trace @ b141c7a5, 1241 x 6 runs, 1 agent/round,
                effort high, max-duration 4h, --max-running 200

## 1. Census

Campaigns (list API pages at 500; totals from local ledger + phase queries):

    Failed 424 · Cancelled 152+ · Completed 85+ · Running ~94 · Reviewing ~47
    · Summarizing ~30

    Cancelled is dominated by the 4h max_duration wall — 110 of the first 152
    carried a >=1x speedup when cut, i.e. normal bounded-cost termination.
    Failed-with-speedup>=1 (62/424) settles as success in the queue (banked
    work rule).

Queue (kfq @ b141c7a5): todo 828 · submitted 389 · success 1 · retry 65.

Runs-per-problem progress (246 problems have >=1 finished run):

    1/6: 128 · 2/6: 83 · 3/6: 20 · 4/6: 8 · 5/6: 6 · 6/6: 1
    finished-run outcomes: success 293 / retry 129

Bookkeeping audit: all 65 early-settled problems are prepare-phase deaths
(never had a campaign); zero problems with runs settled before 6/6. The
repeat logic is behaving to spec.

## 2. The 424 Failed, classified (every STOP_REASON read in full)

| class | n | % | owner |
|---|---|---|---|
| no-pass-submission | 290 | 68% | split: see telemetry below |
| mode5 input_image 400 | 98 | 23% | ours — fixed, activation lagged |
| TransferEncoding stream cut | 23 | 5% | infra lifecycle (preemption) |
| vault-agent-init pod failure | 13 | 3% | KF k8s infra (new mode 7) |

### 2a. no-pass-submission (290) — 30% telemetry sample (87, seed 922)

`kf campaign cudagym <id> --window 2d --format json`, worst endpoint:

    compile-starved (avg 300-1200s, rf==count)   55   63%
    compile-starved hard (avg >=1200s, max 2106s) 2    2%
    cancelled-other                                7    8%
    genuinely healthy loops (compile 15-29s)      22   25%
    no-calls                                       1    1%

Extrapolated: **~190 of 290 died waiting on the KF compile pool; only ~74 are
honest "model could not beat the baseline".** Constants mapped by the KF
session (a-compile-starvation-constants-mapped): 840s = eval_timeout(600)+240
harness deadline; 1440s = the server-side x2 of the same (my uniform-600
census forced that correction); per-attempt compile ceiling is 420s, so the
walls are compile retries looping into the whole-eval cancel. Root-cause
candidates: eval_image digest mismatch (documented silent hang) or starved
b300 compile pool, masked by REAPI's 25-attempt retry.
**Escalation package ready for the KF team** (a's finding; forwarding is the
human's, per the no-channel constraint).

### 2b. mode5 input_image (98) — ours, and the activation gap is the lesson

Codex screenshots arrive as `input_image`; our input union had no image
member; pydantic 400s the whole request deterministically; codex retries all
fail; campaign dies. Fix (degrade to text placeholder, no payload; plus 4KB
truncation of validation-error bodies so base64 stops leaking into
STOP_REASON) was deployed at 09:45Z — **but editable installs bind at import,
and the instances kept running pre-fix code for hours while 96 more campaigns
died.** Rolling restarts (fleetctl roll, zero-gap) are activating it now.
KF-side complement: the codex modality catalog override exists but is
chat-route-only; the responses-route MR is drafted on branch
`dev-junyix-fix-byom-responses-text-only` (push pending human).

### 2c. TransferEncoding cuts (23) — not a bug anywhere

Worker logs: 0 tracebacks, 0 terminated streams. Gateway log: the cuts align
with backend lifecycle — 558519 preempted 07:16, requeued to a NEW IP in 3
minutes (as designed), killing every in-flight stream. HTTP cannot resume a
half-sent response; the only mitigations are fewer transitions (7d wall
already does this) and KF-side stream retry policy for BYOM.

### 2d. vault-agent-init (13) — new KF mode 7, root-caused (KF session)

`Pod <agentrun>: init container vault-agent-init terminated` — the agent pod
never starts; nothing reaches our endpoint. Source-level mechanism
(agent_run.rs:422-463): EVERY agent pod carries a HashiCorp vault-agent init
container (`agent-init-first` + `pre-populate-only`) whose only job is
templating the GitLab credential for external-Git knowledge cloning. If
prod.vault.nvidia.com is unreachable or JWT auth fails, the pod dies before
the agent exists, and BYOM's backoffLimit=0 makes that fatal. 13 clustered in
one window = almost certainly a vault outage, not a KF code bug. Suggested KF
improvement (in the escalation): make the vault sidecar conditional on the
knowledge set actually referencing external Git — built-in-skills campaigns
currently pay the availability tax for nothing. Sample:
y88xhanqq51p52ecs3hwqyg4bc.

## 3. TRT-LLM verdict for this window

Clean, with evidence rather than absence-of-complaints: zero Python
exceptions on both gen workers over the whole window, zero
terminated-before-completion, zero proxy client-error records; tool-call
delivery healthy in every sampled Failed campaign's logs. The two prior
serving bugs (call_id remint, bare-name delivery) remain fixed in live
traffic.

## 4. Prepare-phase failures (65 problems parked in retry)

    61  CudaGym baseline validation (mixed custom/non-custom workload inputs;
        the schema class — regeneration recipe with in-repo precedents in
        a-prepare-fail-class-b-resolved)
     4  complex64 dtype unsupported (re-encode recipe ditto)
     2  HTTP 408 + 1 timeout (transient capacity)

Owner: problem-set (upstream dl/flashinfer/kernel-trace). Human decision.

## 5. Actions

Done during this sweep: rolling instance restarts to activate mode5 fix +
error-body truncation (i00 successor up, i01 next); sample-backed starvation
rates added to the escalation; mode 7 reported to the KF session.

Pending human: (1) forward compile-starvation escalation to KF team;
(2) problem-set regeneration for the 65; (3) push the solswarm modality MR.

Watch next: mode5 count must flatten once both instances run the fix — if it
does not, the degrade path has a gap and I want to know within a day, not a
week.

## Tooling corrections recorded (so the next sweep is faster)

- `kf campaign cudagym` defaults to a 1-hour server window — pass
  `--window 2d` or historical campaigns read as "no calls" (an earlier pass
  scored 87/87 as no-calls on exactly this).
- `kf campaign list` pages at 500; per-phase `--phase` queries or the local
  ledger give true totals.
- STOP_REASON must be read in full: every BYOM failure shares the
  "backoff limit" prefix, and a 200-char truncation collapsed 424 campaigns
  into one class on the first pass.
