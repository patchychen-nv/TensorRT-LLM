# Kernel-Trace `ktrace-0922` — the 142 problems that ended in retry

The 1241-problem pass (mirror `b141c7a5`, 2026-09-22 → 10-01, up to 8 runs
per problem plus two retry passes) ended at **1099 success / 142 retry**.
This register classifies the 142, records what was measured about each
class, and states their disposition in the follow-on `ktrace-1001` pass
(mirror `4d4563d3`). Every count below is measured from `stages.jsonl`,
the kfrun state file, and the queue lists — not estimated.

## Disposition at a glance (what happened to them in ktrace-1001)

| class | n | content changed in the new mirror? | disposition |
|---|---|---|---|
| A. prepare-phase deaths | 107 | **no** (`git diff b141c7a5b..4d4563d30` is empty for all of them) | carried straight into `list_retry` of the new queue — re-running an unchanged broken definition burns ~4 prepares each for nothing |
| B. all-runs-negative | 35 | **no** | returned to `list_todo` for 4 fresh runs each (operator preference: the previous all-negative rematch flipped 9 of 39) |

(The split is 107 + 35 = 142. Five problems from the old register appear
under A here but were counted loosely as "112" in earlier running notes;
107 is the measured number of retry-listed problems with zero recorded
runs.)

## Class A — 107 prepare-phase deaths (never ran a single campaign)

The problem's own baseline fails Kernel Factory's prepare step, so no
agent ever started. Failure signatures from the last recorded prepare
attempt of each problem:

| n | signature | meaning |
|---|---|---|
| 86 | `baseline solution did not pass CudaGym validation/measurement` | two sub-families (the stages log truncates the detail; both were root-caused with the KF-source session in `tmp/cowork/findings/a-prepare-fail-class-b-resolved.md`): **(a) ValidateRequest schema rejection** — a workload mixes custom and non-custom inputs, which CudaGym's schema forbids (`Value error: A workload cannot have both custom and non-custom inputs`); **(b) baseline fails its own correctness** — `0/N workloads passed correctness`, i.e. the recorded reference kernel does not reproduce its own trace output (INCORRECT_NUMERICAL / RUNTIME_ERROR) |
| 14 | `internal error: baseline CudaGym evaluation failed: connection error: Transport ...` | KF-side infra during baseline evaluation, not a problem-content defect — these are the only Class-A members worth a blind retry if KF confirms the outage window |
| 4 | `complex64 dtype unsupported` | the trace uses a dtype CudaGym's workload schema cannot encode |
| 2 | `GET kernelfactory .../operations → 401: invalid token: unknown kid ... after JWKS refresh` | transient KF auth-rotation failure |
| 1 | `Baseline evaluation on h100 did not finish within 600 seconds` | baseline itself exceeds the per-evaluation wall (includes GPU queue time) |

**Owner: the problem-set generator** (upstream kernel-trace export), not
serving and not the model. The regeneration recipes — with in-repo
precedents for both big sub-families — are in the cowork finding
`a-prepare-fail-class-b-resolved.md`:

- mixed custom/non-custom inputs → re-export the workload with the
  non-custom tensors folded into the custom-input dict (precedent cited in
  the finding);
- self-failing baselines → re-record the reference output on the pinned
  hardware, or mark the trace lossy;
- complex64 → re-encode as paired float32 planes, the format CudaGym
  accepts.

Until regeneration happens these problems are dead on arrival, which is
why they are parked rather than re-queued: the previous pass burned four
prepare attempts (~each a KF GPU allocation) per problem per retry wave
re-proving the same rejection.

## Class B — 35 problems, every run lost (270 losing runs total)

These ran fully — 6 to 14 campaigns each across the original pass and two
rematches — and never once beat the baseline (no run produced speedup ≥ 1
or a passing submission).

Distribution:

    by GPU     gb300 12 · h100 9 · b300 9 · gb10 5
    by model   wan2.2-ti2v-5b 5 · qwen-image-edit-2511 5 · qwen-image 4 ·
               glm-image 3 · nemotron-3-nano 3 · cosmos3-super 2 ·
               gemma-4-26b 2 · cosmos3-nano 2 · (others 1 each)

**The pattern worth reporting:** the list is dominated by
**image/video-model VAE and convolution kernels** (wan/qwen-image/
glm-image/cosmos3 conv, groupnorm, vae-output-conv shapes — e.g.
`cosmos3-edge_..._vae_output_conv_cold_cache`,
`qwen-image_..._vae_decoder_input_conv`). These baselines are typically
cuDNN/cuBLAS-backed convolutions already near roofline, where a
hand-written replacement has little room — a different regime from the
GEMM/attention/norm problems where the pass earned most of its 1099 wins.
A 23% flip rate was measured when 39 all-negative problems got a fresh
rematch (9 flipped), so "all-negative" is not proof of impossibility —
but the conv/VAE cluster's persistence across up to 14 runs suggests many
of these are genuinely at parity with their baselines.

Disposition: back in `list_todo` for ktrace-1001 (4 more runs each, new
fleet, new serving fixes). If they return all-negative again at 18 total
runs, recommend retiring them as measured-at-parity rather than
re-queueing a third time.

## Asks

1. **Problem-set owners:** run the regeneration recipes over Class A's
   90 schema/baseline problems (86 + 4). The 14 + 2 + 1 infra-flavored
   failures can simply be re-prepared once KF confirms the windows.
2. **KF:** the 14 `internal error: Transport` baseline evaluations and the
   2 JWKS-rotation 401s are yours to confirm as outages (timestamps in
   `stages.jsonl`, tag `ktrace-0922`).
3. **Data consumers:** Class B's losing trajectories are still valid
   agentic SFT negatives (the agents worked correctly and lost honestly —
   verified during the audits); only Class A problems produced nothing
   usable.
