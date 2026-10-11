<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# analysis

Per-request and engine-side metrics for one serving run, from what an attempt directory holds:
`request_trace/` (client-facing requests and responses), `perf_metrics/` (proxy and worker timing),
`adp_route_trace*.jsonl` (routing decisions) and the worker logs with the iteration lines:
`ctx-N.log` / `gen-N.log` on a disaggregated run, `server.log` on an aggregated one.

```
python3 analysis/report.py <attempt_dir>                        # whole run
python3 analysis/report.py <attempt_dir>/request_trace/<hour>   # one UTC hour
```

Writes `requests.csv`, `ctx_iters.csv`, `ctx_rank_iters.csv`, `gen_iters.csv`, `gen_rank_iters.csv`
and `REPORT.html` under `_reports/<run>[_<hour>]/`. The page opens with §0 · Metrics (below), then §1 requests, then the engine sections. On an aggregated run the four engine CSVs are
replaced by `worker_iters.csv` / `worker_rank_iters.csv` and the page has one engine section,
"Worker (prefill + decode)", instead of the ctx and gen sections.

Everything joins on the server's request id: `disagg_request_id` (the proxy's) on a disaggregated
run, `client_id` (the server's own `request_id`, unique per server process) on an aggregated one.
The mode is decided from the perf records (only a disaggregated run has proxy records), the engine
side from the log names.

| file | role |
|---|---|
| `requests_table.py` | one row per request: identity, usage tokens, latency phases, KV blocks, routing |
| `engine_iters.py` | iteration lines → per-rank rows and per-instance (ranks pooled) rows, pads removed |
| `charts.py` | SVG line charts and histograms, no dependencies |
| `report.py` | entry point; CSVs plus the HTML |
| `common.py` | time parsing, percentiles, CSV writer |
| `modeling.py` | request / engine models, scheduling and observed conversation diagnostics |
| `model_fit.py` | bounded linear fitting with chronological validation |
| `model_view.py` | shared model input preparation, JSON output and HTML rendering |
| `model_report.py` | model existing CSV reports, optionally merging consecutive hours |
| `modeling.py` | optional request / engine models and observed conversation diagnostics |
| `model_fit.py` | bounded sampling, chronological validation and linear fitting using the standard library |
| `model_report.py` | recompute models from compatible hourly report CSVs |
| `model_view.py` | HTML presentation of model results |

## Optional empirical modeling

Add `--model` to build `model.json` and include a modeling section in `REPORT.html`:

```bash
python3 analysis/report.py <attempt_dir> --hour 2026-09-28T17 --log-tz America/Los_Angeles --model
```

This reuses the parsed request and engine rows, preserving their in-memory timestamp precision. Existing
reports can be modeled without scanning the logs again:

```bash
python3 analysis/report.py <attempt_dir> --hour 2026-09-28T17 --out <report_dir> --from-csv --model
```

CSV timestamps have whole-second precision; the model records this limitation. The separate millisecond latency
columns retain fractional values, subject to CSV numeric rounding. `--json --model` also writes the usual `rollup.json`;
`--json --json-only --model` writes `model.json` and `rollup.json` without the CSVs or HTML.
Without `--model`, the report follows the existing analysis path.

The models describe associations within the observed workload and deployment:

| Model / diagnostic | Inputs and interpretation |
|---|---|
| Request prefill latency | Effective new-input and total-context tokens; separate fits by context worker, new-token source, model and route |
| KV transfer latency | Transfer bytes on disaggregated runs; no transfer fit on aggregated runs |
| Engine device step | Instance iteration workload, decode token slots and cached-context tokens when available; separate fits by role, worker, engine lifetime, rank count and iteration kind |
| Scheduling | Rank skew, budget / batch occupancy, paused requests, KV state and host / device step summaries |
| Observed conversation activity | Request occupancy, session spans and gaps between observed requests |

Each linear fit reports the intercept and coefficients in the original units, training feature ranges, retained
and dropped features, sample and filter counts, and MAE / RMSE / R² on the held-out time interval. Samples are
ordered chronologically with an approximately 80% / 20% training / validation split, keeping equal timestamps
together and requiring at least 20 training and 5 validation samples. A deterministic reservoir bounds each
group's samples at 20,000. Constant or collinear
features are dropped; insufficient data is reported instead of presenting unsupported coefficients.

Before interpreting the fits, inspect their data-quality metadata and the report's join, padding, topology and
timezone checks. Request fits use completed observations wholly within the engine window; hourly responses can
extend past that window and are therefore excluded from these fits. Effective new-input tokens keep their
provenance: usage-derived values and the disaggregated `ctx_blocks_new × tokens_per_block` fallback belong to
separate groups. The fallback is a block-granularity estimate. Decode token slots include MTP / CUDA-graph padding
and are not accepted output tokens. Host and device step values are elapsed spans, not GPU busy time. Coefficients
are neither causal effects nor hardware-capacity estimates, and predictions outside the training ranges need
separate validation.

Observed sessions can contain tools, user think time or client scheduling between turns. They can also be
truncated by the selected time window. `model.json` explicitly leaves full job metrics unavailable: computing
job latency, jobs/h/user, jobs/h/GPU or a parallel critical path requires explicit job boundaries and the tool /
branch dependency graph. Do not treat conversation spans as completed-job latency.

To fit compatible consecutive hours together, use their report directories:

```bash
python3 analysis/model_report.py \
  _reports/<run>_2026-09-28T17 _reports/<run>_2026-09-28T18 \
  --out _reports/<run>_model_T17_T18
```

The command reads CSVs, `engine_meta.json` and `summary.json`, checks that the reports describe the same run,
attempt, configuration, GPU count and log timezone, removes duplicate boundary iterations, and writes
`model.json` plus `MODEL.html`. The UTC hours must be consecutive. This avoids reparsing the worker logs but
retains the CSV timestamp precision limit. Fit combined details again; hourly coefficients, percentiles and
compressed `rollup.json` distributions cannot be averaged into a model for the combined window.
The CSV entry loads the selected detail tables into memory; select a small number of hours on a compute node.
Fitting samples are bounded, but the input tables are not.

## REPORT.html §0 · Metrics

The headline numbers of the run as mean / p10 / p25 / p50 / p75 / p90 / p99 (mean, p10, p50 and p90 in red), over
completed requests only, in three blocks. Every input is a column of `requests.csv`; the formulas are the same in both
deployment modes and only the inputs differ (see the second table).

| block | row | formula | distribution over |
|---|---|---|---|
| E2E | E2E latency | `e2e_ms` | requests |
| E2E | Output tokens / E2E | `osl / e2e_s` | requests |
| E2E | Σ output tokens / Σ E2E | `Σ osl / Σ e2e_s` | 1-min buckets |
| E2E | (ISL new + OSL) / E2E | `(isl_new + osl) / e2e_s` | requests |
| E2E | Σ (ISL new + OSL) / Σ E2E | `Σ (isl_new + osl) / Σ e2e_s` | 1-min buckets |
| Throughput | Requests / s / GPU | `N / T / G` | 1-min buckets |
| Throughput | (ISL new + OSL) tokens / s / GPU | `Σ (isl_new + osl) / T / G` | 1-min buckets |
| Throughput | Output tokens / s / GPU | `Σ osl / T / G` | 1-min buckets |
| Throughput | Input tokens (ISL new) / s / GPU | `Σ isl_new / T / G` | 1-min buckets |
| Token-level SLO | TTFT | `ttft_ms` | requests |
| Token-level SLO | TPS / user | `(osl − 1) / (e2e_s − ttft_s)` = 1 / TPOT, requests with `osl ≥ 2` | requests |

Bucketed rows: a request belongs to the minute of its `finished_at`; the first and last partial minutes are dropped;
an empty minute is 0 in the Throughput block and skipped in the Σ / Σ rows (0 / 0). Their percentiles run over the
minutes, but their **mean cell is the whole-window value** — Σ over every completed request divided by Σ E2E, or by
the window length `T` (first start to last finish) × `G`. That is the request-second-weighted mean of the minutes,
not their plain average; under a fluctuating load the two differ (36 % on an overloaded hour), because a heavy
minute carries more request-seconds and a lower per-user speed. `Σ osl / Σ e2e` is also output throughput ÷ mean
concurrency (Little's law).

| input | disaggregated | aggregated |
|---|---|---|
| `e2e_ms` | trace `finished_at − started_at` | same |
| `ttft_ms` | proxy arrival → the ctx worker's first token, before the KV transfer; so `e2e − ttft` under TPS / user includes the transfer and the proxy hop | worker `server_arrival_time → first_token_time` |
| `osl`, `isl_total`, `isl_cached` | the usage block the client received | same |
| `isl_new` | `isl_total − isl_cached`; where a /v1/responses usage reports `cached == total` (the proxy bug fixed by 2026-09-16) the row falls back to `ctx_blocks_new × tokens_per_block` | same, no fallback: the one worker record's new blocks include the decode blocks |
| `G` | `Σ role: instances × tp × pp × cp` from `ctx_config.yaml`, `gen_config.yaml` and `disagg_config.yaml` (6P1D: 6 × 4 + 1 × 8 = 32); cross-checked against the distinct ranks per worker log | `tp × pp × cp` of `server_config.yaml` (8); ranks of `server.log` |

The GPU derivation and the cross-check are printed in the page notes; a mismatch between config and logs is flagged there.

**In the instance over time** (chart + table at the end of §0): four occupancy series sampled once a second.
*Client requests in the instance* counts conversations — turns grouped by `session`, the conversation_id the serve path
resolves for routing affinity — from the first turn's arrival to the last turn's end, so the tool-call gaps between
turns are included with no cap: the client is away but will come back for the same KV. *Turns in flight* counts
server-side requests (arrival → response end); *turns in prefill* runs from ctx first scheduled (`first token −
prefill_ms`) to the first token; *turns in decode* from the first token to the response end. Rejected and unanswered requests are excluded; a turn still open at the window's end is
cut there, and a conversation whose next turn falls in the next hour looks finished on an hourly report.

## REPORT.html §1 · Requests

**Latency** lists the phases of a request in the order they happen, indented by containment (an indented row lies
inside the span above it: TTFT and Decode inside E2E; proxy pre-dispatch, ctx overhead, prefill queue and prefill inside
TTFT; KV transfer and the gen decode inside Decode), with n / mean / p50 / p90 / p99 and each phase's **share of E2E** =
Σ phase / Σ E2E over the completed requests that have the phase (weighted by request length). **Tokens** is ISL total /
cached / new, OSL, cache hit ratio, KV transfer bytes (disaggregated) and the draft acceptance rate; the KV block counts
stay in `requests.csv`. The page uses Inter and JetBrains Mono from Google Fonts and falls back to the local sans-serif
stack offline. On an aggregated run **By rank (from the routing trace)** gives, per attention-DP rank, the request
count and the p50 of TTFT, engine queue and decode, each followed by its ratio to the lowest rank in brackets.

## REPORT.html §2 / §3 on a disaggregated run

Both engine sections open with a highlighted budget line (`max_num_tokens`, `max_batch_size`, `tokens_per_block`, workers ×
ranks, KV quotas of that worker type), then a scheduling table (ctx: iterations, pad share, average prefill requests and
ctx tokens per rank, budget utilization, busy ranks, rank skew, Σ paused, mean host / device step; gen: iterations,
idle-rank share, average decode requests and token slots per rank, tokens per request, batch occupancy, Σ paused, mean
host / device step) and a separate KV table (kv_cache_util, pool filled peak, cumulative hit rate, cross-tier totals).
The per-rank tables carry, per rank, Σ prefill requests / Σ real ctx tokens / share (ctx), average scheduled requests
and tokens over all of the rank's iterations (a pad or idle dummy counting 0), idle iterations, Σ paused, cumulative hit
rate (ctx), kv_cache_util (gen), pool filled, capacity blocks and the mean host / device step — every cell followed by its
ratio to the lowest rank in brackets.

## REPORT.html §2 on an aggregated run

The section opens with the budgets in a highlighted line (`max_num_tokens`, `max_batch_size`, `tokens_per_block`, rank
count, KV quotas), then a scheduling table (iterations, prefill / decode-only counts, idle-rank share, average prefill and
decode requests, ctx tokens and decode token slots per rank, budget utilization, batch occupancy, Σ paused, mean host
and device step) and a separate KV table (kv_cache_util, pool filled peak, cumulative hit rate, cross-tier totals). The
per-rank table lists Σ prefill requests, Σ real ctx tokens and their share, average scheduled requests and tokens
(prefill + decode, over all of the rank's iterations), idle iterations, Σ paused, cumulative hit rate, pool filled,
capacity blocks and the mean host / device step, every cell followed by its ratio to the lowest rank.

**Step times** (`tensorrt_llm/_torch/pyexecutor/py_executor.py`, `profile_step`): `host_step_time` is `time.time()`
across the loop body that just finished, i.e. the wall clock of the iteration on the log line; Σ host step over an hour
equals the hour. `prev_device_step_time` is the elapsed time between two CUDA events recorded at the top of consecutive
loops on the GPU stream; the ping-pong event pair means the value printed on iteration N describes iteration N − 1, and
because the events bracket the whole loop it is the GPU-timeline span of the iteration, not GPU busy time (its sum also
equals the hour). `engine_iters.realign_device_step` shifts the device value back one iteration, so `device_step_ms`
on a row is the GPU-timeline span of that same iteration and the two columns are comparable row by row; a rank's last
row has no successor and keeps None.

## requests.csv

| column | definition |
|---|---|
| `rid` | join key to perf_metrics and the routing trace: `disagg_request_id`, or `client_id` when that is null (aggregated) |
| `session` | the conversation key the trace resolved for the request (headers, else `prompt_cache_key` / `client_metadata` in the body: the Codex thread id); `_no_session` when nothing was found |
| `started_at` | request arrival: middleware `server_arrival_time` (steady clock) plus the wall-clock offset recovered as `min(recorded_at − server_arrival_time)` over the attempt |
| `handler_entry_at` | the trace's `recorded_at`, stamped after the body was read and validated |
| `finished_at` | response `finished_at` |
| `parse_ms` | `handler_entry_at − started_at`: body read and validation before the handler ran (about 1.5 s per MB) |
| `server_overhead_ms` | worker `arrival_time − server_arrival_time`: HTTP handler to engine enqueue (the ctx worker's in a disaggregated run) |
| `proxy_dispatch_ms` | proxy `disagg_ctx_dispatch_time − disagg_server_arrival_time` (disaggregated only) |
| `status` | `rejected_400`, or the response status: `completed`, `error`, `client_disconnected`, `no_response`. In single-hour mode the next hour's response files are read as well (a request arriving at 09:59 ends in T10), so `no_response` means no response line in either hour; only the newest hour of a live instance still shows boundary spill as `no_response` |
| `isl_total`, `isl_cached`, `osl` | usage the client received (chat: last chunk; responses: `response.completed`) |
| `isl_new` | `isl_total − isl_cached` |
| `cache_hit_ratio` | `isl_cached / isl_total` |
| `ttft_ms` | proxy `disagg_server_first_token_time − disagg_server_arrival_time`; aggregated: worker `first_token_time − server_arrival_time` (not `server_first_token_time`, which that server stamps at response end) |
| `e2e_ms` | `finished_at − started_at` of the same trace_id |
| `decode_ms` | `e2e_ms − ttft_ms` |
| `prefill_queue_ms` | worker `first_scheduled_time − arrival_time` (ctx worker / the aggregated worker: "engine queue") |
| `prefill_ms` | worker `first_token_time − first_scheduled_time` (several chunked iterations on a cold long prompt) |
| `gpu_prefill_ms` | ctx worker `time_breakdown_metrics.ctx_gpu_forward_time` (disaggregated only) |
| `kv_transfer_ms`, `kv_transfer_bytes` | gen worker `kv_cache_transfer_end − start`, `kv_cache_size` (disaggregated only) |
| `gen_decode_ms` | `last_token_time − first_token_time` on the gen worker record, or on the one worker record when aggregated |
| `mtp_acceptance` | `speculative_decoding.acceptance_rate` of the decode-side record: accepted / proposed draft tokens over the request |
| `ctx_blocks_total/new/reused`, `gen_blocks_total` | worker `kv_cache_metrics` (aggregated: the one record fills both) |
| `ctx_instance`, `routed_rank`, `route_phase`, `route_iter`, `route_log_iter` | routing decision (`best_rank`, `phase`, batch iteration); aggregated: `ctx_instance` is `server` and the rank is where prefill and decode ran |

Caveat: on `/v1/responses` the server currently reports `cached_tokens == input_tokens`; the value is
recorded as received. `ctx_blocks_reused × tokens_per_block` is the engine-side measurement.

## ctx_rank_iters.csv / gen_rank_iters.csv / worker_rank_iters.csv (one row per worker, iteration, rank)

| column | definition |
|---|---|
| `engine_instance` | engine lifetime inside the log (the KV-sizing dry run is 0, the real engine 1) |
| `ctx_requests`, `ctx_tokens` | `states.num_ctx_requests` / `states.num_ctx_tokens` as logged: the rank's prefill work this iteration |
| `gen_tokens` | `states.num_generation_tokens`: decode token **slots**, requests × (draft length + 1) rounded up to the CUDA-graph batch size and, in a pure-decode attention-DP iteration, aligned across ranks. Not the tokens accepted; that is `mtp_acceptance` in requests.csv |
| `is_adp_pad` | idle-rank dummy. ctx: all four `kv_*_blocks` counters unchanged since this rank's previous iteration and `ctx_tokens ≥ 0.9 × max_num_tokens`. gen: one scheduled request with `kv_cache_util == 0`. aggregated worker: the gen rule, and no context request or tokens on the rank (it never gets a near-budget context pad) |
| `ctx_tokens_real` | `ctx_tokens`, or 0 for a pad |
| `decode_requests_real` | gen: `scheduled_requests`; aggregated: `scheduled_requests − ctx_requests`; 0 on an idle rank |
| `gen_tokens_real` | `gen_tokens`, or 0 on an idle rank |
| `total_tokens_real`, `total_budget_util` | `ctx_tokens_real + gen_tokens_real`, what the scheduler charged against `max_num_tokens` this iteration, and that share (aggregated worker only) |
| `token_budget_util` | `ctx_tokens_real / max_num_tokens` |
| `paused_requests` | `num_paused_requests`: admitted requests the scheduler set aside this iteration for lack of KV blocks or budget |
| `kv_hit_rate_iter` | `Δkv_reused_blocks / (Δreused + Δmissed)` for this iteration |
| `kv_hit_rate_cum` | the log's `kv_hit_rate`: the same ratio over the engine lifetime |
| `kv_cache_util` | as logged: share of blocks pinned by in-flight requests (`1 − available / max`) |
| `kv_capacity_blocks` | `max(kv_free_blocks + kv_evictable_blocks)` over the engine lifetime |
| `kv_pool_filled_ratio` | `1 − kv_free_blocks / kv_capacity_blocks`: blocks holding content, pinned or reusable |
| `kv_{offload,onboard,host_dropped}_blocks_delta/_total` | cross-tier movement this iteration (the log prints these drained per iteration, not cumulative) / running total since engine start. Unit is pages = pool-group slots, the same unit as `kv_free_blocks`; one page is one KV block of `tokens_per_block` tokens across all layers of the pool group |
| `host_step_ms`, `device_step_ms` | `host_step_time` (wall clock of this iteration); `prev_device_step_time` **of the next line of the same rank**, i.e. the GPU-timeline span of this iteration — the log prints it one iteration late and the table shifts it back; None on a rank's last row (see §2 notes) |

## ctx_iters.csv / gen_iters.csv / worker_iters.csv (ranks pooled per iteration)

| column | definition |
|---|---|
| `busy_ranks`, `has_prefill` | ranks with `ctx_tokens_real > 0`; any |
| `ctx_requests_sum` | Σ `ctx_requests` (prefill requests scheduled this iteration) |
| `ctx_tokens_mean` | mean of `ctx_tokens_real` over all ranks, pads included as 0 |
| `token_budget_util` | `ctx_tokens_mean / max_num_tokens` |
| `*_rank_skew` | `(max − mean) / mean` across ranks; 0 = even, `ranks − 1` = one rank has everything |
| `idle_ranks` | ranks flagged as pads this iteration |
| `decode_requests_sum`, `has_decode` | Σ `decode_requests_real`; any (gen) |
| `batch_occupancy` | mean `decode_requests_real` / `max_batch_size` (gen) |
| `tokens_per_request` | `Σ gen_tokens_real / Σ decode_requests_real` (gen): token slots per request, draft length + 1 under MTP plus whatever CUDA-graph padding shows |
| `decode_only` | `has_decode and not has_prefill` (aggregated worker) |
| `total_tokens_mean`, `total_budget_util` | mean of `total_tokens_real` over ranks; `/ max_num_tokens` (aggregated worker) |
| `kv_hit_rate_iter` / `kv_hit_rate_cum` | counters summed over ranks first, then divided |
| `kv_pool_filled_mean` | mean of the per-rank ratio |

## Aggregated deployments

One server (`server.log`, `server_config.yaml`, `adp_route_trace.jsonl`) runs prefill and decode on
the same ranks, so there is neither a proxy nor a KV transfer, and every request has exactly one
worker perf record. What changes:

- **Join.** `rid = client_id` (the response line's `disagg_request_id` is null); it equals the worker
  record's `request_id` and the routing decision's `client_id` (not its `req_id`, which is the engine's
  small sequential counter there). Verified 100 % on the traces this was built on.
- **Latency.** TTFT = worker `server_arrival_time → first_token_time`; server overhead = `→ arrival_time`;
  engine queue = `arrival_time → first_scheduled_time`; prefill = `→ first_token_time`; decode =
  `first_token_time → last_token_time`; E2E from the trace as before. Draft acceptance from the same record.
- **Report §1** shows the aggregated phase tree (no proxy dispatch, KV transfer or GPU-prefill rows; the body parse
  sits inside the server overhead) and "By rank (from the routing trace)" instead of "By context instance".
- **Report §2 "Worker (prefill + decode)"** replaces the ctx and gen sections: per worker, iterations with a
  prefill chunk vs decode-only, average prefill and decode requests per rank, ctx tokens and decode token
  slots per rank, budget utilization (ctx + decode) against `max_num_tokens`, batch occupancy against
  `max_batch_size`, paused requests, kv_cache_util, pool fill, hit rates, cross-tier totals, and step time
  split by whether the iteration carried a prefill chunk (a chunk on any rank stretches the step for all).
  Per-rank charts: decode requests, ctx tokens, decode token slots, ctx + decode tokens with the
  `max_num_tokens` line, budget utilization, paused, kv_cache_util, pool filled, cumulative hit rate, device
  step by iteration kind, and the three cross-tier charts.
- **Idle ranks** are the generation-style dummy (one scheduled request, `kv_cache_util` 0); there are no
  near-budget context pads on this worker. `rollup.json` carries `engine.mode` and an `engine.mixed` side.
