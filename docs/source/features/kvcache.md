<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV Cache System

The KV cache stores previously computed key-value pairs for reuse during generation in order to avoid redundant calculations. The TensorRT LLM KV cache system also supports reuse across requests and uses a suite of tools like offloading and prioritized eviction to increase reuse. It supports variable attention window sizes and Multi-Head Attention (MHA) optimization techniques such as MQA and GQA.

## The Basics

The KV cache is a pool of blocks that can hold KV state for a fixed number of tokens. Multiple layers are packed within a single block, which requires all the layers to have the same number of heads and the same attention window size. A separate pool is created for each combination of attention window size and number of heads to support variable attention window size and optimization techniques like GQA.

The number of tokens that can be stored in a single block can be set by user when the model engine is created. It must be a power of two greater than 1. Blocks are assigned to requests as needed. Blocks are stored in a search structure as they are filled by requests, this allows later requests to reuse KV state if they have a matching prefix.

If more than one pool is created, available memory is divided among the pools. The fraction to assign to each pool is determined during initialization and is static. This is not optimal and we are working on providing a better solution.

## Reuse Across Requests

Blocks containing KV state computed for previous requests are stored in a radix search tree as soon as they are filled. A search is performed when a new request is added, and matched blocks are reused instead of calculated. Blocks that are reused can be shared among multiple requests, so reuse saves memory as well as computations.

Blocks remain reusable until they are evicted from the search tree. Eviction happens when a new (blank) block is needed. The core eviction scheme is prioritized LRU. All blocks are assigned a priority between 0 and 100 (100 being most important). All blocks of the lowest priority must be evicted before any blocks of the next priority can be evicted. If all blocks have the same priority, the least recently used block is evicted.

When a block is evicted from primary memory, its KV state is copied to a block in secondary memory. The secondary memory block remains in the search tree, so the block remains reusable until it is evicted from secondary memory. Eviction from secondary memory happens when a new block in secondary memory is needed to offload a primary block. The eviction scheme is the same for primary and secondary blocks.

One caveat in the current code is that only leaf blocks can be evicted (leaves are blocks with no descendants in the radix tree). This design works well for full attention layers, but not for limited attention layers. This will be fixed in a future version.

### Retention Policy

Blocks are assigned priority in line with the [retention policy](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.KvCacheRetentionConfig) of the request. Blocks with lower priority scores will be freed preferentially to blocks with higher priority. The retention policy is a list of [TokenRangeRetentionConfig](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.KvCacheRetentionConfig.TokenRangeRetentionConfig) objects, each specifying priority for a given range of tokens, such as "assign priority X to tokens 10 through 61". You can also assign a duration in milliseconds for this to remain in effect. Priority reverts to the default of 35 after a period of ```duration_ms``` has elapsed from the first time the block was made available for reuse. TokenRangeRetentionConfig only applies to input (prompt) tokens. The property ```decode_retention_policy``` specifies what priority to assign to blocks with generated (decoded) tokens and ```decode_duration_ms``` specifies how long this should remain in effect. Priority reverts to the default after expiration. Any property that expects a duration can be set to None. This indicates that particular part of the retention policy never expires.

Not in use: ```transfer_mode``` is a debug option and should not be used.

See [this example](../examples/kvcacheretentionconfig.md) for an example of how to change block priorities of specific requests by altering their retention policy.

### Speculative Decoding

Reuse across requests is supported by all speculative decoding models. Please see [speculative decoding](speculative-decoding.md) for more details.

## Limited Attention Window Size

TensorRT LLM takes advantage of layers with limited attention window size in order to reduce computations and memory usage. Blocks that leave the attention window are freed and placed on the radix search tree so they can be reused.

## MQA / GQA

TensorRT LLM takes advantage of grouped query attention in order to save memory. KV cache will create blocks with only enough space to store state for the discrete query head groups. For MHA, there is one group per head, for MQA there is a single group for all the heads. GQA strikes a balance between these two.

## Controlling KV Cache Behavior

Many of the features in the KV cache system are optional or have user defined properties that alter how they work. Users can control KV cache features through class [KVCacheConfig](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.KvCacheConfig). The remainder of this section describes how to change the most important behaviors of the KV cache system.

See [this example](../examples/kvcacheconfig.md) for an example of how to use KvCacheConfig to control KV cache behavior.

### Datatype

Perhaps the most important property is ```dtype``` which specifies what data type is held in KV cache. The default 'auto' specifies that data type should be inferred from model config.

### How Much Memory is Allocated to KV Cache

Property ```free_gpu_memory_fraction``` is a ratio > 0 and < 1 that specifies how much of free GPU memory should be allocated to KV cache. The default is 90% (ratio of 0.9). If ```max_tokens``` is also set, KV cache will determine how much memory is needed to hold ```max_tokens``` and will allocate the lesser of ```max_tokens``` and ```free_gpu_memory_fraction```.

### Enable/Disable Cross Request Reuse

Block reuse across requests is enabled by default, but can be disabled by setting ```enable_block_reuse``` to False.

`scheduler_config.enable_prefix_aware_scheduling` controls only scheduler-side use of prefix-reuse estimates. When it is
`True` (the default), schedulers can use estimated reusable KV tokens to defer duplicate first-chunk context requests and
to reduce token-budget accounting for requests that are expected to reuse cached prefix blocks. When it is `False`,
these scheduler estimates are disabled and reusable-token estimates remain zero, but actual KV block reuse is still
controlled by `kv_cache_config.enable_block_reuse`.

For example, this keeps runtime KV block reuse enabled while disabling prefix-aware scheduler admission and token-budget
credit:

```yaml
kv_cache_config:
  enable_block_reuse: true
scheduler_config:
  enable_prefix_aware_scheduling: false
```

### Selecting the KV Cache Manager

TensorRT LLM ships two KV cache manager implementations. `use_kv_cache_manager_v2`
selects between them and defaults to `auto`, which adopts the model's own
preference and falls back to the V1 C++ manager for models that do not declare
one. Set it to `true` or `false` to override the model default.

Models that select the V2 manager by default:

| Model | Reason |
| --- | --- |
| Hybrid Mamba (NemotronH and its multimodal models, Qwen3-Next) | Attention KV and Mamba state pools must be sized together |
| DeepSeek-V4 | Sparse attention attaches auxiliary per-layer buffers |
| GPT-OSS | Sliding window on every other layer (VSWA), so the sliding-window and full-attention pools are sized independently |
| Gemma3 / Gemma4 (text and multimodal) | Alternating sliding-window and full-attention layers (VSWA); same independent pool sizing |
| Llama / Llama4 | Uniform KV pool layout; chunked attention does not partition the pools |

Separately, Gemma4 hybrid attention and sparse-attention models are routed to
V2 unconditionally: their per-layer buffer layouts cannot be represented by V1's
unified pool, so `use_kv_cache_manager_v2` does not apply to them.

For a model whose `layer_types` mixes sliding-window and full-attention layers
and that publishes a single `sliding_window` (GPT-OSS, Gemma3), the V2 manager
derives one attention window per layer from `layer_types` when
`max_attention_window` is not set: sliding layers get `sliding_window`, full
layers get `max_seq_len`, and the two window sizes form two layer groups whose
pools are sized independently. The derived list is logged at startup. Set
`max_attention_window` explicitly to override the derivation; a single entry
restores one full-context pool for every layer. With derived windows,
`pool_ratio` must carry one entry per layer group (two for such a model). If a
configured `pool_ratio` does not match the derived group count, the manager
logs a warning and keeps the single-window default, so existing configurations
continue to run.

For the native V2 cold-storage representation and codec extension contract, see
[KVCacheManagerV2 Cold-Page Codec Design](../developer-guide/kv-cache-cold-page-codec.md).

### SWA Endpoint Retention

To prefer cached sliding-window attention (SWA) blocks near a prompt's endpoint
under cache pressure, opt in with the prototype
`kv_cache_config.block_reuse_config.swa_endpoint_rewind_tokens` option:

```yaml
kv_cache_config:
  enable_block_reuse: true
  use_kv_cache_manager_v2: true
  block_reuse_config:
    policy: all_reusable
    swa_endpoint_rewind_tokens: 1024
```

For newly created pages, positive values assign priority `70` to sink blocks
and SWA blocks overlapping the final `window_size + swa_endpoint_rewind_tokens`
tokens of the reusable prompt prefix, excluding the final prompt token that is
recomputed. Other SWA pages receive priority `0`; full-attention and other
life cycles retain the default priority `35`. Blocks remain reusable until
evicted. Within each eviction pool, lower priorities are evicted first, with
LRU ordering among pages of equal priority.

The endpoint is fixed for each request. Existing reused pages retain their
assigned priorities: advancing the conversation does not automatically promote
or demote them, and decoding does not advance the callback's endpoint.
This preference does not guarantee residency or change attention windows or
prefix matching. It excludes dummy and draft requests. The default, `0`, disables
the entire endpoint-priority callback, including its preference for the final window.
This option requires V2, block reuse, and the `all_reusable` policy.

### Mamba Snapshot Boundaries

Hybrid Mamba models must retain the recurrent Mamba state together with the
attention KV prefix. Snapshot policy is grouped under
`kv_cache_config.mamba_state_config`. `periodic_snapshot_interval` controls
periodic boundaries. They are disabled by default; set the interval to a
positive value to enable them. The deprecated
`kv_cache_config.mamba_state_cache_interval` alias remains accepted for
compatibility and is copied to the nested field during validation. New code and
configuration files should use the nested field. The prototype
`additional_snapshot_offsets_from_start` and
`additional_snapshot_offsets_from_end` options add fixed boundaries. Start
offsets count tokens from the beginning of the prompt. End offsets count
backward from the prompt end, and an end offset of `0` selects the final
prompt boundary. The `per_conversation` block reuse policy disables periodic
Mamba snapshots, so configure one or more explicit stable boundaries (usually
an end offset of `0`) when using it with a hybrid Mamba model. For example:

```yaml
kv_cache_config:
  enable_block_reuse: true
  use_kv_cache_manager_v2: true
  avg_seq_len: 2048
  block_reuse_config:
    policy: per_conversation
    max_num_turns: 2
  mamba_state_config:
    periodic_snapshot_interval: 0
    additional_snapshot_offsets_from_start: [128]
    additional_snapshot_offsets_from_end: [0, 32]
```

This retains snapshots after the first 128 tokens, at the end of the prompt,
and before the final 32 prompt tokens. Positions outside a particular prompt
are ignored. Set `avg_seq_len` to the workload's average total sequence length
so V2 can size the attention KV and Mamba state pools in the right proportion.
`pool_ratio` contains one positive, normalized cache-tier quota weight per
layer group in layer-group ID order.
If neither `avg_seq_len` nor an explicit `pool_ratio` is configured, hybrid
Mamba models warn and fall back to half of `max_seq_len`, which can produce a
suboptimal pool split. Exact explicit boundaries currently require
`MambaHybridCacheManagerV2` and `max_beam_width=1`. Hybrid
Mamba models select V2 by default (see
[Selecting the KV Cache Manager](#selecting-the-kv-cache-manager)); set
`use_kv_cache_manager_v2` to `false` to select the V1 C++
compatibility manager. In disaggregated serving, V2 Mamba requires the Python
NIXL transceiver (`transceiver_runtime: PYTHON`); V1 routes support periodic
snapshots only.

### KV Cache Salting for Secure Reuse

KV cache salting provides a security mechanism to control which requests can reuse cached KV states. When a `cache_salt` parameter is provided with a request, the KV cache system will only allow reuse of cached blocks given the same cache salt value. This prevents potential security issues such as prompt theft attacks, where malicious users might try to infer information from cached states of other users' requests.

To use cache salting, specify the `cache_salt` parameter as a string when creating requests. Only requests with matching cache salt values can share cached KV blocks. The salt value can be any non-empty string, such as a user ID, tenant ID, or hash string.

This isolation is enforced entirely by the block-key hash: the salt is mixed into the hashed input and prefix matching is decided by digest equality alone (blocks are not re-compared token-by-token). The block-key hash is therefore required to be a cryptographic hash with strong collision resistance and a 256-bit digest (SHA-256 provides ~128-bit collision resistance, which is ample here); substituting a non-cryptographic hash would allow crafted collisions to bypass salt isolation and must not be done.

### Multimodal UUID Support for Cache Identification

When working with multimodal models (e.g., vision-language models), the KV cache system needs to identify which cached blocks correspond to which multimodal inputs (images, videos, etc.). By default, the system uses content-based hashing to generate unique identifiers for each multimodal input. However, this approach has limitations for cache management across sessions, as the same content must be re-processed to generate the same hash.

You can provide custom UUID strings for your multimodal data using the `multi_modal_uuids` parameter when creating requests. Both cache managers compute the item digest from **both** the UUID and content together for correctness. V1 returns the original UUID in the KV cache event's `mm_keys[].hash` field when one is supplied. V2 returns the item digest as a hexadecimal string, including for items with UUIDs.

**Usage Example:**

```python
from tensorrt_llm.inputs import TextPrompt

# Provide custom UUIDs for your images
prompt = TextPrompt(
    prompt="Describe these images.",
    multi_modal_data={"image": [image1, image2]},
    multi_modal_uuids={"image": ["image-uuid-001", "image-uuid-002"]}
)
```

**Key Features:**

- **Cache Correctness**: When a UUID is provided, the cache key is computed from both the UUID and content together using `BLAKE3(UUID || Content)`. This ensures different content always produces different cache entries, even with the same UUID.
- **User Isolation**: Same content with different UUIDs produces different cache entries, enabling per-user or per-session cache isolation.
- **Stable Event Identifiers**: `get_kv_cache_events()` returns the original UUID for V1, or the item digest for V2. V2 consumers can use the same digest that appears in its cache-key token sequence.
- **Partial UUID Support**: You can provide UUIDs for some items and use `None` for others to fall back to content-only hashing.
- **Cross-Modality Support**: Different modalities (images, videos) can each have their own UUIDs.

**UUID Format:**

- Can be any string (e.g., "image-123", "user-session-img-a", database keys)
- Original UUID strings are preserved in request metadata and returned in V1 KV cache events

V2 derives `mm_keys` directly from the cached token sequence. Each entry identifies a continuous multimodal segment within that block: `hash` is the item's digest, and `start_offset` is the segment's first token offset within the item. An item spanning multiple blocks retains the same digest with increasing offsets. Text may separate segments of the same item. Items are processed in prompt order; one item cannot resume after another item has started. The item digest is distinct from `block_hash`, which also depends on the preceding token sequence.


### Enable Offloading to Host Memory

Before a block is evicted from GPU memory, it can optionally be offloaded to host (CPU) memory. The block remains reusable until it is evicted from host memory. When an offloaded block is reused, it is first copied back into GPU memory. Offloading is controlled with property ```host_cache_size``` which specifies how much host memory (in bytes) should be allocated for offloading. The default is 0.

When offloading is enabled, the client can prevent specific blocks from being offloaded by toggling block priority. Blocks with lower priority than a certain threshold are not offloaded; they are evicted directly from GPU memory to reduce traffic between GPU and host. This priority is set with ```secondary_offload_min_priority```. Default value is 35, meaning any block with lower priority than 35 will not be offloaded.

Here is an [example](../../../examples/llm-api/llm_kv_cache_offloading.py) to show how to enable host offloading.

KV cache compression can reduce the storage and transfer cost of offloaded
Pages, or reduce the amount of KV retained by an algorithm. Compression is
configured separately from `KvCacheConfig`; see
[KV Cache Compression](kv-cache-compression.md) for the available methods and
their activation points.

### Partial Reuse

Partial reuse of a block can happen when some but not all tokens are matched. It is enabled by default, but can be disabled by setting ```enable_partial_reuse``` to False.

The property ```copy_on_partial_reuse``` specifies whether a block should be copied or not in order to allow partial reuse. If copying is disabled, a partially matched block can only be reused if no other request is using it. If copying is enabled, partially matched blocks are not reused directly, instead a new block is created and the matched tokens are copied into the new block. This allows multiple requests to partially reuse a block.

### Attention Window Size

Property ```max_attention_window``` specifies the maximum attention window size for each layer in the model as a list of integer values. If the length of this list is less than number of layers, the list is repeated as many times as necessary. For instance, if the model has only full attention layers and maximum sequence length is 4096, you can specify this as ```max_attention_window = [4096]```. If the first layer is full attention, the second layer is limited attention with window size 256 and then this repeats for the remaining layers, you specify this as ```max_attention_window = [4096,256]```. This means first layer is full attention, second layer is limited attention, third layer is full attention, fourth layer is limited attention and so on.

### Debugging Aids

Two opt-in environment variables help when a result looks like it came from KV
pages the request does not own -- a stale page handed over by a previous owner,
or a page-table slot the attention mask was supposed to cover. Both are off
unless set, and when unset nothing about allocation, page contents or reported
block counts changes. Both are diagnostic only: they cost extra work and are
not meant for production serving. They apply to `KVCacheManagerV2`.

Each accepts the same fill value: `1`, `on`, `true` or `zero` for zeros, `nan`
for NaN (any uncovered read then fails immediately and visibly rather than
producing a plausible number), or any number for that constant.

> **Packed NVFP4 (e2m1) limitation.** A packed sub-byte pool cannot store an
> arbitrary sentinel: every non-zero fill value collapses to the byte pattern
> `0x7f`. For most packed formats `0x7f` is non-finite, so a `nan`/`inf`
> sentinel still poisons the page. Packed NVFP4 (e2m1) has no NaN/Inf encoding,
> so `0x7f` decodes to `+6.0` (the largest finite magnitude) instead. On such a
> pool the sentinel lands as `6.0` -- still a recognisable out-of-band pattern,
> but not one that an `isnan`/`isinf` check will flag. Only zero fills carry
> over exactly for packed e2m1.

- `TRTLLM_KV_GUARD_PAGE` reserves one page that no request can be given, fills
  it with the chosen pattern, and publishes its per-layer index. Attention
  backends that have to keep masked-out page-table entries in range park them
  on this page instead of on page 0, which is a live page belonging to whatever
  request holds it. The mask still decides the result; what changes is that a
  masking bug reads a recognisable pattern instead of a stranger's keys and
  values. Costs one page and one index-mapper slot.
- `TRTLLM_KV_FRESH_PAGE_FILL` writes the pattern into pages as a request is
  given them, so a read past what the request itself wrote returns the pattern
  rather than the previous owner's data. Pages covering the reused/committed
  prefix are never overwritten. Costs one fill per layer per new allocation
  plus a device synchronization.

Both announce themselves once at warning level when they take effect, so a log
shows whether the switch actually did anything.

### KV Cache Events

KV cache events report block **stored**, **removed**, **created** and **updated** operations
so an external KV-cache-aware router (for example NVIDIA Dynamo) can route a request to the
engine that already holds its prefix. Two delivery paths are available.

#### Buffered path (default)

Set ```event_buffer_max_size``` to a positive integer and ```enable_block_reuse``` to True.
Events are buffered per rank, gathered onto rank 0 under attention data parallelism, and
pulled per iteration through `LLM.get_kv_cache_events()` / `LLM.get_kv_cache_events_async()`,
or over the `/kv_cache_events` endpoint of `trtllm-serve`.

#### Streaming path (unsupported)

```{note}
The streaming path has no implementation: `kv_cache_config.kv_events_config` is rejected
at startup. Use the buffered path via `kv_cache_config.event_buffer_max_size` instead. The
wire format and endpoint convention below describe the contract a future native event sink
must satisfy.
```

Configured with ```kv_cache_config.kv_events_config```. Each rank encodes its own events and
publishes them directly over a ZeroMQ `PUB` socket from a background thread, so there is no
rank-0 gather and no per-iteration pull.

```python
from tensorrt_llm.llmapi import KvCacheConfig, KVEventsConfig

kv_cache_config = KvCacheConfig(
    enable_block_reuse=True,
    kv_events_config=KVEventsConfig(
        enable_kv_cache_events=True,
        endpoint="tcp://*:5557",
        replay_endpoint="tcp://*:5657",
    ),
)
```

**Constraints.** Enabling the streaming path raises at startup. A Python event sink cannot
serve it, because the KV cache manager V2 radix tree calls its sink natively rather than
through Python; re-enabling it needs a native sink. Pipeline parallelism and context
parallelism are rejected independently.

**Endpoint convention.** Every attention-DP rank binds `base_port + rank` using its
**global** rank, so `N` ranks occupy `[base_port, base_port + N - 1]` cluster-wide and
each rank's port is distinct — on a multi-node deployment, rank 8 binds `base_port + 8`
whichever node it runs on. Co-located engines — for example disaggregated prefill and
decode on one host — must use base ports at least `N` apart.

```replay_endpoint``` follows the same convention. Because only ranks co-located on one
host actually contend for a port, and a host holds a contiguous run of ranks, its base
port must be at least *ranks-per-host* away from ```endpoint```'s rather than `N` away.
Overlapping ranges are rejected at startup. For `ipc://` and `inproc://` endpoints, which
have no port, each rank appends a `_dp<rank>` suffix instead.

**Wire format.** Each batch is sent as three ZeroMQ frames: the subscription ```topic```,
an 8-byte big-endian sequence number, and a msgpack payload
`[timestamp, [events], data_parallel_rank]`. Each event is a map tagged with a `type` key —
`BlockStored`, `BlockRemoved` or `AllBlocksCleared` — carrying int64 block hashes derived
from the V2 radix block keys. This is the format documented for custom router backends; it
differs from vLLM's positional-array encoding of the individual events, though the batch
envelope is positional in both.

**Delivery guarantees.** Delivery is best effort, but loss is observable. Every accepted
batch reserves a sequence number up front, so a batch dropped by a full publisher queue
(```max_queue_size```) or by a failed send leaves a hole in the sequence. Subscribers must
treat any gap as lost KV-cache state and resynchronize rather than assuming continuity.

**Replay.** If ```replay_endpoint``` is set, the publisher also binds a `ROUTER` socket. A
subscriber sends an empty delimiter frame plus an 8-byte big-endian start sequence, and
receives each retained batch as `[delimiter, topic, seq, payload]`, terminated by a sentinel
with an empty payload. Only the last ```buffer_steps``` batches are retained, so a replay
can legitimately start above the requested sequence — that too is a gap.

### Experimental DKV replicated lifecycle

`dkv_config` enables a prototype that replicates KV cache metadata and request
lifecycles across an attention-DP group. The group size is the tensor-parallel
size. Each rank allocates storage for every request in the group, including
requests assigned to another compute rank. Model execution, sampling, sequence
slots, and request statistics use only the local compute-rank batch. An idle
rank forwards its resident dummy so all ranks participate in model collectives.
Only the compute rank constructs a response. A DKV context worker can send its
computed KV to an ordinary generation worker. `dkv_config.kv_layout` names the
layout of the KV data: `replicated`, the default, keeps every layer on every rank;
`layer_split` keeps each layer on its owner rank only and moves the KV of a layer
between ranks around its attention (see **Layer split** below).

Two collectives keep the replicas aligned. **S-sample** is the single all-gather
after sampling in every iteration that has a batch: each compute rank publishes
the completion state and finish reasons of its requests (never token values)
together with a digest of the global batch, so every rank commits the same state
before KV is committed or released. **S-control** is the all-gather at the top of
every iteration, idle ones included; it carries the capacity digest, fatal
errors, pending-response flags, and transfer events.

Use this mode only for development. The tests run the MPI executor proxy with
`num_postprocess_workers=0`; a two-rank check also answered requests with
postprocess workers and with the RPC orchestrator. Sampling completion and finish reasons are
replicated before global KV commit and release. Sampler failures currently
fail only the affected compute-rank requests after S-sample. Each originating
rank charges its own error budget; a fatal decision is shared through the next
S-control exchange before every rank shuts down together. Error responses are
produced once, by the request's compute rank, and replicas release resources
at the same commit point. Forward exceptions remain unrecoverable: another
rank may be blocked in a model collective, so the existing crash handler
terminates the group.

S-control runs before scheduling on every iteration, including idle iterations.
It compares the iteration number, active-request count, used index slots, free
pages in every cache level and pool, and the count and digest of requests in
transfer even when debug checks are disabled. S-sample likewise rejects a rank
whose view of the global batch (request IDs, compute ranks, context positions)
differs. A mismatch raises an error with the per-rank summaries on every rank
in the same iteration. Pending error responses
are flushed only when this exchange indicates that at least one rank has work
or commits a transfer result that requires a response or termination.
Cancellation uses replicated request state and is processed even when no batch
can run. Context requests with replicated transfer ownership retain a pending
cancellation until transfer completion or timeout. Transfer completion, failure,
and timeout events share this exchange and are committed in request-ID order.

The scheduler uses per-rank token and request budgets by default:
`TRTLLM_DKV_DUAL_LEDGER=1`. Each compute rank has its own `max_num_tokens` and
`max_batch_size`, while KV allocation remains global. Exhausting one rank's
compute budget does not stop admission on other ranks; KV allocation failures
retain the V2 scheduler's global stop/skip semantics. Setting the internal
switch to `0` restores a single shared compute budget for diagnostics while
retaining local execution and response ownership. If an iteration schedules
nothing while started context requests hold KV pages, every rank releases the
last started request and restarts its context; the repeat count of one request
is tracked and a warning is logged each time it doubles, because the KV cache is
then probably too small for the active context requests.

DKV activates requests in waiting-queue order. Routing assigns compute-rank
tags without reordering that global list. The default router balances new
tokens, subtracting prefix matches from incoming prompts and cached tokens
from active local requests. Prefix probes use the replicated local cache;
the existing rank-state collective still carries load and iteration statistics.

The initial validation configuration uses at least two ranks, a
`max_batch_size` of at least two, non-chunked prefill, and one generated token
per request:

```yaml
tensor_parallel_size: 2
enable_attention_dp: true
disable_overlap_scheduler: true
enable_chunked_prefill: false
max_batch_size: 2
dkv_config:
  attention_mode: dp
kv_cache_config:
  use_kv_cache_manager_v2: true
  enable_block_reuse: false
  block_reuse_config:
    policy: per_request
```

Generation-only requests, `max_tokens` other than one, multiple returned
sequences, multimodal inputs, and generation-first disaggregation are rejected.
Context transfer requires the NIXL backend and Python V2 transceiver; `auto`
resolves to `PYTHON` under DKV. Pipelined transfer and the FP4 MLA ownership
bridge are unsupported. Pipeline/context parallelism, speculative and guided
decoding, LoRA, KV connectors, attention-DP balancing and cache-aware routing,
SWA scratch reuse, pool rebalancing, and disk prefetch are also unsupported.
The runtime accepts the base V2 manager and the DeepSeek-V4 manager; other V2
subclasses have additional state that is not covered by this prototype.

**Disaggregated context worker.** Add this configuration on the context worker,
and configure a generation worker without `dkv_config` using the matching model
and KV layout:

```yaml
cache_transceiver_config:
  backend: NIXL
  transceiver_runtime: PYTHON
  kv_transfer_timeout_ms: 60000
```

Every context rank releases its index slot and enters the transfer lifecycle
after the last context chunk. Only the compute rank sends KV and observes the
transport result. The next S-control exchange commits that result on every
rank, releases the transfer claim, and flushes the owner's response before
freeing resources. Context polling does not add a transceiver TP collective.

A finite transfer deadline is required. Expiration requests transport
cancellation, but pages stay pinned until the transport confirms that it no
longer reads them. A completion reported by the transport wins over a deadline
that expired in the same poll. A send that still has no terminal result after
twice the deadline, and any failure while polling send status on a compute rank,
become a fatal error that S-control delivers to every rank, because the pages
cannot be released while the fabric may read them. An exception raised only on
the compute rank while committing a transfer, such as while building the
response, is not replicated; it ends the rank's event loop and the executor's
crash handler terminates the group. User cancellation during transfer follows the same shared
completion path instead of freeing pages locally. A fatal executor error with
outstanding transfer claims terminates the process group without freeing those
pages in the request cleanup path; asynchronous transport may still reference
them. Include the global volume of in-flight transfers in capacity planning.

**Capacity.** For a group of size `G`, admission and index-table capacity cover
the global request count, with another `G` index slots for resident forward
dummies. Each rank reserves one additional sequence slot for its local dummy.
These larger tables consume additional pinned host memory, which is logged
during initialization. Physical KV memory remains bounded by
each GPU's memory budget. If one rank can hold `P` blocks, the replicated group
can hold approximately `P` distinct blocks, compared with up to `G * P` blocks
in ordinary attention DP. Account for this capacity difference when comparing
reuse rates; use an equal-capacity baseline or an interval without eviction.
A host cache tier works under DKV: every replica offloads and onboards the same
blocks, the capacity digest covers the free pages of every cache level, and a
prefix that left the GPU pool but stays on the host still hits on the other ranks.
A host cache that is too small drops the offloaded blocks, so size
`host_cache_size` for the replicated working set of one rank. State whether
measurements include host-resident blocks.

The integration tests run groups of two, four and eight ranks. Eight ranks take two
four-GPU nodes started with `trtllm-llmapi-launch`: ranks started that way do not see
the environment a test sets, so the tests hand the DKV switches to every worker through
the LLM's `env_overrides`.

**Reuse and metrics.** Keep reuse disabled for numerical comparisons. Remote
replicas hold matching KV metadata but do not receive the computed KV tensors;
a remote prefix hit can therefore read invalid data, and the executor logs a
warning at startup when reuse is enabled. Reuse-enabled runs are
limited to metadata and cache-hit experiments, with
`block_reuse_config.policy=per_request`; a configuration that needs correct
outputs should set `kv_cache_config.enable_block_reuse=false` instead. Per-rank request and token statistics
count only local real requests and exclude dummies. Replicated KV statistics
must be read from one rank, even when `TLLM_METRICS_ALL_RANKS` is enabled.
Stored-block events are likewise replicated; consumers should select one
`attention_dp_rank` when counting logical cache insertions. In a closed-loop run
with one client, ordinary attention DP can place every request on the same rank
and then reaches the same hit rate; the replicated metadata matters when the
requests of one prefix are served by different ranks, so compare the modes with
requests pinned across ranks or under concurrent load.

Numerical gates run sequential, rank-pinned requests with reuse disabled and
compare complete generation logits. The prompt set is 14 real texts of different
lengths spread over the ranks of the group, including four sentences with known answers; a
control that cannot answer them, or prompts whose distributions are not far
apart relative to the model's run-to-run noise, fail the gate because they could
not tell a request that read another request's KV from a correct one. They
repeat ordinary ADP first. For a model whose ADP runs reproduce, identical ADP
logits require identical DKV logits; otherwise both controls and DKV must
produce the same tokens and satisfy the fixed logits tolerance. A model whose ADP
runs do not reproduce (DeepSeek-V4 today) is judged against its own noise: tokens
must agree wherever the top-two logit margin is at least 0.5, and the
total-variation distance of the softmax distributions between DKV and ADP may
exceed the ADP run-to-run distance by at most 0.02 on average and 0.25 for any
prompt. A continuation of several tokens is compared up to the first position
where the three runs choose different tokens, and such a flip is accepted only
where the top-two margin is inside the noise. The shared test policy also
rejects missing requests, nonfinite logits, and inconsistent tensor shapes.
For a reuse-enabled numerical experiment, partial-block reuse must be disabled
and the complete prompt history must have no shared prefix as long as one KV
block, including repeated requests.

The disaggregated context-transfer test applies the same policy to a DKV context
worker that sends its KV to an ordinary generation worker: the eight tokens that
generation produces must match those it produces after an ordinary ADP context
worker, on the same prompt set. Its timeouts, a cancellation race and the two
requests that follow them run in the same job.

DKV supports prefill only, so the aggregate gate never reads back the KV it stores:
only the transfer gate can see a request whose KV is wrong. A mutation test keeps that
gate honest. The sender of the context worker overwrites the transferred KV of one
prompt, with zeros or with the KV of the previous request, before the transport reads
it, and the gate must reject the run. The requests that break the policy on their own
must then be that prompt and no other.

The V2 event tests separately check a request pinned to rank 0: ordinary ADP
stores it only on rank 0, while DKV emits matching stored-block hashes on both
ranks, grouped by layer group. A second test serves more distinct prompts than the
cache keeps: the DKV replicas then emit the same stored and removed events with the
same event IDs, every removed block was stored before, and the stored prompt blocks
of one replica equal the union over the ranks of ordinary ADP. Matching event hashes
establish replicated prefix metadata and lifecycle, not equality of the KV tensors.

**Measurement experiments.** The internal `TRTLLM_DKV_MEASUREMENT=1` observer
records owner-local preparation, reuse, allocation failures, and scheduled
context tokens. It is opt-in for both DKV and its ordinary ADP control, without
adding public configuration or telemetry fields. Each rank's own manager
counters are exported as `dkvMeasurement` on the iteration-stats row of that
rank, read with `llm.get_stats()` (the rows of a rank that had no work in an
iteration carry its latest counters); they ride on the attention-DP stats
payload that is already gathered every iteration, so no collective is added.
They are not the rank-0 KV statistics copied into ordinary ADP's per-rank
iteration rows.

Run the two-rank experiment in the GPU test environment, with outputs outside
the source checkout:

```bash
python tests/integration/defs/dkv/dkv_measurement_runner.py \
  --model "$LLM_MODELS_ROOT/llama-models-v2/TinyLlama-1.1B-Chat-v1.0" \
  --work-dir "/scratch/$USER/dkv-measurements"
```

The runner compares cold, cross-rank, and local prefix reuse, then exceeds the
cache capacity to check that polluted intervals are rejected. Reports (schema
version 2) include per-rank and global matched-prefix lengths and hit rates, the
cross-rank load imbalance inside each iteration averaged over the interval (and
the variance of the interval totals, which can hide a persistent imbalance),
prepare/resize failure attempts, actual pool capacities, and logical
resident-block duplication. Request counters count each compute owner once;
physical storage counters remain per rank, and removals and dropped pages are
also given per replica (one rank's value under DKV when all ranks agree, the
rank sum under ADP). Duplicate storage is a logical
block ratio, not a byte ratio or proof of valid replicated KV data. The hit rate
covers every cache tier listed in the report; no per-tier split is measured, so
state whether a host tier was configured.

Capacity-drop counts describe direct LRU victims dropped from the last cache
tier; GPU-to-host offload is reported separately. Removed-block events are an
additional conservative signal and are not substituted for physical drop
counts. An accepted no-eviction interval requires complete events and statistics,
fixed capacity, no reuse-state reset, and no failed prepare/resize attempts.
Missing observations remain unknown and invalidate the measurement. Capacity
comparisons use each pool's actual page sizes across all tiers and compare the
usable (free plus evictable) pages of an idle pool, summed over the ADP ranks,
with one DKV replica, so a replica's resident dummies do not count as capacity.
The gap is reported per pool and may be allowed a declared tolerance in pages;
unequal per-rank pool sizes between the modes are flagged.

The runner uses `TRTLLM_KV_FRESH_PAGE_FILL=zero` so remote reuse reads initialized
pages. Reports explicitly mark timing as perturbed and output correctness as
unvalidated. These experiments establish metadata visibility and scheduling
load; they do not measure layer-split throughput or validate generated text.

**Workload experiments.** `tests/integration/defs/dkv/dkv_phase6_runner.py` places
one deterministic workload under ordinary attention DP (`adp`), attention DP with
the KV-aware router (`adp_kv:<beta>`) and DKV (`dkv`, whose router is always the
default one). For every mode it reports the prefix hit rate, the context tokens that
still had to be computed, the load of every rank and the logical duplicate storage,
with the validity controls above. The `chat` workload is conversations whose every
turn re-sends the whole history; `zipf` is requests that share one of a few
prefixes, and `--prime` sends every prefix alone once first. A report also states
the hit ceiling of a cache that keeps everything and that every rank can read. DKV
reaches it exactly for conversations and primed prefixes when nothing is evicted; for
an unprimed prefix it is an upper bound.

```bash
python tests/integration/defs/dkv/dkv_phase6_runner.py run --group 2 \
  --model "$LLM_MODELS_ROOT/llama-models-v2/TinyLlama-1.1B-Chat-v1.0" \
  --tokens-per-block 32 --workload chat --first-tokens 256,512 --turn-tokens 32,64 \
  --warmup-tokens 256 --max-seq-len 1024 --max-num-tokens 1024 --concurrency 8 --warmup 4 \
  --modes adp,adp_kv:1,dkv --out "/scratch/$USER/dkv-phase6"
python tests/integration/defs/dkv/dkv_phase6_runner.py report "/scratch/$USER/dkv-phase6"
```

The longest prompt, a warm-up prompt included, must fit `--max-seq-len`,
`--max-num-tokens` and the context window of the model; the runner checks this
before it loads the model.

`run` starts one process per mode on one node. A group that spans nodes starts
`worker` on every task of the job behind `trtllm-llmapi-launch`, once per mode. A
run's capacity is `--kv-max-tokens` or, with `--kv-quota-gib`, a byte quota per
rank. For an equal-capacity comparison with evictions, give each ADP mode a quota of
the DKV quota divided by the group size and pass the `result.json` of the `adp` run
to the DKV run as `--adp-reference`; its report then states whether the usable pages
of one replica match those of all ADP ranks together, and
`--capacity-tolerance-pages` declares a gap that is accepted. The `capacity` command
lists the pages, hit rates and evictions of such runs side by side; evictions are
counted per replica under DKV, where every rank drops the same pages, and summed over
the ranks under ADP. `summary` puts runs of one workload side by side and `check`
lists every run and flags the incomplete ones.

DeepSeek-V4 reuses a prefix only where a stored sequence ended. A shared prefix
followed by a different suffix never hits, whereas the history of a conversation
and a prefix that was sent alone do, so use `chat`, or `zipf --prime`, on that
model. Multi-turn runs on it are listed as invalid by the no-eviction check, because
blocks are also removed as a conversation grows; the number of capacity-dropped
pages, which stays zero, shows that nothing was evicted.

**Staging loopback (DeepSeek-V4).** The layer-split layout runs the attention of a layer on
a staging area and not on the pages of the cache manager: before the layer, its cached
pages are copied into its slot of the staging area, and after the layer the pages the new
tokens wrote are copied back. `TRTLLM_DKV_STAGING_LOOPBACK=1` runs the replicated layout
through that path, with every rank owning every layer, so the staging path is judged alone
and without any transfer between ranks. It needs the DeepSeek-V4 cache manager, an FP8 or
BF16 KV cache (not `fp8_ds_mla` or NVFP4) and `cuda_graph_config: null`. The slots hold the
cached and the new tokens of one iteration: `TRTLLM_DKV_STAGING_TOKENS` bounds their sum
over the requests of a rank, and the scheduler admits a request to an iteration only while
the sum fits (default: `max_seq_len` plus `max_num_tokens`, and no more than every request
of the batch at `max_seq_len`; a value below `max_seq_len` is rejected, since a request of
that length would never be admitted). `TRTLLM_DKV_STAGING_DEPTH` sets how many
layers of a kind can use the staging area at once (default 2).
`TRTLLM_DKV_STAGING_FILL=nan` overwrites the slots of a layer with NaN before its pages are
fetched, so a page that the layer reads without it having been fetched or written turns the
output into NaN. The ranks compare these switches when they start. The loopback gates run
DeepSeek-V4 with and without it and judge the two groups by the rules above, and a
layer-level test runs the attention layers on the cache manager and on its staged view and
requires the same outputs and pages.

**Layer split (DeepSeek-V4).** With `kv_layout: layer_split` the model's layers are divided
into contiguous ranges over the ranks of the group, and the cache manager of a rank holds the
KV of the layers it owns only, so a group stores one copy of the KV of every request instead
of one per rank. The request is still computed on its compute rank: around the attention of
each layer, the rank that owns the layer sends the cached KV the layer reads to the compute
rank (a fetch), and the compute rank sends the pages its new tokens wrote back (a writeback). The pages of a sliding window or
of a compressor state that fall out of their window with the iteration are not sent, since the
cache manager releases them before anything reads them.
Both run on a data stream of their own over NCCL point-to-point messages, in an order that
every rank derives from the replicated scheduling state, so no rank exchanges the plan; the
host waits for the data stream at the end of every iteration, before a page can be freed or
moved. The ranks agree on the number of pages of every KV life cycle, since the layers they
hold differ in bytes. The layout needs DeepSeek-V4 on SM100 or SM103, `cuda_graph_config: null`,
no `torch_compile_config` and every rank to own layers of every KV life cycle, which limits the
group size. As the context worker of a disaggregated deployment, every rank sends the layers it
owns of every request. The group is published to the generation workers as one tensor-parallel
rank of a pipeline with as many stages as the group has ranks, so a generation worker is
configured as for any other context worker, but both sides have to run a version that knows the
layout. A request is done when every rank has finished its send; the failure or the timeout of one
rank cancels the sends of all.

`TRTLLM_DKV_STAGING_TOKENS`, `TRTLLM_DKV_STAGING_DEPTH` and `TRTLLM_DKV_STAGING_FILL` work as
for the loopback. `TRTLLM_DKV_TRANSPORT_TIMEOUT_S` (default 60) is how long the host waits for
the data plane at the end of an iteration before it fails; set it below the hang detector
timeout. The plan of an iteration is resolved to device addresses once, when the staged batch is
known, so the hooks of the forward pass only enqueue work. The LLM launches its workers with
`CUDA_DEVICE_MAX_CONNECTIONS=32` in their environment (an `env_overrides` entry it adds unless the
caller set one): the data stream and the stream of the forward pass must not share a hardware queue
of the GPU, since a forward kernel queued behind a receive that still waits for its peer would wait
with it. The driver reads the variable once, when a process starts, and hands streams to the queues
as they are created, so a run can still draw a sharing pair; 32 queues make that much less likely
than the default 8. Workers launched ahead of time, with `mpirun` or `trtllm-llmapi-launch`, need
the variable in their launch environment; the executor warns when a worker was launched with fewer
queues. `TRTLLM_DKV_P2P_GROUPS=1` issues the
messages a rank sends or receives in one step as one NCCL group, one kernel for all of them; it is
off by default because a group finishes only when every peer has posted its side, which under a
concurrent load makes the data stream of an owner, and the forward pass queued behind it, wait for
the slowest peer. With `TRTLLM_DKV_DEBUG=1` every message is checksummed where it was packed, where it
arrived and on the pages it was unpacked into, the ranks compare the checksums after every
iteration, and a damaged message is named with its layer, direction and requests. To see that
the check works, `TRTLLM_DKV_FAULT=<iteration or *>:<layer>:<fetch|writeback>[:<flip|zero>[:<rank>]]`
damages the received messages of one step (`*`: of every iteration). With `TRTLLM_DKV_MEASUREMENT=1` the `dkvMeasurement` of an
iteration-stats row also carries `data_plane`, the messages and bytes the rank has sent,
received and copied locally, the host seconds of the hooks of the forward pass (`hook_seconds`:
what the data plane adds to a pass that is bound by the host; `compile_seconds` is the part of
it that resolved the plan of the iteration to addresses) and the host seconds that the end
of an iteration waited for the data stream (`drain_seconds`). Local bytes are also separated into
`bytes_local_fetch` and `bytes_local_writeback`; their sum is `bytes_local`. Set
`TRTLLM_DKV_WAIT_TIMING=1` to measure the compute stream's GPU wait for layer fetches with CUDA
events. The cumulative `fetch_wait_seconds` is collected at the end of each iteration; it is zero
when the switch is off (the default), which creates no timing events.
With `TRTLLM_DKV_MEASUREMENT=1`, `control_plane` also reports cumulative host time and call counts
for S-control (`control_seconds`, `control_calls`) and S-sample (`sample_seconds`, `sample_calls`),
including time waiting for other ranks. These host timers are disabled otherwise.

The tests are `tests/integration/defs/dkv/test_dkv_layer_split.py` (DeepSeek-V4 on two to eight
ranks, judged against the replicated group by the rules above: precision with and without chunked
prompts, prefixes that a request reads from the cache of another rank and from the host tier,
conversations whose turns alternate over the ranks, the bursts and the aggregate soak of the
replicated layout, a message that is damaged on arrival in either direction, and a layer-split
context group that hands its KV to an ordinary generation worker, with a timeout and a
cancellation, with prefix reuse, and with the KV that all ranks or one rank send corrupted), the
multi-GPU probe `tests/unittest/_torch/multi_gpu/test_dkv_dataplane_tp.py`,
which runs the data plane between real ranks without a model and compares every page with a
pattern (`DKV_DATAPLANE_SOAK=<iterations>` turns on a long run), and CPU tests that run every
rank's streamer on threads (`tests/unittest/_torch/executor/test_dkv_streamer.py`).

**Consistency checks.** The checker always rejects inconsistent enable flags and
inconsistent process-level settings (the dual-ledger switch, `TLLM_METRICS_ALL_RANKS`,
and the KV manager backend) at construction. Set `TRTLLM_DKV_DEBUG=1` on every
rank before starting the workers for the per-iteration checks. They compare the
cache configuration (startup configuration), ordered request/compute-rank
assignments (global request order), prefix-probe results (prefix probes), and
scheduled requests, chunk sizes, and ordered KV operations (scheduling
decisions). At the end of every iteration they also compare the KV fingerprint
of every cache including the resident dummies, the set of requests in transfer,
and the progress of every active request, and a mismatch names the first
differing record of each rank. Debug traces retain KV
operation order and logical state without requiring physical slot IDs to
match. Free operations must occur in a replicated commit window; the freed
request IDs are compared through S-control and checked again at shutdown.
Invariants are raised as `RuntimeError`, so they also hold under `python -O`.
The V2 core's additional checks use
`TLLM_DEBUG_MODE`, which must be set before importing the package.

### Deprecated Properties

Property ```use_uvm``` has been deprecated and will be removed in a future release.

Property ```sink_token_length``` is deprecated and silently ignored on the PyTorch backend.
The PyTorch attention kernels do not support StreamingLLM, so any non-``None`` value is
dropped before reaching the executor.
