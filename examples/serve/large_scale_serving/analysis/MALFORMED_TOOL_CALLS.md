# Malformed tool names: three causes, and what the model actually emitted

Measured 2026-09-21 on the GLM-5.3-FP8 test fleet (instances 536392 / 536393,
10 Kernel Factory campaigns, tag `test-callid2`), after the bare-vs-qualified
name fix and the call-id fix were both live.

Every finding below is joined against `raw_output` — the detokenized model
text *before* the reasoning and tool parsers see it — so "the model did it"
and "we did it" are separated by evidence rather than by argument. The join
is `response.disagg_request_id -> raw.disagg_request_id`; that is the only id
shared across the proxy/worker boundary.

## Scale

    distinct tool calls                3329
    distinct results                   3329     (1:1 — no orphans, no leaks)

    Script completed                   2930   88.0%
    other (wait/collaboration status)   209    6.3%
    Script failed (model's own JS)      169    5.1%
    unsupported call                     17    0.5%   <- this document
    <empty>                               4    0.1%
    aborted                               0    0.0%

`aborted` — the signature of the original bare-name bug, previously 32.1% —
is gone.

## First: it is NOT interleaved thinking

The shape `<tool_call> ... </think> ... <tool_call>` invites the reading that
the model starts a tool call and then resumes thinking. **It does not.**

    <tool_call> appearing before the first </think>      0 of 3461
    <tool_call> appearing after  the first </think>   3461 of 3461

and every one of the 36 malformed raws begins `</think>` before any
`<tool_call>`. No `<think>` opener ever appears in the output at all — the
template prefills it into the prompt, so a second block is never opened.

What is really happening: the model closes its reasoning block, then keeps
writing reasoning-*like* prose in the content region, punctuated by stray
`</think>` tags and false-start `<tool_call>` markers. Marker sequences in the
malformed raws:

    x8   </think> <tool_call>
    x6   </think> <tool_call> </think> <tool_call> <arg_key>
    x5   </think> </think> <tool_call>
    x4   </think> <tool_call> <arg_key>
    x4   </think> </think> <tool_call> <arg_key>
    x2   </think> </think> <tool_call> <tool_call>
    x1   </think> <tool_call> <tool_call> </think> <tool_call> <arg_key>

The repeated `</think>` is normal GLM behaviour, not an anomaly — across
**all** responses:

    one </think>    2482      two   1077      three  22      four  2      none 12

So ~31% of every response carries more than one closing tag, and
`GlmReasoningParser._without_stray_end` (reasoning_parser.py:405) handles
essentially all of them. Only the 36 that *also* contain a false-start
`<tool_call>` go wrong.

## Class A — a false-start `<tool_call>`, then a well-formed one (9 of 25, recoverable)

The model opens a tool call, writes prose retracting it, and then issues a
correct call. We take the **first** opener and read everything up to
`<arg_key>` as the name.

    raw       …</think><tool_call></think><tool_call>exec<arg_key>input</arg_key><arg_value>for (const p of […
    delivered name = '<tool_call>exec'
    result    unsupported call: <tool_call>exec

    raw       <tool_call>collab? no. Use exec.</think><tool_call>exec<arg_key>input</arg_key><arg_value>// @exec: …
    delivered name = 'collab? no. Use exec.<tool_call>exec'

    raw       <tool_call>\exec_request denied? no direct such tool. Must call functions.exec nest read.</think>
              <tool_call>exec<arg_key>input</arg_key><arg_value>// @exec: {"yield_time_ms": 10000 …
    delivered name = '\exec_request denied? no direct such tool. Must call functions.exec nest read.<tool_call>exec'

    raw       <tool_call> exec might be large. Let's call.\n###</think><tool_call>exec<arg_key>input</arg_key>…
    delivered name = "exec might be large. Let's call.\n###<tool_call>exec"

    raw       <tool_call>exec</arg_value><arg_key>input</arg_key><arg_value>const r = await tools.exec_command(…
    delivered name = 'exec</arg_value>'                    (stray close tag, not a second opener)

In all of these the payload that follows is intact. The rule that recovers
them is narrow: **when the span between `<tool_call>` and `<arg_key>` contains
another `<tool_call>`, the last one is the real opener.** Same lesson as the
bugs already fixed — do not trust the first boundary when the markup is noisy.

## Class B — generation degeneracy, no valid call anywhere (16 of 25, not recoverable)

    raw   <tool_call>exec surg? No, actually use proper functions.exec tool with patch string.
          <tool_call>exec surg? Let's do properly.<tool_call>exec surg? We'll use apply_patch…
          <tool_call>exec surg? Here we go.<tool_call>exec surg? Tool call now.   … (x35, no <arg_key> ever)

    raw   <tool_call>exec_command_placeholder</arg_value></tool_call>
    raw   <tool_call>collarian.exec? no. Use exec.</arg_value></tool_call>
    raw   <tool_call>execález… / execattle.waitfoil? Actually no… / exec ۲input…     (۲ = Arabic-Indic digit)
    raw   <tool_call>commentary_to=functions.exec]<tool_call>commentary_summary:…   (harmony channel syntax)

Nothing to recover: the model never emits arguments. Current handling — warn,
deliver it, let the client answer `unsupported call` — is correct. It does not
guess and it does not fail silently.

## Class C — RETRACTED

An earlier version of this document claimed `exec_command` (and `write_stdin`)
were a third class: well-formed markup carrying the inner JS API name instead
of the tool name. **That was an artifact of a hand-written list of "declared"
tools.** Checked against the client's own verdict, no `exec_command` call is
ever answered `unsupported call` — in the 2026-09-20 paired run they come back
`Chunk ID: …`, which is a normal chunked-output handle.

The tool set must be read from the declaration the client actually sends —
an `additional_tools` item inside `input[0]`, with `namespace` groups — and
never reconstructed from memory. For these campaigns it is 18 names:

    exec  wait  spawn_agent  wait_agent  list_agents  interrupt_agent
    followup_task  send_message  request_user_input
    functions.exec  functions.wait  functions.request_user_input
    collaboration.{spawn_agent,wait_agent,list_agents,interrupt_agent,
                   followup_task,send_message}

Better still, do not classify by name at all: **the client's first result line
is ground truth.** Everything below is counted that way.

## Rate

Client-confirmed rejections, deduped by `call_id`, over the 10 campaigns of
2026-09-21:

    calls with a result            3695
    answered "unsupported call"      99      2.68%

    per campaign   mean 9.9   median 2   max 82   three of ten had none

The distribution is not the mean. One agent (`kernelbook-10191`, the 61.42x
campaign, which also ran the longest) accounts for **82 of the 99** at 10.0%
of its own calls; across the other nine campaigns it is 17 of 2876 — **0.6%**.
So the typical campaign sees about two, and one degenerate agent produces the
rest.

## Recommendation

Fix Class A only: 9 of 25, with intact payloads, under a rule that cannot
misfire on well-formed markup. Classes B and C stay as they are.

Overall impact is small — 25 malformed of 3329 calls (0.75%), dropping to
~0.48%. It is worth doing mainly because these traces are training data, and
nine mislabelled samples are nine wrong lessons.

## Samples

`data/malformed_tool_calls_samples.json` holds one complete
request -> raw -> response triple per class, keyed by `trace_id` and
`disagg_request_id`:

    class  trace_id                              disagg_request_id  delivered name
    A      tr_29a28549d64b45edbc2d0fe57abe882d   8924903842252890   '<tool_call>exec'
    B      tr_d503818fe6544be3973c7a1b64db5555   8934260947420570   'exec_command_placeholder</arg_value>'
    C      tr_b6e315b4bfa840079ec4fb26988330b3   9012278710766427   'exec_command'

Each entry carries the delivered item, the `response.output_item.done` event,
the raw model text, and the tail of the request that produced it.

Note `tools_declared` reads 0 in all three: Codex declares its tools as an
`additional_tools` item inside `input[0]`, not in the top-level `tools` field,
so counting that field says nothing about whether tools were offered.

## Reproducing

    jx_garbled.py      malformed names joined to raw text
    jx_recover.py      recoverable vs not
    jx_interleave.py   <tool_call> before/after </think>
    jx_markers.py      marker sequences and </think> counts

under the scratch user dir. Two traps they encode, both of which produced a
confidently wrong answer first:

- **Dedupe tool calls by `call_id`.** A Responses conversation resends its
  whole history every turn, so counting input items counts each call once per
  later request — 30x inflation, and every conversation looks like it is
  spinning.
- **Classify a tool result on its status line, not the whole body.** The
  client's verdict is the first line and the script's stdout follows it;
  searching the body scored 73 successful calls as `aborted` because a skill
  document they printed contained the word.
