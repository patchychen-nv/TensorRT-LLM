# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve one chunked prompt on a DKV group in a process of its own.

A fault that the data plane reports as fatal stops every rank of the group, and the client of the
group then waits for an answer that never comes. A test runs the group here, bounds the process and
judges what the ranks wrote. The options are one JSON argument: the checkpoint, the block size, the
MoE backend, the group size, the longest sequence and the prompt.
"""

import json
import sys

from dkv_models import dkv_worker_env, make_dkv_llm


def main() -> None:
    options = json.loads(sys.argv[1])
    from tensorrt_llm import SamplingParams
    from tensorrt_llm.llmapi.llm_args import MoeConfig
    from tensorrt_llm.scheduling_params import SchedulingParams

    with make_dkv_llm(
        options["model"],
        dkv=True,
        tokens_per_block=options["tokens_per_block"],
        moe_config=MoeConfig(
            backend=options["moe_backend"],
            disable_finalize_fusion=options["disable_finalize_fusion"],
        ),
        group_size=options["group_size"],
        chunked_prefill=True,
        max_num_tokens=512,
        max_seq_len=options["max_seq_len"],
        env_overrides=dkv_worker_env(),
    ) as llm:
        llm.generate(
            options["prompt"],
            sampling_params=SamplingParams(max_tokens=1, temperature=0),
            scheduling_params=SchedulingParams(attention_dp_rank=0, attention_dp_relax=False),
            use_tqdm=False,
        )
    print("DKV_GROUP_CHILD_DONE", flush=True)


if __name__ == "__main__":
    main()
