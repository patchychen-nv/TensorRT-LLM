# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def _validate_dkv_request(
    *,
    max_tokens: int | None,
    is_generation_only: bool = False,
    n: int = 1,
    best_of: int | None = None,
    has_multimodal_input: bool = False,
    is_generation_first: bool = False,
) -> None:
    """Validate a prefill-only DKV request after context-only normalization."""
    unsupported = (
        (is_generation_only, "generation-only requests"),
        (max_tokens != 1, "max_tokens != 1"),
        (n > 1, "n > 1"),
        (best_of is not None and best_of > 1, "best_of > 1"),
        (has_multimodal_input, "multimodal inputs"),
        (is_generation_first, "GENERATION_FIRST scheduling"),
    )
    for enabled, feature in unsupported:
        if enabled:
            raise ValueError(f"{feature} is not supported with dkv_config yet")
