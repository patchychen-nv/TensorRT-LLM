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

from typing import Any, Mapping, Optional, Protocol

# Supported HTTP header protocol for external clients, gateways, or proxies
# that carry a stable multi-turn identifier outside the JSON body. Body
# ``conversation_params.conversation_id`` is canonical when both body and
# headers are set; the serve edge copies the first non-empty header value into
# ``request.conversation_params`` only when the body omits it. Routers then read
# ``conversation_params.conversation_id`` to keep later turns of the same
# conversation on the same backend when sticky conversation routing is enabled.
#
# The Claude Code headers are listed first because a client that sends one also
# sends nothing else on this list; ordering them ahead of the generic names
# keeps the common case a single lookup. They mirror the set the Anthropic
# adapter already reads for audit records, so the Messages API gets the same
# conversation identity the audit log records rather than a second notion of a
# session.
CONVERSATION_ID_HEADERS = (
    "x-claude-code-session-id",
    "x-claude-session-id",
    "x-session-id",
    "x-correlation-id",
    "x-session-affinity",
    "x-multi-turn-session-id",
)


# Body fields that carry a stable conversation identity for clients that speak
# no header protocol at all. Consulted only after the body's own
# ``conversation_params`` and the headers above have come up empty, so a gateway
# that sets a header still wins. Each entry is a path into the request body.
#
# Codex CLI is the client that motivated the list. Over the Responses API it
# sets ``prompt_cache_key`` -- OpenAI's cache-affinity hint -- to its thread id;
# over chat completions the same id sits in ``client_metadata.thread_id``. A
# thread is one conversation and stays constant across context compaction and
# resumed sessions, which is exactly what sticky routing needs.
# ``client_metadata.session_id`` is one Codex *run*, which can hold several
# threads (multi-agent), so it is only the last resort: in a single-conversation
# run it equals thread_id and the result is the same.
CONVERSATION_ID_BODY_FIELDS = (
    ("prompt_cache_key",),
    ("client_metadata", "thread_id"),
    ("client_metadata", "session_id"),
)


class RequestWithConversationParams(Protocol):
    conversation_params: Any


def get_request_conversation_id(request: RequestWithConversationParams) -> Optional[str]:
    conversation_params = request.conversation_params
    return None if conversation_params is None else conversation_params.conversation_id


def extract_conversation_id_from_headers(headers: Optional[Mapping[str, str]]) -> Optional[str]:
    if headers is None:
        return None
    lower_headers = {str(key).lower(): value for key, value in headers.items()}
    for header_name in CONVERSATION_ID_HEADERS:
        conversation_id = lower_headers.get(header_name)
        if conversation_id is None:
            continue
        conversation_id = str(conversation_id).strip()
        if conversation_id:
            return conversation_id
    return None


def _field(value: Any, name: str) -> Any:
    """``name`` off a request model or a raw body dict, or None.

    Request models are read with ``getattr`` so declared fields and pydantic extras
    (``ResponsesRequest`` allows unknown fields) look the same; a dict is read by key.
    """
    if value is None:
        return None
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def extract_conversation_id_from_params(body: Any) -> Optional[str]:
    """Body ``conversation_params.conversation_id``, off a request model or a raw body dict.

    Normalized the way ``ConversationParams`` validates it -- ``str()``, then
    stripped -- so a raw body yields the value its parsed request would hold.
    """
    value = _field(_field(body, "conversation_params"), "conversation_id")
    if value is None:
        return None
    conversation_id = str(value).strip()
    return conversation_id or None


def extract_conversation_id_from_body(body: Any) -> Optional[str]:
    """First non-empty ``CONVERSATION_ID_BODY_FIELDS`` value.

    Read off a request model or a raw body dict, as ``_field`` reads either.
    """
    for path in CONVERSATION_ID_BODY_FIELDS:
        value: Any = body
        for name in path:
            value = _field(value, name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def find_conversation_id(body: Any, headers: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The id sticky routing keys a request on, without recording it anywhere.

    The one precedence every caller shares: body ``conversation_params.conversation_id``,
    then ``CONVERSATION_ID_HEADERS``, then ``CONVERSATION_ID_BODY_FIELDS``. ``body`` is a
    request model or the raw JSON body, so the request trace, which holds only the
    latter, names the conversation the router does.
    """
    return (
        extract_conversation_id_from_params(body)
        or extract_conversation_id_from_headers(headers)
        or extract_conversation_id_from_body(body)
    )


def resolve_request_conversation_id(
    request: RequestWithConversationParams,
    headers: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return conversation_params.conversation_id populated at the serve edge.

    Precedence is ``find_conversation_id``'s. An id found anywhere but the body's own
    ``conversation_params`` is written back there for the router to read.
    """
    conversation_params = request.conversation_params
    if conversation_params is not None:
        return conversation_params.conversation_id

    conversation_id = find_conversation_id(request, headers)
    if conversation_id is not None:
        from tensorrt_llm.serve.openai_protocol import ConversationParams

        request.conversation_params = ConversationParams(conversation_id=conversation_id)
    return conversation_id
