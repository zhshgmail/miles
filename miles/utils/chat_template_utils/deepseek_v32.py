"""DeepSeek V3.2 prompt rendering bridge.

DeepSeek V3.2 ships no jinja chat_template; sglang renders prompts through
``encoding_dsv32.encode_messages`` instead. To keep TITO append-only
tokenization byte-aligned with the runtime, miles' ``apply_chat_template``
delegates to this module for any V3.2 tokenizer. The logic here mirrors
``sglang.srt.entrypoints.openai.serving_chat.OpenAIServingChat._process_messages``'s
``use_dpsk_v32_encoding=True`` branch.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from pydantic import TypeAdapter
from sglang.srt.entrypoints.openai.protocol import Tool

_GENERATION_PROMPT_SUFFIX = "<｜Assistant｜>"


def _canonicalize_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match sglang's V3.2 tool canonicalization before encoding.

    sglang's ``serving_chat`` runs request tools through
    ``ChatCompletionRequest`` (which validates them as ``Tool`` pydantic
    models), then dumps back to ``dict``. That fills in defaults like
    ``strict: false`` and reorders fields, which the V3.2 encoder
    serializes verbatim into the ``<functions>`` block — so caller-side
    dicts must go through the same canonicalization or token IDs drift.
    """
    wrapped = [
        tool if isinstance(tool, dict) and "function" in tool else {"type": "function", "function": tool}
        for tool in tools
    ]
    validated = TypeAdapter(list[Tool]).validate_python(copy.deepcopy(wrapped))
    return [tool.model_dump() for tool in validated]


def is_deepseek_v32_tokenizer(tokenizer: Any) -> bool:
    """Detect DeepSeek V3.2 via ``name_or_path`` or initialization kwargs.

    Mirrors sglang's auto-detect for ``use_dpsk_v32_encoding``: the V3.2
    HuggingFace repo ships no jinja chat_template, so the only way to
    recognize it before launching the runtime is by name. We accept any
    casing or separator that resolves to ``deepseek-v3.2`` /
    ``deepseek-v32``.
    """
    names: list[Any] = [getattr(tokenizer, "name_or_path", None)]
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        names.append(init_kwargs.get("name_or_path"))

    for name in names:
        if not name:
            continue
        normalized = str(name).lower().replace("_", "-")
        if "deepseek-v3.2" in normalized or "deepseek-v32" in normalized:
            return True
    return False


def render_messages(
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> str:
    """Render a chat trajectory through sglang's ``encoding_dsv32``.

    Accepts the same call surface as miles' ``apply_chat_template``:
    ``thinking`` / ``enable_thinking`` controls chat-vs-thinking mode and
    ``drop_thinking`` controls whether sglang strips prior assistant
    thinking content before the last user turn (must be ``False`` for
    multi-turn append-only trajectories — see DeepSeekV32TITOTokenizer's
    ``{tool, user}`` SUPPORTED_TEMPLATES row).
    """
    from sglang.srt.entrypoints.openai import encoding_dsv32
    from sglang.srt.parser.jinja_template_utils import process_content_for_template_format

    remaining_kwargs = dict(kwargs)
    thinking = remaining_kwargs.pop("thinking", remaining_kwargs.pop("enable_thinking", False))
    drop_thinking = remaining_kwargs.pop("drop_thinking", True)
    if remaining_kwargs:
        raise ValueError(f"Unsupported DeepSeek V3.2 chat-template kwargs: {sorted(remaining_kwargs)}")

    rendered_messages = copy.deepcopy(messages)
    for i, msg in enumerate(rendered_messages):
        if msg.get("content") is None:
            msg["content"] = ""
        processed = process_content_for_template_format(msg, "string", [], [], [], [], use_dpsk_v32_encoding=True)
        rendered_messages[i] = {**msg, **processed}
        msg = rendered_messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tool_call in msg["tool_calls"]:
                function = tool_call.get("function", {})
                arguments = function.get("arguments")
                if isinstance(arguments, dict):
                    function["arguments"] = json.dumps(arguments, ensure_ascii=False)

    # sglang's V3.2 path requires a system message to anchor tool rendering.
    if not rendered_messages or rendered_messages[0].get("role") != "system":
        rendered_messages.insert(0, {"role": "system", "content": ""})
    if tools:
        rendered_messages[0]["tools"] = _canonicalize_tools(tools)

    prompt = encoding_dsv32.encode_messages(
        rendered_messages,
        thinking_mode="thinking" if thinking else "chat",
        drop_thinking=drop_thinking,
    )
    if add_generation_prompt:
        return prompt

    # encoding_dsv32 always appends the assistant-start sentinel; strip it
    # when the caller wants a raw prefix (e.g. fixed-template verification).
    if prompt.endswith(_GENERATION_PROMPT_SUFFIX):
        return prompt[: -len(_GENERATION_PROMPT_SUFFIX)]
    return prompt
