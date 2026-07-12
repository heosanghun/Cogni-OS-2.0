"""Pinned text-only Gemma 4 chat and response contract.

The shipped local E4B mirror intentionally stays offline, but it currently
does not contain ``chat_template.jinja`` or a response schema.  This module
implements the documented, text-only, thinking-disabled turn contract without
falling back to transcript-like ``USER:/ASSISTANT:`` labels.

Rendered prompts deliberately omit ``<bos>``.  ``ModelService`` tokenizes the
string with ``add_special_tokens=True`` and therefore inserts BOS exactly once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor


SOT = "<|turn>"
EOT = "<turn|>"
SOC = "<|channel>"
EOC = "<channel|>"
TOOL_RESPONSE = "<|tool_response>"

_REQUIRED_CONTROL_TOKENS = (SOT, EOT)
_KNOWN_CONTROL_TOKENS = (
    SOT,
    EOT,
    SOC,
    EOC,
    TOOL_RESPONSE,
    "<|think|>",
    "<bos>",
    "<eos>",
)
_RESERVED_OUTPUT_SEQUENCES = (
    "<|endoftext|>",
    "<|startoftext|>",
)


class PromptContractError(ValueError):
    """Raised when a tokenizer or message violates the local chat contract."""


def _token_id(tokenizer: Any, token: str) -> int | None:
    attributes = {
        EOT: "eot_token_id",
        SOT: "sot_token_id",
        EOC: "eoc_token_id",
        SOC: "soc_token_id",
    }
    value = getattr(tokenizer, attributes.get(token, ""), None)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    try:
        value = convert(token)
    except (KeyError, TypeError, ValueError):
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    unknown = getattr(tokenizer, "unk_token_id", None)
    if value == unknown:
        reverse = getattr(tokenizer, "convert_ids_to_tokens", None)
        if not callable(reverse) or reverse(value) != token:
            return None
    return value


def is_gemma4_tokenizer(tokenizer: Any) -> bool:
    """Return True only when the tokenizer exposes the required turn tokens."""

    return all(
        _token_id(tokenizer, token) is not None for token in _REQUIRED_CONTROL_TOKENS
    )


def _neutralize_control_tokens(text: str, tokenizer: Any) -> str:
    if not isinstance(text, str) or not text:
        raise PromptContractError("chat content must be non-empty text")
    if any(ord(character) < 32 and character not in "\t\r\n" for character in text):
        raise PromptContractError(
            "chat content contains unsupported control characters"
        )
    candidates = list(_KNOWN_CONTROL_TOKENS)
    extra = getattr(tokenizer, "all_special_tokens", ())
    if isinstance(extra, (list, tuple)):
        candidates.extend(token for token in extra if isinstance(token, str))
    for token in sorted(set(candidates), key=len, reverse=True):
        if token:
            text = text.replace(token, token.replace("<", "＜").replace(">", "＞"))
    return text


def render_gemma4_chat(
    tokenizer: Any,
    system_prompt: str,
    messages: Sequence[Mapping[str, str]],
    *,
    partial_assistant: str | None = None,
) -> str:
    """Render one open Gemma 4 model turn, omitting BOS by contract."""

    if not is_gemma4_tokenizer(tokenizer):
        raise PromptContractError("tokenizer lacks the Gemma 4 turn-token contract")
    parts = [
        f"{SOT}system\n{_neutralize_control_tokens(system_prompt, tokenizer)}{EOT}\n"
    ]
    previous_role: str | None = None
    for message in messages:
        if not isinstance(message, Mapping):
            raise PromptContractError("chat messages must be mappings")
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise PromptContractError("chat message role or content is invalid")
        if role == previous_role:
            raise PromptContractError("chat roles must alternate")
        gemma_role = "user" if role == "user" else "model"
        safe = _neutralize_control_tokens(content, tokenizer)
        parts.append(f"{SOT}{gemma_role}\n{safe}{EOT}\n")
        previous_role = role
    if not messages or previous_role != "user":
        raise PromptContractError("generation requires a final user message")
    parts.append(f"{SOT}model\n")
    if partial_assistant is not None:
        parts.append(_neutralize_control_tokens(partial_assistant, tokenizer))
    rendered = "".join(parts)
    if rendered.startswith("<bos>"):
        raise PromptContractError("rendered chat must omit BOS before tokenization")
    return rendered


def render_chat_prompt(
    tokenizer: Any,
    system_prompt: str,
    messages: Sequence[Mapping[str, str]],
    *,
    partial_assistant: str | None = None,
) -> str:
    """Render Gemma 4 locally or a declared third-party template for tests.

    There is deliberately no ASCII transcript fallback.  A production
    tokenizer with neither contract must fail closed instead of teaching the
    model to generate fake user turns.
    """

    if is_gemma4_tokenizer(tokenizer):
        return render_gemma4_chat(
            tokenizer,
            system_prompt,
            messages,
            partial_assistant=partial_assistant,
        )
    template = getattr(tokenizer, "apply_chat_template", None)
    # Lightweight injected tokenizers used by tests may provide the method
    # without a serialized template. Production Gemma is gated above.
    if callable(template):
        payload = [{"role": "system", "content": system_prompt}, *messages]
        options: dict[str, Any] = {"tokenize": False}
        if partial_assistant is None:
            options["add_generation_prompt"] = True
        else:
            payload.append({"role": "assistant", "content": partial_assistant})
            options["continue_final_message"] = True
        try:
            rendered = template(payload, **options)
        except (KeyError, TypeError, ValueError) as exc:
            raise PromptContractError("declared chat template failed") from exc
        if isinstance(rendered, str) and rendered:
            bos = getattr(tokenizer, "bos_token", None)
            if isinstance(bos, str) and bos and rendered.startswith(bos):
                rendered = rendered[len(bos) :]
            return rendered
    raise PromptContractError("no verified local chat template is available")


def stop_token_ids(tokenizer: Any) -> Tensor:
    """Return official stops plus quarantined reserved text tokens.

    The local six-file mirror does not declare every reserved ``<unusedN>``
    token as special. A greedy decoder can emit and repeat one after an
    otherwise complete answer. Treating such tokens as terminal keeps them out
    of the public response and prevents a false length continuation.
    """

    values: list[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int) and not isinstance(eos, bool) and eos >= 0:
        values.append(eos)
    for token in (EOT, TOOL_RESPONSE):
        value = _token_id(tokenizer, token)
        if value is not None and value not in values:
            values.append(value)
    multimodal = _token_id(tokenizer, "[multimodal]")
    if multimodal is not None and multimodal not in values:
        values.append(multimodal)
    # The low control band (<|think|> / turn markers begin at 98+) contains
    # unused0..unused88. Quarantine that whole band while keeping the protocol
    # frame compact; higher multimodal vocabulary is filtered on decode.
    for index in range(89):
        value = _token_id(tokenizer, f"<unused{index}>")
        if value is not None and value not in values:
            values.append(value)
    if not values:
        raise PromptContractError("tokenizer has no verified response stop token")
    channel_end = _token_id(tokenizer, EOC)
    values = [value for value in values if value != channel_end]
    if len(values) > 128:
        raise PromptContractError("response stop contract exceeds its tensor bound")
    return torch.tensor(values, dtype=torch.int64)


def truncate_at_stop(token_ids: Tensor, stops: Tensor) -> Tensor:
    """Remove the first terminal token and every token after it."""

    tokens = torch.as_tensor(token_ids).detach().to("cpu", dtype=torch.int64).flatten()
    stop_values = set(map(int, torch.as_tensor(stops).flatten().tolist()))
    for index, value in enumerate(tokens.tolist()):
        if value in stop_values:
            return tokens[:index].contiguous()
    return tokens.contiguous()


def reserved_stop_sequences(tokenizer: Any) -> Tensor:
    """Encode bounded multi-token pseudo-markers produced by bad artifacts."""

    rows: list[list[int]] = []
    for marker in _RESERVED_OUTPUT_SEQUENCES:
        try:
            encoded = tokenizer(
                marker,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=False,
            )
            values = torch.as_tensor(encoded["input_ids"], dtype=torch.int64).flatten()
        except (KeyError, TypeError, ValueError):
            continue
        if 1 < values.numel() <= 16 and bool((values >= 0).all()):
            row = values.tolist()
            if row not in rows:
                rows.append(row)
    if not rows:
        return torch.empty((0, 0), dtype=torch.int64)
    width = max(map(len, rows))
    result = torch.full((len(rows), width), -1, dtype=torch.int64)
    for index, row in enumerate(rows):
        result[index, : len(row)] = torch.tensor(row, dtype=torch.int64)
    return result


def final_channel_tokens(tokenizer: Any, token_ids: Tensor) -> Tensor:
    """Extract a structured final channel, suppressing any analysis channel."""

    tokens = torch.as_tensor(token_ids).detach().to("cpu", dtype=torch.int64).flatten()
    soc = _token_id(tokenizer, SOC)
    eoc = _token_id(tokenizer, EOC)
    if soc is None or eoc is None or soc not in tokens.tolist():
        return tokens.contiguous()
    positions = [index for index, value in enumerate(tokens.tolist()) if value == soc]
    for position_index, start in enumerate(positions):
        try:
            label_end = tokens.tolist().index(eoc, start + 1)
        except ValueError:
            return torch.empty(0, dtype=torch.int64)
        label = tokenizer.decode(
            tokens[start + 1 : label_end].tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if isinstance(label, str) and label.strip().lower() == "final":
            end = (
                positions[position_index + 1]
                if position_index + 1 < len(positions)
                else tokens.numel()
            )
            return tokens[label_end + 1 : end].contiguous()
    return torch.empty(0, dtype=torch.int64)


def decode_response(tokenizer: Any, token_ids: Tensor, stops: Tensor) -> str:
    """Token-boundary parser for one public model response."""

    bounded = truncate_at_stop(token_ids, stops)
    public = final_channel_tokens(tokenizer, bounded)
    text = tokenizer.decode(
        public.tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(text, str):
        raise PromptContractError("tokenizer decode did not return text")
    return text


__all__ = [
    "EOC",
    "EOT",
    "PromptContractError",
    "SOC",
    "SOT",
    "TOOL_RESPONSE",
    "decode_response",
    "final_channel_tokens",
    "is_gemma4_tokenizer",
    "render_chat_prompt",
    "render_gemma4_chat",
    "reserved_stop_sequences",
    "stop_token_ids",
    "truncate_at_stop",
]
