"""Explicit, single-request OpenRouter transport for a fresh-context QA oracle."""

import argparse
import http.client
import json
import os
import sys
from urllib import error, request

from round_trace import identifier


_URL = "https://openrouter.ai/api/v1/chat/completions"
_MAX_KEY_BYTES = 4096
_MAX_REQUEST_BYTES = 1024 * 1024
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class OracleGatewayError(ValueError):
    """A gateway failure whose message is safe to include in a diagnostic."""


class _NoRedirect(request.HTTPRedirectHandler):
    # Reject the response before urllib interprets the untrusted Location header.
    http_error_301 = http_error_302 = http_error_303 = http_error_307 = http_error_308 = (
        request.HTTPDefaultErrorHandler.http_error_default
    )


def api_key() -> str:
    """Read the credential only for an explicitly selected OpenRouter transport."""
    filename = os.environ.get("OPENROUTER_API_KEY_FILE")
    if filename is not None:
        try:
            with open(filename, "rb") as source:
                data = source.read(_MAX_KEY_BYTES + 1)
            if len(data) > _MAX_KEY_BYTES:
                raise OracleGatewayError("OpenRouter key file exceeds the size limit")
            key = data.decode("ascii").strip()
        except (OSError, UnicodeError):
            raise OracleGatewayError("cannot read the OpenRouter key file") from None
    else:
        key = os.environ.get("OPENROUTER_API_KEY", "")
    if not (0 < len(key) <= _MAX_KEY_BYTES and key.isascii() and key.isprintable() and " " not in key):
        raise OracleGatewayError("a valid OpenRouter API key is required")
    return key


def _model_id(value: str) -> str:
    try:
        return identifier(value)
    except (TypeError, argparse.ArgumentTypeError):
        raise OracleGatewayError("invalid OpenRouter model identifier") from None


def _reject_constant(value: str):
    raise ValueError("invalid JSON number")


def _parse_json(text: str):
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        raise OracleGatewayError("expected complete JSON from OpenRouter") from None


def _token_count(usage: dict, name: str) -> int | None:
    count = usage.get(name)
    if count is not None and (type(count) is not int or not 0 <= count < 2**63):
        raise OracleGatewayError("invalid token count from OpenRouter")
    return count


def _cost(usage: dict) -> float | None:
    value = usage.get("cost")
    if value is None:
        return None
    if type(value) not in (int, float) or not 0 <= value <= sys.float_info.max:
        raise OracleGatewayError("invalid cost from OpenRouter")
    return float(value)


def _completion(data: bytes, shape: str) -> tuple[str, str | None, int | None, int | None, float | None]:
    try:
        body = _parse_json(data.decode("utf-8"))
    except UnicodeDecodeError:
        raise OracleGatewayError("invalid UTF-8 from OpenRouter") from None
    if not isinstance(body, dict):
        raise OracleGatewayError("invalid completion envelope from OpenRouter")
    choices = body.get("choices")
    if body.get("error") is not None or not isinstance(choices, list) or len(choices) != 1:
        raise OracleGatewayError("OpenRouter did not return one completion")
    choice = choices[0]
    if (not isinstance(choice, dict) or choice.get("finish_reason") != "stop"
            or choice.get("error") is not None):
        raise OracleGatewayError("OpenRouter did not finish the completion successfully")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant" or message.get("refusal"):
        raise OracleGatewayError("OpenRouter did not return an assistant answer")
    answer = message.get("content")
    if not isinstance(answer, str):
        raise OracleGatewayError("OpenRouter did not return answer text")
    try:
        answer.encode("utf-8")
    except UnicodeEncodeError:
        raise OracleGatewayError("invalid Unicode answer from OpenRouter") from None
    if not isinstance(_parse_json(answer), dict if shape == "object" else list):
        raise OracleGatewayError(f"expected a JSON {shape} answer from OpenRouter")
    model = body.get("model")
    usage = body.get("usage")
    if usage is None:
        usage = {}
    if not isinstance(usage, dict):
        raise OracleGatewayError("invalid usage from OpenRouter")
    return (answer, _model_id(model) if model is not None else None,
            _token_count(usage, "prompt_tokens"), _token_count(usage, "completion_tokens"), _cost(usage))


def complete(
    prompt: str, *, model: str, max_tokens: int, timeout: float, shape: str = "object"
) -> tuple[str, str | None, int | None, int | None, float | None]:
    """Send only this prompt and return JSON answer text and observed provider metadata."""
    if not isinstance(prompt, str) or len(prompt) > _MAX_REQUEST_BYTES or not prompt.strip():
        raise OracleGatewayError("expected a nonempty prompt within the request size limit")
    if type(max_tokens) is not int or not 0 < max_tokens < 2**31:
        raise OracleGatewayError("invalid maximum token count")
    if type(timeout) not in (int, float) or not 0 < timeout < 2**31:
        raise OracleGatewayError("invalid request timeout")
    if shape not in ("object", "array"):
        raise OracleGatewayError("expected JSON shape object or array")
    payload = {
        "model": _model_id(model),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "provider": {"require_parameters": True},
    }
    if shape == "object":
        payload["response_format"] = {"type": "json_object"}
    try:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        raise OracleGatewayError("invalid Unicode prompt") from None
    if len(data) > _MAX_REQUEST_BYTES:
        raise OracleGatewayError("OpenRouter request exceeds the size limit")
    req = request.Request(_URL, method="POST", data=data,
                          headers={"Content-Type": "application/json", "Accept": "application/json"})
    req.add_unredirected_header("Authorization", "Bearer " + api_key())
    try:
        opener = request.build_opener(_NoRedirect())
        with opener.open(req, timeout=timeout) as response:
            status = response.status
            raw = response.read(_MAX_RESPONSE_BYTES + 1) if status == 200 else b""
    except error.HTTPError as exc:
        exc.close()
        raise OracleGatewayError(f"OpenRouter returned HTTP {exc.code:d}") from None
    except (OSError, ValueError, http.client.HTTPException):
        raise OracleGatewayError("OpenRouter request failed") from None
    if status != 200:
        raise OracleGatewayError(f"OpenRouter returned HTTP {status:d}")
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise OracleGatewayError("OpenRouter response exceeds the size limit")
    return _completion(raw, shape)
