from __future__ import annotations

import base64
import json
import os
import random
from dataclasses import dataclass
from typing import Callable

import httpx

from .config import Settings
from .tasks import intended_writes

Recorder = Callable[[dict], None]


class BackendError(RuntimeError):
    """Transport/protocol failure, excluded from tool-failure training data."""


@dataclass
class Completion:
    message: dict
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None


class ScriptedBackend:
    """Deterministic engineering fixture. Never represents measured LLM behavior."""

    name = "scripted"

    def __init__(self, seed: int):
        self.random = random.Random(seed)
        self.calls = 0
        # Seeded imperfect behavior exercises both classes even without injected errors.
        self.behavior = self.random.choice(["normal"] * 6 + ["give_up", "repeat", "wrong_write"])

    def complete(self, messages: list[dict], tools: list[dict], record: Recorder,
                 remaining_tokens: int) -> Completion:
        self.calls += 1
        request = json.loads(messages[1]["content"])
        observed: dict[str, dict] = {}
        successful_writes = []
        last_call = None
        failures = 0
        for msg in messages:
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                last_call = msg["tool_calls"][0]["function"]
            elif msg["role"] == "tool" and last_call:
                response = json.loads(msg["content"])
                if response.get("error"):
                    failures += 1
                elif response.get("ok"):
                    successful_writes.append({
                        "tool": last_call["name"],
                        "arguments": json.loads(last_call["arguments"]),
                    })
                else:
                    observed[last_call["name"]] = response
        operation = None
        if request["workflow"] == "ticket" and "get_ticket" not in observed:
            operation = {"tool": "get_ticket", "arguments": {"ticket_id": request["ticket_id"]}}
        elif "get_service_status" not in observed:
            operation = {
                "tool": "get_service_status", "arguments": {"service_id": request["service_id"]},
            }
        elif not all(observed.get("lookup_policy", {}).get(key) for key in ("action", "owner")):
            operation = {"tool": "lookup_policy", "arguments": {"service_id": request["service_id"]}}
        else:
            plan = intended_writes(
                request["goals"], request["ticket_id"], request["service_id"],
                observed["get_service_status"]["state"], observed["lookup_policy"],
            )
            if len(successful_writes) < len(plan):
                operation = plan[len(successful_writes)]
                if self.behavior == "wrong_write" and operation["tool"] == "update_ticket":
                    operation = {"tool": "update_ticket", "arguments": {
                        **operation["arguments"], "ticket_id": "T-decoy",
                    }}
        if self.behavior == "repeat" and self.calls > 3:
            operation = {
                "tool": "get_service_status", "arguments": {"service_id": request["service_id"]},
            }
        if self.behavior == "give_up" and failures >= 2:
            operation = None
        message = {"role": "assistant", "content": "Finished."}
        if operation:
            message = {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": f"call_{self.calls}", "type": "function", "function": {
                    "name": operation["tool"],
                    "arguments": json.dumps(operation["arguments"], sort_keys=True),
                }}],
            }
        # Estimated counts are explicitly synthetic in session metadata.
        input_tokens = max(1, len(json.dumps(messages)) // 4)
        output_tokens = max(1, len(json.dumps(message)) // 4)
        record({"kind": "scripted_response", "message": message})
        return Completion(message, input_tokens, output_tokens)


class OpenAIBackend:
    """OpenAI-compatible Chat Completions, streamed SSE with lossless body chunks."""

    name = "openai"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.api_key or not settings.model:
            raise ValueError("请在项目 .env 配置 OPENAI_API_KEY 和 OPENAI_MODEL。")

    def complete(self, messages: list[dict], tools: list[dict], record: Recorder,
                 remaining_tokens: int) -> Completion:
        cfg = self.settings
        payload = {
            "model": cfg.model, "messages": messages, "tools": tools,
            "parallel_tool_calls": False, "temperature": cfg.temperature,
            "max_tokens": min(cfg.max_completion_tokens, remaining_tokens),
            "stream": True, "stream_options": {"include_usage": True},
        }
        # No automatic API retries: every invocation and failure remains observable.
        try:
            with httpx.Client(timeout=cfg.request_timeout) as client:
                with client.stream(
                    "POST", cfg.base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        # Do not print response bodies, request headers, or credentials.
                        raise BackendError(f"LLM HTTP {response.status_code}")
                    return self._consume(response.iter_bytes(), record)
        except (httpx.HTTPError, UnicodeError, json.JSONDecodeError) as exc:
            raise BackendError(f"LLM transport/protocol error: {type(exc).__name__}") from None

    @staticmethod
    def _consume(chunks, record: Recorder) -> Completion:
        buffer = b""
        data_lines: list[bytes] = []
        content: list[str] = []
        calls: dict[int, dict] = {}
        usage: dict = {}
        done = False

        def dispatch() -> None:
            nonlocal usage, done
            if not data_lines:
                return
            raw = b"\n".join(data_lines)
            data_lines.clear()
            if raw == b"[DONE]":
                done = True
                return
            event = json.loads(raw.decode("utf-8"))
            if "error" in event:
                raise BackendError("LLM returned a streamed error")
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("index", 0) != 0:
                    raise BackendError("Multiple completion choices are unsupported")
                delta = choice.get("delta", {})
                if delta.get("content"):
                    content.append(delta["content"])
                for call in delta.get("tool_calls", []):
                    entry = calls.setdefault(call["index"], {
                        "id": "", "type": "function", "function": {"name": "", "arguments": ""},
                    })
                    if call.get("id"):
                        entry["id"] += call["id"]
                    for key in ("name", "arguments"):
                        entry["function"][key] += call.get("function", {}).get(key) or ""

        for chunk in chunks:
            # Raw SSE response body can be reconstructed by concatenating decoded chunks.
            record({"kind": "llm_sse", "body_base64": base64.b64encode(chunk).decode("ascii")})
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.rstrip(b"\r")
                if not line:
                    dispatch()
                elif line.startswith(b"data:"):
                    data_lines.append(line[5:].removeprefix(b" "))
        if buffer:
            line = buffer.rstrip(b"\r")
            if line.startswith(b"data:"):
                data_lines.append(line[5:].removeprefix(b" "))
        dispatch()
        if not done:
            raise BackendError("SSE stream ended before [DONE]")
        if len(calls) > 1:
            raise BackendError("Provider ignored parallel_tool_calls=false")
        message = {"role": "assistant", "content": "".join(content) or None}
        if calls:
            ordered = [calls[index] for index in sorted(calls)]
            if any(not c["id"] or not c["function"]["name"] for c in ordered):
                raise BackendError("Incomplete streamed tool call")
            message["tool_calls"] = ordered
        return Completion(
            message=message,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        )
