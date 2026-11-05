from __future__ import annotations

import asyncio
import base64
import json
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import httpx

from .config import Settings
from .tasks import SEVERITY, aggregate_writes, intended_writes

Recorder = Callable[[dict], None]


class BackendError(RuntimeError):
    """Transport/protocol failure, excluded from tool-failure training data."""


class SessionDeadline(BackendError):
    pass


@dataclass
class Completion:
    message: dict
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    finish_reason: str | None = "stop"


class ScriptedBackend:
    """Deterministic engineering fixture. Never represents measured LLM behavior."""

    name = "scripted"

    def __init__(self, seed: int):
        self.random = random.Random(seed)
        self.calls = 0
        # 带种子的非理想行为仅用于工程验证，不代表真实模型表现。
        self.behavior = self.random.choice(["normal"] * 6 + ["give_up", "repeat", "wrong_write"])

    @staticmethod
    def _plan(request: dict, observed: dict):
        # 每次从公开请求和已观察响应重建计划，不访问任务私有状态或标准答案。
        def read(tool: str, object_id: str, fields: tuple[str, ...] = ()):
            id_field = "ticket_id" if tool == "get_ticket" else "service_id"
            response = observed.get((tool, object_id), {})
            if (response.get(id_field) != object_id
                    or any(field not in response or response[field] is None for field in fields)):
                yield {"tool": tool, "arguments": {id_field: object_id}}
            return response

        def ticket(tid: str, fields: tuple[str, ...] = ()):
            return (yield from read("get_ticket", tid, ("service_id", *fields)))

        def service(sid: str, fields: tuple[str, ...] = ()):
            return (yield from read("get_service_status", sid, ("state", *fields)))

        def policy(sid: str):
            return (yield from read("lookup_policy", sid, ("action", "owner")))

        tid, sid = request["ticket_id"], request["service_id"]
        structure, goals = request["structure"], request["goals"]
        if request["workflow"] == "ticket":
            root_fields = {
                "linked": ("related_ticket_id",), "compare": ("candidate_ticket_ids",),
                "branch": ("escalation_ticket_id",),
            }.get(structure, ())
            target = yield from ticket(tid, root_fields)
            if structure == "linked":
                tid = target["related_ticket_id"]
                target = yield from ticket(tid)
            elif structure == "compare":
                candidates = []
                for candidate_id in target["candidate_ticket_ids"]:
                    candidates.append((yield from ticket(candidate_id, ("impact_score",))))
                target = min(candidates, key=lambda t: (-t["impact_score"], t["ticket_id"]))
                tid = target["ticket_id"]
            elif structure == "branch":
                root_service = yield from service(target["service_id"])
                if root_service["state"] == "healthy":
                    goals = ["status"]
                else:
                    tid, goals = target["escalation_ticket_id"], ["owner", "note"]
                    target = yield from ticket(tid)
            elif structure == "chain":
                while True:
                    if "parent_ticket_id" not in target:
                        yield {"tool": "get_ticket", "arguments": {"ticket_id": tid}}
                    parent = target["parent_ticket_id"]
                    if parent is None:
                        break
                    tid = parent
                    target = yield from ticket(tid)
            sid = target["service_id"]
            selected = yield from service(sid)
        else:
            root_fields = {
                "linked": ("response_ticket_id",), "compare": ("related_service_ids",),
                "branch": ("dependency_service_id",), "aggregate": ("dependency_service_ids",),
            }.get(structure, ())
            selected = yield from service(sid, root_fields)
            if structure == "linked":
                tid = selected["response_ticket_id"]
                target = yield from ticket(tid)
                sid = target["service_id"]
                selected = yield from service(sid)
            elif structure == "compare":
                candidates = []
                for candidate_id in selected["related_service_ids"]:
                    candidates.append((yield from service(candidate_id, ("error_count",))))
                selected = min(candidates, key=lambda s: (
                    -SEVERITY[s["state"]], -s["error_count"], s["service_id"],
                ))
                sid = selected["service_id"]
            elif structure == "branch":
                if selected["state"] == "healthy":
                    goals = ["notify"]
                else:
                    sid = selected["dependency_service_id"]
                    selected = yield from service(sid)
            elif structure == "aggregate":
                services = [selected]
                for dependency in selected["dependency_service_ids"]:
                    services.append((yield from service(dependency)))
                policies = {}
                for item in services:
                    policies[item["service_id"]] = yield from policy(item["service_id"])
                yield from aggregate_writes(sid, services, policies)
                return
        selected_policy = {}
        if any(goal in {"owner", "note", "incident", "notify"} for goal in goals):
            selected_policy = yield from policy(sid)
        yield from intended_writes(goals, tid, sid, selected["state"], selected_policy)

    def complete(self, messages: list[dict], tools: list[dict], record: Recorder,
                 remaining_tokens: int | None) -> Completion:
        self.calls += 1
        request = json.loads(next(msg["content"] for msg in messages if msg["role"] == "user"))
        observed: dict[tuple[str, str], dict] = {}
        successful_writes = []
        calls = {}
        failures = 0
        for msg in messages:
            if msg["role"] == "assistant":
                for call in msg.get("tool_calls", []):
                    calls[call["id"]] = call["function"]
            elif msg["role"] == "tool" and msg.get("tool_call_id") in calls:
                call = calls[msg["tool_call_id"]]
                response = json.loads(msg["content"])
                arguments = json.loads(call["arguments"])
                if response.get("error"):
                    failures += 1
                elif response.get("ok"):
                    successful_writes.append({"tool": call["name"], "arguments": arguments})
                else:
                    object_id = arguments.get("ticket_id", arguments.get("service_id"))
                    observed[(call["name"], object_id)] = response
        # 按完整操作匹配成功写入，不能把另一个对象的成功响应当成当前操作完成。
        remaining_successes = list(successful_writes)
        operation = None
        for candidate in self._plan(request, observed):
            if candidate in remaining_successes:
                remaining_successes.remove(candidate)
            else:
                operation = candidate
                break
        if self.behavior == "wrong_write" and operation:
            arguments = operation["arguments"]
            if operation["tool"] in {"update_ticket", "append_note"}:
                operation = {"tool": operation["tool"], "arguments": {
                    **arguments, "ticket_id": "T-decoy",
                }}
            elif operation["tool"] == "record_incident":
                operation = {"tool": operation["tool"], "arguments": {
                    **arguments, "action": "incorrect-action",
                }}
            elif operation["tool"] == "send_notification":
                operation = {"tool": operation["tool"], "arguments": {
                    **arguments, "body": "incorrect-notification",
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
        # 合成计数在会话元数据中明确标记，不用作真实模型测量。
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
                 remaining_tokens: int | None, remaining_seconds: float | None = None) -> Completion:
        return asyncio.run(self._complete(messages, tools, record, remaining_tokens,
                                         remaining_seconds))

    async def _complete(self, messages, tools, record, remaining_tokens, remaining_seconds):
        cfg = self.settings
        payload = {
            "model": cfg.model, "messages": messages, "tools": tools,
            "parallel_tool_calls": False, "temperature": cfg.temperature,
            "stream": True, "stream_options": {"include_usage": True},
        }
        limits = [limit for limit in (cfg.max_completion_tokens, remaining_tokens)
                  if limit is not None]
        if limits:
            payload["max_tokens"] = min(limits)
        # No automatic API retries: every invocation and failure remains observable.
        try:
            async with (asyncio.timeout(remaining_seconds)
                        if remaining_seconds is not None else nullcontext()):
                async with httpx.AsyncClient(
                    timeout=(cfg.request_timeout if remaining_seconds is None
                             else min(cfg.request_timeout, remaining_seconds)),
                ) as client:
                    async with client.stream(
                        "POST", cfg.base_url.rstrip("/") + "/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"}, json=payload,
                    ) as response:
                        if response.status_code >= 400:
                            raise BackendError(f"LLM HTTP {response.status_code}")
                        chunks = []
                        async for chunk in response.aiter_bytes():
                            record({"kind": "llm_sse", "body_base64":
                                    base64.b64encode(chunk).decode("ascii")})
                            chunks.append(chunk)
                        return self._consume(chunks, lambda event: None)
        except TimeoutError:
            raise SessionDeadline("Session wall-clock deadline during HTTP request") from None
        except (httpx.HTTPError, UnicodeError, json.JSONDecodeError,
                KeyError, TypeError, AttributeError, ValueError) as exc:
            raise BackendError(f"LLM transport/protocol error: {type(exc).__name__}") from None

    @staticmethod
    def _consume(chunks, record: Recorder) -> Completion:
        buffer = b""
        data_lines: list[bytes] = []
        content: list[str] = []
        calls: dict[int, dict] = {}
        usage: dict = {}
        done = False
        finish_reason = None

        def dispatch() -> None:
            nonlocal usage, done, finish_reason
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
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
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
            if finish_reason != "length" and any(
                not c["id"] or not c["function"]["name"] for c in ordered
            ):
                raise BackendError("Incomplete streamed tool call")
            message["tool_calls"] = ordered
        return Completion(
            message=message,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            finish_reason=finish_reason,
        )
