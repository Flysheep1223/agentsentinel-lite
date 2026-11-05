import asyncio
import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from agentsentinel.backends import Completion, OpenAIBackend, SessionDeadline
from agentsentinel.config import Settings
from agentsentinel.environment import FaultConfig
from agentsentinel.harness import run_session
from agentsentinel.tasks import make_tasks


def fake_backend(monkeypatch, completions):
    backend = Mock()
    backend.complete.side_effect = completions
    monkeypatch.setattr("agentsentinel.harness.OpenAIBackend", lambda settings: backend)
    return backend


def test_defaults_and_cache_pricing():
    cfg = Settings(currency="CNY", input_price=13, cached_input_price=1.3, output_price=68)
    assert (cfg.max_steps, cfg.max_tokens, cfg.session_timeout) == (12, 30000, 120)
    assert cfg.cost(1000, 100, 800) == pytest.approx(0.01044)
    assert cfg.cost(1000, 100) == pytest.approx(0.0198)


@pytest.mark.parametrize("finish", ["length", None, "content_filter"])
def test_nonstandard_finish_never_normal_completion(tmp_path, monkeypatch, finish):
    fake_backend(monkeypatch, [Completion(
        {"role": "assistant", "content": "unfinished"}, 20, 10, finish_reason=finish,
    )])
    result = run_session(make_tasks()[0], Settings(), FaultConfig(), tmp_path,
                         backend_name="openai")
    assert result["termination"] == (
        "output_truncated" if finish == "length" else "invalid_finish_reason"
    )
    assert result["valid_for_analysis"] == (finish == "length")
    assert not result["success"]


def test_missing_usage_preserves_prior_known_spend(tmp_path, monkeypatch):
    call = {"role": "assistant", "tool_calls": [{
        "id": "one", "function": {"name": "get_ticket", "arguments": '{"ticket_id":"T-0"}'},
    }]}
    fake_backend(monkeypatch, [Completion(call, 1000, 100, 50, 800, "tool_calls"),
                               Completion({"role": "assistant"}, None, None)])
    result = run_session(make_tasks()[0], Settings(
        currency="CNY", input_price=13, cached_input_price=1.3, output_price=68,
    ), FaultConfig(), tmp_path, backend_name="openai")
    totals = result["totals"]
    assert not result["valid_for_analysis"]
    assert totals["known_cost_estimate"] == pytest.approx(0.01044)
    assert totals["known_input_tokens"] == 1000
    assert totals["total_tokens"] is None
    assert totals["cost_usd"] is None
    assert totals["unknown_usage_requests"] == 1


def test_reasoning_not_double_counted_and_cny_not_usd(tmp_path, monkeypatch):
    fake_backend(monkeypatch, [Completion(
        {"role": "assistant", "content": "done"}, 1000, 100, 50, 800,
    )])
    result = run_session(make_tasks()[0], Settings(
        currency="CNY", input_price=13, cached_input_price=1.3, output_price=68,
    ), FaultConfig(), tmp_path, backend_name="openai")
    assert result["totals"]["total_tokens"] == 1100
    assert result["totals"]["cost_usd"] is None
    assert result["totals"]["cost_cny"] == pytest.approx(0.01044)


@pytest.mark.parametrize("input_tokens,output_tokens,reasoning,cached", [
    (-1, 10, None, None), (10, "bad", None, None), (10, 5, 6, 0), (10, 5, 0, 11),
])
def test_invalid_usage_is_not_training_failure(
    tmp_path, monkeypatch, input_tokens, output_tokens, reasoning, cached,
):
    fake_backend(monkeypatch, [Completion(
        {"role": "assistant"}, input_tokens, output_tokens, reasoning, cached,
    )])
    result = run_session(make_tasks()[0], Settings(), FaultConfig(), tmp_path,
                         backend_name="openai")
    assert result["termination"] == "invalid_usage"
    assert not result["valid_for_analysis"]


def test_token_budget_blocks_write_and_next_request(tmp_path, monkeypatch):
    backend = fake_backend(monkeypatch, [Completion(
        {"role": "assistant", "tool_calls": [{"id": "one", "function": {
            "name": "update_ticket", "arguments": "{}",
        }}]}, 30000, 1, finish_reason="tool_calls",
    )])
    result = run_session(make_tasks()[0], Settings(), FaultConfig(), tmp_path,
                         backend_name="openai")
    assert result["termination"] == "token_budget"
    assert result["totals"]["tool_calls"] == 0
    assert backend.complete.call_count == 1


def test_session_deadline_passed_to_http_and_stops(tmp_path, monkeypatch):
    backend = fake_backend(monkeypatch, [SessionDeadline("deadline")])
    result = run_session(make_tasks()[0], Settings(session_timeout=0.5), FaultConfig(),
                         tmp_path, backend_name="openai")
    assert 0 < backend.complete.call_args.args[-1] <= 0.5
    assert result["termination"] == "time_budget"
    assert result["totals"]["unknown_usage_requests"] == 1
    assert backend.complete.call_count == 1


def test_sse_finish_reason_and_truncated_tool_arguments():
    body = ('data: ' + json.dumps({"choices": [{"index": 0, "delta": {},
                                               "finish_reason": "length"}],
                                  "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
            + '\n\ndata: [DONE]\n\n').encode()
    completion = OpenAIBackend._consume([body], lambda event: None)
    assert completion.finish_reason == "length"


def test_wall_clock_timeout_cancels_continuously_streaming_http(monkeypatch):
    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(100):
                await asyncio.sleep(0.01)
                yield b": heartbeat\n\n"

    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=SlowStream()))
    monkeypatch.setattr("agentsentinel.backends.httpx.AsyncClient", lambda **kwargs:
                        real_client(transport=transport, **kwargs))
    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    backend = OpenAIBackend(Settings(model="offline", request_timeout=1))
    recorded = []
    with pytest.raises(SessionDeadline):
        backend.complete([], [], recorded.append, 100, 0.05)
    assert recorded


def test_time_boundary_prevents_tool_execution(tmp_path, monkeypatch):
    import time

    backend = fake_backend(monkeypatch, [])

    def slow(*args):
        time.sleep(0.03)
        return Completion({"role": "assistant", "tool_calls": [{"id": "one", "function": {
            "name": "update_ticket", "arguments": "{}",
        }}]}, 1, 1, finish_reason="tool_calls")

    backend.complete.side_effect = slow
    result = run_session(make_tasks()[0], Settings(session_timeout=0.02), FaultConfig(),
                         tmp_path, backend_name="openai")
    assert result["termination"] == "time_budget"
    assert result["totals"]["tool_calls"] == 0
    recorded = [json.loads(line) for line in
                (Path(result["path"]) / "events.jsonl").read_text().splitlines()]
    assert not any(event["kind"] == "tool_end" for event in recorded)


def test_resumable_plan_pilot_freeze_and_no_duplicate_requests(tmp_path, monkeypatch):
    from agentsentinel import collection_run
    from agentsentinel.backends import ScriptedBackend
    from agentsentinel.config import read_json, write_json

    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    monkeypatch.setattr(Settings, "from_env", lambda: Settings(model="offline"))
    root = tmp_path / "experiment"
    collection_run.initialize(root)
    original = collection_run.run_session

    def offline_session(*args, **kwargs):
        args = list(args)
        args[4] = "scripted"
        return original(*args, **kwargs)

    def normal(seed):
        backend = ScriptedBackend(seed)
        backend.behavior = "normal"
        return backend

    monkeypatch.setattr("agentsentinel.harness.ScriptedBackend", normal)
    stub = Mock(side_effect=offline_session)
    monkeypatch.setattr(collection_run, "run_session", stub)
    pilot = collection_run.run_phase(root, "pilot")
    assert pilot["completed_sessions"] == 30
    assert pilot["by_split"]["train"]["sessions"] == 30
    collection_run.run_phase(root, "pilot")
    assert stub.call_count == 30
    collection_run.freeze_after_pilot(root)
    formal = collection_run.run_phase(root, "collection")
    assert formal["completed_sessions"] == 600
    assert [formal["by_split"][s]["sessions"] for s in
            ("train", "validation", "test")] == [360, 120, 120]
    assert stub.call_count == 630
    collection_run.run_phase(root, "collection")
    assert stub.call_count == 630
    ledger_path = root / "collection/ledger.json"
    ledger = read_json(ledger_path)
    first = next(iter(ledger["entries"]))
    ledger["entries"][first] = {"state": "started"}
    write_json(ledger_path, ledger)
    collection_run.run_phase(root, "collection")
    assert stub.call_count == 630


def test_ambiguous_started_attempt_is_not_retried(tmp_path, monkeypatch):
    from agentsentinel import collection_run
    from agentsentinel.config import read_json, write_json

    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    monkeypatch.setattr(Settings, "from_env", lambda: Settings(model="offline"))
    root = tmp_path / "experiment"
    collection_run.initialize(root)
    ledger_path = root / "pilot/ledger.json"
    ledger = read_json(ledger_path)
    ledger["entries"][next(iter(ledger["entries"]))] = {"state": "started"}
    write_json(ledger_path, ledger)
    stub = Mock(side_effect=AssertionError("No paid retry allowed"))
    monkeypatch.setattr(collection_run, "run_session", stub)
    with pytest.raises(ValueError, match="Paused"):
        collection_run.run_phase(root, "pilot")
    stub.assert_not_called()


@pytest.mark.parametrize("field", ["max_tokens", "session_timeout", "max_completion_tokens"])
def test_optional_budgets_null_roundtrip_and_invalid_values(field):
    cfg = Settings(**{field: None})
    assert json.loads(cfg.model_dump_json())[field] is None
    assert Settings.model_validate_json(cfg.model_dump_json()) == cfg
    for invalid in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            Settings(**{field: invalid})


@pytest.mark.parametrize("output_limit,remaining,expected", [
    (None, None, None), (None, 73, 73), (1024, None, 1024), (1024, 73, 73),
])
def test_optional_payload_and_network_timeout(monkeypatch, output_limit, remaining, expected):
    captured = []
    body = ('data: ' + json.dumps({"choices": [{"delta": {"content": "done"},
                                               "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
            + '\n\ndata: [DONE]\n\n').encode()

    def respond(request):
        captured.append(json.loads(request.content))
        assert request.extensions["timeout"] == dict.fromkeys(("connect", "read", "write", "pool"), 60)
        return httpx.Response(200, content=body)

    real_client = httpx.AsyncClient
    monkeypatch.setattr("agentsentinel.backends.httpx.AsyncClient", lambda **kwargs:
                        real_client(transport=httpx.MockTransport(respond), **kwargs))
    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    monkeypatch.setattr("agentsentinel.backends.asyncio.timeout", Mock(
        side_effect=AssertionError("No session timeout may be installed"),
    ))
    backend = OpenAIBackend(Settings(model="offline", session_timeout=None,
                                     max_completion_tokens=output_limit))
    assert backend.complete([], [], lambda event: None, remaining, None).output_tokens == 5
    assert len(captured) == 1
    if expected is None:
        assert "max_tokens" not in captured[0]
    else:
        assert captured[0]["max_tokens"] == expected


def test_network_timeout_remains_effective_without_session_deadline(monkeypatch):
    from agentsentinel.backends import BackendError

    attempts = []

    def timeout(request):
        attempts.append(request)
        assert request.extensions["timeout"]["read"] == 60
        raise httpx.ReadTimeout("isolated timeout")

    real_client = httpx.AsyncClient
    monkeypatch.setattr("agentsentinel.backends.httpx.AsyncClient", lambda **kwargs:
                        real_client(transport=httpx.MockTransport(timeout), **kwargs))
    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    with pytest.raises(BackendError, match="ReadTimeout"):
        OpenAIBackend(Settings(model="offline", session_timeout=None)).complete(
            [], [], lambda event: None, None, None,
        )
    assert len(attempts) == 1


@pytest.mark.parametrize("limit,termination,expected_calls", [
    ({"max_steps": 2}, "step_budget", 2),
    ({"max_model_calls": 2}, "model_call_budget", 2),
])
def test_unlimited_budgets_keep_step_and_call_limits(tmp_path, monkeypatch, limit,
                                                   termination, expected_calls):
    call = {"role": "assistant", "tool_calls": [{"id": "one", "function": {
        "name": "get_ticket", "arguments": json.dumps({"ticket_id": make_tasks()[0].ticket_id}),
    }}]}
    backend = fake_backend(monkeypatch, [Completion(call, 100000, 2000, finish_reason="tool_calls")
                                       for _ in range(expected_calls)])
    settings = Settings(max_tokens=None, session_timeout=None, max_completion_tokens=None, **limit)
    result = run_session(make_tasks()[0], settings, FaultConfig(), tmp_path, backend_name="openai")
    assert result["termination"] == termination
    assert result["totals"]["total_tokens"] == 102000 * expected_calls
    assert all(call.args[-2:] == (None, None) for call in backend.complete.call_args_list)
    recorded = [json.loads(line) for line in
                (Path(result["path"]) / "events.jsonl").read_text().splitlines()]
    eligibility = [e["decision_eligible"] for e in recorded if e["kind"] == "tool_end"]
    assert eligibility == [True, False]


def test_expired_deadline_is_not_replaced_by_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    backend = OpenAIBackend(Settings(model="offline"))

    async def inspect(*args):
        assert args[-1] == 0
        return "offline"

    monkeypatch.setattr(backend, "_complete", inspect)
    assert backend.complete([], [], lambda event: None, 10, 0) == "offline"
