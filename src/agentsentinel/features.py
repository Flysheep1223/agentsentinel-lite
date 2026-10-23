from __future__ import annotations

import json

NUMERIC_FEATURES = [
    "step", "tool_calls", "error_count", "retry_count", "schema_error_count",
    "input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens",
    "total_tokens", "cost_usd", "write_count", "elapsed_ms",
    "recent_error_count", "recent_latency_ms", "recent_same_tool_count",
    "recent_repeat_ratio", "consecutive_errors",
]
CATEGORICAL_FEATURES = ["tool", "previous_tool", "workflow", "status"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def signature(event: dict) -> str:
    return event["tool"] + ":" + json.dumps(
        event["arguments"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


class FeatureBuilder:
    """Accepts observable tool events only; never task labels or injector metadata."""

    def __init__(self, workflow: str):
        self.workflow = workflow
        self.history: list[dict] = []
        self.errors = self.retries = self.schema_errors = self.writes = self.streak = 0

    def add(self, event: dict) -> dict:
        previous = self.history[-1] if self.history else None
        sig = signature(event)
        self.retries += int(
            previous is not None and previous["error"] and signature(previous) == sig
        )
        self.errors += int(event["error"])
        self.schema_errors += int(event["schema_error"])
        self.writes += int(event["is_write"])
        self.streak = self.streak + 1 if event["error"] else 0
        self.history.append(event)
        recent = self.history[-3:]
        signatures = [signature(item) for item in recent]
        totals = event["totals"]
        return {
            "step": event["step"], "tool_calls": len(self.history),
            "error_count": self.errors, "retry_count": self.retries,
            "schema_error_count": self.schema_errors,
            **{key: totals.get(key) for key in (
                "input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens",
                "total_tokens", "cost_usd",
            )},
            "write_count": self.writes, "elapsed_ms": event["elapsed_ms"],
            "recent_error_count": sum(int(e["error"]) for e in recent),
            "recent_latency_ms": sum(e["latency_ms"] for e in recent) / len(recent),
            "recent_same_tool_count": sum(e["tool"] == event["tool"] for e in recent),
            "recent_repeat_ratio": (len(signatures) - len(set(signatures))) / len(signatures),
            "consecutive_errors": self.streak, "tool": event["tool"],
            "previous_tool": previous["tool"] if previous else "__start__",
            "workflow": self.workflow, "status": event["status"],
        }
