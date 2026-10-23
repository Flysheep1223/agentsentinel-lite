from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


class Settings(BaseModel):
    model: str = ""
    base_url: str = "https://api.openai.com/v1"
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_steps: int = Field(default=18, ge=1)
    max_model_calls: int = Field(default=24, ge=1)
    max_tokens: int = Field(default=24000, ge=1)
    max_completion_tokens: int = Field(default=1024, ge=1)
    request_timeout: float = Field(default=60, gt=0)
    input_price: float | None = Field(default=None, ge=0)
    output_price: float | None = Field(default=None, ge=0)
    monitor_price_per_second: float = Field(default=0, ge=0)

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(ROOT / ".env", override=True)
        mapping = {
            "model": "OPENAI_MODEL",
            "base_url": "OPENAI_BASE_URL",
            "temperature": "AGENT_TEMPERATURE",
            "max_steps": "AGENT_MAX_STEPS",
            "max_model_calls": "AGENT_MAX_MODEL_CALLS",
            "max_tokens": "AGENT_MAX_TOKENS",
            "max_completion_tokens": "AGENT_MAX_COMPLETION_TOKENS",
            "request_timeout": "AGENT_REQUEST_TIMEOUT",
            "input_price": "INPUT_PRICE_PER_MILLION",
            "output_price": "OUTPUT_PRICE_PER_MILLION",
            "monitor_price_per_second": "MONITOR_PRICE_PER_SECOND",
        }
        return cls(**{k: os.environ[v] for k, v in mapping.items() if os.getenv(v)})

    def cost(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        if None in (self.input_price, self.output_price, input_tokens, output_tokens):
            return None
        return (input_tokens * self.input_price + output_tokens * self.output_price) / 1e6
