"""Generate one decorative poster asset with the user-authorized Ark endpoint."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENDPOINT = "ep-m-20261106162157-7d2jm"
PROMPT = (
    "Create a refined editorial scientific illustration, landscape 3:2 composition, "
    "for a university research poster about monitoring a tool-using AI agent. "
    "A short sequence of small geometric blocks and circular nodes travels from left to right "
    "through one elegant translucent monitoring gate. The gate gently scans the sequence. "
    "After the gate, the path forks: a continuous teal route and a shorter amber route ending "
    "at a simple pause symbol formed by two bars. Suggest observation, sequential decisions, "
    "and selective stopping. Restrained dimensional paper-and-glass aesthetic, soft shadows, "
    "precise clean edges, sparse composition, ample negative space. "
    "Color palette: deep ink navy #12313B, muted teal #26867C, warm amber #D89845, "
    "and pale sage. Flat warm ivory background #F6F4EF, evenly lit, no vignette. "
    "Suitable for the upper right of a sophisticated academic poster. "
    "No text, no letters, no numbers, no axes, no statistical charts, no logos, no watermarks, "
    "no robot faces, no futuristic city, no neon, no people. "
    "This is a conceptual illustration, not experimental data."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=Path)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=True)
    key = os.environ.get("OPENAI_API_KEY")
    base = os.environ.get("OPENAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    if not key:
        raise SystemExit("Missing project API key.")
    if urlsplit(base).hostname != "ark.cn-beijing.volces.com":
        raise SystemExit("Unexpected API host; refusing to send the configured key.")
    if args.output.exists():
        raise SystemExit("Output directory already exists; use a new revision directory.")
    prompt = args.prompt.read_text() if args.prompt else PROMPT
    args.output.mkdir(parents=True)
    request = {
        "model": args.endpoint,
        "prompt": prompt,
        "size": "2K",
        "response_format": "b64_json",
        "watermark": False,
    }
    (args.output / "request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n"
    )
    started = time.monotonic()
    print("Submitting one Seedream visual generation request.", flush=True)
    try:
        with httpx.Client(timeout=httpx.Timeout(240, connect=20)) as client:
            response = client.post(
                base.rstrip("/") + "/images/generations",
                headers={"Authorization": f"Bearer {key}"},
                json=request,
            )
            data = response.json()
            if response.is_error:
                error = data.get("error", {})
                message = str(error.get("message", "Image request failed")).replace(key, "[redacted]")
                (args.output / "error.json").write_text(json.dumps({
                    "status": response.status_code, "code": error.get("code"), "message": message,
                }, ensure_ascii=False, indent=2))
                raise SystemExit(f"Image API {response.status_code}: {message[:400]}")
            item = data["data"][0]
            if item.get("b64_json"):
                content = base64.b64decode(item["b64_json"])
            elif item.get("url"):
                download = client.get(item["url"])
                download.raise_for_status()
                content = download.content
            else:
                raise SystemExit("Image API returned no image content.")
        image = Image.open(io.BytesIO(content)).convert("RGB")
        destination = args.output / "seedream_visual.png"
        image.save(destination)
        metadata = {
            "requested_endpoint": args.endpoint,
            "returned_model": data.get("model"),
            "usage": data.get("usage"),
            "created": data.get("created"),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "image_size": list(image.size),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "purpose": "Conceptual poster decoration; never used as a source of study results.",
        }
        (args.output / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    except httpx.HTTPError as error:
        raise SystemExit(f"Image request failed: {type(error).__name__}; no automatic retry.") from None


if __name__ == "__main__":
    main()
