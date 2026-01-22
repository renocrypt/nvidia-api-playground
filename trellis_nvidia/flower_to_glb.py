"""
Minimal Trellis client: local PNG -> GLB -> `trellis_nvidia/output/`.

Usage:
  uv run python -m trellis_nvidia.flower_to_glb --in trellis_nvidia/assets/flower.png

Important:
- The hosted endpoint at `ai.api.nvidia.com` currently rejects arbitrary base64 image inputs and will
  likely return HTTP 422. This script is kept to quickly test/confirm behavior.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


DEFAULT_INVOKE_URL = "https://ai.api.nvidia.com/v1/genai/microsoft/trellis"


def _load_env(project_root: Path) -> None:
    load_dotenv(project_root / ".env", override=False)
    load_dotenv(project_root / ".env.local", override=False)


def _get_api_key() -> str:
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if api_key:
        return api_key
    return getpass.getpass("Enter NVIDIA_API_KEY (input hidden): ").strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a GLB from a local image (Trellis).")
    default_in = str(Path(__file__).resolve().parent / "assets" / "flower.png")
    parser.add_argument("--in", dest="in_path", default=default_in, help="Input PNG path.")
    parser.add_argument("--seed", type=int, default=0, help="0 means random seed.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout seconds.")
    parser.add_argument("--slat-cfg-scale", type=float, default=3.0)
    parser.add_argument("--ss-cfg-scale", type=float, default=7.5)
    parser.add_argument("--slat-sampling-steps", type=int, default=25)
    parser.add_argument("--ss-sampling-steps", type=int, default=25)
    parser.add_argument(
        "--out",
        default="flower.glb",
        help="Output filename (relative to trellis_nvidia/output/).",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    _load_env(project_root)
    api_key = _get_api_key()
    if not api_key:
        print("Error: Missing NVIDIA_API_KEY (set env var or put it in .env.local).")
        return 2

    in_path = Path(args.in_path).expanduser().resolve()
    if not in_path.exists():
        print(f"Error: input image not found: {in_path}")
        return 2

    img_b64 = base64.b64encode(in_path.read_bytes()).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "image": f"data:image/png;base64,{img_b64}",
        "slat_cfg_scale": args.slat_cfg_scale,
        "ss_cfg_scale": args.ss_cfg_scale,
        "slat_sampling_steps": args.slat_sampling_steps,
        "ss_sampling_steps": args.ss_sampling_steps,
        "seed": args.seed,
    }

    t0 = time.perf_counter()
    resp = requests.post(DEFAULT_INVOKE_URL, headers=headers, json=payload, timeout=args.timeout)
    dt = time.perf_counter() - t0
    print(f"HTTP {resp.status_code} in {dt:.2f}s")

    try:
        body = resp.json()
    except ValueError:
        print(resp.text[:4000])
        return 2

    if resp.status_code < 200 or resp.status_code >= 300:
        print(json.dumps(body, indent=2)[:4000])
        return 2

    artifacts = body.get("artifacts") if isinstance(body, dict) else None
    if not isinstance(artifacts, list) or not artifacts or not isinstance(artifacts[0], dict):
        print("Error: response did not contain artifacts[0].")
        print(json.dumps(body, indent=2)[:4000])
        return 2

    artifact = artifacts[0]
    if artifact.get("finishReason") and artifact["finishReason"] != "SUCCESS":
        print(f"Error: finishReason={artifact.get('finishReason')!r}")
        print(json.dumps(body, indent=2)[:4000])
        return 2

    b64 = artifact.get("base64")
    if not isinstance(b64, str) or not b64:
        print("Error: artifacts[0].base64 missing/empty.")
        print(json.dumps(body, indent=2)[:4000])
        return 2

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out
    data = base64.b64decode(b64)
    out_path.write_bytes(data)
    print(f"Wrote {out_path} ({len(data)} bytes) seed_used={artifact.get('seed')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

