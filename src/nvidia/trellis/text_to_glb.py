"""
Trellis text-to-3D: text -> GLB.

Usage:
  uv run nvidia-trellis --prompt "Toy armored vehicle, olive green"
  uv run python -m nvidia.trellis.text_to_glb --prompt "A chair"

Notes:
- Hosted endpoint expects prompt length <= 77 characters.
- Reads NVIDIA_API_KEY from environment or `.env.local` in project root.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv


DEFAULT_INVOKE_URL = "https://ai.api.nvidia.com/v1/genai/microsoft/trellis"
MAX_PROMPT_LENGTH = 77


def _load_env(project_root: Path | None = None) -> None:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env", override=False)
    load_dotenv(project_root / ".env.local", override=False)


def _get_api_key() -> str:
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if api_key:
        return api_key
    return getpass.getpass("Enter NVIDIA_API_KEY (input hidden): ").strip()


@dataclass
class TrellisResult:
    """Result from Trellis 3D generation."""

    glb_data: bytes
    seed_used: int


class TrellisError(Exception):
    """Error from Trellis API."""

    pass


def generate_glb(
    prompt: str,
    seed: int = 0,
    slat_cfg_scale: float = 3.0,
    ss_cfg_scale: float = 7.5,
    slat_sampling_steps: int = 25,
    ss_sampling_steps: int = 25,
    timeout: float = 300.0,
    truncate: bool = True,
) -> TrellisResult:
    """
    Generate a GLB 3D model from a text prompt using Trellis API.

    Args:
        prompt: Text description of the 3D object (max 77 chars)
        seed: Random seed (0 for random)
        slat_cfg_scale: Classifier-free guidance scale for structured latent diffusion
        ss_cfg_scale: Classifier-free guidance scale for sparse structure diffusion
        slat_sampling_steps: Number of sampling steps for structured latent
        ss_sampling_steps: Number of sampling steps for sparse structure
        timeout: Request timeout in seconds
        truncate: If True, truncate prompt to 77 chars; if False, raise error

    Returns:
        TrellisResult with glb_data bytes and seed_used

    Raises:
        TrellisError: If API call fails or returns invalid response
        ValueError: If prompt is empty or too long (when truncate=False)
    """
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt cannot be empty")

    if len(prompt) > MAX_PROMPT_LENGTH:
        if truncate:
            prompt = prompt[:MAX_PROMPT_LENGTH]
        else:
            raise ValueError(f"prompt too long ({len(prompt)} chars). Must be <= {MAX_PROMPT_LENGTH}.")

    api_key = _get_api_key()
    if not api_key:
        raise TrellisError("Missing NVIDIA_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "slat_cfg_scale": slat_cfg_scale,
        "ss_cfg_scale": ss_cfg_scale,
        "slat_sampling_steps": slat_sampling_steps,
        "ss_sampling_steps": ss_sampling_steps,
        "seed": seed,
    }

    resp = requests.post(DEFAULT_INVOKE_URL, headers=headers, json=payload, timeout=timeout)

    try:
        body = resp.json()
    except ValueError:
        raise TrellisError(f"Invalid JSON response: {resp.text[:1000]}")

    if resp.status_code < 200 or resp.status_code >= 300:
        raise TrellisError(f"HTTP {resp.status_code}: {json.dumps(body, indent=2)[:1000]}")

    artifacts = body.get("artifacts") if isinstance(body, dict) else None
    if not isinstance(artifacts, list) or not artifacts or not isinstance(artifacts[0], dict):
        raise TrellisError(f"Response did not contain artifacts[0]: {json.dumps(body, indent=2)[:1000]}")

    artifact = artifacts[0]
    if artifact.get("finishReason") and artifact["finishReason"] != "SUCCESS":
        raise TrellisError(f"finishReason={artifact.get('finishReason')!r}")

    b64 = artifact.get("base64")
    if not isinstance(b64, str) or not b64:
        raise TrellisError("artifacts[0].base64 missing/empty")

    glb_data = base64.b64decode(b64)
    seed_used = artifact.get("seed", seed)

    return TrellisResult(glb_data=glb_data, seed_used=seed_used)


def get_output_dir() -> Path:
    """Get the default output directory for GLB files."""
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a GLB from a text prompt (Trellis).")
    parser.add_argument(
        "--prompt",
        default="Ergonomic lab chair, metal frame, cushioned seat",
        help="Text prompt (<= 77 chars).",
    )
    parser.add_argument(
        "--truncate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Truncate prompt to 77 chars (prevents server-side errors).",
    )
    parser.add_argument("--seed", type=int, default=0, help="0 means random seed.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Request timeout seconds.")
    parser.add_argument(
        "--slat-cfg-scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale for structured latent diffusion.",
    )
    parser.add_argument(
        "--ss-cfg-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale for sparse structure diffusion.",
    )
    parser.add_argument("--slat-sampling-steps", type=int, default=25)
    parser.add_argument("--ss-sampling-steps", type=int, default=25)
    parser.add_argument(
        "--out",
        default="text.glb",
        help="Output filename (relative to trellis_nvidia/output/).",
    )
    args = parser.parse_args(argv)

    _load_env()

    prompt = args.prompt.strip()
    if len(prompt) > MAX_PROMPT_LENGTH:
        if args.truncate:
            print(f"Warning: truncating prompt from {len(prompt)} to {MAX_PROMPT_LENGTH} characters.")
        else:
            print(f"Error: prompt too long ({len(prompt)} chars). Must be <= {MAX_PROMPT_LENGTH}.")
            return 2

    t0 = time.perf_counter()
    try:
        result = generate_glb(
            prompt=prompt,
            seed=args.seed,
            slat_cfg_scale=args.slat_cfg_scale,
            ss_cfg_scale=args.ss_cfg_scale,
            slat_sampling_steps=args.slat_sampling_steps,
            ss_sampling_steps=args.ss_sampling_steps,
            timeout=args.timeout,
            truncate=args.truncate,
        )
    except (ValueError, TrellisError) as e:
        print(f"Error: {e}")
        return 2

    dt = time.perf_counter() - t0
    print(f"Generated in {dt:.2f}s")

    out_path = get_output_dir() / args.out
    out_path.write_bytes(result.glb_data)
    print(f"Wrote {out_path} ({len(result.glb_data)} bytes) seed_used={result.seed_used}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
