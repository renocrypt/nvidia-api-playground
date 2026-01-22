"""
Chat with Google Gemma 3 27B via NVIDIA API.

Usage:
  uv run python -m nvidia.llm.chat --prompt "What is Python?"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "google/gemma-3-27b-it"


def _load_env() -> None:
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env.local", override=False)


def chat(prompt: str, max_tokens: int = 1024) -> str:
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not set")

    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with Gemma 3 27B")
    parser.add_argument("--prompt", required=True, help="Your message")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args(argv)

    _load_env()
    response = chat(args.prompt, args.max_tokens)
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
