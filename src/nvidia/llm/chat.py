"""
Chat with NVIDIA multimodal LLMs.

Usage:
  uv run python -m nvidia.llm.chat --prompt "What is Python?"
  uv run python -m nvidia.llm.chat --prompt "Describe this" --image photo.png
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import requests
from dotenv import load_dotenv

from .models import DEFAULT_MODEL

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def _load_env() -> None:
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env.local", override=False)


def _get_api_key() -> str:
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not set")
    return api_key


def _get_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }


# --- Image helpers ---


def get_clipboard_image() -> Path | None:
    """Get image from clipboard, save to temp file. Returns None if no image."""
    try:
        from PIL import Image, ImageGrab
    except ImportError:
        return None

    try:
        clip = ImageGrab.grabclipboard()
    except Exception:
        return None

    if clip is None:
        return None

    # On macOS, grabclipboard() returns:
    # - PIL Image for screenshots
    # - List of file paths for copied files
    if isinstance(clip, Image.Image):
        path = Path(tempfile.mktemp(suffix=".png"))
        clip.save(path, "PNG")
        return path

    # Handle list of file paths (macOS file copy)
    if isinstance(clip, list) and clip:
        for item in clip:
            p = Path(str(item))
            if p.exists() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                return p

    return None


def download_image(url: str) -> Path:
    """Download URL to temp file."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    ext = ".png" if "png" in content_type else ".jpg"
    path = Path(tempfile.mktemp(suffix=ext))
    path.write_bytes(resp.content)
    return path


def encode_image(path: Path) -> tuple[str, str]:
    """Encode image to base64. Returns (base64_data, mime_type)."""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    ext = path.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(
        ext, "image/png"
    )
    return b64, mime


# --- Content builders ---


def build_content(prompt: str, image_path: Path | None = None) -> str | list:
    """Build message content, handling text-only vs multimodal."""
    if not image_path:
        return prompt

    b64, mime = encode_image(image_path)
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ]


# --- Chat functions ---


def chat(
    prompt: str,
    image_path: Path | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
) -> str:
    """Send a chat message and return the full response."""
    resp = requests.post(
        API_URL,
        headers=_get_headers(),
        json={
            "model": model,
            "messages": [{"role": "user", "content": build_content(prompt, image_path)}],
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def chat_stream(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
) -> Iterator[str]:
    """Stream chat response, yielding text chunks."""
    resp = requests.post(
        API_URL,
        headers=_get_headers(),
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        },
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8")
        if line_str.startswith("data: ") and line_str != "data: [DONE]":
            try:
                chunk = json.loads(line_str[6:])
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
            except (json.JSONDecodeError, KeyError, IndexError):
                continue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with NVIDIA multimodal LLMs")
    parser.add_argument("--prompt", required=True, help="Your message")
    parser.add_argument("--image", type=Path, help="Path to image file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args(argv)

    _load_env()
    response = chat(args.prompt, args.image, args.model, args.max_tokens)
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
