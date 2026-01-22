# nvidia-gen

CLI for generative AI via NVIDIA APIs: 3D models, LLMs, and more.

## Agent Guidelines

- Use TodoWrite to plan and track multi-step tasks
- Experiment freely with `.venv/` - run `uv run` or `source .venv/bin/activate`

## Commands

```bash
uv sync                                              # Install deps
uv run nvidia-trellis --prompt "A chair"             # Text-to-3D
uv run python -m nvidia.llm.chat --prompt "Hello"    # LLM chat
uv add <package>                                     # Add dependency
```

## Structure

```
src/nvidia/
├── trellis/
│   ├── text_to_glb.py   # Text-to-3D
│   └── output/          # Generated GLB files
└── llm/
    └── chat.py          # Gemma 3 27B chat
```

## Key Files

- `src/nvidia/trellis/text_to_glb.py` - Trellis text-to-3D
- `src/nvidia/llm/chat.py` - LLM chat
- `pyproject.toml` - deps and config (src layout, `nvidia.*` packages)
- `.env.local` - API keys (not committed)

## APIs

- `NVIDIA_API_KEY` in `.env.local`
- LLM: `https://integrate.api.nvidia.com/v1/chat/completions`
- Trellis: `https://ai.api.nvidia.com/v1/genai/microsoft/trellis`
- Docs: `docs/nvidia-nim-api-reference.md`

## Adding Modules

1. Create `src/nvidia/new_module/` with `__init__.py`
2. Add code files
3. Optionally add CLI entry in `pyproject.toml` under `[project.scripts]`

## Known Issues

- Trellis prompt max 77 chars (auto-truncated)
- Trellis image-to-3D not supported on hosted endpoint
