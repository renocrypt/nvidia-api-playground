from __future__ import annotations

import sys

from trellis_nvidia.flower_to_glb import main as flower_main
from trellis_nvidia.text_to_glb import main as text_main


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "trellis-nvidia\n\n"
            "Commands:\n"
            "  text   Generate a GLB from a text prompt\n"
            "  flower Try image->GLB using trellis_nvidia/assets/flower.png\n\n"
            "Examples:\n"
            "  uv run trellis-nvidia text --prompt \"Toy armored vehicle, olive green\" --out toy.glb\n"
            "  uv run trellis-nvidia flower --in trellis_nvidia/assets/flower.png\n\n"
            "Or run the modules directly:\n"
            "  uv run python -m trellis_nvidia.text_to_glb --prompt \"Toy armored vehicle\"\n"
        )
        return 0

    cmd, *rest = argv
    if cmd == "text":
        return text_main(rest)
    if cmd == "flower":
        return flower_main(rest)

    print(f"Unknown command: {cmd!r}. Use --help.")
    return 2

