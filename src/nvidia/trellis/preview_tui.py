"""
Simple TUI to test GLB rendering.

Usage:
  uv run python -m nvidia.trellis.preview_tui
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from PIL import Image
from rich.console import RenderableType
from rich_pixels import Pixels
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Button, Footer, Header, Static

from .preview import render_glb_to_image

# Default GLB file to render
DEFAULT_GLB = Path(__file__).parent / "output" / "Cylindrical_white_ceramic_base.glb"

IMAGE_PREVIEW_WIDTH = 60


def render_image_preview(path: Path, max_width: int = IMAGE_PREVIEW_WIDTH) -> Pixels | None:
    """Render an image as terminal pixels."""
    try:
        img = Image.open(path)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        aspect = img.height / img.width
        new_width = min(max_width, img.width)
        new_height = int(new_width * aspect)

        img = img.resize((new_width, max(1, new_height)), Image.Resampling.LANCZOS)
        return Pixels.from_image(img)
    except Exception as e:
        print(f"Preview error: {e}")
        return None


class ImagePreview(Static):
    """Widget to display an image preview."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pixels: Pixels | None = None

    def set_image(self, path: Path | None) -> None:
        """Set or clear the image."""
        if path and path.exists():
            self._pixels = render_image_preview(path)
            if self._pixels:
                self.update(self._pixels)
            else:
                self.update(f"[red]Could not preview: {path.name}[/]")
        else:
            self._pixels = None
            self.update("[dim]No image[/]")

    def render(self) -> RenderableType:
        if self._pixels:
            return self._pixels
        return "[dim]No image loaded[/]"


class PreviewApp(App):
    """Simple GLB preview TUI."""

    TITLE = "GLB Preview Test"
    CSS = """
    #main {
        padding: 1;
    }

    #preview {
        height: auto;
        min-height: 10;
        border: round green;
        padding: 1;
    }

    #status {
        height: auto;
        margin-top: 1;
    }

    #btn-render {
        margin-top: 1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("r", "render", "Render"),
    ]

    def __init__(self, glb_path: Path | None = None):
        super().__init__()
        self.glb_path = glb_path or DEFAULT_GLB

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="main"):
            yield Static(f"[bold]GLB File:[/] {self.glb_path}")
            yield ImagePreview(id="preview")
            yield Static("Press 'r' or click button to render", id="status")
            yield Button("Render GLB to Image", id="btn-render", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        if not self.glb_path.exists():
            self.query_one("#status", Static).update(
                f"[red]File not found: {self.glb_path}[/]"
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-render":
            self.action_render()

    def action_render(self) -> None:
        """Render the GLB file."""
        if not self.glb_path.exists():
            self.notify("GLB file not found", severity="error")
            return

        self.query_one("#status", Static).update("[yellow]Rendering...[/]")
        self.query_one("#preview", ImagePreview).update("[yellow]Rendering GLB...[/]")

        # Run rendering (this calls subprocess internally)
        self.run_worker(self._do_render, thread=True)

    async def _do_render(self) -> None:
        """Worker to render GLB."""
        try:
            png_path = render_glb_to_image(self.glb_path)

            if png_path and png_path.exists():
                self.call_from_thread(
                    self.query_one("#preview", ImagePreview).set_image, png_path
                )
                self.call_from_thread(
                    self.query_one("#status", Static).update,
                    f"[green]Rendered successfully![/] ({png_path})",
                )
                self.call_from_thread(self.notify, "Render complete!")
            else:
                self.call_from_thread(
                    self.query_one("#status", Static).update,
                    "[red]Rendering failed[/]",
                )
                self.call_from_thread(
                    self.query_one("#preview", ImagePreview).update,
                    "[red]Rendering failed - check console for errors[/]",
                )
        except Exception as e:
            self.call_from_thread(
                self.query_one("#status", Static).update,
                f"[red]Error: {e}[/]",
            )


def main() -> int:
    import sys

    glb_path = None
    if len(sys.argv) > 1:
        glb_path = Path(sys.argv[1])

    app = PreviewApp(glb_path)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
