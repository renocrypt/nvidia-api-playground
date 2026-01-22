"""
Textual TUI for Trellis 3D generation from images.

Workflow:
1. User attaches image (Ctrl+U paste or /image <path>)
2. VLM generates description (≤77 chars)
3. User can edit the description before generation
4. Trellis API generates GLB
5. GLB rendered to 2D image and displayed in terminal
6. GLB saved to output directory

Usage:
  uv run nvidia-trellis-tui
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from PIL import Image
from rich.console import RenderableType
from rich_pixels import Pixels
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import Button, Footer, Header, Input, Static

from ..llm.chat import (
    _load_env,
    build_content,
    chat_stream,
    get_clipboard_image,
)
from .preview import get_glb_info, open_in_viewer, render_glb_to_image
from .text_to_glb import MAX_PROMPT_LENGTH, TrellisError, generate_glb, get_output_dir

# VLM model for image description
VLM_MODEL = "nvidia/nemotron-nano-12b-v2-vl"

# Prompt for VLM to generate concise descriptions
DESCRIPTION_PROMPT = """Describe this object for 3D model generation in under 70 characters.
Focus on: shape, material, color. Be specific and concise.
Example: "Wooden desk lamp with brass adjustable arm and white shade"
Just output the description, nothing else."""

# Max width for image preview (in terminal columns)
IMAGE_PREVIEW_WIDTH = 50


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
    except Exception:
        return None


class ImagePreview(Static):
    """Widget to display an image preview."""

    def __init__(self, path: Path | None = None, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self._pixels: Pixels | None = None

    def set_image(self, path: Path | None) -> None:
        """Set or clear the image."""
        self.path = path
        if path and path.exists():
            self._pixels = render_image_preview(path)
            if self._pixels:
                self.update(self._pixels)
            else:
                self.update(f"[dim](Could not preview: {path.name})[/]")
        else:
            self._pixels = None
            self.update("[dim]No image attached[/]")

    def render(self) -> RenderableType:
        if self._pixels:
            return self._pixels
        if self.path:
            return f"[dim](Image: {self.path.name})[/]"
        return "[dim]No image attached[/]"


class DescriptionInput(Input):
    """Input with live character counter in border title."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._update_counter()

    def _update_counter(self) -> None:
        count = len(self.value)
        if count <= MAX_PROMPT_LENGTH:
            self.border_title = f"Description ({count}/{MAX_PROMPT_LENGTH})"
            self.styles.border = ("round", "green" if count > 0 else "gray")
        else:
            self.border_title = f"Description ({count}/{MAX_PROMPT_LENGTH}) [red]TOO LONG[/]"
            self.styles.border = ("round", "red")

    def on_input_changed(self, event: Input.Changed) -> None:
        self._update_counter()


class StatusPanel(Static):
    """Status panel showing current operation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._status = "Ready"

    def set_status(self, status: str) -> None:
        self._status = status
        self.update(f"[bold]Status:[/] {status}")

    def render(self) -> RenderableType:
        return f"[bold]Status:[/] {self._status}"


class TrellisApp(App):
    """Trellis 3D Generator TUI."""

    TITLE = "Trellis 3D Generator"
    CSS = """
    #main-container {
        height: 1fr;
        padding: 1;
    }

    #input-section {
        height: auto;
        margin-bottom: 1;
    }

    #image-preview {
        height: auto;
        min-height: 5;
        border: round gray;
        padding: 1;
    }

    #description-input {
        margin-top: 1;
        border: round gray;
    }

    #button-row {
        height: 3;
        margin-top: 1;
    }

    #button-row Button {
        margin-right: 1;
    }

    #output-section {
        height: auto;
        margin-top: 1;
        border: round gray;
        padding: 1;
    }

    #glb-preview {
        height: auto;
        min-height: 5;
    }

    #glb-info {
        height: auto;
        margin-top: 1;
    }

    #status-panel {
        dock: bottom;
        height: 1;
        padding: 0 1;
        background: $surface;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+u", "paste_image", "Paste Img", priority=True),
        Binding("ctrl+g", "generate_3d", "Generate 3D"),
        Binding("ctrl+r", "regenerate_desc", "Regen Desc"),
        Binding("ctrl+o", "open_viewer", "Open Viewer"),
    ]

    def __init__(self):
        super().__init__()
        self.source_image: Path | None = None
        self.generated_glb: Path | None = None
        self.is_generating = False

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer(id="main-container"):
            with Vertical(id="input-section"):
                yield Static("[bold]Source Image[/]")
                yield ImagePreview(id="image-preview")
                yield DescriptionInput(
                    placeholder="Description will appear here after image analysis...",
                    id="description-input",
                )
                with Horizontal(id="button-row"):
                    yield Button("Generate 3D (^G)", id="btn-generate", variant="primary")
                    yield Button("Regen Description (^R)", id="btn-regen")
                    yield Button("Open in Viewer (^O)", id="btn-open")

            with Vertical(id="output-section"):
                yield Static("[bold]Generated 3D Model[/]", id="output-title")
                yield ImagePreview(id="glb-preview")
                yield Static(
                    "[dim]Workflow: ^U paste image → edit description → ^G generate 3D[/]",
                    id="glb-info",
                )

        yield StatusPanel(id="status-panel")
        yield Footer()

    def on_mount(self) -> None:
        _load_env()
        self._update_status("Ready - press ^U to paste an image from clipboard")
        self.query_one("#description-input", DescriptionInput).focus()

    def _update_status(self, status: str) -> None:
        self.query_one("#status-panel", StatusPanel).set_status(status)

    def _attach_image(self, path: Path) -> None:
        """Attach an image and trigger description generation."""
        self.source_image = path
        self.query_one("#image-preview", ImagePreview).set_image(path)
        self.notify(f"Attached: {path.name}")
        self._update_status(f"Image attached: {path.name}")

        # Auto-generate description
        self._generate_description()

    def action_paste_image(self) -> None:
        """Paste image from clipboard."""
        path = get_clipboard_image()
        if path:
            self._attach_image(path)
        else:
            self.notify("No image in clipboard", severity="warning")

    def action_regenerate_desc(self) -> None:
        """Regenerate description from current image."""
        if not self.source_image:
            self.notify("No image attached", severity="warning")
            return
        self._generate_description()

    def action_generate_3d(self) -> None:
        """Generate 3D model from description."""
        if self.is_generating:
            self.notify("Generation in progress...", severity="warning")
            return

        desc_input = self.query_one("#description-input", DescriptionInput)
        description = desc_input.value.strip()

        if not description:
            self.notify("No description - attach an image first", severity="warning")
            return

        if len(description) > MAX_PROMPT_LENGTH:
            self.notify(
                f"Description too long ({len(description)} chars). Max {MAX_PROMPT_LENGTH}.",
                severity="error",
            )
            return

        self._generate_3d_model(description)

    def action_open_viewer(self) -> None:
        """Open generated GLB in system viewer."""
        if not self.generated_glb or not self.generated_glb.exists():
            self.notify("No GLB file generated yet", severity="warning")
            return

        if open_in_viewer(self.generated_glb):
            self.notify(f"Opened: {self.generated_glb.name}")
        else:
            self.notify("Failed to open viewer", severity="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-generate":
            self.action_generate_3d()
        elif event.button.id == "btn-regen":
            self.action_regenerate_desc()
        elif event.button.id == "btn-open":
            self.action_open_viewer()

    @work(thread=True)
    def _generate_description(self) -> None:
        """Generate description from image using VLM."""
        if not self.source_image:
            return

        self.call_from_thread(self._update_status, "Generating description...")
        desc_input = self.query_one("#description-input", DescriptionInput)
        self.call_from_thread(setattr, desc_input, "value", "")

        try:
            content = build_content(DESCRIPTION_PROMPT, self.source_image)
            messages = [{"role": "user", "content": content}]

            full_response = ""
            for chunk in chat_stream(messages, model=VLM_MODEL, max_tokens=30):
                full_response += chunk
                # Clean up the response (remove quotes, newlines)
                cleaned = full_response.strip().strip('"').strip("'")
                # Truncate at sentence boundary if over limit
                if len(cleaned) > MAX_PROMPT_LENGTH:
                    # Try to find a good break point
                    for sep in [". ", ", ", " "]:
                        idx = cleaned[:MAX_PROMPT_LENGTH].rfind(sep)
                        if idx > 20:
                            cleaned = cleaned[: idx + (1 if sep == ". " else 0)].strip()
                            break
                    else:
                        cleaned = cleaned[:MAX_PROMPT_LENGTH]

                self.call_from_thread(setattr, desc_input, "value", cleaned)

            # Final cleanup
            final = full_response.strip().strip('"').strip("'")
            if len(final) > MAX_PROMPT_LENGTH:
                for sep in [". ", ", ", " "]:
                    idx = final[:MAX_PROMPT_LENGTH].rfind(sep)
                    if idx > 20:
                        final = final[: idx + (1 if sep == ". " else 0)].strip()
                        break
                else:
                    final = final[:MAX_PROMPT_LENGTH]

            self.call_from_thread(setattr, desc_input, "value", final)
            self.call_from_thread(
                self._update_status, f"Description ready ({len(final)} chars) - edit if needed"
            )

        except Exception as e:
            self.call_from_thread(self.notify, f"Error: {e}", severity="error")
            self.call_from_thread(self._update_status, f"Error: {e}")

    @work(thread=True)
    def _generate_3d_model(self, description: str) -> None:
        """Generate 3D model from description using Trellis."""
        self.is_generating = True
        self.call_from_thread(self._update_status, "Generating 3D model (this may take a while)...")
        self.call_from_thread(
            self.query_one("#glb-preview", ImagePreview).update,
            "[yellow]Generating 3D model...[/]\n\n[dim]This typically takes 30-300 seconds.[/]",
        )
        self.call_from_thread(
            self.query_one("#glb-info", Static).update,
            "[dim]Waiting for Trellis API response...[/]",
        )

        try:
            result = generate_glb(prompt=description, timeout=300.0)

            # Save GLB file
            output_dir = get_output_dir()
            # Generate filename from description
            safe_name = re.sub(r"[^\w\s-]", "", description[:30]).strip().replace(" ", "_")
            if not safe_name:
                safe_name = "model"
            output_path = output_dir / f"{safe_name}.glb"

            # Handle existing files
            counter = 1
            while output_path.exists():
                output_path = output_dir / f"{safe_name}_{counter}.glb"
                counter += 1

            output_path.write_bytes(result.glb_data)
            self.generated_glb = output_path

            # Update GLB info
            info = get_glb_info(output_path)
            info_text = f"Saved: {output_path}\n"
            if info:
                info_text += f"Size: {info.get('file_size', 0) / 1024:.1f} KB"
                if "vertices" in info:
                    info_text += f" | Vertices: {info['vertices']:,}"
                if "faces" in info:
                    info_text += f" | Faces: {info['faces']:,}"
                info_text += f" | Seed: {result.seed_used}"

            self.call_from_thread(
                self.query_one("#glb-info", Static).update, info_text
            )

            # Render GLB preview using pyvista
            preview_path = render_glb_to_image(output_path)
            if preview_path:
                self.call_from_thread(
                    self.query_one("#glb-preview", ImagePreview).set_image, preview_path
                )
            else:
                self.call_from_thread(
                    self.query_one("#glb-preview", ImagePreview).update,
                    "[green]3D model generated successfully![/]\n\n[dim]Press ^O to open in 3D viewer[/]",
                )

            self.call_from_thread(self._update_status, f"Done! Saved to {output_path.name}")
            self.call_from_thread(self.notify, f"Generated: {output_path.name}")

        except TrellisError as e:
            self.call_from_thread(self.notify, f"Trellis error: {e}", severity="error")
            self.call_from_thread(self._update_status, f"Error: {e}")
        except Exception as e:
            self.call_from_thread(self.notify, f"Error: {e}", severity="error")
            self.call_from_thread(self._update_status, f"Error: {e}")
        finally:
            self.is_generating = False


def main() -> int:
    app = TrellisApp()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
