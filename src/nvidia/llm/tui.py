"""
Textual TUI for multimodal LLM chat with NVIDIA APIs.

Usage:
  uv run nvidia-chat
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
from textual.containers import ScrollableContainer
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, OptionList, Static
from textual.widgets.option_list import Option

from .chat import (
    _load_env,
    build_content,
    chat_stream,
    download_image,
    get_clipboard_image,
)
from .models import DEFAULT_MODEL, VISION_MODELS, get_short_name

# Pattern to match image tag at start of input
IMAGE_TAG_PATTERN = re.compile(r"^\[image: [^\]]+\] ")

# Max width for image preview (in terminal columns)
IMAGE_PREVIEW_WIDTH = 60


def render_image_preview(path: Path) -> Pixels | None:
    """Render an image as terminal pixels."""
    try:
        img = Image.open(path)
        # Convert to RGB if necessary (handles RGBA, P mode, etc.)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        # Resize to fit terminal width while maintaining aspect ratio
        # rich-pixels uses half-block chars (▀▄) - each char is 1 wide x 2 pixels tall
        # Terminal chars are roughly 2:1 (height:width), so half-blocks make it ~1:1
        # No adjustment needed - just use natural aspect ratio
        aspect = img.height / img.width
        new_width = min(IMAGE_PREVIEW_WIDTH, img.width)
        new_height = int(new_width * aspect)

        img = img.resize((new_width, max(1, new_height)), Image.Resampling.LANCZOS)
        return Pixels.from_image(img)
    except Exception:
        return None


class ImagePreview(Static):
    """Widget to display an image preview."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self._pixels: Pixels | None = None

    def on_mount(self) -> None:
        self._pixels = render_image_preview(self.path)
        if self._pixels:
            self.update(self._pixels)
        else:
            self.update(f"[dim](Could not preview: {self.path.name})[/]")

    def render(self) -> RenderableType:
        if self._pixels:
            return self._pixels
        return f"[dim](Image: {self.path.name})[/]"


class ChatMessage(Static):
    """A single chat message."""

    DEFAULT_CSS = """
    ChatMessage {
        layout: vertical;
    }
    ChatMessage > .message-header {
        padding: 0 1;
    }
    ChatMessage > ImagePreview {
        padding: 0 1;
        height: auto;
    }
    ChatMessage > .message-content {
        padding: 0 1;
    }
    """

    def __init__(
        self,
        role: str,
        content: str = "",
        image_path: Path | None = None,
        model: str | None = None,
    ):
        super().__init__()
        self.role = role
        self._content = content
        self.image_path = image_path
        self.model = model

    def compose(self) -> ComposeResult:
        # Header line
        yield Static(self._format_header(), classes="message-header")
        # Image preview if present
        if self.image_path and self.image_path.exists():
            yield ImagePreview(self.image_path)
        # Message content
        yield Static(self._content, classes="message-content")

    def _format_header(self) -> str:
        if self.role == "user":
            prefix = "[bold]You:[/]"
        else:
            model_note = f" [dim]({get_short_name(self.model)})[/]" if self.model else ""
            prefix = f"[bold]Assistant{model_note}:[/]"
        image_note = f" [dim][image: {self.image_path.name}][/]" if self.image_path else ""
        return f"{prefix}{image_note}"

    def append_text(self, text: str) -> None:
        """Append text to the message (for streaming)."""
        self._content += text
        try:
            content_widget = self.query_one(".message-content", Static)
            content_widget.update(self._content)
        except Exception:
            pass


class ChatLog(ScrollableContainer):
    """Scrollable container for chat messages."""

    def add_message(
        self,
        role: str,
        content: str = "",
        image_path: Path | None = None,
        model: str | None = None,
    ) -> ChatMessage:
        """Add a new message and return it."""
        msg = ChatMessage(role, content, image_path, model)
        self.mount(msg)
        self.scroll_end(animate=False)
        return msg


class ModelSelectScreen(ModalScreen[str | None]):
    """Modal screen for selecting a model."""

    CSS = """
    ModelSelectScreen {
        align: center middle;
    }
    #model-dialog {
        width: 60;
        height: auto;
        max-height: 80%;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    #model-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    OptionList {
        height: auto;
        max-height: 20;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, current_model: str):
        super().__init__()
        self.current_model = current_model

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="model-dialog"):
            yield Static("Select Model", id="model-title")
            option_list = OptionList()
            for short_name, full_name in VISION_MODELS.items():
                marker = " *" if full_name == self.current_model else ""
                option_list.add_option(Option(f"{short_name}{marker}", id=full_name))
            yield option_list

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ChatApp(App):
    """Multimodal LLM chat TUI."""

    TITLE = "NVIDIA Chat"
    CSS = """
    ChatLog {
        height: 1fr;
        padding: 1;
    }
    ChatMessage {
        margin-bottom: 1;
    }
    #status {
        dock: bottom;
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
    }
    Input {
        dock: bottom;
        margin: 0 1 1 1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+o", "switch_model", "Model"),
        Binding("ctrl+k", "clear_chat", "Clear"),
        Binding("ctrl+u", "paste_image", "Paste Img", priority=True),
    ]

    def __init__(self):
        super().__init__()
        self.model = DEFAULT_MODEL
        self.messages: list[dict] = []
        self.pending_image: Path | None = None
        self.is_streaming = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield ChatLog()
        yield Static(self._status_text(), id="status")
        yield Input(placeholder="Type message... (/image <path> or ^U to paste)")
        yield Footer()

    def _status_text(self) -> str:
        return f"Model: {get_short_name(self.model)}"

    def _update_status(self) -> None:
        self.query_one("#status", Static).update(self._status_text())

    def _get_image_tag(self) -> str:
        """Get the image tag string for the current pending image."""
        if self.pending_image:
            return f"[image: {self.pending_image.name}] "
        return ""

    def _attach_image(self, path: Path) -> None:
        """Attach an image and update the input field."""
        self.pending_image = path
        input_widget = self.query_one(Input)
        current_text = input_widget.value

        # Remove any existing image tag
        current_text = IMAGE_TAG_PATTERN.sub("", current_text)

        # Add new image tag at the start
        new_text = f"[image: {path.name}] {current_text}"
        input_widget.value = new_text
        # Move cursor to end
        input_widget.cursor_position = len(new_text)

        self.notify(f"Attached: {path.name}")

    def _remove_image(self) -> None:
        """Remove the attached image and its tag from input."""
        if self.pending_image:
            self.pending_image = None
            input_widget = self.query_one(Input)
            # Remove the image tag from input
            input_widget.value = IMAGE_TAG_PATTERN.sub("", input_widget.value)
            self.notify("Image removed")

    def on_mount(self) -> None:
        _load_env()
        self._update_status()
        self.query_one(Input).focus()

    def on_key(self, event: Key) -> None:
        """Handle key events to detect when user deletes into image tag."""
        if event.key == "backspace" and self.pending_image:
            input_widget = self.query_one(Input)
            tag = self._get_image_tag()
            # If cursor is at or before the end of the tag, remove the whole image
            if input_widget.cursor_position <= len(tag):
                event.prevent_default()
                self._remove_image()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if self.is_streaming:
            return

        raw_text = event.value.strip()
        if not raw_text:
            return

        # Handle /image command
        if raw_text.startswith("/image "):
            event.input.clear()
            await self._handle_image_command(raw_text[7:].strip())
            return

        # Extract actual message text (remove image tag if present)
        text = IMAGE_TAG_PATTERN.sub("", raw_text).strip()
        if not text:
            self.notify("Please enter a message", severity="warning")
            return

        event.input.clear()

        # Send message with the pending image
        await self._send_message(text)

    async def _handle_image_command(self, path_or_url: str) -> None:
        """Handle /image command to attach an image."""
        try:
            if path_or_url.startswith(("http://", "https://")):
                path = download_image(path_or_url)
                self._attach_image(path)
            else:
                path = Path(path_or_url).expanduser().resolve()
                if not path.exists():
                    self.notify(f"File not found: {path}", severity="error")
                    return
                self._attach_image(path)
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")

    async def _send_message(self, text: str) -> None:
        """Send a message and stream the response."""
        chat_log = self.query_one(ChatLog)

        # Capture the image before clearing
        image_to_send = self.pending_image

        # Add user message to chat log
        chat_log.add_message("user", text, image_to_send)

        # Build message for API
        content = build_content(text, image_to_send)
        self.messages.append({"role": "user", "content": content})

        # Clear pending image
        self.pending_image = None

        # Add assistant message placeholder with model info
        assistant_msg = chat_log.add_message("assistant", "", model=self.model)

        # Stream response
        self._stream_response(assistant_msg)

    @work(thread=True)
    def _stream_response(self, message_widget: ChatMessage) -> None:
        """Stream the response in a background thread."""
        self.is_streaming = True
        full_response = ""

        try:
            for chunk in chat_stream(self.messages, self.model):
                full_response += chunk
                self.call_from_thread(message_widget.append_text, chunk)
                self.call_from_thread(
                    self.query_one(ChatLog).scroll_end, animate=False
                )
        except Exception as e:
            self.call_from_thread(message_widget.append_text, f"\n\n[Error: {e}]")
        finally:
            self.is_streaming = False
            # Save assistant response to history
            if full_response:
                self.messages.append({"role": "assistant", "content": full_response})

    def action_switch_model(self) -> None:
        """Show model selection screen."""

        def on_model_selected(model: str | None) -> None:
            if model:
                self.model = model
                self._update_status()
                self.notify(f"Model: {get_short_name(self.model)}")

        self.push_screen(ModelSelectScreen(self.model), on_model_selected)

    def action_clear_chat(self) -> None:
        """Clear chat history."""
        self.messages.clear()
        chat_log = self.query_one(ChatLog)
        chat_log.remove_children()
        self.pending_image = None
        self.query_one(Input).clear()
        self.notify("Chat history cleared")

    def action_paste_image(self) -> None:
        """Paste image from clipboard."""
        path = get_clipboard_image()
        if path:
            self._attach_image(path)
        else:
            self.notify("No image in clipboard", severity="warning")


def main() -> int:
    app = ChatApp()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
