"""Model registry for NVIDIA multimodal LLMs."""

DEFAULT_MODEL = "nvidia/nemotron-nano-12b-v2-vl"

VISION_MODELS = {
    "nemotron-vl": "nvidia/nemotron-nano-12b-v2-vl",
    "llama-90b-vision": "meta/llama-3.2-90b-vision-instruct",
    "llama-11b-vision": "meta/llama-3.2-11b-vision-instruct",
    "phi-4-mm": "microsoft/phi-4-multimodal-instruct",
    "nemotron-8b-vl": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    "cosmos-reason": "nvidia/cosmos-reason2-8b",
}

# Short names for display
MODEL_SHORT_NAMES = {v: k for k, v in VISION_MODELS.items()}


def get_short_name(model: str) -> str:
    """Get short display name for a model."""
    return MODEL_SHORT_NAMES.get(model, model.split("/")[-1])
