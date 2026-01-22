# NVIDIA NIM API Guide

This guide documents NVIDIA's NIM (NVIDIA Inference Microservices) API structure for programmatic access.

## Overview

NVIDIA provides **two distinct API base URLs** with different purposes:

| Base URL | Purpose | API Style |
|----------|---------|-----------|
| `https://integrate.api.nvidia.com` | LLMs, VLMs, Embeddings | OpenAI-compatible |
| `https://ai.api.nvidia.com` | Specialized models (3D, Speech, etc.) | Custom per-model |

## Authentication

All requests require an API key in the `Authorization` header:

```
Authorization: Bearer $NVIDIA_API_KEY
```

Get your API key from [build.nvidia.com](https://build.nvidia.com) by selecting any model and clicking "Get API Key".

---

## API 1: OpenAI-Compatible Endpoint

**Base URL:** `https://integrate.api.nvidia.com/v1`

This endpoint follows the OpenAI API specification exactly.

### List Available Models

```bash
curl https://integrate.api.nvidia.com/v1/models
```

Response format:
```json
{
  "object": "list",
  "data": [
    {"id": "nvidia/nemotron-nano-12b-v2-vl", "object": "model", "owned_by": "nvidia"},
    {"id": "meta/llama-3.2-90b-vision-instruct", "object": "model", "owned_by": "meta"}
  ]
}
```

### Chat Completions

**Endpoint:** `POST /v1/chat/completions`

#### Text-only Request

```bash
curl -X POST https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.3-70b-instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 1024,
    "temperature": 0.7
  }'
```

#### Multimodal Request (Image + Text)

```bash
curl -X POST https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/nemotron-nano-12b-v2-vl",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example.com/image.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 1024
  }'
```

#### Image as Base64

```bash
curl -X POST https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/nemotron-nano-12b-v2-vl",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this image"},
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,iVBORw0KGgo..."
            }
          }
        ]
      }
    ],
    "max_tokens": 1024
  }'
```

#### Response Format

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "nvidia/nemotron-nano-12b-v2-vl",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The image shows..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
  }
}
```

### Embeddings

**Endpoint:** `POST /v1/embeddings`

```bash
curl -X POST https://integrate.api.nvidia.com/v1/embeddings \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/nv-embed-v1",
    "input": ["Hello world", "How are you?"],
    "encoding_format": "float"
  }'
```

---

## API 2: Specialized Models Endpoint

**Base URL:** `https://ai.api.nvidia.com/v1`

This endpoint hosts specialized models with custom request/response formats.

### Endpoint Pattern

```
POST https://ai.api.nvidia.com/v1/genai/{provider}/{model-name}
```

### 3D Generation (Trellis)

**Endpoint:** `POST /v1/genai/microsoft/trellis`

#### Text-to-3D

```bash
curl -X POST https://ai.api.nvidia.com/v1/genai/microsoft/trellis \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "prompt": "A red sports car",
    "seed": 42,
    "output_format": "glb"
  }'
```

#### Image-to-3D

```bash
curl -X POST https://ai.api.nvidia.com/v1/genai/microsoft/trellis \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -H "Accept": "application/json" \
  -d '{
    "image": "data:image/png;base64,iVBORw0KGgo...",
    "seed": 42,
    "output_format": "glb"
  }'
```

#### Trellis Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | string | Text description (max 77 chars) |
| `image` | string | Base64 image for image-to-3D |
| `seed` | int | Random seed for reproducibility |
| `output_format` | string | Output format: `glb` |
| `sparse_structure_sample_steps` | int | Sampling steps (default: 12) |
| `slat_sample_steps` | int | SLAT sampling steps (default: 12) |
| `sparse_structure_cfg_scale` | float | CFG scale (default: 7.5) |
| `slat_cfg_scale` | float | SLAT CFG scale (default: 3.0) |

#### Response (3D Model)

The response contains base64-encoded GLB data:

```json
{
  "output": {
    "glb": "Z2xURgIAAAA..."
  }
}
```

Decode and save:
```bash
echo "$GLB_BASE64" | base64 -d > model.glb
```

---

## Python Examples

### Using requests (OpenAI-compatible)

```python
import requests
import os

API_KEY = os.environ["NVIDIA_API_KEY"]
BASE_URL = "https://integrate.api.nvidia.com/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Chat completion
response = requests.post(
    f"{BASE_URL}/chat/completions",
    headers=headers,
    json={
        "model": "meta/llama-3.3-70b-instruct",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 100
    }
)
print(response.json()["choices"][0]["message"]["content"])
```

### Using OpenAI SDK

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

# Text completion
response = client.chat.completions.create(
    model="meta/llama-3.3-70b-instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
print(response.choices[0].message.content)

# Vision model
response = client.chat.completions.create(
    model="nvidia/nemotron-nano-12b-v2-vl",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]
        }
    ],
    max_tokens=1024
)
print(response.choices[0].message.content)
```

### Specialized API (Trellis)

```python
import requests
import base64
import os

API_KEY = os.environ["NVIDIA_API_KEY"]

def text_to_3d(prompt: str, output_path: str, seed: int = 42):
    response = requests.post(
        "https://ai.api.nvidia.com/v1/genai/microsoft/trellis",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        },
        json={
            "prompt": prompt,
            "seed": seed,
            "output_format": "glb"
        }
    )
    response.raise_for_status()

    glb_data = base64.b64decode(response.json()["output"]["glb"])
    with open(output_path, "wb") as f:
        f.write(glb_data)

    return output_path

# Usage
text_to_3d("A wooden chair", "chair.glb")
```

---

## Available Models Reference

### Vision/Multimodal Models (17)

| Model ID | Provider | Best For |
|----------|----------|----------|
| `nvidia/nemotron-nano-12b-v2-vl` | NVIDIA | OCR, multi-image, video |
| `nvidia/cosmos-reason2-8b` | NVIDIA | Physical world reasoning |
| `meta/llama-3.2-90b-vision-instruct` | Meta | High-quality vision |
| `meta/llama-3.2-11b-vision-instruct` | Meta | Efficient vision |
| `microsoft/phi-4-multimodal-instruct` | Microsoft | Latest multimodal |
| `microsoft/phi-3.5-vision-instruct` | Microsoft | 128k context vision |
| `nvidia/vila` | NVIDIA | General VLM |
| `google/paligemma` | Google | Image understanding |
| `nvidia/nvclip` | NVIDIA | Image embeddings |
| `google/deplot` | Google | Chart understanding |
| `microsoft/kosmos-2` | Microsoft | Grounding + referring |
| `nvidia/streampetr` | NVIDIA | 3D object detection |
| `adept/fuyu-8b` | Adept | Multimodal |
| `nvidia/neva-22b` | NVIDIA | Vision-language |
| `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` | NVIDIA | Efficient VLM |
| `nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1` | NVIDIA | VLM embeddings |

### Top LLMs

| Model ID | Provider | Parameters |
|----------|----------|------------|
| `meta/llama-3.1-405b-instruct` | Meta | 405B |
| `deepseek-ai/deepseek-v3.2` | DeepSeek | 685B MoE |
| `mistralai/mistral-large-3-675b-instruct-2512` | Mistral | 675B MoE |
| `nvidia/llama-3.1-nemotron-70b-instruct` | NVIDIA | 70B |
| `qwen/qwen3-235b-a22b` | Qwen | 235B |

### Embedding Models

| Model ID | Provider | Use Case |
|----------|----------|----------|
| `nvidia/nv-embed-v1` | NVIDIA | General embeddings |
| `nvidia/embed-qa-4` | NVIDIA | QA embeddings |
| `baai/bge-m3` | BAAI | Multilingual |
| `snowflake/arctic-embed-l` | Snowflake | Large embeddings |

### Specialized Models (ai.api.nvidia.com)

| Model | Endpoint | Purpose |
|-------|----------|---------|
| Trellis | `/v1/genai/microsoft/trellis` | Text/Image to 3D |

---

## Error Handling

### Common HTTP Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Process response |
| 401 | Unauthorized | Check API key |
| 404 | Not found | Check endpoint/model name |
| 405 | Method not allowed | Use correct HTTP method |
| 422 | Validation error | Check request body |
| 429 | Rate limited | Implement backoff |
| 500 | Server error | Retry with backoff |

### Error Response Format

```json
{
  "error": {
    "message": "Invalid API key",
    "type": "authentication_error",
    "code": "invalid_api_key"
  }
}
```

---

## Rate Limits & Credits

- Free tier: 1000 credits on signup, up to 5000 total
- Credits consumed per request vary by model
- Check [build.nvidia.com](https://build.nvidia.com) for current limits

---

## Quick Reference

```bash
# List all models
curl https://integrate.api.nvidia.com/v1/models

# Chat with LLM
curl -X POST https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "meta/llama-3.3-70b-instruct", "messages": [{"role": "user", "content": "Hi"}]}'

# Vision model
curl -X POST https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "nvidia/nemotron-nano-12b-v2-vl", "messages": [{"role": "user", "content": [{"type": "text", "text": "Describe"}, {"type": "image_url", "image_url": {"url": "https://..."}}]}]}'

# Embeddings
curl -X POST https://integrate.api.nvidia.com/v1/embeddings \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "nvidia/nv-embed-v1", "input": ["text"]}'

# 3D generation
curl -X POST https://ai.api.nvidia.com/v1/genai/microsoft/trellis \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A chair", "output_format": "glb"}'
```
