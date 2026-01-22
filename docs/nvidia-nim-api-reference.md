# NVIDIA NIM API Reference

## Decision Tree

```
TASK: What do you need?
├── Text generation / Chat → integrate.api.nvidia.com/v1/chat/completions
├── Image understanding → integrate.api.nvidia.com/v1/chat/completions (use VLM)
├── Video understanding → integrate.api.nvidia.com/v1/chat/completions (use VLM)
├── Text embeddings → integrate.api.nvidia.com/v1/embeddings
├── 3D model generation → ai.api.nvidia.com/v1/genai/microsoft/trellis
└── List available models → integrate.api.nvidia.com/v1/models
```

## Authentication

**Required header for ALL requests:**
```
Authorization: Bearer $NVIDIA_API_KEY
```

---

## Endpoint 1: OpenAI-Compatible

**Base:** `https://integrate.api.nvidia.com/v1`

### GET /models

```bash
curl https://integrate.api.nvidia.com/v1/models
```

### POST /chat/completions

**Text only:**
```json
{
  "model": "meta/llama-3.3-70b-instruct",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 1024
}
```

**With image (VLM required):**
```json
{
  "model": "nvidia/nemotron-nano-12b-v2-vl",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image"},
      {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
    ]
  }],
  "max_tokens": 1024
}
```

**With base64 image:**
```json
{
  "model": "nvidia/nemotron-nano-12b-v2-vl",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}}
    ]
  }],
  "max_tokens": 1024
}
```

**Streaming:**
```json
{
  "model": "meta/llama-3.3-70b-instruct",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true
}
```

### POST /embeddings

```json
{
  "model": "nvidia/nv-embed-v1",
  "input": ["text to embed"],
  "encoding_format": "float"
}
```

---

## Endpoint 2: Specialized (3D Generation)

**Base:** `https://ai.api.nvidia.com/v1`

### POST /genai/microsoft/trellis

**Text-to-3D:**
```json
{
  "prompt": "A wooden chair",
  "seed": 42,
  "output_format": "glb"
}
```

**Image-to-3D:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgo...",
  "seed": 42,
  "output_format": "glb"
}
```

**Response:** `{"output": {"glb": "<base64-encoded-glb>"}}`

**Decode:** `echo "$GLB_BASE64" | base64 -d > model.glb`

---

## Model Selection

### For text generation:
| Model | Use when |
|-------|----------|
| `meta/llama-3.3-70b-instruct` | General purpose, high quality |
| `meta/llama-3.1-8b-instruct` | Fast, cost-effective |
| `deepseek-ai/deepseek-r1` | Complex reasoning |
| `mistralai/codestral-22b-instruct-v0.1` | Code generation |

### For image understanding:
| Model | Use when |
|-------|----------|
| `nvidia/nemotron-nano-12b-v2-vl` | OCR, documents, multi-image |
| `meta/llama-3.2-90b-vision-instruct` | Highest quality |
| `meta/llama-3.2-11b-vision-instruct` | Balance of speed/quality |
| `microsoft/phi-4-multimodal-instruct` | Efficient multimodal |

### For embeddings:
| Model | Use when |
|-------|----------|
| `nvidia/nv-embed-v1` | General text embeddings |
| `baai/bge-m3` | Multilingual |

### For 3D:
| Model | Use when |
|-------|----------|
| `microsoft/trellis` | Text or image to GLB |

---

## Complete curl Examples

### Chat
```bash
curl -X POST https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"meta/llama-3.3-70b-instruct","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'
```

### Vision
```bash
curl -X POST https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/nemotron-nano-12b-v2-vl","messages":[{"role":"user","content":[{"type":"text","text":"Describe"},{"type":"image_url","image_url":{"url":"https://example.com/img.jpg"}}]}],"max_tokens":1024}'
```

### Embeddings
```bash
curl -X POST https://integrate.api.nvidia.com/v1/embeddings \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/nv-embed-v1","input":["hello world"]}'
```

### 3D Generation
```bash
curl -X POST https://ai.api.nvidia.com/v1/genai/microsoft/trellis \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A red car","seed":42,"output_format":"glb"}'
```

---

## Python (requests)

```python
import requests, base64, os

KEY = os.environ["NVIDIA_API_KEY"]
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

# Chat
def chat(prompt: str, model: str = "meta/llama-3.3-70b-instruct") -> str:
    r = requests.post("https://integrate.api.nvidia.com/v1/chat/completions",
        headers=HEADERS, json={"model": model, "messages": [{"role": "user", "content": prompt}]})
    return r.json()["choices"][0]["message"]["content"]

# Vision
def vision(prompt: str, image_url: str, model: str = "nvidia/nemotron-nano-12b-v2-vl") -> str:
    r = requests.post("https://integrate.api.nvidia.com/v1/chat/completions",
        headers=HEADERS, json={"model": model, "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]}], "max_tokens": 1024})
    return r.json()["choices"][0]["message"]["content"]

# Embeddings
def embed(texts: list[str], model: str = "nvidia/nv-embed-v1") -> list:
    r = requests.post("https://integrate.api.nvidia.com/v1/embeddings",
        headers=HEADERS, json={"model": model, "input": texts})
    return [d["embedding"] for d in r.json()["data"]]

# 3D Generation
def text_to_3d(prompt: str, output_path: str, seed: int = 42) -> str:
    r = requests.post("https://ai.api.nvidia.com/v1/genai/microsoft/trellis",
        headers=HEADERS, json={"prompt": prompt, "seed": seed, "output_format": "glb"})
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(r.json()["output"]["glb"]))
    return output_path
```

---

## Response Parsing

### Chat/Vision response:
```python
response["choices"][0]["message"]["content"]  # The text response
response["usage"]["total_tokens"]              # Tokens used
```

### Embeddings response:
```python
response["data"][0]["embedding"]  # Vector for first input
```

### 3D response:
```python
base64.b64decode(response["output"]["glb"])  # Binary GLB data
```

---

## Error Codes

| Code | Meaning | Fix |
|------|---------|-----|
| 401 | Bad API key | Check NVIDIA_API_KEY |
| 404 | Wrong endpoint/model | Verify URL and model ID |
| 422 | Invalid request body | Check JSON schema |
| 429 | Rate limited | Wait and retry |

---

## All Vision Models

```
nvidia/nemotron-nano-12b-v2-vl
nvidia/cosmos-reason2-8b
nvidia/vila
nvidia/neva-22b
nvidia/llama-3.1-nemotron-nano-vl-8b-v1
nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1
meta/llama-3.2-90b-vision-instruct
meta/llama-3.2-11b-vision-instruct
microsoft/phi-4-multimodal-instruct
microsoft/phi-3.5-vision-instruct
microsoft/phi-3-vision-128k-instruct
microsoft/kosmos-2
google/paligemma
google/deplot
adept/fuyu-8b
nvidia/nvclip
nvidia/streampetr
```
