# DFlash: Block Diffusion for Flash Speculative Decoding
[**Paper**](https://arxiv.org/abs/2602.06036) | [**Blog**](https://z-lab.ai/projects/dflash/) | [**Models**](https://huggingface.co/collections/z-lab/dflash)

**DFlash** is a lightweight **block diffusion** model designed for speculative decoding. It enables efficient and high-quality parallel drafting.

![DFlash Architecture](https://raw.githubusercontent.com/jianc99/jianc99.github.io/master/images/dflash_system.png)

https://github.com/user-attachments/assets/5b29cabb-eb95-44c9-8ffe-367c0758de8c

## Supported Models

| Model | DFlash Draft |
|---|---|
| Kimi-K2.5 (Preview) | [z-lab/Kimi-K2.5-DFlash](https://huggingface.co/z-lab/Kimi-K2.5-DFlash) |
| Qwen3.5-4B | [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) |
| Qwen3.5-9B | [z-lab/Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash) |
| Qwen3.5-27B | [z-lab/Qwen3.5-27B-DFlash](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash) |
| Qwen3.5-35B-A3B | [z-lab/Qwen3.5-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash) |
| Qwen3-Coder-Next | [z-lab/Qwen3-Coder-Next-DFlash](https://huggingface.co/z-lab/Qwen3-Coder-Next-DFlash) |
| Qwen3-Coder-30B-A3B | [z-lab/Qwen3-Coder-30B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3-Coder-30B-A3B-DFlash) |
| gpt-oss-20b | [z-lab/gpt-oss-20b-DFlash](https://huggingface.co/z-lab/gpt-oss-20b-DFlash) |
| gpt-oss-120b | [z-lab/gpt-oss-120b-DFlash](https://huggingface.co/z-lab/gpt-oss-120b-DFlash) |
| Qwen3-4B | [z-lab/Qwen3-4B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16) |
| Qwen3-8B | [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) |
| Llama-3.1-8B-Instruct | [z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat](https://huggingface.co/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat) |
| Qwen3.5-122B-A10B | Coming soon |
| Qwen3.5-397B-A17B | Coming soon |
| GLM-5.1 | Coming soon |

> Feel free to open a GitHub issue to request support for additional models. We will also open-source the training recipe soon, so you can train your own DFlash draft model to accelerate any LLM.

## 📦 Installation

Use a separate virtual environment for each to avoid conflict.

| Backend | Install command |
|---|---|
| **Transformers** | `uv pip install -e .` |
| **SGLang** | `uv pip install -e ".[sglang]"` |
| **vLLM** | See below |

**vLLM:** DFlash support requires the nightly build:
```bash
uv pip install -e ".[vllm]"
uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```

> **Note:** I've been testing primarily with the vLLM backend on a single A100 80GB. SGLang setup worked fine too but requires the `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` env var or you'll hit context length errors on longer prompts.

## 🚀 Quick Start

### vLLM

```bash
vllm serve Qwen/Qwen3.5-27B \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.5-27B-DFlash", "num_speculative_tokens": 15}' \
  --attention-backend flash_attn
```

> **Tip:** I found `num_speculative_tokens` between 10 and 20 to be the sweet spot — going higher doesn't always improve throughput and can hurt acceptance rate depending on the task.

### SGLang

```bash
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-27B \
  --speculative-algorithm dflash \
  --speculative-draft-model-path z-lab/Qwen3.5-27B-DFlash \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 15
```

## 📝 Notes (Personal)

- Forked mainly to experiment with Llama-3.1-8B-Instruct + DFlash on my local 3090.
- The Transformers backend is the easiest to get running for quick tests without a full vLLM setup.
- `num_speculative_tokens=10` seems to work well for the 8B model on 24GB VRAM.
