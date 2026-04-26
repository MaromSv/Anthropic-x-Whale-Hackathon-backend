"""Backend abstraction over different ways to run Gemma locally.

The point of this file is to let you swap *how* the model is loaded —
full-precision HF weights, INT4/INT8 GGUF, future MediaPipe `.task` —
without touching the prompt-building or retrieval code.

This mirrors how the Kotlin app will work: the on-device Gemma is one
component behind an interface; the rest of the pipeline (retrieval,
prompt assembly, response handling) stays the same regardless of which
quantization is loaded.

Supported now:
  - "hf"     → google/gemma-2-2b-it (or any HF gemma variant) via transformers
  - "gguf"   → quantized Q2/Q4/Q5/Q8 via llama-cpp-python (optional install)

Test small → big to find the smallest acceptable variant for the phone.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    output_tokens: int
    seconds: float

    @property
    def tokens_per_second(self) -> float:
        return self.output_tokens / self.seconds if self.seconds > 0 else 0.0


class Backend(ABC):
    """Common interface every model backend must satisfy."""
    name: str  # short label used in eval output

    @abstractmethod
    def generate(self, messages: list[dict], max_new_tokens: int = 300) -> GenerationResult:
        """Generate from a list of {role, content} messages.

        Implementations must apply the Gemma chat template themselves —
        the Kotlin app on-device will do the same via MediaPipe's prompt
        formatting. Keep this contract stable.
        """
        raise NotImplementedError


# --- HuggingFace transformers backend ---------------------------------------

class HFBackend(Backend):
    """Full-precision (or bf16) HF Gemma — the dev baseline."""

    def __init__(self, model_id: str = "google/gemma-2-2b-it") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf:{model_id.split('/')[-1]}"
        self.device = self._pick_device(torch)
        print(f"[{self.name}] loading on {self.device}...")

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
            device_map=self.device,
            attn_implementation="eager",  # SDPA/cuDNN fails on MIG slices
        )
        self.model.eval()

    @staticmethod
    def _pick_device(torch) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def generate(self, messages: list[dict], max_new_tokens: int = 300) -> GenerationResult:
        prompt = _format_prompt(self.tokenizer, messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        t0 = time.time()
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        new_tokens = out[0, prompt_tokens:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=new_tokens.shape[0],
            seconds=elapsed,
        )


# Gemma chat template — used when tokenizer.chat_template is missing
# (base / pretrained models don't ship one). This matches the format
# Gemma's instruct variants use, so output stays comparable.
def _format_prompt(tokenizer, messages: list[dict]) -> str:
    """Apply chat template if available, else fall back to Gemma's manual format."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        # Surface the failure so the caller knows the template didn't apply.
        print(f"[warn] apply_chat_template failed ({e}); using manual Gemma format")
        # Gemma format: <bos><start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n
        bos = getattr(tokenizer, "bos_token", "<bos>") or "<bos>"
        parts = [bos]
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            parts.append(f"<start_of_turn>{role}\n{m['content']}<end_of_turn>")
        parts.append("<start_of_turn>model\n")
        return "\n".join(parts)


# --- llama-cpp / GGUF backend (the easy way to test quantization) -----------

class GGUFBackend(Backend):
    """Quantized Gemma via llama.cpp. Lets you A/B Q2/Q4/Q5/Q8 quickly.

    Setup once:
        pip install llama-cpp-python
        # Download a pre-quantized GGUF, e.g. from
        #   https://huggingface.co/bartowski/gemma-2-2b-it-GGUF
        # Pick the variant you want to test (Q4_K_M is a common 'good balance').

    Then:
        backend = GGUFBackend("/path/to/gemma-2-2b-it-Q4_K_M.gguf")
    """

    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1) -> None:
        from llama_cpp import Llama

        import os
        self.name = f"gguf:{os.path.basename(model_path)}"
        print(f"[{self.name}] loading...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,  # -1 = offload all layers to Metal on Mac
            verbose=False,
        )

    def generate(self, messages: list[dict], max_new_tokens: int = 300) -> GenerationResult:
        # llama.cpp's create_chat_completion handles Gemma's template natively.
        t0 = time.time()
        resp = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        elapsed = time.time() - t0
        text = resp["choices"][0]["message"]["content"].strip()
        usage = resp.get("usage", {})
        return GenerationResult(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            seconds=elapsed,
        )


# --- Factory ----------------------------------------------------------------

# --- MLX backend (Apple Silicon native, easy quant testing) -----------------

class MLXBackend(Backend):
    """MLX-quantized Gemma running on Apple Silicon (Metal).

    Convert any HF model to MLX format with quantization in one command:
        mlx_lm.convert --hf-path ./models/gemma-4-E2B \\
            --mlx-path ./models/gemma-4-E2B-mlx-int4 \\
            -q --q-bits 4

    Then point this backend at the output directory:
        backend = MLXBackend("./models/gemma-4-E2B-mlx-int4")

    `q-bits` can be 2, 3, 4, 6, or 8. Test smaller → bigger to find the
    smallest variant that still produces coherent first-aid answers.
    """

    def __init__(self, model_path: str) -> None:
        from mlx_lm import generate as mlx_generate
        from mlx_lm import load as mlx_load

        import os
        self.name = f"mlx:{os.path.basename(model_path.rstrip('/'))}"
        print(f"[{self.name}] loading...")
        self.model, self.tokenizer = mlx_load(model_path)
        self._generate = mlx_generate

    def generate(self, messages: list[dict], max_new_tokens: int = 300) -> GenerationResult:
        # mlx-lm's tokenizer wraps HF's; chat template support depends on the
        # model's tokenizer_config.json. apply_chat_template returns a string.
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = len(self.tokenizer.encode(prompt))

        t0 = time.time()
        text = self._generate(
            self.model, self.tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            verbose=False,
        )
        elapsed = time.time() - t0

        output_tokens = len(self.tokenizer.encode(text))
        return GenerationResult(
            text=text.strip(),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            seconds=elapsed,
        )


# --- Quanto backend (architecture-agnostic Mac quantization) ---------------

class QuantoBackend(Backend):
    """Load a model that was quantized by `scripts/quantize_quanto.py`.

    Quantization is pure PyTorch via Quanto, so this works for any HF model
    on any device — including Gemma 4 E2B on Mac MPS, which architecture-
    specific quantizers (MLX, llama.cpp) can't handle yet.
    """

    def __init__(self, model_path: str) -> None:
        import json
        import os

        import torch
        from optimum.quanto import requantize
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.name = f"quanto:{os.path.basename(model_path.rstrip('/'))}"
        self._torch = torch
        self.device = HFBackend._pick_device(torch)
        print(f"[{self.name}] loading on {self.device}...")

        config = AutoConfig.from_pretrained(model_path)
        # SDPA/cuDNN fails on MIG slices (same fix as HFBackend)
        config._attn_implementation = "eager"
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))
        with open(os.path.join(model_path, "quanto_qmap.json")) as f:
            qmap = json.load(f)

        def _load(device: str):
            with torch.device("meta"):
                m = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
            requantize(m, state_dict, qmap, device=torch.device(device))
            m.eval()
            return m

        try:
            model = _load(self.device)
        except OSError as e:
            # quanto's CUDA int4 unpack kernel requires JIT compilation which
            # needs CUDA_HOME / nvcc. If that's not set up, fall back to CPU.
            if self.device != "cpu":
                print(f"[{self.name}] CUDA load failed ({e}), retrying on cpu...")
                self.device = "cpu"
                model = _load("cpu")
            else:
                raise

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate(self, messages: list[dict], max_new_tokens: int = 300) -> GenerationResult:
        prompt = _format_prompt(self.tokenizer, messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        t0 = time.time()
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        new_tokens = out[0, prompt_tokens:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not text:
            print(f"[{self.name}] WARNING: empty output ({new_tokens.shape[0]} raw tokens generated)")
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=new_tokens.shape[0],
            seconds=elapsed,
        )


# --- Factory ----------------------------------------------------------------

def build_backend(spec: str) -> Backend:
    """Parse a CLI-friendly model spec into a Backend instance.

    Examples:
        hf:google/gemma-2-2b-it
        hf:./models/gemma-4-E2B                  (local HF safetensors)
        quanto:./models/gemma-4-E2B-quanto-int4  (Quanto-quantized; works for any arch)
        mlx:./models/gemma-3-1b-it-mlx-int4      (MLX-quantized; arch-specific)
        gguf:/abs/path/to/foo-Q4_K_M.gguf
    """
    if ":" not in spec:
        raise ValueError(f"Model spec must be '<kind>:<target>', got {spec!r}")
    kind, target = spec.split(":", 1)
    if kind == "hf":
        return HFBackend(target)
    if kind == "quanto":
        return QuantoBackend(target)
    if kind == "mlx":
        return MLXBackend(target)
    if kind == "gguf":
        return GGUFBackend(target)
    raise ValueError(f"Unknown backend {kind!r}. Use 'hf', 'quanto', 'mlx', or 'gguf'.")
