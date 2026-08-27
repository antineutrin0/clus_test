"""Provider-agnostic LLM clients for CLUSE-Test.

Built-in providers:
- ``hf`` / ``huggingface``: local Transformers causal language models;
- ``gemini`` / ``google``: Google Gemini API;
- ``openai`` / ``gpt``: OpenAI Responses API;
- ``mock``: deterministic smoke-test client.

Additional providers can be registered with :func:`register_provider` without
changing layer or pipeline code.
"""

from __future__ import annotations

import ast
import gc
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from src.utils import config
from src.utils.logger import get_logger

log = get_logger(__name__)


def _get_kaggle_secret(name: str) -> str:
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        return UserSecretsClient().get_secret(name) or ""
    except Exception:
        return ""


def _secret_or_env(name: str, configured_value: str = "") -> str:
    return configured_value or os.environ.get(name, "") or _get_kaggle_secret(name)


def _build_timeout_stopping_criteria(timeout_sec: float):
    """A ``StoppingCriteriaList`` that halts local generation once a
    wall-clock budget is exceeded, so a single call can never run
    indefinitely regardless of the token budget or whether the model emits
    EOS. Returns ``None`` (no-op) if ``transformers`` isn't importable yet or
    ``timeout_sec`` is disabled (<= 0); the caller already imports
    ``transformers`` elsewhere so this should normally succeed.
    """
    if timeout_sec is None or timeout_sec <= 0:
        return None
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except ImportError:
        return None

    class _TimeoutStoppingCriteria(StoppingCriteria):
        def __init__(self, deadline: float) -> None:
            self._deadline = deadline

        def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: D401
            return time.time() >= self._deadline

    return StoppingCriteriaList([_TimeoutStoppingCriteria(time.time() + timeout_sec)])


# ---------------------------------------------------------------------------
# Local-model response cleanup (HF provider only).
# ---------------------------------------------------------------------------
#
# Paid API completions (OpenAI's Responses API, Gemini) come back as plain
# text with no Markdown wrapping in practice -- the model follows the
# "output only the function" instruction cleanly. A small locally hosted
# model (e.g. Qwen2.5-Coder-3B-Instruct) reliably does not: it wraps its
# answer in ```python fences, sometimes prefixes a sentence before the code,
# and sometimes appends an explanatory paragraph after it. Rather than push
# that inconsistency downstream into every caller of ``LocalHFClient``, it is
# cleaned once, right here, immediately after decoding -- so
# ``LLMResponse.text`` for the ``hf`` provider is already just the runnable
# function, the same shape a paid provider's response already has. No other
# provider is touched by this.

_CODE_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\s*(.*?)```", re.DOTALL)
_CHECK_SIGNATURE_RE = re.compile(r"(?m)^[ \t]*def\s+check\s*\(")


def _extract_function_source(text: str) -> str:
    """Cut everything before the ``def check(`` signature, and everything
    after the function body ends (the first unindented line following the
    body, or a stray Markdown fence marker). Returns ``""`` if no signature
    is found at all.
    """
    match = _CHECK_SIGNATURE_RE.search(text)
    if not match:
        return ""

    text = text[match.start():]
    lines = text.splitlines()
    if not lines:
        return ""

    function_lines = [lines[0].rstrip()]
    body_seen = False

    for line in lines[1:]:
        stripped = line.strip()

        # A Markdown fence is never part of the function itself.
        if stripped.startswith("```"):
            break

        # Blank lines are retained only once the body has actually started.
        if not stripped:
            if body_seen:
                function_lines.append("")
            continue

        # Everything belonging to the function body is indented.
        if line[0].isspace():
            function_lines.append(line.rstrip())
            body_seen = True
            continue

        # An unindented line after the body has started is trailing prose
        # or another top-level statement -- stop before it.
        break

    return "\n".join(function_lines).strip("\n")


def _clean_check_function_response(raw_text: str) -> str:
    """Reduce a raw local-model completion down to just the runnable
    ``check(...)`` function -- no Markdown fences, no leading/trailing
    prose, no comments. This mirrors what a paid API model already returns.

    Extraction order:
    1. Every fenced ```...``` block that contains a `def check(` signature
       (a local model sometimes emits more than one fenced block).
    2. The raw, unfenced text, as a fallback for a model that forgot the
       closing fence or never used one.

    Once a candidate block is isolated, it is canonicalized through an AST
    parse/unparse round-trip. ``ast.parse`` ignores comments entirely, so
    unparsing back to source drops every comment along with any residual
    stray formatting -- purely as a side effect of AST round-tripping, never
    by regex-stripping ``#`` (which would be unsafe against a ``#`` inside a
    string literal). If the extracted text isn't valid Python, the raw
    extracted text is returned as-is so the caller/downstream validator can
    reject it explicitly rather than this function silently manufacturing
    something that looks plausible.

    If no `def check(` signature is found anywhere, the original text is
    returned unchanged (stripped of surrounding whitespace only) so an
    unexpected response shape fails visibly downstream instead of being
    silently swallowed here.
    """
    if not raw_text or not raw_text.strip():
        return raw_text

    text = raw_text.strip()
    candidates: list[str] = []

    fenced_blocks = _CODE_FENCE_RE.findall(text)
    for block in fenced_blocks:
        if _CHECK_SIGNATURE_RE.search(block):
            candidates.append(block)

    if _CHECK_SIGNATURE_RE.search(text):
        candidates.append(text)

    for candidate in candidates:
        extracted = _extract_function_source(candidate)
        if not extracted:
            continue
        try:
            tree = ast.parse(extracted)
            return ast.unparse(tree) + "\n"
        except SyntaxError:
            return extracted

    return text


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thoughts_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_sec: float = 0.0


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int, thoughts_tokens: int = 0) -> float:
    prices = config.MODEL_PRICING_USD_PER_1K_TOKENS.get(model, {"input": 0.0, "output": 0.0})
    output_billable = completion_tokens + thoughts_tokens
    cost = (prompt_tokens / 1000.0) * float(prices.get("input", 0.0))
    cost += (output_billable / 1000.0) * float(prices.get("output", 0.0))
    return round(cost, 8)


class BaseLLMClient(ABC):
    model_name: str
    provider_name: str

    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    provider_name = "mock"

    def __init__(self, model_name: str = "mock-llm", max_output_tokens: Optional[int] = None, **_: object) -> None:
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens

    def generate(self, prompt: str) -> LLMResponse:
        start = time.time()
        text = "def check(candidate):\n    assert candidate is not None\n"
        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = len(text.split())
        return LLMResponse(
            text=text,
            model=self.model_name,
            provider=self.provider_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_sec=round(time.time() - start, 4),
        )


class LocalHFClient(BaseLLMClient):
    """Local Hugging Face causal LM. Model weights are cached per process."""

    provider_name = "hf"
    _MODEL_CACHE: Dict[str, object] = {}
    _TOKENIZER_CACHE: Dict[str, object] = {}

    def __init__(
        self,
        model_name: str,
        max_output_tokens: Optional[int],
        fallback_models: Optional[list[str]] = None,
        **_: object,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_output_tokens if max_output_tokens and max_output_tokens > 0 else None
        self.fallback_models = [m for m in (fallback_models or []) if m and m != model_name]
        self._model = None
        self._tokenizer = None

    def _clear_model(self) -> None:
        LocalHFClient._MODEL_CACHE.pop(self.model_name, None)
        LocalHFClient._TOKENIZER_CACHE.pop(self.model_name, None)
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install local LLM dependencies: transformers torch accelerate bitsandbytes") from exc

        if self.model_name in self._MODEL_CACHE and self.model_name in self._TOKENIZER_CACHE:
            self._model = self._MODEL_CACHE[self.model_name]
            self._tokenizer = self._TOKENIZER_CACHE[self.model_name]
            log.info("Reusing cached Hugging Face model: %s", self.model_name)
            return

        token = _secret_or_env("HF_TOKEN", config.HF_TOKEN) or None
        log.info("Loading Hugging Face model: %s", self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token, trust_remote_code=True)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        # If a compact prompt still exceeds the local-model budget, preserve the
        # mutation dossiers and output contract placed at the end of the prompt.
        tokenizer.truncation_side = "left"

        kwargs = {"token": token, "trust_remote_code": True, "device_map": "auto"}
        if config.LOAD_IN_4BIT and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception as exc:
                log.warning("4-bit quantization unavailable: %s", exc)
                kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        model.eval()
        self._model, self._tokenizer = model, tokenizer
        self._MODEL_CACHE[self.model_name] = model
        self._TOKENIZER_CACHE[self.model_name] = tokenizer

    def _input_device(self):
        try:
            return next(self._model.parameters()).device
        except Exception:
            return getattr(self._model, "device", "cpu")

    def _generate_once(self, prompt: str) -> LLMResponse:
        self._load()
        import torch

        start = time.time()
        tokenizer, model = self._tokenizer, self._model
        if tokenizer is None or model is None:
            raise RuntimeError("Hugging Face model/tokenizer was not loaded")

        rendered = prompt
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
                )
            except Exception:
                rendered = prompt
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=config.LAYER1_MAX_INPUT_TOKENS,
        )
        device = self._input_device()
        encoded = {k: v.to(device) for k, v in encoded.items() if hasattr(v, "to")}
        input_len = int(encoded["input_ids"].shape[-1])
        generation_kwargs = {
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            # Greedy decoding on a small local model can drift into a long
            # repetitive tail instead of emitting EOS; these discourage that
            # without affecting a normal, well-formed response.
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 6,
        }
        if self.max_new_tokens is not None:
            generation_kwargs["max_new_tokens"] = min(self.max_new_tokens, config.HF_MAX_NEW_TOKENS_HARD_CAP)
        else:
            # No project-level output cap: allow generation up to the model's
            # natural context boundary and rely on EOS plus the output contract
            # -- but never beyond the hard safety cap below, regardless of how
            # large the model's own context window is.
            model_limit = int(
                getattr(getattr(model, "config", None), "max_position_embeddings", 0)
                or getattr(tokenizer, "model_max_length", 0)
                or config.LAYER1_MAX_INPUT_TOKENS + 4096
            )
            if model_limit > 1_000_000:  # Some tokenizers expose a sentinel value.
                model_limit = config.LAYER1_MAX_INPUT_TOKENS + 4096
            generation_kwargs["max_new_tokens"] = min(
                max(64, model_limit - input_len), config.HF_MAX_NEW_TOKENS_HARD_CAP
            )
        stopping_criteria = _build_timeout_stopping_criteria(config.HF_GENERATION_TIMEOUT_SEC)
        if stopping_criteria is not None:
            generation_kwargs["stopping_criteria"] = stopping_criteria
        with torch.inference_mode():
            outputs = model.generate(**encoded, **generation_kwargs)
        generated = outputs[0][input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        # HF-specific cleanup: strip Markdown fences / leading-trailing prose
        # / comments so this provider's LLMResponse.text is pure runnable
        # code, matching what the paid providers already return. completion
        # token accounting below still reflects the true, uncleaned
        # generation length -- cleanup only affects the returned text, not
        # cost/telemetry.
        text = _clean_check_function_response(text)
        completion_tokens = int(generated.shape[-1])
        return LLMResponse(
            text=text,
            model=self.model_name,
            provider=self.provider_name,
            prompt_tokens=input_len,
            completion_tokens=completion_tokens,
            total_tokens=input_len + completion_tokens,
            estimated_cost_usd=estimate_cost_usd(self.model_name, input_len, completion_tokens),
            latency_sec=round(time.time() - start, 4),
        )

    def generate(self, prompt: str) -> LLMResponse:
        candidates = [self.model_name] + self.fallback_models
        errors: list[str] = []
        for model in candidates:
            self.model_name = model
            try:
                return self._generate_once(prompt)
            except Exception as exc:
                errors.append(f"{model}: {type(exc).__name__}: {exc}")
                log.exception("Hugging Face generation failed for %s", model)
                self._clear_model()
        raise RuntimeError("All Hugging Face models failed: " + " | ".join(errors[-5:]))


class GeminiClient(BaseLLMClient):
    provider_name = "gemini"

    def __init__(
        self,
        model_name: str,
        max_output_tokens: Optional[int],
        fallback_models: Optional[list[str]] = None,
        **_: object,
    ) -> None:
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens if max_output_tokens and max_output_tokens > 0 else None
        self.fallback_models = [m for m in (fallback_models or []) if m and m != model_name]

    def _candidate_models(self) -> list[str]:
        return list(dict.fromkeys([self.model_name] + self.fallback_models))

    @staticmethod
    def _error_text(exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc!r} {exc}"

    @staticmethod
    def _retry_delay_sec(exc: Exception, default: float) -> float:
        text = GeminiClient._error_text(exc)
        match = re.search(r"retryDelay[\"']?\s*[:=]\s*[\"']?(\d+(?:\.\d+)?)(ms|s)?", text, flags=re.I)
        if not match:
            match = re.search(r"retry in\s+(\d+(?:\.\d+)?)(ms|s)", text, flags=re.I)
        if not match:
            return default
        value = float(match.group(1))
        if (match.group(2) or "s").lower() == "ms":
            value /= 1000.0
        return max(0.0, min(value, config.GEMINI_MAX_RETRY_SLEEP_SEC))

    def generate(self, prompt: str) -> LLMResponse:
        key = _secret_or_env("GEMINI_API_KEY", config.GEMINI_API_KEY)
        if not key:
            raise ValueError("GEMINI_API_KEY is not set")
        errors: list[str] = []
        for model in self._candidate_models():
            for attempt in range(config.GEMINI_MAX_RETRIES):
                start = time.time()
                try:
                    try:
                        response = self._generate_with_google_genai(prompt, key, start, model)
                    except ImportError:
                        response = self._generate_with_google_generativeai(prompt, key, start, model)
                    self.model_name = response.model
                    return response
                except Exception as exc:
                    text = self._error_text(exc)
                    errors.append(f"{model}: {text}")
                    lower = text.lower()
                    quota_zero = ("429" in lower or "quota" in lower) and "limit: 0" in lower
                    unavailable = "503" in lower or "unavailable" in lower or "high demand" in lower
                    if quota_zero or (unavailable and self.fallback_models):
                        break
                    if attempt < config.GEMINI_MAX_RETRIES - 1:
                        time.sleep(self._retry_delay_sec(exc, float(2**attempt)))
            log.warning("Gemini model %s failed; trying fallback", model)
        raise RuntimeError("All Gemini models failed: " + " | ".join(errors[-5:]))

    def _generate_with_google_genai(self, prompt: str, key: str, start: float, model_name: str) -> LLMResponse:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        kwargs = {"temperature": 0.2}
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens
        if config.GEMINI_THINKING_BUDGET >= 0:
            try:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=config.GEMINI_THINKING_BUDGET)
            except Exception:
                pass
        try:
            generation_config = types.GenerateContentConfig(**kwargs)
        except TypeError:
            kwargs.pop("thinking_config", None)
            generation_config = types.GenerateContentConfig(**kwargs)
        resp = client.models.generate_content(model=model_name, contents=prompt, config=generation_config)
        usage = getattr(resp, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
        thoughts_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0) if usage else 0
        total_tokens = int(getattr(usage, "total_token_count", 0) or 0) if usage else 0
        if not total_tokens:
            total_tokens = prompt_tokens + completion_tokens + thoughts_tokens
        return LLMResponse(
            text=getattr(resp, "text", "") or "",
            model=model_name,
            provider=self.provider_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thoughts_tokens=thoughts_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimate_cost_usd(model_name, prompt_tokens, completion_tokens, thoughts_tokens),
            latency_sec=round(time.time() - start, 4),
        )

    def _generate_with_google_generativeai(self, prompt: str, key: str, start: float, model_name: str) -> LLMResponse:
        import google.generativeai as genai

        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name)
        generation_config = {"temperature": 0.2}
        if self.max_output_tokens is not None:
            generation_config["max_output_tokens"] = self.max_output_tokens
        resp = model.generate_content(prompt, generation_config=generation_config)
        usage = getattr(resp, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
        total_tokens = int(getattr(usage, "total_token_count", 0) or 0) if usage else prompt_tokens + completion_tokens
        return LLMResponse(
            text=getattr(resp, "text", "") or "",
            model=model_name,
            provider=self.provider_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimate_cost_usd(model_name, prompt_tokens, completion_tokens),
            latency_sec=round(time.time() - start, 4),
        )


class OpenAIClient(BaseLLMClient):
    """OpenAI client using the Responses API."""

    provider_name = "openai"

    def __init__(
        self,
        model_name: str,
        max_output_tokens: Optional[int],
        fallback_models: Optional[list[str]] = None,
        **_: object,
    ) -> None:
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens if max_output_tokens and max_output_tokens > 0 else None
        self.fallback_models = [m for m in (fallback_models or []) if m and m != model_name]

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the OpenAI SDK: pip install openai") from exc
        key = _secret_or_env("OPENAI_API_KEY", config.OPENAI_API_KEY)
        if not key:
            raise ValueError("OPENAI_API_KEY is not set")
        kwargs: dict = {
            "api_key": key,
            "timeout": config.OPENAI_TIMEOUT_SEC,
            "max_retries": config.OPENAI_MAX_RETRIES,
        }
        base_url = config.OPENAI_BASE_URL or os.environ.get("OPENAI_BASE_URL", "")
        if base_url:
            kwargs["base_url"] = base_url
        if config.OPENAI_ORGANIZATION:
            kwargs["organization"] = config.OPENAI_ORGANIZATION
        if config.OPENAI_PROJECT:
            kwargs["project"] = config.OPENAI_PROJECT
        return OpenAI(**kwargs)

    def generate(self, prompt: str) -> LLMResponse:
        client = self._client()
        errors: list[str] = []
        for model in list(dict.fromkeys([self.model_name] + self.fallback_models)):
            start = time.time()
            try:
                request_kwargs = {
                    "model": model,
                    "input": prompt,
                    "store": False,
                    "reasoning": {"effort": config.OPENAI_REASONING_EFFORT},
                    "text": {"verbosity": config.OPENAI_TEXT_VERBOSITY},
                }
                # A value of 0/None means no project-imposed output-token cap.
                # The provider/model context window and EOS remain the limits.
                if self.max_output_tokens is not None:
                    request_kwargs["max_output_tokens"] = self.max_output_tokens
                resp = client.responses.create(**request_kwargs)
                usage = getattr(resp, "usage", None)
                prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
                api_output_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
                details = getattr(usage, "output_tokens_details", None) if usage else None
                # OpenAI reports reasoning tokens as a subset of output_tokens.
                # Normalize completion_tokens to visible/non-reasoning output so
                # completion + thoughts equals the billable API output count.
                thoughts_tokens = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
                completion_tokens = max(0, api_output_tokens - thoughts_tokens)
                total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
                if not total_tokens:
                    total_tokens = prompt_tokens + api_output_tokens
                self.model_name = model
                return LLMResponse(
                    text=getattr(resp, "output_text", "") or "",
                    model=model,
                    provider=self.provider_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    thoughts_tokens=thoughts_tokens,
                    total_tokens=total_tokens,
                    estimated_cost_usd=estimate_cost_usd(model, prompt_tokens, completion_tokens, thoughts_tokens),
                    latency_sec=round(time.time() - start, 4),
                )
            except Exception as exc:
                errors.append(f"{model}: {type(exc).__name__}: {exc}")
                log.warning("OpenAI model %s failed; trying fallback if configured", model)
        raise RuntimeError("All OpenAI models failed: " + " | ".join(errors[-5:]))


ProviderFactory = Callable[..., BaseLLMClient]
_PROVIDER_REGISTRY: Dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory, aliases: Optional[list[str]] = None) -> None:
    """Register a custom provider factory.

    The factory receives ``model_name``, ``max_output_tokens``, and
    ``fallback_models`` keyword arguments.
    """
    for key in [name] + list(aliases or []):
        _PROVIDER_REGISTRY[key.strip().lower()] = factory


def available_providers() -> list[str]:
    return sorted(_PROVIDER_REGISTRY)


def infer_provider(model_name: str) -> str:
    name = (model_name or "").lower()
    if name.startswith("gemini"):
        return "gemini"
    if name.startswith(("gpt-", "o1", "o3", "o4", "ft:")):
        return "openai"
    if name.startswith("mock"):
        return "mock"
    return "hf" if "/" in model_name else "openai"


def _clean_model_name(model_name: str, field: str = "model") -> str:
    name = (model_name or "").strip()
    if not name:
        raise ValueError(f"{field} name is empty")
    if name.startswith("<") or name.endswith(">"):
        raise ValueError(
            f"Invalid {field} name {name!r}. Angle brackets are placeholders; "
            f"use the literal API model ID without < or >."
        )
    return name


def create_client(
    provider: str,
    model_name: str,
    max_output_tokens: Optional[int],
    fallback_models: Optional[list[str]] = None,
) -> BaseLLMClient:
    provider = (provider or "auto").strip().lower()
    model_name = _clean_model_name(model_name)
    cleaned_fallbacks = [_clean_model_name(item, "fallback model") for item in (fallback_models or []) if item and item.strip()]
    if provider == "auto":
        provider = infer_provider(model_name)
    if config.USE_MOCK_LLM:
        provider, model_name = "mock", f"mock-{model_name or 'llm'}"
        cleaned_fallbacks = []
    factory = _PROVIDER_REGISTRY.get(provider)
    if factory is None:
        raise ValueError(f"Unknown LLM provider '{provider}'. Available: {available_providers()}")

    # Fallback model IDs are submitted to the same provider client.  Reject
    # obvious cross-provider leakage (e.g., Gemini IDs inside an OpenAI client).
    if provider == "openai":
        bad = [name for name in cleaned_fallbacks if name.lower().startswith("gemini")]
        if bad:
            raise ValueError(f"OpenAI fallback list contains Gemini model IDs: {bad}")
    if provider == "gemini":
        bad = [name for name in cleaned_fallbacks if name.lower().startswith(("gpt-", "o1", "o3", "o4"))]
        if bad:
            raise ValueError(f"Gemini fallback list contains OpenAI model IDs: {bad}")

    return factory(
        model_name=model_name,
        max_output_tokens=max_output_tokens,
        fallback_models=cleaned_fallbacks,
    )


def get_client(
    layer: int,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    fallback_models: Optional[list[str]] = None,
    max_output_tokens: Optional[int] = None,
) -> BaseLLMClient:
    settings = {
        1: (config.LAYER1_PROVIDER, config.LAYER1_MODEL, [config.LAYER1_FALLBACK_MODEL], config.LAYER1_MAX_TOKENS),
        2: (config.LAYER2_PROVIDER, config.LAYER2_MODEL, config.LAYER2_FALLBACK_MODELS, config.LAYER2_MAX_TOKENS),
        3: (config.LAYER3_PROVIDER, config.LAYER3_MODEL, config.LAYER3_FALLBACK_MODELS, config.LAYER3_MAX_TOKENS),
    }
    if layer not in settings:
        raise ValueError(f"Unknown layer: {layer}")
    default_provider, default_model, default_fallbacks, default_tokens = settings[layer]
    provider_overridden = bool(provider)
    model_overridden = bool(model_name)
    resolved_fallbacks = fallback_models
    if resolved_fallbacks is None:
        # Do not inherit provider-specific defaults after the user changes the
        # provider/model.  This fixes Gemini IDs being sent to the OpenAI API.
        resolved_fallbacks = [] if (provider_overridden or model_overridden) else default_fallbacks
    return create_client(
        provider=provider or default_provider,
        model_name=model_name or default_model,
        max_output_tokens=default_tokens if max_output_tokens is None else max_output_tokens,
        fallback_models=resolved_fallbacks,
    )


def get_baseline_client(
    model_name: str = config.BASELINE_MODEL,
    provider: Optional[str] = None,
    fallback_models: Optional[list[str]] = None,
    max_output_tokens: Optional[int] = None,
) -> BaseLLMClient:
    provider_overridden = bool(provider)
    model_overridden = bool(model_name and model_name != config.BASELINE_MODEL)
    resolved_fallbacks = fallback_models
    if resolved_fallbacks is None:
        resolved_fallbacks = [] if (provider_overridden or model_overridden) else config.BASELINE_FALLBACK_MODELS
    return create_client(
        provider=provider or config.BASELINE_PROVIDER,
        model_name=model_name,
        max_output_tokens=config.BASELINE_MAX_TOKENS if max_output_tokens is None else max_output_tokens,
        fallback_models=resolved_fallbacks,
    )


register_provider("hf", LocalHFClient, aliases=["huggingface", "local"])
register_provider("gemini", GeminiClient, aliases=["google"])
register_provider("openai", OpenAIClient, aliases=["gpt"])
register_provider("mock", MockLLMClient)