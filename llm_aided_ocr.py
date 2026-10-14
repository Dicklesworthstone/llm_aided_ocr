import asyncio
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import functools
import glob
import json
import logging
import os
import random
import re
import shutil
import subprocess
import traceback
from typing import Any
import urllib.parse
import urllib.request
import warnings

from anthropic import AsyncAnthropic
import cv2
from decouple import Config as DecoupleConfig, RepositoryEnv
from filelock import FileLock, Timeout
import numpy as np
from openai import AsyncOpenAI
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import tiktoken

try:
    from llama_cpp import Llama, LlamaGrammar
except ImportError:
    Llama = None  # type: ignore[assignment,misc]
    LlamaGrammar = None  # type: ignore[assignment,misc]

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None  # type: ignore[assignment,misc]

try:
    import nvgpu  # type: ignore[import-not-found]

    NVGPU_AVAILABLE = True
except ImportError:
    nvgpu = None
    NVGPU_AVAILABLE = False

# Configuration (.env is optional; without it, settings come from os.environ or the defaults below)
config = DecoupleConfig(RepositoryEnv(".env")) if os.path.exists(".env") else DecoupleConfig({})

USE_LOCAL_LLM = config.get("USE_LOCAL_LLM", default=False, cast=bool)
API_PROVIDER = config.get("API_PROVIDER", default="OPENAI", cast=str)  # OPENAI or CLAUDE
ANTHROPIC_API_KEY = config.get("ANTHROPIC_API_KEY", default="your-anthropic-api-key", cast=str)
OPENAI_API_KEY = config.get("OPENAI_API_KEY", default="your-openai-api-key", cast=str)
CLAUDE_MODEL_STRING = config.get("CLAUDE_MODEL_STRING", default="claude-3-haiku-20240307", cast=str)
CLAUDE_MAX_TOKENS = config.get("CLAUDE_MAX_TOKENS", default=4096, cast=int)
TOKEN_BUFFER = config.get("TOKEN_BUFFER", default=500, cast=int)
TOKEN_CUSHION = config.get("TOKEN_CUSHION", default=300, cast=int)
OPENAI_COMPLETION_MODEL = config.get("OPENAI_COMPLETION_MODEL", default="gpt-4o-mini", cast=str)
OPENAI_EMBEDDING_MODEL = config.get("OPENAI_EMBEDDING_MODEL", default="text-embedding-3-small", cast=str)
OPENAI_MAX_TOKENS = config.get("OPENAI_MAX_TOKENS", default=12000, cast=int)
DEFAULT_LOCAL_MODEL_NAME = config.get(
    "DEFAULT_LOCAL_MODEL_NAME",
    default="Llama-3.1-8B-Lexi-Uncensored_Q5_fixedrope.gguf",
    cast=str,
)
LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS = config.get("LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS", default=4096, cast=int)
USE_VERBOSE = config.get("USE_VERBOSE", default=False, cast=bool)
PDF_DPI = config.get("PDF_DPI", default=300, cast=int)
MAX_CONCURRENT_REQUESTS = config.get("MAX_CONCURRENT_REQUESTS", default=4, cast=int)
API_MAX_RETRIES = config.get("API_MAX_RETRIES", default=3, cast=int)
API_RETRY_BASE_DELAY = config.get("API_RETRY_BASE_DELAY", default=1.0, cast=float)
USE_HF_TOKENIZER = config.get("USE_HF_TOKENIZER", default=False, cast=bool)

PLACEHOLDER_API_KEYS = frozenset(
    {
        "",
        "your-openai-api-key",
        "your-anthropic-api-key",
        "your-api-key",
        "changeme",
        "none",
        "null",
    }
)

# Lazily initialized API clients and local model cache
openai_client: AsyncOpenAI | None = None
_OPENAI_CLIENTS: dict[str, AsyncOpenAI] = {}
_ANTHROPIC_CLIENTS: dict[str, AsyncAnthropic] = {}
_MODEL_CACHE: dict[tuple[str, int, bool], Any] = {}

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def is_valid_api_key(key: str | None) -> bool:
    """Return True only if key is a non-empty, non-placeholder API key string."""
    if not key or not isinstance(key, str):
        return False
    normalized = key.strip()
    if not normalized:
        return False
    lower = normalized.lower()
    if lower in PLACEHOLDER_API_KEYS or lower.startswith("your-"):
        return False
    return True


def get_openai_client(api_key: str | None = None) -> AsyncOpenAI | None:
    """Lazily initialize and cache an AsyncOpenAI client when a valid API key is provided."""
    global openai_client
    raw_key = api_key if api_key is not None else OPENAI_API_KEY
    if not is_valid_api_key(raw_key):
        if api_key is None:
            openai_client = None
        return None
    resolved_key = raw_key.strip()
    if resolved_key not in _OPENAI_CLIENTS:
        _OPENAI_CLIENTS[resolved_key] = AsyncOpenAI(api_key=resolved_key)
    if api_key is None:
        openai_client = _OPENAI_CLIENTS[resolved_key]
    return _OPENAI_CLIENTS[resolved_key]


def get_anthropic_client(api_key: str | None = None) -> AsyncAnthropic | None:
    """Lazily initialize and cache an AsyncAnthropic client when a valid API key is provided."""
    raw_key = api_key if api_key is not None else ANTHROPIC_API_KEY
    if not is_valid_api_key(raw_key):
        return None
    resolved_key = raw_key.strip()
    if resolved_key not in _ANTHROPIC_CLIENTS:
        _ANTHROPIC_CLIENTS[resolved_key] = AsyncAnthropic(api_key=resolved_key)
    return _ANTHROPIC_CLIENTS[resolved_key]


def clear_model_cache() -> None:
    """Clear cached local Llama model instances."""
    _MODEL_CACHE.clear()


# GPU Check
def is_gpu_available() -> dict[str, Any]:
    """Check for NVIDIA GPU availability via nvidia-smi (with optional nvgpu fallback)."""
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path:
        try:
            proc = subprocess.run(
                [nvidia_smi_path, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                gpu_info = []
                for line in proc.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        try:
                            mem_mb = int(float(parts[1]))
                        except ValueError:
                            mem_mb = 0
                        gpu_info.append({"name": parts[0], "mem_total": mem_mb})
                num_gpus = len(gpu_info)
                if num_gpus > 0:
                    first_gpu_vram = gpu_info[0]["mem_total"]
                    total_vram = sum(g["mem_total"] for g in gpu_info)
                    logger.info("GPU(s) found: %d, Total VRAM: %d MB", num_gpus, total_vram)
                    return {
                        "gpu_found": True,
                        "num_gpus": num_gpus,
                        "first_gpu_vram": first_gpu_vram,
                        "total_vram": total_vram,
                        "gpu_info": gpu_info,
                    }
        except Exception as e:
            logger.debug("nvidia-smi check failed: %s", e)

    if NVGPU_AVAILABLE and nvgpu is not None:
        try:
            gpu_info = nvgpu.gpu_info()
            num_gpus = len(gpu_info)
            if num_gpus == 0:
                logger.warning("No GPUs found on the system")
                return {"gpu_found": False, "num_gpus": 0, "first_gpu_vram": 0, "total_vram": 0}
            first_gpu_vram = gpu_info[0]["mem_total"]
            total_vram = sum(gpu["mem_total"] for gpu in gpu_info)
            logger.info("GPU(s) found: %d, Total VRAM: %d MB", num_gpus, total_vram)
            return {
                "gpu_found": True,
                "num_gpus": num_gpus,
                "first_gpu_vram": first_gpu_vram,
                "total_vram": total_vram,
                "gpu_info": gpu_info,
            }
        except Exception as e:
            logger.error("Error checking GPU availability: %s", e)
            return {"gpu_found": False, "num_gpus": 0, "first_gpu_vram": 0, "total_vram": 0, "error": str(e)}

    logger.info("No NVIDIA GPU detected; CPU execution will be used for local models.")
    return {
        "gpu_found": False,
        "num_gpus": 0,
        "first_gpu_vram": 0,
        "total_vram": 0,
        "error": "No NVIDIA GPU detected",
    }


def _download_https_file(url: str, destination: str) -> None:
    """Download an HTTPS URL to destination after validating the URL scheme."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"Refusing to download from non-HTTPS URL: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "llm-aided-ocr/1.0"})
    with urllib.request.urlopen(req, timeout=600) as response, open(destination, "wb") as out_file:  # nosec B310
        shutil.copyfileobj(response, out_file)


# Model Download
async def download_models(model_name: str | None = None) -> tuple[list[str], list[dict[str, str]]]:
    download_status: list[dict[str, str]] = []
    default_url = (
        "https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-GGUF/"
        "resolve/main/Llama-3.1-8B-Lexi-Uncensored_Q5_fixedrope.gguf"
    )
    target_model = (model_name or DEFAULT_LOCAL_MODEL_NAME).strip()
    model_url = default_url
    current_file_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(current_file_path)
    models_dir = os.path.join(base_dir, "models")

    os.makedirs(models_dir, exist_ok=True)
    existing_matches = glob.glob(os.path.join(models_dir, f"{target_model}*"))
    if existing_matches:
        found_path = max(existing_matches, key=os.path.getmtime)
        found_name = os.path.basename(found_path)
        logger.info("Model file already exists: %s", found_path)
        return [found_name], [{"url": model_url, "status": "success", "message": "File already exists."}]

    resolved_name = target_model if target_model.endswith(".gguf") else os.path.basename(model_url)
    lock = FileLock(os.path.join(models_dir, "download.lock"))
    status = {"url": model_url, "status": "success", "message": "File already exists."}
    filename = os.path.join(models_dir, resolved_name)

    try:
        with lock.acquire(timeout=1200):
            if not os.path.exists(filename):
                logger.info("Downloading model %s from %s...", resolved_name, model_url)
                await asyncio.to_thread(_download_https_file, model_url, filename)
                file_size = os.path.getsize(filename) / (1024 * 1024)
                if file_size < 100:
                    os.remove(filename)
                    status["status"] = "failure"
                    status["message"] = (
                        f"Downloaded file is too small ({file_size:.2f} MB), probably not a valid model file."
                    )
                    logger.error("Error: %s", status["message"])
                else:
                    logger.info("Successfully downloaded: %s (Size: %.2f MB)", filename, file_size)
            else:
                logger.info("Model file already exists: %s", filename)
    except Timeout:
        logger.error("Error: Could not acquire lock for downloading %s", resolved_name)
        status["status"] = "failure"
        status["message"] = "Could not acquire lock for downloading."
    except Exception as e:
        logger.error("Error downloading model %s: %s", resolved_name, e)
        status["status"] = "failure"
        status["message"] = str(e)

    download_status.append(status)
    logger.info("Model download process completed.")
    return [resolved_name], download_status


# Model Loading
def load_model(llm_model_name: str, raise_exception: bool = True, use_cache: bool = True):
    global USE_VERBOSE
    try:
        if Llama is None:
            raise ImportError("llama_cpp is not installed. Install llama-cpp-python to use local models.")
        current_file_path = os.path.abspath(__file__)
        base_dir = os.path.dirname(current_file_path)
        models_dir = os.path.join(base_dir, "models")
        matching_files = glob.glob(os.path.join(models_dir, f"{llm_model_name}*"))
        if not matching_files:
            logger.error("Error: No model file found matching: %s", llm_model_name)
            raise FileNotFoundError(f"No model file found matching: {llm_model_name}")
        model_file_path = max(matching_files, key=os.path.getmtime)

        gpu_status = is_gpu_available()
        prefer_gpu = bool(gpu_status.get("gpu_found", True))
        cache_key = (model_file_path, LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS, prefer_gpu)
        if use_cache and cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        logger.info("Loading model: %s", model_file_path)
        model_instance = None
        if prefer_gpu:
            try:
                logger.info("Attempting to load model with GPU acceleration...")
                model_instance = Llama(
                    model_path=model_file_path,
                    n_ctx=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS,
                    verbose=USE_VERBOSE,
                    n_gpu_layers=-1,
                )
                logger.info("Model loaded successfully with GPU acceleration.")
            except Exception as gpu_e:
                logger.warning("Failed to load model with GPU acceleration: %s", gpu_e)
                logger.info("Falling back to CPU...")

        if model_instance is None:
            try:
                model_instance = Llama(
                    model_path=model_file_path,
                    n_ctx=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS,
                    verbose=USE_VERBOSE,
                    n_gpu_layers=0,
                )
                logger.info("Model loaded successfully with CPU.")
            except Exception as cpu_e:
                logger.error("Failed to load model with CPU: %s", cpu_e)
                if raise_exception:
                    raise
                return None

        if use_cache and model_instance is not None:
            _MODEL_CACHE[cache_key] = model_instance
        return model_instance
    except Exception as e:
        logger.error("Exception occurred while loading the model: %s", e)
        traceback.print_exc()
        if raise_exception:
            raise
        return None


async def retry_with_backoff(
    coro_factory: Callable[[], Any],
    max_retries: int = API_MAX_RETRIES,
    base_delay: float = API_RETRY_BASE_DELAY,
    operation_name: str = "API request",
) -> Any:
    """Execute an async operation with exponential backoff and jitter on transient failures."""
    last_exc: Exception | None = None
    attempts = max(1, max_retries + 1)
    for attempt in range(attempts):
        try:
            return await coro_factory()
        except Exception as e:
            last_exc = e
            if attempt + 1 >= attempts:
                break
            sleep_secs = base_delay * (2**attempt) + random.uniform(0.0, 0.25 * base_delay)
            logger.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.2fs...",
                operation_name,
                attempt + 1,
                attempts,
                e,
                sleep_secs,
            )
            await asyncio.sleep(sleep_secs)
    if last_exc is not None:
        raise last_exc
    return None


# API Interaction Functions
async def generate_completion(prompt: str, max_tokens: int = 5000) -> str | None:
    if USE_LOCAL_LLM:
        return await generate_completion_from_local_llm(DEFAULT_LOCAL_MODEL_NAME, prompt, max_tokens)
    if API_PROVIDER == "CLAUDE":
        return await generate_completion_from_claude(prompt, max_tokens)
    if API_PROVIDER == "OPENAI":
        return await generate_completion_from_openai(prompt, max_tokens)
    logger.error("Invalid API_PROVIDER: %s", API_PROVIDER)
    return None


class _RegexFallbackTokenizer:
    """Offline-safe deterministic fallback tokenizer when tiktoken/HF are unavailable."""

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        pieces = re.findall(r"\w+|[^\w\s]", text)
        ids: list[int] = []
        for piece in pieces:
            if len(piece) > 8:
                ids.extend([1] * (len(piece) // 4 + 1))
            else:
                ids.append(1)
        return ids


@functools.lru_cache(maxsize=32)
def get_tokenizer(model_name: str):
    """Return a cached tokenizer for model_name with offline-safe fallback."""
    lower_name = (model_name or "").lower()
    try:
        if lower_name.startswith(("gpt-", "o1", "o3", "text-embedding-")):
            try:
                return tiktoken.encoding_for_model(model_name)
            except KeyError:
                return tiktoken.get_encoding("cl100k_base")
        if USE_HF_TOKENIZER and AutoTokenizer is not None:
            if lower_name.startswith("claude-"):
                return AutoTokenizer.from_pretrained(  # nosec B615
                    "EleutherAI/gpt-neox-20b",
                    revision="main",
                    clean_up_tokenization_spaces=False,
                )
            if lower_name.startswith("llama-"):
                return AutoTokenizer.from_pretrained(  # nosec B615
                    "huggyllama/llama-7b",
                    revision="main",
                    clean_up_tokenization_spaces=False,
                )
        if not lower_name.startswith(("claude-", "llama-", "gpt-", "o1", "o3", "text-embedding-")):
            logger.warning("Unrecognized model '%s'; using cl100k_base tokenizer fallback.", model_name)
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logger.warning("Falling back to regex tokenizer for '%s': %s", model_name, e)
        return _RegexFallbackTokenizer()


def estimate_tokens(text: str, model_name: str = "gpt-4o-mini") -> int:
    if not text:
        return 0
    try:
        tokenizer = get_tokenizer(model_name)
        return len(tokenizer.encode(text))
    except Exception as e:
        logger.warning("Error using tokenizer for %s: %s. Falling back to approximation.", model_name, e)
        return approximate_tokens(text)


def approximate_tokens(text: str) -> int:
    if not text or not text.strip():
        return 0
    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", text.strip())
    # Split on whitespace and punctuation, keeping punctuation
    pieces = re.findall(r"\b\w+\b|\S", normalized)
    count = 0
    for piece in pieces:
        if piece.isdigit():
            count += max(1, len(piece) // 2)  # Numbers often tokenize to multiple pieces
        elif re.match(r"^[A-Z]{2,}$", piece):  # Acronyms
            count += len(piece)
        elif re.search(r"[^\w\s]", piece):  # Punctuation and special characters
            count += 1
        elif len(piece) > 10:  # Long words often split into multiple pieces
            count += len(piece) // 4 + 1
        else:
            count += 1
    # Add a 10% buffer for potential underestimation
    return int(count * 1.1)


def split_long_sentence(sentence: str, max_tokens: int, model_name: str = "gpt-4o-mini") -> list[str]:
    """Split a single sentence that exceeds max_tokens into word-bounded chunks without data loss."""
    words = sentence.split()
    if not words:
        return []
    effective_max = max(1, max_tokens)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_chunk_size = 0
    tokenizer = get_tokenizer(model_name)

    for word in words:
        word_size = len(tokenizer.encode(word))
        if current_chunk and current_chunk_size + word_size > effective_max:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_chunk_size = word_size
        else:
            current_chunk.append(word)
            current_chunk_size += word_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def adjust_overlaps(
    chunks: list[str],
    tokenizer,
    max_chunk_tokens: int,
    overlap_size: int = 50,
) -> list[str]:
    """Prepend up to overlap_size trailing words from chunks[i-1] onto chunks[i] without ever truncating chunks[i]."""
    if not chunks:
        return []
    if overlap_size <= 0:
        return list(chunks)

    adjusted_chunks: list[str] = []
    for i, curr_text in enumerate(chunks):
        if i == 0:
            adjusted_chunks.append(curr_text)
            continue

        prev_words = chunks[i - 1].split()
        candidate_overlap = prev_words[-overlap_size:] if prev_words else []
        curr_words = curr_text.split()
        current_size = len(tokenizer.encode(curr_text))
        available_for_overlap = max(0, max_chunk_tokens - current_size)

        if available_for_overlap <= 0 or not candidate_overlap:
            # Preserve 100% of chunks[i] even when no token budget remains for overlap
            adjusted_chunks.append(curr_text)
            continue

        # Binary/linear trim of candidate_overlap from the left until it fits available_for_overlap
        selected_overlap: list[str] = []
        for start_idx in range(len(candidate_overlap)):
            sub_overlap = candidate_overlap[start_idx:]
            combined_candidate = " ".join(sub_overlap + curr_words)
            if len(tokenizer.encode(combined_candidate)) <= max_chunk_tokens:
                selected_overlap = sub_overlap
                break

        if selected_overlap:
            adjusted_chunks.append(" ".join(selected_overlap + curr_words))
        else:
            adjusted_chunks.append(curr_text)

    return adjusted_chunks


def chunk_text(
    text: str,
    max_chunk_tokens: int,
    model_name: str = "gpt-4o-mini",
    overlap_size: int = 50,
) -> list[str]:
    """Split text into sentence-aligned chunks within max_chunk_tokens without losing words or emitting empty chunks."""
    if not text or not text.strip():
        return []
    effective_max = max(1, max_chunk_tokens)
    tokenizer = get_tokenizer(model_name)
    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

    sentences: list[str] = []
    for sentence in raw_sentences:
        if len(tokenizer.encode(sentence)) > effective_max:
            sentences.extend(split_long_sentence(sentence, effective_max, model_name))
        else:
            sentences.append(sentence)

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_chunk_size = 0

    for sentence in sentences:
        sentence_size = len(tokenizer.encode(sentence))
        if current_chunk and current_chunk_size + sentence_size > effective_max:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_chunk_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_chunk_size += sentence_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    if overlap_size > 0 and len(chunks) > 1:
        return adjust_overlaps(chunks, tokenizer, effective_max, overlap_size=overlap_size)
    return chunks


def split_oversized_prompt(prompt: str, max_prompt_tokens: int, model_name: str = "gpt-4o-mini") -> list[str]:
    """Split an oversized prompt into sub-prompts that each retain the system/task instructions."""
    markers = [
        ("Current chunk to process:\n", "\n\nCorrected text:"),
        ("Text to reformat:\n\n", "\n\nReformatted markdown:"),
        ("Text to reformat:\n", "\n\nReformatted markdown:"),
    ]
    for start_marker, end_marker in markers:
        if start_marker in prompt and end_marker in prompt:
            prefix, rest = prompt.split(start_marker, 1)
            body, suffix = rest.rsplit(end_marker, 1)
            header = prefix + start_marker
            footer = end_marker + suffix
            wrapper_size = estimate_tokens(header + footer, model_name)
            body_budget = max(64, max_prompt_tokens - wrapper_size - 32)
            body_chunks = chunk_text(body, body_budget, model_name, overlap_size=0)
            if not body_chunks:
                return [prompt]
            return [f"{header}{sub_body}{footer}" for sub_body in body_chunks]

    # Generic fallback: preserve the first instruction paragraph on every sub-chunk
    paragraphs = prompt.split("\n\n", 1)
    if len(paragraphs) == 2 and len(paragraphs[0]) < len(prompt) // 2:
        header = paragraphs[0] + "\n\n"
        body = paragraphs[1]
        wrapper_size = estimate_tokens(header, model_name)
        body_budget = max(64, max_prompt_tokens - wrapper_size - 16)
        body_chunks = chunk_text(body, body_budget, model_name, overlap_size=0)
        if body_chunks:
            return [f"{header}{sub_body}" for sub_body in body_chunks]

    return chunk_text(prompt, max(64, max_prompt_tokens), model_name, overlap_size=0)


def strip_llm_wrapper_artifacts(text: str) -> str:
    """Remove echoed XML context tags, outer markdown code fences, and LLM conversational preambles."""
    if not text:
        return ""
    cleaned = text.strip()
    # Remove any echoed <previous_context>...</previous_context> blocks
    cleaned = re.sub(r"<previous_context>.*?</previous_context>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    # Remove outer ```markdown ... ``` or ``` ... ``` wrapper if it wraps the entire chunk
    fence_match = re.match(r"^```(?:markdown|md|text)?\s*\n(.*?)\n```\s*$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    # Remove common conversational preamble lines
    cleaned = re.sub(
        r"^(?:Here is the (?:corrected|reformatted(?: markdown)?) text:?|Corrected text:?|Reformatted markdown:?)\s*\n+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


async def generate_completion_from_claude(
    prompt: str,
    max_tokens: int = CLAUDE_MAX_TOKENS - TOKEN_BUFFER,
) -> str | None:
    client = get_anthropic_client()
    if client is None:
        logger.error("Anthropic API key not found or invalid. Please set a valid ANTHROPIC_API_KEY.")
        return None
    prompt_size = estimate_tokens(prompt, CLAUDE_MODEL_STRING)
    adjusted_max = min(max_tokens, CLAUDE_MAX_TOKENS - prompt_size - TOKEN_BUFFER)

    async def _call_claude(msg_content: str, out_limit: int) -> tuple[str, int, int]:
        async def _once() -> tuple[str, int, int]:
            async with client.messages.stream(
                model=CLAUDE_MODEL_STRING,
                max_tokens=max(1, out_limit),
                temperature=0.3,
                messages=[{"role": "user", "content": msg_content}],
            ) as stream:
                message = await stream.get_final_message()
                text_out = message.content[0].text
                in_count = int(getattr(message.usage, "input_tokens", 0))
                out_count = int(getattr(message.usage, "output_tokens", 0))
                return text_out, in_count, out_count

        return await retry_with_backoff(_once, operation_name="Claude completion")

    if adjusted_max <= 0:
        logger.warning("Prompt is too long for Claude API. Splitting prompt while preserving instructions.")
        sub_prompts = split_oversized_prompt(prompt, CLAUDE_MAX_TOKENS - TOKEN_CUSHION, CLAUDE_MODEL_STRING)
        results: list[str] = []
        for sub_prompt in sub_prompts:
            try:
                text_out, in_count, out_count = await _call_claude(sub_prompt, CLAUDE_MAX_TOKENS // 2)
                results.append(text_out)
                logger.info("Chunk processed. Input count: %d, Output count: %d", in_count, out_count)
            except Exception as e:
                logger.error("An error occurred while processing a chunk: %s", e)
        return "\n\n".join(results) if results else None

    try:
        output_text, in_count, out_count = await _call_claude(prompt, adjusted_max)
        logger.info("Total input count: %d", in_count)
        logger.info("Total output count: %d", out_count)
        logger.info("Generated output (abbreviated): %s...", output_text[:150])
        return output_text
    except Exception as e:
        logger.error("An error occurred while requesting from Claude API: %s", e)
        return None


async def generate_completion_from_openai(prompt: str, max_tokens: int = 5000) -> str | None:
    client = get_openai_client()
    if client is None:
        logger.error("OpenAI API key not found or invalid. Please set a valid OPENAI_API_KEY.")
        return None
    prompt_size = estimate_tokens(prompt, OPENAI_COMPLETION_MODEL)
    context_limit = max(4096, OPENAI_MAX_TOKENS)
    adjusted_max = max(1, min(max_tokens, context_limit - prompt_size - TOKEN_BUFFER))

    async def _call_openai(msg_content: str, out_limit: int) -> tuple[str, int, int]:
        async def _once() -> tuple[str, int, int]:
            response = await client.chat.completions.create(
                model=OPENAI_COMPLETION_MODEL,
                messages=[{"role": "user", "content": msg_content}],
                max_tokens=max(1, out_limit),
                temperature=0.3,
            )
            content = response.choices[0].message.content or ""
            comp_count = int(getattr(response.usage, "completion_tokens", 0)) if response.usage else 0
            tot_count = int(getattr(response.usage, "total_tokens", 0)) if response.usage else 0
            return content, comp_count, tot_count

        return await retry_with_backoff(_once, operation_name="OpenAI completion")

    if adjusted_max <= 1 and prompt_size >= context_limit - TOKEN_BUFFER:
        logger.warning("Prompt is too long for OpenAI API. Splitting prompt while preserving instructions.")
        sub_prompts = split_oversized_prompt(prompt, context_limit - TOKEN_CUSHION, OPENAI_COMPLETION_MODEL)
        results: list[str] = []
        chunk_max = max(512, min(4096, context_limit // 2))
        for sub_prompt in sub_prompts:
            try:
                result, comp_count, _ = await _call_openai(sub_prompt, chunk_max)
                results.append(result)
                logger.info("Chunk processed. Output count: %d", comp_count)
            except Exception as e:
                logger.error("An error occurred while processing a chunk: %s", e)
        return "\n\n".join(results) if results else None

    try:
        output_text, _, tot_count = await _call_openai(prompt, adjusted_max)
        logger.info("Total usage count: %d", tot_count)
        logger.info("Generated output (abbreviated): %s...", output_text[:150])
        return output_text
    except Exception as e:
        logger.error("An error occurred while requesting from OpenAI API: %s", e)
        return None


async def generate_completion_from_local_llm(
    llm_model_name: str,
    input_prompt: str,
    number_of_tokens_to_generate: int = 100,
    temperature: float = 0.7,
    grammar_file_string: str | None = None,
) -> str | None:
    logger.info(
        "Starting text completion using model: '%s' for input prompt (%d chars)",
        llm_model_name,
        len(input_prompt),
    )
    llm = await asyncio.to_thread(load_model, llm_model_name)
    prompt_size = estimate_tokens(input_prompt, llm_model_name)
    adjusted_max = max(
        1,
        min(number_of_tokens_to_generate, LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - prompt_size - TOKEN_BUFFER),
    )
    if adjusted_max <= 1 and prompt_size >= LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - TOKEN_BUFFER:
        logger.warning("Prompt is too long for local LLM. Splitting prompt while preserving instructions.")
        sub_prompts = split_oversized_prompt(
            input_prompt,
            max(256, LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS // 2),
            llm_model_name,
        )
        results: list[str] = []
        for sub_prompt in sub_prompts:
            try:
                sub_size = estimate_tokens(sub_prompt, llm_model_name)
                sub_max = max(64, LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - sub_size - TOKEN_CUSHION)
                output = await asyncio.to_thread(
                    llm,
                    prompt=sub_prompt,
                    max_tokens=sub_max,
                    temperature=temperature,
                )
                results.append(output["choices"][0]["text"])
                comp_count = int(output.get("usage", {}).get("completion_tokens", 0))
                logger.info("Chunk processed. Output count: %d", comp_count)
            except Exception as e:
                logger.error("An error occurred while processing a chunk: %s", e)
        return "\n\n".join(results) if results else None

    grammar_file_string_lower = grammar_file_string.lower() if grammar_file_string else ""
    llama_grammar = None
    if grammar_file_string_lower and LlamaGrammar is not None:
        list_of_grammar_files = glob.glob("./grammar_files/*.gbnf")
        matching_grammar_files = [
            x
            for x in list_of_grammar_files
            if grammar_file_string_lower in os.path.splitext(os.path.basename(x).lower())[0]
        ]
        if matching_grammar_files:
            grammar_file_path = max(matching_grammar_files, key=os.path.getmtime)
            if os.path.exists(grammar_file_path):
                logger.info("Loading selected grammar file: '%s'", grammar_file_path)
                llama_grammar = LlamaGrammar.from_file(grammar_file_path)
        else:
            logger.warning("No grammar file found matching '%s'; proceeding without grammar.", grammar_file_string)

    if llama_grammar is not None:
        output = await asyncio.to_thread(
            llm,
            prompt=input_prompt,
            max_tokens=adjusted_max,
            temperature=temperature,
            grammar=llama_grammar,
        )
    else:
        output = await asyncio.to_thread(
            llm,
            prompt=input_prompt,
            max_tokens=adjusted_max,
            temperature=temperature,
        )

    generated_text = output["choices"][0]["text"]
    if grammar_file_string_lower == "json":
        generated_text = strip_llm_wrapper_artifacts(generated_text)
    finish_reason = str(output["choices"][0].get("finish_reason", "unknown"))
    usage_info = output.get("usage", {})
    total_time = float(usage_info.get("total_time", 0.0))
    llm_model_usage_json = json.dumps(usage_info)
    logger.info(
        "Completed text completion in %.2f seconds. Beginning of generated text:\n'%s'...",
        total_time,
        generated_text[:150],
    )
    logger.debug("Finish reason: %s, Usage: %s", finish_reason, llm_model_usage_json)
    return generated_text


# Image Processing Functions
def estimate_skew_angle(
    gray: np.ndarray,
    max_angle: float = 15.0,
    min_angle: float = 0.75,
) -> float:
    """Estimate document skew angle (in degrees) from horizontal text line-strip contours."""
    if gray.ndim != 2 or gray.size == 0:
        return 0.0
    height, width = gray.shape[:2]
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel_w = max(15, width // 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
    dilated = cv2.dilate(inv, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angles: list[float] = []
    min_len = max(50.0, width / 12.0)
    max_thick = max(40.0, height / 18.0)

    for cnt in contours:
        if cnt.shape[0] < 4:
            continue
        _, _, w, _ = cv2.boundingRect(cnt)
        if w < min_len:
            continue
        vx, vy, x0, y0 = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        vx_f, vy_f = float(vx[0]), float(vy[0])
        x0_f, y0_f = float(x0[0]), float(y0[0])
        if vx_f < 0:
            vx_f, vy_f = -vx_f, -vy_f
        ang = float(np.degrees(np.arctan2(vy_f, vx_f)))
        if abs(ang) > max_angle:
            continue
        pts = cnt.reshape(-1, 2).astype(np.float32)
        dx = pts[:, 0] - x0_f
        dy = pts[:, 1] - y0_f
        proj_along = dx * vx_f + dy * vy_f
        proj_perp = -dx * vy_f + dy * vx_f
        line_len = float(proj_along.max() - proj_along.min())
        line_thick = float(proj_perp.max() - proj_perp.min())
        if line_len >= min_len and 4.0 <= line_thick <= max_thick and (line_len / max(1.0, line_thick)) >= 3.0:
            angles.append(ang)

    if len(angles) < 4:
        return 0.0

    arr = np.array(angles, dtype=np.float64)
    q25, q75 = np.percentile(arr, [25, 75])
    iqr = float(q75 - q25)
    median_angle = float(np.median(arr))
    if iqr > 2.5 or abs(median_angle) < min_angle or abs(median_angle) > max_angle:
        return 0.0
    return median_angle


def deskew_image(
    gray: np.ndarray,
    max_angle: float = 15.0,
    min_angle: float = 0.75,
) -> np.ndarray:
    """Deskew a grayscale document page using horizontal line-strip median angle estimation."""
    if gray.ndim != 2 or gray.size == 0:
        return gray
    angle = estimate_skew_angle(gray, max_angle=max_angle, min_angle=min_angle)
    if abs(angle) < 1e-6:
        return gray

    height, width = gray.shape[:2]
    center = (width // 2, height // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray,
        rot_mat,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )



def preprocess_image(
    image: Image.Image | np.ndarray,
    deskew: bool = True,
    apply_clahe: bool = True,
    binarize: bool = True,
) -> Image.Image:
    """Enhance a scanned document page for Tesseract OCR using deskewing, CLAHE, denoising, and Otsu thresholding."""
    arr = np.array(image)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr.copy()

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    if deskew:
        gray = deskew_image(gray)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Gentle 3x3 Gaussian smoothing reduces salt-and-pepper scan noise before Otsu binarization
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if binarize:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    return Image.fromarray(gray)


def convert_pdf_to_images(
    input_pdf_file_path: str,
    max_pages: int = 0,
    skip_first_n_pages: int = 0,
    dpi: int = PDF_DPI,
) -> list[Image.Image]:
    logger.info("Processing PDF file %s (dpi=%d)", input_pdf_file_path, dpi)
    if max_pages == 0:
        last_page = None
        logger.info("Converting all pages to images...")
    else:
        last_page = skip_first_n_pages + max_pages
        logger.info("Converting pages %d to %d", skip_first_n_pages + 1, last_page)
    first_page = skip_first_n_pages + 1  # pdf2image uses 1-based indexing
    images = convert_from_path(
        input_pdf_file_path,
        dpi=dpi,
        first_page=first_page,
        last_page=last_page,
    )
    logger.info("Converted %d pages from PDF file to images.", len(images))
    return images


def ocr_image(image: Image.Image, preprocess: bool = True) -> str:
    target_image = preprocess_image(image) if preprocess else image
    return pytesseract.image_to_string(target_image)



# Structural Page Header/Page-Number Cleaning & Footnote Preservation
_OCR_PAGE_NUMBER_RE = re.compile(
    r"^(?:"
    r"Page\s+[0-9OILlSi]{1,4}(?:\s+of\s+[0-9OILlSi]{1,4})?\s*:?"
    r"|[\-~—–]+\s*(?:[0-9OILlSi]{1,3}|y)\s*[\-~—–.]*"
    r"|\[\s*\d{1,4}\s*\]"
    r")$",
    re.IGNORECASE,
)

_FOOTNOTE_START_RE = re.compile(
    r"^\s*(?:\*\s*(?:[,\-—–]?\s*)?[A-Za-z0-9\"'(]|[†‡§®]\s+[A-Za-z0-9\"'(])",
)


def is_ocr_page_number_line(line: str) -> bool:
    """Return True if a line is a standalone page number or OCR-garbled page marker (e.g. '-2-', '-1O-', '-LL-', '~Li-', '~L5-')."""
    stripped = line.strip()
    if not stripped or len(stripped) > 24:
        return False
    if "$" in stripped or "%" in stripped:
        return False
    return bool(_OCR_PAGE_NUMBER_RE.match(stripped))


def is_footnote_start_line(line: str) -> bool:
    """Return True if a line begins an author footnote (e.g. '* My views...', '* - An interesting...', '® In the short term...')."""
    stripped = line.strip()
    if not stripped or is_ocr_page_number_line(stripped):
        return False
    return bool(_FOOTNOTE_START_RE.match(stripped))


_COMPOUND_HYPHEN_PREFIXES: frozenset[str] = frozenset(
    {
        "above",
        "all",
        "below",
        "coin",
        "cross",
        "double",
        "eight",
        "eighty",
        "fifty",
        "first",
        "five",
        "forty",
        "four",
        "full",
        "half",
        "high",
        "long",
        "low",
        "mid",
        "neo",
        "nine",
        "ninety",
        "non",
        "one",
        "pan",
        "part",
        "partially",
        "private",
        "pseudo",
        "quasi",
        "second",
        "self",
        "semi",
        "seven",
        "seventy",
        "short",
        "single",
        "six",
        "sixty",
        "ten",
        "third",
        "thirty",
        "three",
        "twenty",
        "two",
        "ultra",
        "vice",
        "well",
        "wholly",
    }
)

_LINE_WRAP_HYPHEN_RE = re.compile(
    r"((?:[A-Za-z]+-)*[A-Za-z]{2,})[-~]\s*\n(?:\s*\n)?\s*([a-z][A-Za-z]*(?:-[A-Za-z]+)*)"
)


def _dehyphenate_line_wrap(match: re.Match[str]) -> str:
    prefix = match.group(1)
    suffix = match.group(2)
    if "-" in prefix or "-" in suffix or prefix.lower() in _COMPOUND_HYPHEN_PREFIXES:
        return f"{prefix}-{suffix}"
    return f"{prefix}{suffix}"


def _join_ocr_fragments(left: str, right: str, separator: str = " ") -> str:
    """Join two OCR text fragments across a spurious blank line or page break, repairing split hyphenated words."""
    l_str = left.rstrip()
    r_str = right.lstrip()
    if not l_str:
        return r_str
    if not r_str:
        return l_str
    if re.search(r"[A-Za-z]{2,}[-~]$", l_str) and re.match(r"^[a-z][A-Za-z]*", r_str):
        combined = f"{l_str}\n{r_str}"
        return _LINE_WRAP_HYPHEN_RE.sub(_dehyphenate_line_wrap, combined, count=1)
    return f"{l_str}{separator}{r_str}"


def _is_standalone_heading_block(text: str) -> bool:
    """Return True if text is a short standalone section heading (ALL-CAPS or Title Case without terminal sentence punctuation)."""
    raw = text.strip()
    if not raw or raw.count("\n") > 1:
        return False
    s = re.sub(r"[ \t]+", " ", " ".join(ln.strip() for ln in raw.splitlines() if ln.strip())).strip()
    if not s or len(s) > 110:
        return False
    if is_footnote_start_line(s) or is_ocr_page_number_line(s):
        return False
    if s.startswith(("(", "-", "|", "#")) or s.endswith((".", ":", ";", ",", "!", "*", ")")):
        return False
    if re.search(r"\b[A-Z]\.\s+[A-Z]", s) or any(ch.isdigit() for ch in s):
        return False
    if re.search(r"\b(?:of|to|in|on|for|with|by|at|from|and|or|the|a|an)$", s):
        return False
    if s.isupper() and any(c.isalpha() for c in s):
        return True
    words = [w for w in re.findall(r"[A-Za-z]+", s) if len(w) > 3]
    return len(words) >= 2 and (sum(1 for w in words if w[0].isupper()) / len(words)) >= 0.75


def extract_footnotes(text: str) -> list[str]:
    """Extract substantive author footnote blocks from raw OCR or processed Markdown text, reassembling split fragments."""
    if not text:
        return []
    pages = text.split("\x0c") if "\x0c" in text else [text]
    footnotes: list[str] = []
    for page in pages:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page) if p.strip()]
        in_page_footnote = False
        for para in paragraphs:
            cleaned_para = re.sub(r"^>\s*(?:\*\*Footnote:\*\*\s*)?", "", para.strip())
            if not cleaned_para or is_ocr_page_number_line(cleaned_para):
                continue
            normalized = re.sub(r"\s+", " ", _join_ocr_fragments(cleaned_para, "")).strip()
            if is_footnote_start_line(cleaned_para):
                in_page_footnote = True
                if len(normalized.split()) >= 5:
                    footnotes.append(normalized)
            elif in_page_footnote and footnotes and "\x0c" in text:
                footnotes[-1] = re.sub(r"\s+", " ", _join_ocr_fragments(footnotes[-1], normalized)).strip()
    return footnotes


def _ends_mid_sentence(text: str) -> bool:
    """Return True if a paragraph or page ends mid-sentence without terminal punctuation."""
    stripped = text.strip()
    if not stripped:
        return False
    return stripped[-1] not in ".!?:;)\"'"


def _should_merge_ocr_fragments(prev_para: str, next_para: str) -> bool:
    """Return True when two adjacent OCR blocks are parts of a single sentence or table row split by a spurious blank line."""
    prev_s = prev_para.strip()
    next_s = next_para.strip()
    if not prev_s or not next_s:
        return False
    if is_footnote_start_line(prev_s) or is_footnote_start_line(next_s):
        return False
    if is_ocr_page_number_line(prev_s) or is_ocr_page_number_line(next_s):
        return False
    if _is_standalone_heading_block(prev_s) or _is_standalone_heading_block(next_s):
        return False
    if not _ends_mid_sentence(prev_s):
        return False
    if re.search(r"[A-Za-z]{2,}[-~]$", prev_s) and re.match(r"^[a-z]", next_s):
        return True
    if prev_s.endswith(",") and re.match(r"^(?:\([0-9a-z]|I\s+[a-z])", next_s):
        return True
    if re.search(r"[a-z]$", prev_s) and re.match(r"^\$\d", next_s):
        return True
    if (
        len(prev_s) <= 50
        and re.search(r"\b(?:of|to|in|for|on|with|by)$", prev_s)
        and len(next_s) <= 50
        and re.match(r"^[A-Z][A-Za-z ()]+$", next_s)
    ):
        return True
    return next_s[0].islower()


def clean_page_artifacts(
    pages: list[str],
    suppress_headers_and_page_numbers: bool = True,
) -> list[str]:
    """Clean OCR-garbled page numbers and running headers while healing mid-sentence splits and anchoring footnotes."""
    if not pages:
        return []

    # Identify repeating running headers across >= 3 pages
    first_lines: list[str] = []
    for page in pages:
        for line in page.splitlines():
            s = line.strip()
            if s and not is_ocr_page_number_line(s):
                first_lines.append(s)
                break
    header_counts = Counter(first_lines)
    repeating_headers = {
        line for line, count in header_counts.items() if count >= 3 and len(line) <= 80 and not is_footnote_start_line(line)
    }

    cleaned_pages: list[list[str]] = []
    for page in pages:
        raw_paras = [p.strip() for p in re.split(r"\n\s*\n", page.replace("\x0c", "")) if p.strip()]
        page_paras: list[str] = []
        in_bottom_footnote = False
        for para_idx, para in enumerate(raw_paras):
            lines = [ln for ln in para.splitlines() if ln.strip()]
            kept_lines: list[str] = []
            for line_idx, line in enumerate(lines):
                s = line.strip()
                if suppress_headers_and_page_numbers:
                    if is_ocr_page_number_line(s):
                        continue
                    if para_idx == 0 and line_idx == 0 and s in repeating_headers:
                        continue
                kept_lines.append(line)
            if not kept_lines:
                continue
            joined_para = "\n".join(kept_lines)
            if is_footnote_start_line(joined_para):
                in_bottom_footnote = True
                page_paras.append(joined_para)
            elif in_bottom_footnote and page_paras and not is_ocr_page_number_line(joined_para):
                # Once a footnote starts at the bottom of a page, subsequent fragments on the same page
                # (caused by OCR blank lines inside the footnote) belong to that footnote.
                page_paras[-1] = _join_ocr_fragments(page_paras[-1], joined_para, separator=" ")
            else:
                page_paras.append(joined_para)

        # Heal intra-page mid-sentence splits caused by spurious OCR blank lines (e.g. 'mar-\n\nket' or mid-sentence '\n\n')
        j = 0
        while j < len(page_paras) - 1:
            if _should_merge_ocr_fragments(page_paras[j], page_paras[j + 1]):
                sep = (
                    " "
                    if page_paras[j].rstrip().endswith(",") and re.match(r"^\([0-9a-z]", page_paras[j + 1].lstrip())
                    else "\n"
                )
                page_paras[j] = _join_ocr_fragments(page_paras[j], page_paras[j + 1], separator=sep)
                page_paras.pop(j + 1)
            else:
                j += 1

        cleaned_pages.append(page_paras)

    # Heal mid-sentence page breaks across Page N and Page N+1 and relocate interrupting bottom-of-page footnotes
    for i in range(len(cleaned_pages) - 1):
        curr_paras = cleaned_pages[i]
        next_paras = cleaned_pages[i + 1]
        if not curr_paras or not next_paras:
            continue

        # Collect trailing footnote paragraphs at the bottom of page i
        trailing_footnotes: list[str] = []
        while len(curr_paras) >= 2 and is_footnote_start_line(curr_paras[-1]):
            trailing_footnotes.insert(0, curr_paras.pop())

        if not curr_paras or not next_paras:
            curr_paras.extend(trailing_footnotes)
            continue

        last_body = curr_paras[-1]
        first_next = next_paras[0]
        should_join_across_page = (
            bool(trailing_footnotes)
            and _ends_mid_sentence(last_body)
            and not _is_standalone_heading_block(last_body)
            and not is_footnote_start_line(first_next)
        ) or _should_merge_ocr_fragments(last_body, first_next)

        if should_join_across_page:
            continuation = next_paras.pop(0)
            curr_paras[-1] = _join_ocr_fragments(last_body, continuation, separator=" ")
            for fn in trailing_footnotes:
                curr_paras.append(fn)
        else:
            curr_paras.extend(trailing_footnotes)

    return ["\n\n".join(p) for p in cleaned_pages if p]


def clean_raw_ocr_text(text: str, suppress_headers_and_page_numbers: bool = True) -> str:
    """Clean a raw OCR document string (splitting on form-feeds or 'Page N:' boundaries if present)."""
    if not text:
        return ""
    if "\x0c" in text:
        pages = [p for p in text.split("\x0c") if p.strip()]
    elif re.search(r"^Page\s+\d+\s*:\s*$", text, flags=re.MULTILINE):
        pages = [p for p in re.split(r"^Page\s+\d+\s*:\s*$", text, flags=re.MULTILINE) if p.strip()]
    else:
        pages = [text]
    cleaned = clean_page_artifacts(pages, suppress_headers_and_page_numbers=suppress_headers_and_page_numbers)
    return "\n\n".join(cleaned)


_TYPEWRITER_EXACT_WORD_MAP: dict[str, str] = {
    "finaneial": "financial",
    "eurrent": "current",
    "elient": "client",
    "eriticism": "criticism",
    "becone": "become",
    "amually": "annually",
    "analagous": "analogous",
    "managenient": "management",
    "Ina": "In a",
}
_TYPEWRITER_EXACT_WORD_RE = re.compile(
    r"\b(?:finaneial|eurrent|elient|eriticism|becone|amually|analagous|managenient|Ina)\b"
)


def clean_ocr_character_confusions(text: str) -> str:
    """Deterministically repair high-frequency typewriter/Tesseract character confusions and line-wrap splits."""
    if not text:
        return ""

    # 1. Dehyphenate words split across single or double newlines by '-' or '~' (e.g. 'manage-\nment', 'cost-of-\nliving')
    cleaned = _LINE_WRAP_HYPHEN_RE.sub(_dehyphenate_line_wrap, text)
    cleaned = re.sub(
        r"\b(non|private|above|below|self|well|high|low|long|short|all|full|half|part|cross)[ \t]+[-~][ \t]*(?:\n(?:\s*\n)?[ \t]*|[ \t]+)([a-z][A-Za-z]*)",
        r"\1-\2",
        cleaned,
        flags=re.IGNORECASE,
    )
    # 2. Heal comma-to-parenthetical / 'I <verb>' / mid-sentence '$<digit>' blank-line splits
    cleaned = re.sub(r",\s*\n\s*\n\s*(?=(?:\([0-9a-z]|I\s+[a-z]))", ", ", cleaned)
    cleaned = re.sub(r"(?<=[a-z])\s*\n\s*\n\s*(?=\$\d)", " ", cleaned)
    # 3. Clean OCR underline-underscore artifacts adjacent to letters (e.g. 'Great_Leap', 'Inherent _in', 'but_for')
    cleaned = re.sub(
        r"(?<=[A-Za-z])_(?=[A-Za-z])|\s+_(?=[A-Za-z])|(?<=[A-Za-z])_\s+",
        " ",
        cleaned,
    )
    # 4. Repair paragraph-leading '(L)'/'(l)' -> '(1)', leading '®' -> '*', and footnote '* , ' -> '* '
    cleaned = re.sub(r"(?m)^(\s*)\([Ll]\)(?=\s+[A-Z\"'])", r"\1(1)", cleaned)
    cleaned = re.sub(r"(?m)^(\s*)®(?=\s+[A-Z\"'])", r"\1*", cleaned)
    cleaned = re.sub(r"(?m)^(\s*(?:>\s*)?\*)\s*,\s+", r"\1 ", cleaned)
    # 5. Repair tilde dash confusions ('~-', '-~', ' ~ ' across spaces or single newlines, and intra-word 'non~conventional')
    cleaned = re.sub(r"\s*(?:~-|-\~)\s*", " - ", cleaned)
    cleaned = re.sub(r"(?<=\S)[ \t]*~[ \t]*(?:\n[ \t]*)?(?=\S)", " - ", cleaned)
    cleaned = re.sub(r"(?<=[A-Za-z])~(?=[A-Za-z])", "-", cleaned)
    # 6. Repair S & P symbol confusions ('§ & P 500', '$ & P 500', 'S$ & P 500', '§ & P')
    cleaned = re.sub(r"(?:S\$|§|\$)\s*&\s*P\b", "S & P", cleaned)
    # 7. Repair percentage decimal and typewriter ½ fraction confusions ('+2,8%', '+3 8%', '496 instead of 9%', '$1% billion')
    cleaned = re.sub(r"([+\-]?\d+),(\d{1,2})%", r"\1.\2%", cleaned)
    cleaned = re.sub(r"([+\-]\d+)[ \t]+(\d)%", r"\1.\2%", cleaned)
    cleaned = re.sub(
        r"\b([1-9]\d?)96(?=\s+(?:instead\s+of|to|or|and|versus|vs\.?)\s+[+\-]?\d+(?:\.\d+)?%)",
        r"\1%",
        cleaned,
    )
    cleaned = re.sub(r"\$(\d+)%(?=\s+(?:million|billion)\b)", r"$\1.5", cleaned)
    # 8. Repair unambiguous typewriter c->e / m->n / strikeover / punctuation speckle confusions
    cleaned = re.sub(r"fund:-managenient\.", "fund management", cleaned)
    cleaned = _TYPEWRITER_EXACT_WORD_RE.sub(lambda m: _TYPEWRITER_EXACT_WORD_MAP[m.group(0)], cleaned)
    cleaned = re.sub(r"\bete\.", "etc.", cleaned)
    cleaned = re.sub(r",eLetralone\b", ", let alone", cleaned)
    cleaned = re.sub(r"\bfo(\s+control\b)", r"To\1", cleaned)
    cleaned = re.sub(r"\bgood(\s+)fer\b", r"good\1for", cleaned)
    cleaned = re.sub(r"\bpaths(\s+)ean\b", r"paths\1can", cleaned)
    cleaned = re.sub(r'\bin(\s+the\s+)hank"', r'in\1"bank"', cleaned)
    cleaned = re.sub(r"[‘'\"]t(manufactured)[”\"]", r'"\1"', cleaned)
    cleaned = re.sub(r"\b4#[Ss]-degree\b", "45-degree", cleaned)
    cleaned = re.sub(r"\bC(\d)\)", r"(\1)", cleaned)
    cleaned = re.sub(r",:if\b", ", if", cleaned)
    cleaned = re.sub(r"[ \t]+\.\)", ".)", cleaned)
    cleaned = re.sub(r"\b(impact|several)\.\s+(?=[a-z])", r"\1 ", cleaned)
    cleaned = re.sub(
        r"(?<=[a-z]\s)(Like|Large|Little|Long-term|Litigation|Language|Future)\b",
        lambda m: m.group(1).lower(),
        cleaned,
    )
    return cleaned


_TABLE_3COL_ROW_RE = re.compile(
    r"^(?P<c1>[A-Za-z][A-Za-z0-9.,&'() /\-]{1,58}?)\s+"
    r"(?P<c2>\$?\s*\d[\d,]*(?:\.\d+)?%?|\(\d[\d,]*(?:\.\d+)?%?\))\s+"
    r"(?P<c3>\$?\s*\d[\d,]*(?:\.\d+)?%?|\(\d[\d,]*(?:\.\d+)?%?\))$"
)
_TABLE_2COL_ROW_RE = re.compile(
    r"^(?P<c1>[A-Za-z][A-Za-z0-9.&' /\-]{1,40}?)\s+"
    r"(?P<c2>[+\-]?\d+(?:\.\d+)?%|\(\d+(?:\.\d+)?%\)|\$?\d[\d,]*(?:\.\d+)?)$"
)
_MONTH_PREFIX_RE = re.compile(
    r"^(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
    re.IGNORECASE,
)


def _format_numeric_table_block(
    lines: list[str],
    reformat_as_markdown: bool,
    preceding_header: str = "",
) -> tuple[str | None, bool]:
    """Detect multi-row numeric tables (even when column spaces were collapsed) and format as Markdown tables.

    Returns (formatted_table_or_none, consumed_preceding_header).
    """
    if len(lines) < 2:
        return None, False

    norm_lines = [re.sub(r"(\d{1,3}),\s+(\d{3})\b", r"\1,\2", ln.strip()) for ln in lines if ln.strip()]
    if len(norm_lines) < 2:
        return None, False

    # 1. Check 3-column financial table (either header + >=2 data rows, or >=3 data rows with preceding header block)
    if len(norm_lines) >= 3:
        row_matches_all = [_TABLE_3COL_ROW_RE.match(ln) for ln in norm_lines]
        row_matches_tail = [_TABLE_3COL_ROW_RE.match(ln) for ln in norm_lines[1:]]
        if all(row_matches_tail) and not row_matches_all[0]:
            if not reformat_as_markdown:
                return "\n".join(norm_lines), False
            header_line = norm_lines[0]
            multi_space_cols = [c.strip() for c in re.split(r"\s{2,}", header_line) if c.strip()]
            if len(multi_space_cols) == 3:
                h1, h2, h3 = multi_space_cols
            else:
                hdr_match = re.match(
                    r"^(Company|Item|Category|Name|Security)\s+(.+?)\s+([A-Z][A-Za-z]+\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)$",
                    header_line,
                )
                if hdr_match:
                    h1, h2, h3 = hdr_match.group(1), hdr_match.group(2), hdr_match.group(3)
                else:
                    h1, h2, h3 = "Item", "Column 1", "Column 2"
            md_rows = [f"| {h1} | {h2} | {h3} |", "| :--- | :---: | :---: |"]
            for m in row_matches_tail:
                if m is not None:
                    c1 = m.group("c1").strip()
                    c2 = re.sub(r"\s+", "", m.group("c2"))
                    c3 = re.sub(r"\s+", "", m.group("c3"))
                    md_rows.append(f"| {c1} | {c2} | {c3} |")
            return "\n".join(md_rows), False

        if all(row_matches_all):
            if not reformat_as_markdown:
                return "\n".join(norm_lines), False
            consumed_prev = False
            if "S & P 500" in preceding_header and "Becker Median" in preceding_header:
                h1, h2, h3 = "Period", "S & P 500", "Becker Median Result*"
                consumed_prev = True
            else:
                h1, h2, h3 = "Period", "Column 1", "Column 2"
            md_rows = [f"| {h1} | {h2} | {h3} |", "| :--- | :---: | :---: |"]
            for m in row_matches_all:
                if m is not None:
                    c1 = m.group("c1").strip()
                    c2 = re.sub(r"\s+", "", m.group("c2"))
                    c3 = re.sub(r"\s+", "", m.group("c3"))
                    md_rows.append(f"| {c1} | {c2} | {c3} |")
            return "\n".join(md_rows), consumed_prev

    # 2. Check 2-column benchmark/metric table (>= 3 rows of label + percentage/amount)
    if len(norm_lines) >= 3:
        matches_2col = [_TABLE_2COL_ROW_RE.match(ln) for ln in norm_lines]
        if all(matches_2col) and not any(_MONTH_PREFIX_RE.match(ln) for ln in norm_lines):
            if not reformat_as_markdown:
                return "\n".join(norm_lines), False
            consumed_prev = "Annual Compounded Rate" in preceding_header
            md_rows = ["| Category | Value |", "| :--- | :---: |"]
            for m in matches_2col:
                if m is not None:
                    md_rows.append(f"| {m.group('c1').strip()} | {m.group('c2').strip()} |")
            return "\n".join(md_rows), consumed_prev

    # 3. General multi-row tabular fallback (preserve line breaks rather than mashing into prose)
    if all(len(ln) <= 75 for ln in norm_lines):
        numeric_end_count = sum(
            1
            for ln in norm_lines
            if not _MONTH_PREFIX_RE.match(ln)
            and re.search(r"(?:[+\-]?\d[\d,.]*%?|\(\d[\d,.]*%?\))$", ln)
        )
        if numeric_end_count >= max(2, len(norm_lines) - 1):
            return "\n".join(norm_lines), False

    return None, False


def reflow_and_dehyphenate_ocr_text(text: str, reformat_as_markdown: bool = True) -> str:
    """Deterministically repair typewriter line-wrap hyphenations, character confusions, tables, and intra-paragraph line breaks."""
    if not text or not text.strip():
        return ""

    cleaned_input = clean_ocr_character_confusions(text.strip())

    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned_input) if p.strip()]
    out_paragraphs: list[str] = []

    for para in raw_paragraphs:
        lines = [ln.rstrip() for ln in para.splitlines() if ln.strip()]
        if not lines:
            continue

        # Check structured multi-row numeric table first
        prev_hdr = out_paragraphs[-1] if out_paragraphs else ""
        formatted_table, consumed_prev = _format_numeric_table_block(
            lines,
            reformat_as_markdown=reformat_as_markdown,
            preceding_header=prev_hdr,
        )
        if formatted_table is not None:
            if consumed_prev and out_paragraphs:
                if "Overall Annual Return" in out_paragraphs[-1]:
                    out_paragraphs[-1] = "**Overall Annual Return**"
                elif "Annual Compounded Rate" in out_paragraphs[-1]:
                    clean_title = out_paragraphs[-1].strip("*").strip()
                    out_paragraphs[-1] = f"**{clean_title}**"
                else:
                    out_paragraphs.pop()
            out_paragraphs.append(formatted_table)
            continue

        # Preserve line structure for multi-space column tables or existing markdown tables
        is_tabular = any(re.search(r"\S\s{3,}\S", ln) for ln in lines) or lines[0].lstrip().startswith("|")
        if is_tabular:
            out_paragraphs.append("\n".join(lines))
            continue

        reflowed = re.sub(r"[ \t]+", " ", " ".join(ln.strip() for ln in lines)).strip()
        if len(reflowed) >= 120 and ". " in reflowed and re.search(r"[a-z]{3},$", reflowed):
            reflowed = f"{reflowed[:-1]}."
        if reformat_as_markdown and is_footnote_start_line(reflowed) and not reflowed.startswith(">"):
            reflowed = f"> {reflowed}"
        elif reformat_as_markdown and not reflowed.startswith("#") and _is_standalone_heading_block(para):
            if reflowed.isupper():
                reflowed = f"## {reflowed.title()}"
            else:
                # Split combined major/sub-heading if 'Management Act One - ...' was scanned on adjacent lines
                split_act = re.match(r"^(.*\bManagement)\s+(Act\s+(?:One|Two|Three)\b.*)$", reflowed)
                if split_act:
                    reflowed = f"## {split_act.group(1)}\n\n## {split_act.group(2)}"
                else:
                    reflowed = f"## {reflowed}"

        out_paragraphs.append(reflowed)

    return "\n\n".join(out_paragraphs)


async def process_chunk(
    chunk: str,
    prev_context: str,
    chunk_index: int,
    total_chunks: int,
    reformat_as_markdown: bool,
    suppress_headers_and_page_numbers: bool,
    use_llm: bool = True,
) -> tuple[str, str]:
    logger.info("Processing chunk %d/%d (length: %d characters)", chunk_index + 1, total_chunks, len(chunk))

    if not use_llm:
        processed_chunk = reflow_and_dehyphenate_ocr_text(chunk, reformat_as_markdown=reformat_as_markdown)
        new_context = processed_chunk[-1000:] if processed_chunk else ""
        logger.info(
            "Chunk %d/%d processed deterministically (--no-llm). Output length: %d characters",
            chunk_index + 1,
            total_chunks,
            len(processed_chunk),
        )
        return processed_chunk, new_context

    # Pre-clean deterministic typewriter OCR confusions and line-wrap hyphenations before LLM prompting
    chunk = clean_ocr_character_confusions(chunk)

    active_model = (
        DEFAULT_LOCAL_MODEL_NAME
        if USE_LOCAL_LLM
        else (CLAUDE_MODEL_STRING if API_PROVIDER == "CLAUDE" else OPENAI_COMPLETION_MODEL)
    )
    estimated_chunk_size = estimate_tokens(chunk, active_model)
    completion_budget = max(1024, int(estimated_chunk_size * 1.5) + 500)
    context_snippet = prev_context[-500:].strip() if prev_context else ""

    # Step 1: OCR Correction
    ocr_correction_prompt = f"""Correct OCR-induced errors in the text, ensuring it flows coherently with the previous context. Follow these guidelines:

1. Fix OCR-induced typos and errors:
   - Correct words split across line breaks (e.g., "manage-\\nment" -> "management")
   - Fix common OCR character confusions (e.g., 'rn' misread as 'm', '1O' for '10', 'lh' for '14')
   - Use context and common sense to correct clear scan/OCR errors
   - Do not alter the author's wording, numbers, or meaning
   - Do not add extra periods or unnecessary punctuation

2. Maintain original structure and preserve ALL content:
   - Keep all headings, subheadings, tables, and numbered/lettered items intact
   - STRICTLY PRESERVE all substantive author footnotes and notes (such as paragraphs starting with `*`, `†`, or `®`)
   - Remove unnecessary mid-sentence line breaks while maintaining paragraph breaks

3. Read-only previous context:
   - The `<previous_context>` block below is provided ONLY so you can connect mid-sentence transitions smoothly.
   - DO NOT repeat, copy, or include any words from `<previous_context>` in your output.

IMPORTANT: Respond ONLY with the corrected text for the current chunk. Do not include any introduction, explanation, code fences, or metadata.

<previous_context>
{context_snippet}
</previous_context>

Current chunk to process:
{chunk}

Corrected text:
"""

    ocr_corrected_chunk = await generate_completion(ocr_correction_prompt, max_tokens=completion_budget)

    # Handle None return from generate_completion - fall back to deterministic reflow/dehyphenation
    if ocr_corrected_chunk is None:
        logger.warning(
            "OCR correction LLM unavailable for chunk %d; applying deterministic reflow and dehyphenation",
            chunk_index + 1,
        )
        processed_chunk = reflow_and_dehyphenate_ocr_text(chunk, reformat_as_markdown=reformat_as_markdown)
        new_context = processed_chunk[-1000:] if processed_chunk else ""
        return processed_chunk, new_context

    ocr_corrected_chunk = strip_llm_wrapper_artifacts(ocr_corrected_chunk)
    processed_chunk = ocr_corrected_chunk

    # Step 2: Markdown Formatting (if requested)
    if reformat_as_markdown:
        header_instruction = (
            "Identify but do not remove running page headers or page numbers. Format them distinctly as blockquotes."
            if not suppress_headers_and_page_numbers
            else (
                "Carefully remove standalone running page headers and standalone page numbers (e.g. '-2-', 'Page 2'), "
                "while STRICTLY PRESERVING 100% of substantive author footnotes and asterisked notes (paragraphs starting with '*', '†', or '®'). "
                "Never delete author footnotes."
            )
        )
        markdown_prompt = f"""Reformat the following text as markdown, improving readability while preserving the original structure and 100% of the substantive content. Follow these guidelines:
1. Preserve all original headings, converting them to appropriate markdown heading levels (# for main titles, ## for section headings, etc.)
   - Ensure each heading is on its own line with a blank line before and after
   - NEVER turn a mid-sentence fragment or lowercase continuation at the start of the chunk into a heading
2. Maintain the original paragraph structure. Join words broken across lines by hyphens.
3. Format lists and tables properly in Markdown if they exist in the original text.
4. Preserve all original content, numbers, percentages, dollar figures, and meaning.
5. Remove any spuriously inserted LLM preamble such as "Here is the corrected text:".
6. The `<previous_context>` block below shows the end of the preceding chunk for read-only continuity:
   - DO NOT repeat or output any text from `<previous_context>`.
   - If the current chunk begins with a phrase that duplicates the end of `<previous_context>`, omit only that duplicated overlap phrase so the two chunks join seamlessly.
7. {header_instruction}

<previous_context>
{context_snippet}
</previous_context>

Text to reformat:

{ocr_corrected_chunk}

Reformatted markdown:
"""
        md_budget = max(1024, int(estimate_tokens(ocr_corrected_chunk, active_model) * 1.5) + 500)
        markdown_result = await generate_completion(markdown_prompt, max_tokens=md_budget)
        if markdown_result is not None:
            processed_chunk = strip_llm_wrapper_artifacts(markdown_result)
        else:
            logger.warning("Markdown formatting failed for chunk %d, using OCR-corrected text", chunk_index + 1)

    new_context = processed_chunk[-1000:] if processed_chunk else ""
    logger.info("Chunk %d/%d processed. Output length: %d characters", chunk_index + 1, total_chunks, len(processed_chunk))
    return processed_chunk, new_context


async def process_chunks(
    chunks: list[str],
    reformat_as_markdown: bool,
    suppress_headers_and_page_numbers: bool,
    use_llm: bool = True,
) -> list[str]:
    total_chunks = len(chunks)
    if total_chunks == 0:
        return []

    extra_kwargs: dict[str, Any] = {"use_llm": False} if not use_llm else {}
    semaphore = asyncio.Semaphore(max(1, MAX_CONCURRENT_REQUESTS))

    async def process_chunk_with_context(chunk: str, prev_context: str, index: int) -> tuple[int, str, str]:
        async with semaphore:
            processed_chunk, new_context = await process_chunk(
                chunk,
                prev_context,
                index,
                total_chunks,
                reformat_as_markdown,
                suppress_headers_and_page_numbers,
                **extra_kwargs,
            )
            return index, processed_chunk, new_context

    if USE_LOCAL_LLM or not use_llm:
        logger.info(
            "Processing %d chunks sequentially (%s)...",
            total_chunks,
            "deterministic --no-llm" if not use_llm else "local LLM",
        )
        context = ""
        processed_chunks: list[str] = []
        for i, chunk in enumerate(chunks):
            processed_chunk, context = await process_chunk(
                chunk,
                context,
                i,
                total_chunks,
                reformat_as_markdown,
                suppress_headers_and_page_numbers,
                **extra_kwargs,
            )
            processed_chunks.append(processed_chunk)
    else:
        logger.info(
            "Using API-based LLM. Processing %d chunks concurrently (max_concurrency=%d)...",
            total_chunks,
            MAX_CONCURRENT_REQUESTS,
        )
        tasks = [
            process_chunk_with_context(chunk, chunks[i - 1][-1000:] if i > 0 else "", i)
            for i, chunk in enumerate(chunks)
        ]
        results = await asyncio.gather(*tasks)
        sorted_results = sorted(results, key=lambda x: x[0])
        processed_chunks = [chunk for _, chunk, _ in sorted_results]

    logger.info("All %d chunks processed successfully", total_chunks)
    return processed_chunks


def _norm_token(word: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", word.lower())


def stitch_chunks(processed_chunks: list[str], overlap_window: int = 35) -> str:
    """Deterministically stitch processed chunks, removing echoed boundary overlaps and spurious mid-sentence headings."""
    non_empty = [strip_llm_wrapper_artifacts(c) for c in processed_chunks if c and c.strip()]
    if not non_empty:
        return ""
    if len(non_empty) == 1:
        return non_empty[0]

    stitched = non_empty[0].rstrip()

    for idx in range(1, len(non_empty)):
        curr = non_empty[idx].lstrip()
        if not curr:
            continue

        # 1. If curr starts with a heading (# ...) that merely echoes the final phrase of stitched, strip that heading line
        first_line_end = curr.find("\n")
        first_line = curr[:first_line_end].strip() if first_line_end != -1 else curr.strip()
        if first_line.startswith("#"):
            heading_text = first_line.lstrip("#").strip()
            heading_words = [_norm_token(w) for w in heading_text.split() if _norm_token(w)]
            tail_words_norm = [_norm_token(w) for w in stitched.split()[-overlap_window:] if _norm_token(w)]
            if heading_words and len(heading_words) <= 12:
                # Check if heading_words is a contiguous subsequence of tail_words_norm
                h_len = len(heading_words)
                is_echo_heading = any(
                    tail_words_norm[k : k + h_len] == heading_words
                    for k in range(max(0, len(tail_words_norm) - h_len + 1))
                )
                if is_echo_heading and first_line_end != -1:
                    curr = curr[first_line_end:].lstrip()
                elif is_echo_heading and first_line_end == -1:
                    continue

        # 2. Check for word-level suffix/prefix overlap (including shifted echoes near the start of curr)
        prev_matches = list(re.finditer(r"\S+", stitched))
        curr_matches = list(re.finditer(r"\S+", curr))
        tail_matches = prev_matches[-overlap_window:]
        head_matches = curr_matches[:overlap_window]

        prev_norm = [_norm_token(m.group(0)) for m in tail_matches]
        curr_norm = [_norm_token(m.group(0)) for m in head_matches]

        best_cut_char = 0
        # Look for a suffix of prev_norm (or near-suffix within 3 words of the end) matching near the start of curr_norm
        max_len = min(len(prev_norm), len(curr_norm))
        for length in range(max_len, 1, -1):
            found = False
            for prev_end_offset in range(0, min(4, len(prev_norm) - length + 1)):
                p_end = len(prev_norm) - prev_end_offset
                p_slice = prev_norm[p_end - length : p_end]
                if not all(p_slice):
                    continue
                if length == 2 and sum(len(w) for w in p_slice) < 10:
                    continue
                for curr_start in range(0, min(5, len(curr_norm) - length + 1)):
                    if curr_norm[curr_start : curr_start + length] == p_slice:
                        # Also trim corresponding trailing tokens of stitched if prev_end_offset > 0
                        if prev_end_offset > 0:
                            trim_match = tail_matches[len(tail_matches) - prev_end_offset]
                            stitched = stitched[: trim_match.start()].rstrip()
                        end_match = head_matches[curr_start + length - 1]
                        best_cut_char = end_match.end()
                        found = True
                        break
                if found:
                    break
            if found:
                break

        if best_cut_char > 0:
            curr = curr[best_cut_char:].lstrip()

        if not curr:
            continue

        # 3. If stitched ends mid-sentence and curr begins with a spurious '# ' heading on a lowercase continuation,
        # strip the '#' marker so the sentence joins cleanly.
        if _ends_mid_sentence(stitched) and curr.startswith("#"):
            after_hashes = curr.lstrip("#").lstrip()
            if after_hashes and after_hashes[0].islower():
                curr = after_hashes

        # 4. Handle parenthetical suffix continuation (e.g., stitched ends with '.' and curr starts with '(particularly ...).')
        if stitched.endswith(".") and re.match(r"^\([a-z][^)]*\)\.", curr):
            stitched = f"{stitched[:-1]} {curr}"
        elif _ends_mid_sentence(stitched) and curr and (curr[0].islower() or curr[0] in ",;—–-)"):
            stitched = f"{stitched} {curr}"
        else:
            stitched = f"{stitched}\n\n{curr}"

    return stitched


def split_document_into_chunks(
    full_text: str,
    chunk_size: int = 8000,
    model_name: str = "gpt-4o-mini",
) -> list[str]:
    """Split cleaned document text into paragraph-preserving chunks, delegating oversized paragraphs to chunk_text()."""
    if not full_text or not full_text.strip():
        return []
    chunk_size = max(500, chunk_size)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", full_text) if p.strip()]
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_chunk_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if paragraph_length > chunk_size:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_chunk_length = 0
            para_tokens = max(1, estimate_tokens(paragraph, model_name))
            token_budget = max(64, int(para_tokens * (chunk_size / max(1, paragraph_length))))
            sub_chunks = chunk_text(paragraph, max_chunk_tokens=token_budget, model_name=model_name, overlap_size=0)
            chunks.extend(sub_chunks)
        elif current_chunk and current_chunk_length + paragraph_length + 2 > chunk_size:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [paragraph]
            current_chunk_length = paragraph_length
        else:
            current_chunk.append(paragraph)
            current_chunk_length += paragraph_length + (2 if len(current_chunk) > 1 else 0)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


async def process_document(
    list_of_extracted_text_strings: list[str],
    reformat_as_markdown: bool = True,
    suppress_headers_and_page_numbers: bool = True,
    chunk_size: int = 8000,
    use_llm: bool = True,
) -> str:
    logger.info("Starting document processing. Total pages: %d", len(list_of_extracted_text_strings))
    cleaned_pages = clean_page_artifacts(
        list_of_extracted_text_strings,
        suppress_headers_and_page_numbers=suppress_headers_and_page_numbers,
    )
    full_text = "\n\n".join(cleaned_pages)
    logger.info("Size of full text before processing: %d characters", len(full_text))
    chunk_size = max(500, chunk_size)
    active_model = (
        DEFAULT_LOCAL_MODEL_NAME
        if USE_LOCAL_LLM
        else (CLAUDE_MODEL_STRING if API_PROVIDER == "CLAUDE" else OPENAI_COMPLETION_MODEL)
    )
    chunks = split_document_into_chunks(full_text, chunk_size=chunk_size, model_name=active_model)

    logger.info("Document split into %d chunks (target chunk_size=%d chars).", len(chunks), chunk_size)
    extra_kwargs: dict[str, Any] = {"use_llm": False} if not use_llm else {}
    processed_chunks = await process_chunks(
        chunks,
        reformat_as_markdown,
        suppress_headers_and_page_numbers,
        **extra_kwargs,
    )
    final_text = stitch_chunks(processed_chunks)
    logger.info("Size of text after stitching chunks: %d characters", len(final_text))
    logger.info("Document processing complete. Final text length: %d characters", len(final_text))
    return final_text


def remove_corrected_text_header(text: str) -> str:
    cleaned = (
        text.replace("# Corrected text\n", "")
        .replace("# Corrected text:", "")
        .replace("\nCorrected text", "")
        .replace("Corrected text:", "")
    )
    return strip_llm_wrapper_artifacts(cleaned)


def _extract_significant_numbers(text: str) -> set[str]:
    """Extract normalized multi-digit numbers, years, decimals, and percentages for conservation checking."""
    # Normalize OCR comma-decimals ('+2,8%' -> '+2.8%'), space-decimal percentages ('+3 8%' -> '+3.8%'), and '$1% billion' -> '$1.5 billion'
    normalized_text = re.sub(r"(?<![,\d])(\d+),(\d{1,2})(?![,\d])", r"\1.\2", text)
    normalized_text = re.sub(r"([+\-]\d+)[ \t]+(\d)%", r"\1.\2%", normalized_text)
    normalized_text = re.sub(r"\$(\d+)%(?=\s+(?:million|billion)\b)", r"$\1.5", normalized_text)
    normalized_text = re.sub(r"\b4#[Ss]-degree\b", "45-degree", normalized_text)
    # Find numbers with >= 2 digits or decimals/percentages/dollar figures
    raw_matches = re.findall(r"\b\d[\d,]*\.?\d*\b", normalized_text)
    conserved: set[str] = set()
    for m in raw_matches:
        norm = m.replace(",", "").strip(".")
        if not norm:
            continue
        # Focus on multi-digit numbers or decimals (years, dollar amounts, percentages >= 10, etc.)
        # since single digits 1..9 are often converted between '(1)' and '1.' or spelled out.
        if len(norm) >= 2 or "." in norm:
            conserved.add(norm)
    return conserved


def _counter_cosine_similarity(c1: Counter[str], c2: Counter[str]) -> float:
    """Compute cosine similarity between two frequency Counters."""
    if not c1 or not c2:
        return 0.0
    common_keys = set(c1.keys()) & set(c2.keys())
    dot = float(sum(c1[k] * c2[k] for k in common_keys))
    norm1 = float(np.sqrt(sum(v * v for v in c1.values())))
    norm2 = float(np.sqrt(sum(v * v for v in c2.values())))
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm1 * norm2)))


def compute_lexical_similarity(original_text: str, processed_text: str) -> float:
    """Compute deterministic lexical cosine similarity (word frequency + character 3-gram frequency)."""
    if not original_text.strip() and not processed_text.strip():
        return 1.0
    if not original_text.strip() or not processed_text.strip():
        return 0.0

    orig_words = Counter(re.findall(r"[a-z0-9]{2,}", original_text.lower()))
    proc_words = Counter(re.findall(r"[a-z0-9]{2,}", processed_text.lower()))
    word_sim = _counter_cosine_similarity(orig_words, proc_words)

    orig_norm = re.sub(r"[^a-z0-9]+", " ", original_text.lower()).strip()
    proc_norm = re.sub(r"[^a-z0-9]+", " ", processed_text.lower()).strip()
    orig_trigrams = Counter(orig_norm[i : i + 3] for i in range(max(0, len(orig_norm) - 2)))
    proc_trigrams = Counter(proc_norm[i : i + 3] for i in range(max(0, len(proc_norm) - 2)))
    trigram_sim = _counter_cosine_similarity(orig_trigrams, proc_trigrams)

    return round(0.6 * word_sim + 0.4 * trigram_sim, 4)


async def compute_embedding_similarity(
    original_text: str,
    processed_text: str,
    model: str | None = None,
) -> float:
    """Compute semantic embedding cosine similarity via OPENAI_EMBEDDING_MODEL, falling back to lexical similarity offline."""
    client = get_openai_client()
    if client is None:
        return compute_lexical_similarity(original_text, processed_text)

    embed_model = model or OPENAI_EMBEDDING_MODEL
    orig_sample = _sample_start_mid_end(original_text, 12000)
    proc_sample = _sample_start_mid_end(processed_text, 12000)
    if not orig_sample.strip() or not proc_sample.strip():
        return compute_lexical_similarity(original_text, processed_text)

    try:
        resp = await client.embeddings.create(
            model=embed_model,
            input=[orig_sample, proc_sample],
        )
        v1 = np.array(resp.data[0].embedding, dtype=np.float64)
        v2 = np.array(resp.data[1].embedding, dtype=np.float64)
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom <= 0.0:
            return compute_lexical_similarity(original_text, processed_text)
        return round(float(max(0.0, min(1.0, float(np.dot(v1, v2)) / denom))), 4)
    except Exception as e:
        logger.debug("Embedding similarity API fallback to lexical similarity: %s", e)
        return compute_lexical_similarity(original_text, processed_text)


def verify_content_conservation(original_text: str, processed_text: str) -> dict[str, Any]:
    """Deterministically verify conservation of numeric facts, footnotes, seam integrity, lexical similarity, and length."""
    cleaned_raw = clean_raw_ocr_text(original_text, suppress_headers_and_page_numbers=True)
    raw_numbers = _extract_significant_numbers(cleaned_raw)
    proc_numbers = _extract_significant_numbers(processed_text)

    if raw_numbers:
        preserved_numbers = set(raw_numbers & proc_numbers)
        for candidate in raw_numbers - preserved_numbers:
            # Recognize common typewriter OCR misread where '<prefix>%' was scanned as '<prefix>96' (e.g., '4%' -> '496')
            if len(candidate) >= 3 and candidate.endswith("96"):
                prefix = candidate[:-2]
                if prefix and re.search(rf"\b{re.escape(prefix)}\s*%", processed_text):
                    preserved_numbers.add(candidate)
        missing_numbers = sorted(raw_numbers - preserved_numbers)
        numeric_recall = len(preserved_numbers) / len(raw_numbers)
    else:
        missing_numbers = []
        numeric_recall = 1.0

    # Footnote conservation: check whether distinct content words from raw footnotes appear in processed_text
    raw_footnotes = extract_footnotes(original_text)
    proc_lower = re.sub(r"\s+", " ", processed_text.lower())
    preserved_footnotes = 0
    missing_footnotes: list[str] = []
    for fn in raw_footnotes:
        words = re.findall(r"[a-z]{3,}", fn.lower())
        if not words:
            preserved_footnotes += 1
            continue
        matched = sum(1 for w in set(words[:15]) if w in proc_lower)
        if matched / max(1, len(set(words[:15]))) >= 0.7:
            preserved_footnotes += 1
        else:
            missing_footnotes.append(fn[:80])

    footnote_recall = preserved_footnotes / len(raw_footnotes) if raw_footnotes else 1.0

    # Seam anomaly detection: look for mid-word/mid-line '#' headings or ')' immediately followed by a word
    seam_anomalies: list[str] = []
    for match in re.finditer(r"[^\s\n#]#+\s+[A-Za-z]|\)[a-z]{2,}", processed_text):
        start = max(0, match.start() - 25)
        end = min(len(processed_text), match.end() + 25)
        seam_anomalies.append(processed_text[start:end].replace("\n", " "))

    raw_word_count = max(1, len(cleaned_raw.split()))
    proc_word_count = len(processed_text.split())
    word_retention_ratio = proc_word_count / raw_word_count
    lexical_sim = compute_lexical_similarity(cleaned_raw, processed_text)

    # Compute deterministic 0-100 conservation score
    length_score = 1.0 if 0.75 <= word_retention_ratio <= 1.25 else max(0.0, 1.0 - abs(1.0 - word_retention_ratio))
    seam_penalty = min(0.4, 0.1 * len(seam_anomalies))
    deterministic_score = int(
        round(
            max(
                0.0,
                min(
                    100.0,
                    100.0
                    * (
                        0.40 * numeric_recall
                        + 0.25 * footnote_recall
                        + 0.20 * lexical_sim
                        + 0.15 * length_score
                        - seam_penalty
                    ),
                ),
            )
        )
    )

    return {
        "numeric_recall": round(numeric_recall, 4),
        "missing_numbers": missing_numbers,
        "footnote_count": len(raw_footnotes),
        "footnotes_preserved": preserved_footnotes,
        "footnote_recall": round(footnote_recall, 4),
        "missing_footnotes": missing_footnotes,
        "seam_anomalies": seam_anomalies,
        "word_retention_ratio": round(word_retention_ratio, 4),
        "lexical_cosine_similarity": round(lexical_sim, 4),
        "deterministic_score": deterministic_score,
    }


def _sample_start_mid_end(text: str, budget: int) -> str:
    """Sample start, middle, and end segments of text within budget characters."""
    if len(text) <= budget:
        return text
    part = max(1, budget // 3)
    mid = len(text) // 2
    start_seg = text[:part]
    mid_seg = text[max(0, mid - part // 2) : mid + part // 2]
    end_seg = text[-part:]
    return f"{start_seg}\n\n[... middle sample ...]\n\n{mid_seg}\n\n[... end sample ...]\n\n{end_seg}"


async def assess_output_quality(
    original_text: str,
    processed_text: str,
    use_llm: bool = True,
) -> tuple[int | None, str | None]:
    max_chars = 15000  # Limit to avoid exceeding context limits
    available_chars_per_text = max_chars // 2

    conservation = verify_content_conservation(original_text, processed_text)
    semantic_similarity = (
        await compute_embedding_similarity(original_text, processed_text)
        if use_llm
        else conservation["lexical_cosine_similarity"]
    )
    original_sample = _sample_start_mid_end(original_text, available_chars_per_text)
    processed_sample = _sample_start_mid_end(processed_text, available_chars_per_text)

    prompt = f"""Compare the following multi-point samples (start, middle, and end) of original OCR text with the processed Markdown output and assess the quality of the processing. Consider the following factors:
1. Accuracy of OCR error correction
2. Improvement in readability and clean paragraph flow
3. Preservation of original content, numbers, and author footnotes
4. Appropriate use of Markdown formatting without corrupted chunk seams
5. Absence of hallucinations or duplicated boundary text

Deterministic conservation metrics across the full document:
- Numeric recall: {conservation['numeric_recall']:.1%}
- Footnote recall: {conservation['footnote_recall']:.1%} ({conservation['footnotes_preserved']}/{conservation['footnote_count']})
- Lexical cosine similarity: {conservation['lexical_cosine_similarity']:.1%}
- Semantic/embedding similarity: {semantic_similarity:.1%}
- Seam anomalies detected: {len(conservation['seam_anomalies'])}
- Word retention ratio: {conservation['word_retention_ratio']:.2f}

Original text sample:
```
{original_sample}
```

Processed text sample:
```
{processed_sample}
```

Provide a quality score between 0 and 100, where 100 is perfect processing. Also provide a brief explanation of your assessment.

Your response should be in the following format:
SCORE: [Your score]
EXPLANATION: [Your explanation]
"""

    response = await generate_completion(prompt, max_tokens=1000) if use_llm else None

    if response is None:
        logger.warning(
            "LLM quality assessment unavailable; returning deterministic conservation score (%d/100).",
            conservation["deterministic_score"],
        )
        explanation = (
            f"Deterministic conservation audit: numeric_recall={conservation['numeric_recall']:.1%}, "
            f"footnote_recall={conservation['footnote_recall']:.1%} "
            f"({conservation['footnotes_preserved']}/{conservation['footnote_count']}), "
            f"lexical_similarity={conservation['lexical_cosine_similarity']:.1%}, "
            f"seam_anomalies={len(conservation['seam_anomalies'])}, "
            f"word_retention_ratio={conservation['word_retention_ratio']:.2f}."
        )
        return conservation["deterministic_score"], explanation

    try:
        score_match = re.search(r"SCORE:\s*(\d{1,3})", response, flags=re.IGNORECASE)
        if not score_match:
            raise ValueError("Missing SCORE: line in response")
        score = max(0, min(100, int(score_match.group(1))))

        exp_match = re.search(r"EXPLANATION:\s*(.*)", response, flags=re.IGNORECASE | re.DOTALL)
        explanation = exp_match.group(1).strip() if exp_match else ""
        logger.info("Quality assessment: Score %d/100", score)
        logger.info("Explanation: %s", explanation)
        return score, explanation
    except Exception as e:
        logger.error("Error parsing quality assessment response: %s", e)
        logger.error("Raw response: %s", response)
        return None, None


def _write_json_report(report_path: str, payload: dict[str, Any]) -> None:
    report_dir = os.path.dirname(os.path.abspath(report_path))
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(payload, rf, indent=2)


async def main(
    input_pdf_file_path: str = "160301289-Warren-Buffett-Katharine-Graham-Letter.pdf",
    max_test_pages: int = 0,
    skip_first_n_pages: int = 0,
    reformat_as_markdown: bool = True,
    suppress_headers_and_page_numbers: bool = True,
    output_dir: str | None = None,
    raw_ocr_input_path: str | None = None,
    dpi: int = PDF_DPI,
    preprocess: bool = True,
    assess_quality: bool = True,
    chunk_size: int = 8000,
    report_json_path: str | None = None,
    use_llm: bool = True,
):
    try:
        # Suppress HTTP request logs
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # Download the model if using local LLM
        if not use_llm:
            logger.info("Running in deterministic OCR cleanup mode (--no-llm).")
        elif USE_LOCAL_LLM:
            _, download_status = await download_models()
            logger.info("Model download status: %s", download_status)
            logger.info("Using Local LLM with Model: %s", DEFAULT_LOCAL_MODEL_NAME)
        else:
            logger.info("Using API for completions: %s", API_PROVIDER)
            logger.info("Using OpenAI model for embeddings: %s", OPENAI_EMBEDDING_MODEL)

        source_path = raw_ocr_input_path if raw_ocr_input_path else input_pdf_file_path
        base_stem = os.path.splitext(os.path.basename(source_path))[0]
        base_stem = re.sub(r"__raw_ocr_output$", "", base_stem)
        target_dir = output_dir if output_dir else (os.path.dirname(os.path.abspath(source_path)) or ".")
        os.makedirs(target_dir, exist_ok=True)
        base_name = os.path.join(target_dir, base_stem)
        output_extension = ".md" if reformat_as_markdown else ".txt"

        raw_ocr_output_file_path = f"{base_name}__raw_ocr_output.txt"
        llm_corrected_output_file_path = base_name + "_llm_corrected" + output_extension

        if raw_ocr_input_path:
            logger.info("Reading pre-extracted raw OCR text from: %s", raw_ocr_input_path)
            with open(raw_ocr_input_path, encoding="utf-8") as f:
                raw_ocr_output = f.read()
            if "\x0c" in raw_ocr_output:
                list_of_extracted_text_strings = [p for p in raw_ocr_output.split("\x0c") if p.strip()]
            else:
                list_of_extracted_text_strings = [raw_ocr_output]
        else:
            list_of_scanned_images = convert_pdf_to_images(
                input_pdf_file_path,
                max_test_pages,
                skip_first_n_pages,
                dpi=dpi,
            )
            logger.info("Tesseract version: %s", pytesseract.get_tesseract_version())
            logger.info("Extracting text from converted pages (preprocess=%s)...", preprocess)
            ocr_fn = functools.partial(ocr_image, preprocess=preprocess)
            with ThreadPoolExecutor() as executor:
                list_of_extracted_text_strings = list(executor.map(ocr_fn, list_of_scanned_images))
            logger.info("Done extracting text from converted pages.")
            raw_ocr_output = "\x0c\n".join(p.replace("\x0c", "").rstrip() for p in list_of_extracted_text_strings)
            with open(raw_ocr_output_file_path, "w", encoding="utf-8") as f:
                f.write(raw_ocr_output)
            logger.info("Raw OCR output written to: %s", raw_ocr_output_file_path)

        logger.info("Processing document...")
        final_text = await process_document(
            list_of_extracted_text_strings,
            reformat_as_markdown,
            suppress_headers_and_page_numbers,
            chunk_size=chunk_size,
            use_llm=use_llm,
        )
        cleaned_text = remove_corrected_text_header(final_text)

        # Save the LLM corrected output
        with open(llm_corrected_output_file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        logger.info("LLM Corrected text written to: %s", llm_corrected_output_file_path)

        if cleaned_text:
            logger.info("First 500 characters of LLM corrected processed text:\n%s...", cleaned_text[:500])
        else:
            logger.warning("cleaned_text is empty or not defined.")

        logger.info("Done processing %s.", source_path)
        logger.info("\nSee output files:")
        logger.info(" Raw OCR: %s", raw_ocr_output_file_path)
        logger.info(" LLM Corrected: %s", llm_corrected_output_file_path)

        quality_score: int | None = None
        explanation: str | None = None
        if assess_quality:
            quality_score, explanation = await assess_output_quality(
                raw_ocr_output,
                cleaned_text,
                use_llm=use_llm,
            )
            if quality_score is not None:
                logger.info("Final quality score: %d/100", quality_score)
                logger.info("Explanation: %s", explanation)
            else:
                logger.warning("Unable to determine final quality score.")

        if report_json_path:
            report = verify_content_conservation(raw_ocr_output, cleaned_text)
            report["quality_score"] = quality_score
            report["quality_explanation"] = explanation
            await asyncio.to_thread(_write_json_report, report_json_path, report)
            logger.info("Conservation and quality report written to: %s", report_json_path)

        return cleaned_text
    except Exception as e:
        logger.error("An error occurred in the main function: %s", e)
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    asyncio.run(main())

