import os
import glob
import traceback
import asyncio
import json
import re
import time
import urllib.request
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings
from typing import List, Dict, Tuple, Optional, Any
from pdf2image import convert_from_path
import pytesseract
from llama_cpp import Llama, LlamaGrammar
import httpx
import tiktoken
import numpy as np
from PIL import Image
from decouple import Config as DecoupleConfig, RepositoryEnv
import cv2
from filelock import FileLock, Timeout
from transformers import AutoTokenizer
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from langdetect import detect, LangDetectException
import faiss
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
try:
    import nvgpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    
# Configuration
config = DecoupleConfig(RepositoryEnv('.env'))

USE_LOCAL_LLM = config.get("USE_LOCAL_LLM", default=False, cast=bool)
API_PROVIDER = config.get("API_PROVIDER", default="OPENAI", cast=str)
ANTHROPIC_API_KEY = config.get("ANTHROPIC_API_KEY", default="your-anthropic-api-key", cast=str)
OPENAI_API_KEY = config.get("OPENAI_API_KEY", default="your-openai-api-key", cast=str)
OPENROUTER_API_KEY = config.get("OPENROUTER_API_KEY", default="your-openrouter-api-key", cast=str)
CLAUDE_MODEL_STRING = config.get("CLAUDE_MODEL_STRING", default="claude-3-haiku-20240307", cast=str)
CLAUDE_MAX_TOKENS = 4096  # Maximum allowed tokens for Claude API
TOKEN_BUFFER = 500  # Buffer to account for token estimation inaccuracies
TOKEN_CUSHION = 100 # Don't use the full max tokens to avoid hitting the limit
OPENAI_COMPLETION_MODEL = config.get("OPENAI_COMPLETION_MODEL", default="gpt-4o-mini", cast=str)
OPENAI_EMBEDDING_MODEL = config.get("OPENAI_EMBEDDING_MODEL", default="text-embedding-3-small", cast=str)
OPENAI_MAX_TOKENS = 4096  # Maximum allowed tokens for OpenAI API
OPENROUTER_MODEL = config.get("OPENROUTER_MODEL", default="meta-llama/llama-3.1-8b-instruct:free", cast=str)
DEFAULT_LOCAL_MODEL_NAME = "Llama-3.1-8B-Lexi-Uncensored_Q5_fixedrope.gguf"
LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS = 2048
USE_VERBOSE = False

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GPU Check
def is_gpu_available():
    if not GPU_AVAILABLE:
        logging.warning("GPU support not available: nvgpu module not found")
        return {"gpu_found": False, "num_gpus": 0, "first_gpu_vram": 0, "total_vram": 0, "error": "nvgpu module not found"}
    try:
        gpu_info = nvgpu.gpu_info()
        num_gpus = len(gpu_info)
        if num_gpus == 0:
            logging.warning("No GPUs found on the system")
            return {"gpu_found": False, "num_gpus": 0, "first_gpu_vram": 0, "total_vram": 0}
        first_gpu_vram = gpu_info[0]['mem_total']
        total_vram = sum(gpu['mem_total'] for gpu in gpu_info)
        logging.info(f"GPU(s) found: {num_gpus}, Total VRAM: {total_vram} MB")
        return {"gpu_found": True, "num_gpus": num_gpus, "first_gpu_vram": first_gpu_vram, "total_vram": total_vram, "gpu_info": gpu_info}
    except Exception as e:
        logging.error(f"Error checking GPU availability: {e}")
        return {"gpu_found": False, "num_gpus": 0, "first_gpu_vram": 0, "total_vram": 0, "error": str(e)}

# Model Download
async def download_models() -> Tuple[List[str], List[Dict[str, str]]]:
    download_status = []    
    model_url = "https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-GGUF/resolve/main/Llama-3.1-8B-Lexi-Uncensored_Q5_fixedrope.gguf"
    model_name = os.path.basename(model_url)
    current_file_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(current_file_path)
    models_dir = os.path.join(base_dir, 'models')
    
    os.makedirs(models_dir, exist_ok=True)
    lock = FileLock(os.path.join(models_dir, "download.lock"))
    status = {"url": model_url, "status": "success", "message": "File already exists."}
    filename = os.path.join(models_dir, model_name)
    
    try:
        with lock.acquire(timeout=1200):
            if not os.path.exists(filename):
                logging.info(f"Downloading model {model_name} from {model_url}...")
                urllib.request.urlretrieve(model_url, filename)
                file_size = os.path.getsize(filename) / (1024 * 1024)
                if file_size < 100:
                    os.remove(filename)
                    status["status"] = "failure"
                    status["message"] = f"Downloaded file is too small ({file_size:.2f} MB), probably not a valid model file."
                    logging.error(f"Error: {status['message']}")
                else:
                    logging.info(f"Successfully downloaded: {filename} (Size: {file_size:.2f} MB)")
            else:
                logging.info(f"Model file already exists: {filename}")
    except Timeout:
        logging.error(f"Error: Could not acquire lock for downloading {model_name}")
        status["status"] = "failure"
        status["message"] = "Could not acquire lock for downloading."
    
    download_status.append(status)
    logging.info("Model download process completed.")
    return [model_name], download_status

# Model Loading
def load_model(llm_model_name: str, raise_exception: bool = True):
    global USE_VERBOSE
    try:
        current_file_path = os.path.abspath(__file__)
        base_dir = os.path.dirname(current_file_path)
        models_dir = os.path.join(base_dir, 'models')
        matching_files = glob.glob(os.path.join(models_dir, f"{llm_model_name}*"))
        if not matching_files:
            logging.error(f"Error: No model file found matching: {llm_model_name}")
            raise FileNotFoundError
        model_file_path = max(matching_files, key=os.path.getmtime)
        logging.info(f"Loading model: {model_file_path}")
        try:
            logging.info("Attempting to load model with GPU acceleration...")
            model_instance = Llama(
                model_path=model_file_path,
                n_ctx=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS,
                verbose=USE_VERBOSE,
                n_gpu_layers=-1
            )
            logging.info("Model loaded successfully with GPU acceleration.")
        except Exception as gpu_e:
            logging.warning(f"Failed to load model with GPU acceleration: {gpu_e}")
            logging.info("Falling back to CPU...")
            try:
                model_instance = Llama(
                    model_path=model_file_path,
                    n_ctx=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS,
                    verbose=USE_VERBOSE,
                    n_gpu_layers=0
                )
                logging.info("Model loaded successfully with CPU.")
            except Exception as cpu_e:
                logging.error(f"Failed to load model with CPU: {cpu_e}")
                if raise_exception:
                    raise
                return None
        return model_instance
    except Exception as e:
        logging.error(f"Exception occurred while loading the model: {e}")
        traceback.print_exc()
        if raise_exception:
            raise
        return None

def improved_sentence_split(text):
    return sent_tokenize(text)

# API Interaction Functions
async def generate_completion(prompt: str, max_tokens: int = 1000) -> Optional[str]:
    if USE_LOCAL_LLM:
        return await generate_completion_from_local_llm(DEFAULT_LOCAL_MODEL_NAME, prompt, max_tokens)
    elif API_PROVIDER == "CLAUDE":
        return await generate_completion_from_claude(prompt, max_tokens)
    elif API_PROVIDER == "OPENAI":
        return await generate_completion_from_openai(prompt, max_tokens)
    elif API_PROVIDER == "OPENROUTER":
        return await generate_completion_from_openrouter(prompt, max_tokens)
    else:
        logging.error(f"Invalid API_PROVIDER: {API_PROVIDER}")
        return None

async def generate_embedding(text: str) -> Optional[List[float]]:
    if USE_LOCAL_LLM:
        llm = load_model(DEFAULT_LOCAL_MODEL_NAME)
        return llm.embed(text)
    elif API_PROVIDER in ["CLAUDE", "OPENAI", "OPENROUTER"]:
        return await generate_embedding_with_retry(text)
    else:
        logging.error(f"Invalid API_PROVIDER: {API_PROVIDER}")
        return None

def get_tokenizer(model_name: str):
    if model_name.startswith("gpt-"):
        return tiktoken.encoding_for_model(model_name)
    elif model_name.startswith("claude-"):
        return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", clean_up_tokenization_spaces=False)
    elif model_name.startswith("llama-"):
        return AutoTokenizer.from_pretrained("huggyllama/llama-7b", clean_up_tokenization_spaces=False)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def estimate_tokens(text: str, model_name: str) -> int:
    try:
        tokenizer = get_tokenizer(model_name)
        return len(tokenizer.encode(text))
    except Exception as e:
        logging.warning(f"Error using tokenizer for {model_name}: {e}. Falling back to approximation.")
        return approximate_tokens(text)

def approximate_tokens(text: str) -> int:
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Split on whitespace and punctuation, keeping punctuation
    tokens = re.findall(r'\b\w+\b|\S', text)
    count = 0
    for token in tokens:
        if token.isdigit():
            count += max(1, len(token) // 2)  # Numbers often tokenize to multiple tokens
        elif re.match(r'^[A-Z]{2,}$', token):  # Acronyms
            count += len(token)
        elif re.search(r'[^\w\s]', token):  # Punctuation and special characters
            count += 1
        elif len(token) > 10:  # Long words often split into multiple tokens
            count += len(token) // 4 + 1
        else:
            count += 1
    # Add a 10% buffer for potential underestimation
    return int(count * 1.1)

def chunk_text(text: str, max_chunk_tokens: int, model_name: str) -> List[str]:
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    for sentence in re.split(r'(?<=[.!?])\s+', text):
        sentence_tokens = estimate_tokens(sentence, model_name)
        if sentence_tokens > max_chunk_tokens:
            # If a single sentence exceeds max_chunk_tokens, split it
            sentence_chunks = split_long_sentence(sentence, max_chunk_tokens, model_name)
            chunks.extend(sentence_chunks)
        elif current_chunk_tokens + sentence_tokens > max_chunk_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_chunk_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_chunk_tokens += sentence_tokens
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def split_long_sentence(sentence: str, max_tokens: int, model_name: str) -> List[str]:
    words = sentence.split()
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    for word in words:
        word_tokens = estimate_tokens(word, model_name)
        if current_chunk_tokens + word_tokens > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_chunk_tokens = word_tokens
        else:
            current_chunk.append(word)
            current_chunk_tokens += word_tokens
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

async def generate_completion_from_claude(prompt: str, max_tokens: int = CLAUDE_MAX_TOKENS - TOKEN_BUFFER) -> Optional[str]:
    if not ANTHROPIC_API_KEY:
        logging.error("Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
        return None
    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    prompt_tokens = estimate_tokens(prompt, CLAUDE_MODEL_STRING)
    adjusted_max_tokens = min(max_tokens, CLAUDE_MAX_TOKENS - prompt_tokens - TOKEN_BUFFER)
    if adjusted_max_tokens <= 0:
        logging.warning("Prompt is too long for Claude API. Chunking the input.")
        chunks = chunk_text(prompt, CLAUDE_MAX_TOKENS - TOKEN_CUSHION, CLAUDE_MODEL_STRING)
        results = []
        for chunk in chunks:
            try:
                async with client.messages.stream(
                    model=CLAUDE_MODEL_STRING,
                    max_tokens=CLAUDE_MAX_TOKENS // 2,
                    temperature=0.7,
                    messages=[{"role": "user", "content": chunk}],
                ) as stream:
                    message = await stream.get_final_message()
                    results.append(message.content[0].text)
                    logging.info(f"Chunk processed. Input tokens: {message.usage.input_tokens:,}, Output tokens: {message.usage.output_tokens:,}")
            except Exception as e:
                logging.error(f"An error occurred while processing a chunk: {e}")
        return " ".join(results)
    else:
        try:
            async with client.messages.stream(
                model=CLAUDE_MODEL_STRING,
                max_tokens=adjusted_max_tokens,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                message = await stream.get_final_message()
                output_text = message.content[0].text
                logging.info(f"Total input tokens: {message.usage.input_tokens:,}")
                logging.info(f"Total output tokens: {message.usage.output_tokens:,}")
                logging.info(f"Generated output (abbreviated): {output_text[:150]}...")
                return output_text
        except Exception as e:
            logging.error(f"An error occurred while requesting from Claude API: {e}")
            return None

async def generate_completion_from_openai(prompt: str, max_tokens: int = 1000) -> Optional[str]:
    if not OPENAI_API_KEY:
        logging.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return None
    prompt_tokens = estimate_tokens(prompt, OPENAI_COMPLETION_MODEL)
    adjusted_max_tokens = min(max_tokens, 4096 - prompt_tokens - TOKEN_BUFFER)  # 4096 is typical max for GPT-3.5 and GPT-4
    if adjusted_max_tokens <= 0:
        logging.warning("Prompt is too long for OpenAI API. Chunking the input.")
        chunks = chunk_text(prompt, OPENAI_MAX_TOKENS - TOKEN_CUSHION, OPENAI_COMPLETION_MODEL) 
        results = []
        for chunk in chunks:
            try:
                response = await openai_client.chat.completions.create(
                    model=OPENAI_COMPLETION_MODEL,
                    messages=[{"role": "user", "content": chunk}],
                    max_tokens=adjusted_max_tokens,
                    temperature=0.7,
                )
                result = response.choices[0].message.content
                results.append(result)
                logging.info(f"Chunk processed. Output tokens: {response.usage.completion_tokens:,}")
            except Exception as e:
                logging.error(f"An error occurred while processing a chunk: {e}")
        return " ".join(results)
    else:
        try:
            response = await openai_client.chat.completions.create(
                model=OPENAI_COMPLETION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=adjusted_max_tokens,
                temperature=0.7,
            )
            output_text = response.choices[0].message.content
            logging.info(f"Total tokens: {response.usage.total_tokens:,}")
            logging.info(f"Generated output (abbreviated): {output_text[:150]}...")
            return output_text
        except Exception as e:
            logging.error(f"An error occurred while requesting from OpenAI API: {e}")
            return None

async def generate_completion_from_openrouter(prompt: str, max_tokens: int = 1000) -> Optional[str]:
    if not OPENROUTER_API_KEY:
        logging.error("OpenRouter API key not found. Please set the OPENROUTER_API_KEY environment variable.")
        return None
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    }
    data = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=data,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error occurred: {e}")
        except httpx.RequestError as e:
            logging.error(f"An error occurred while requesting: {e}")
        except KeyError:
            logging.error("Unexpected response format from OpenRouter API")
        return None

async def generate_completion_from_local_llm(llm_model_name: str, input_prompt: str, number_of_tokens_to_generate: int = 100, temperature: float = 0.7, grammar_file_string: str = None):
    logging.info(f"Starting text completion using model: '{llm_model_name}' for input prompt: '{input_prompt}'")
    llm = load_model(llm_model_name)
    prompt_tokens = estimate_tokens(input_prompt, llm_model_name)
    adjusted_max_tokens = min(number_of_tokens_to_generate, LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - prompt_tokens - TOKEN_BUFFER)
    if adjusted_max_tokens <= 0:
        logging.warning("Prompt is too long for LLM. Chunking the input.")
        chunks = chunk_text(input_prompt, LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - TOKEN_CUSHION, llm_model_name)
        results = []
        for chunk in chunks:
            try:
                output = llm(
                    prompt=chunk,
                    max_tokens=LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS - TOKEN_CUSHION,
                    temperature=temperature,
                )
                results.append(output['choices'][0]['text'])
                logging.info(f"Chunk processed. Output tokens: {output['usage']['completion_tokens']:,}")
            except Exception as e:
                logging.error(f"An error occurred while processing a chunk: {e}")
        return " ".join(results)
    else:
        grammar_file_string_lower = grammar_file_string.lower() if grammar_file_string else ""
        if grammar_file_string_lower:
            list_of_grammar_files = glob.glob("./grammar_files/*.gbnf")
            matching_grammar_files = [x for x in list_of_grammar_files if grammar_file_string_lower in os.path.splitext(os.path.basename(x).lower())[0]]
            if len(matching_grammar_files) == 0:
                logging.error(f"No grammar file found matching: {grammar_file_string}")
                raise FileNotFoundError
            grammar_file_path = max(matching_grammar_files, key=os.path.getmtime)
            logging.info(f"Loading selected grammar file: '{grammar_file_path}'")
            llama_grammar = LlamaGrammar.from_file(grammar_file_path)
            output = llm(
                prompt=input_prompt,
                max_tokens=adjusted_max_tokens,
                temperature=temperature,
                grammar=llama_grammar
            )
        else:
            output = llm(
                prompt=input_prompt,
                max_tokens=adjusted_max_tokens,
                temperature=temperature
            )
        generated_text = output['choices'][0]['text']
        if grammar_file_string == 'json':
            generated_text = generated_text.encode('unicode_escape').decode()
        finish_reason = str(output['choices'][0]['finish_reason'])
        llm_model_usage_json = json.dumps(output['usage'])
        logging.info(f"Completed text completion in {output['usage']['total_time']:.2f} seconds. Beginning of generated text: \n'{generated_text[:150]}'...")
        return {
            "generated_text": generated_text,
            "finish_reason": finish_reason,
            "llm_model_usage_json": llm_model_usage_json
        }

async def generate_embedding_from_openai(text: str) -> Optional[List[float]]:
    if not OPENAI_API_KEY:
        logging.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return None
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENAI_EMBEDDING_MODEL,
                    "input": text
                },
                timeout=30.0  # Add a timeout to prevent hanging
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]  # This is already a list of floats
        except httpx.HTTPStatusError as e:  # noqa: F841
            pass  # Let the error handling code below handle HTTP errors
        except httpx.RequestError as e:  # noqa: F841
            pass  # Let the error handling code below handle request errors
        except KeyError:
            logging.error("Unexpected response format from OpenAI API")
        except Exception as e:
            logging.error(f"An unexpected error occurred while generating embedding: {e}")
        return None
    
async def generate_embedding_with_retry(text: str, max_retries: int = 3, base_delay: float = 1.0) -> Optional[List[float]]:
    for attempt in range(max_retries):
        try:
            embedding = await generate_embedding_from_openai(text)
            if embedding is not None:
                return embedding
        except Exception as e:
            logging.warning(f"Error generating embedding (attempt {attempt + 1}/{max_retries}): {e}")
        
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)  # Exponential backoff
            logging.info(f"Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
    logging.error(f"Failed to generate embedding after {max_retries} attempts")
    return None    
    
# Image Processing Functions
def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    return Image.fromarray(gray)

def convert_pdf_to_images(input_pdf_file_path: str, max_pages: int = 0, skip_first_n_pages: int = 0) -> List[Image.Image]:
    logging.info(f"Processing PDF file {input_pdf_file_path}")
    if max_pages == 0:
        last_page = None
        logging.info("Converting all pages to images...")
    else:
        last_page = skip_first_n_pages + max_pages
        logging.info(f"Converting pages {skip_first_n_pages + 1} to {last_page}")
    first_page = skip_first_n_pages + 1  # pdf2image uses 1-based indexing
    images = convert_from_path(input_pdf_file_path, first_page=first_page, last_page=last_page)
    logging.info(f"Converted {len(images)} pages from PDF file to images.")
    return images

def ocr_image(image):
    preprocessed_image = preprocess_image(image)
    return pytesseract.image_to_string(preprocessed_image)

def is_valid_english(text: str) -> bool:
    if not text:
        return False
    try:
        return detect(text) == 'en'
    except LangDetectException:
        logging.warning("Language detection failed. Assuming text is not valid English.")
        return False

async def calculate_sentence_embedding(text):
    max_retries = 3
    for attempt in range(max_retries):
        embedding = await generate_embedding_with_retry(text)
        if embedding is not None:
            return embedding  # This is already a list of floats
        logging.warning(f"Failed to generate embedding, attempt {attempt + 1} of {max_retries}")
        await asyncio.sleep(2 ** attempt)  # Exponential backoff
    logging.error(f"Failed to calculate embedding after {max_retries} attempts")
    return None

async def calculate_embeddings_batch(sentences: List[str]) -> np.ndarray:
    batch_size = 30  # Adjust based on API limits and performance
    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        batch_embeddings = await asyncio.gather(*[calculate_sentence_embedding(s) for s in batch])
        # Process and add valid embeddings
        for embedding in batch_embeddings:
            if embedding is not None:
                try:
                    embedding_array = np.array(embedding).astype('float32')
                    all_embeddings.append(embedding_array)
                except ValueError:
                    logging.warning(f"Failed to convert embedding to numpy array for a sentence in batch {i//batch_size + 1}")
    if not all_embeddings:
        logging.error("No valid embeddings were generated in the batch.")
        return np.array([])
    return np.vstack(all_embeddings)

def log_filtering_results(filtered_text: str, original_text: str) -> None:
    filtered_length = len(filtered_text)
    original_length = len(original_text)
    diff = original_length - filtered_length
    logging.info(f"Filtered text length: {filtered_length:,}; original text length: {original_length:,}")
    logging.info(f"Difference in length: {diff:,} characters ({(diff / original_length) * 100:.2f}% reduction)")
    logging.info(f"Percentage of original text retained: {(filtered_length / original_length) * 100:.2f}%")

async def smart_split_into_chunks(text: str, chunk_size: int = 1000, max_retries: int = 3) -> List[Tuple[str, bool]]:
    async def get_valid_json_response(prompt: str) -> Optional[List[Dict[str, Any]]]:
        for attempt in range(max_retries):
            try:
                response = await generate_completion(prompt, max_tokens=2000)
                # Remove any non-JSON content
                json_start = response.find('[')
                json_end = response.rfind(']') + 1
                if json_start != -1 and json_end != -1:
                    response = response[json_start:json_end]
                chunks = json.loads(response)
                return chunks
            except json.JSONDecodeError:
                logging.warning(f"Failed to parse JSON response (attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    logging.error("Max retries reached. Failed to get valid JSON response.")
                    return None
            await asyncio.sleep(1)  # Add a small delay between retries

    prompt = f"""Analyze the following text and split it into logical chunks of approximately {chunk_size} characters each. 
    Preserve the document structure, including headers, paragraphs, and formatting.
    For each chunk, determine if it's a header (true) or content (false).
    Output the result as a JSON array of objects, each with 'text' and 'is_header' properties.
    Ensure that headers are always in their own chunk.

    IMPORTANT: YOU MUST RESPOND ONLY WITH VALID JSON. DO NOT INCLUDE ANY INTRODUCTION, EXPLANATION, OR EXTRA TEXT. ONLY PROVIDE THE JSON ARRAY.

    Text to analyze:
    {text[:10000]}  # Limit input to avoid exceeding token limits

    Output format:
    [
        {{"text": "chunk1 text here", "is_header": false}},
        {{"text": "## Header", "is_header": true}},
        {{"text": "chunk2 text here", "is_header": false}}
    ]
    """

    chunks = await get_valid_json_response(prompt)
    
    if chunks is None:
        logging.error("Failed to get valid JSON response after multiple attempts. Falling back to simple split.")
        return [(text[i:i+chunk_size], False) for i in range(0, len(text), chunk_size)]
    
    return [(chunk['text'], chunk['is_header']) for chunk in chunks]

async def filter_hallucinations(corrected_text: str, raw_text: str, threshold: float = 0.7, pdf_file_path: Optional[str] = None, db_path: Optional[str] = None) -> str:
    logging.info(f"Starting optimized FAISS-based hallucination filtering with smart chunking. Threshold: {threshold}")
    
    corrected_chunks = await smart_split_into_chunks(corrected_text)
    raw_chunks = await smart_split_into_chunks(raw_text)

    # Process non-header chunks
    corrected_text_chunks = [chunk for chunk, is_header in corrected_chunks if not is_header]
    raw_text_chunks = [chunk for chunk, is_header in raw_chunks if not is_header]

    # Build FAISS index for original chunks
    raw_embeddings = await calculate_embeddings_batch(raw_text_chunks)
    if raw_embeddings.size == 0:
        logging.error("No valid embeddings were generated for the original text.")
        return corrected_text

    faiss.normalize_L2(raw_embeddings)
    index = faiss.IndexFlatIP(raw_embeddings.shape[1])
    index.add(raw_embeddings)

    # Process corrected chunks
    corrected_embeddings = await calculate_embeddings_batch(corrected_text_chunks)
    if corrected_embeddings.size == 0:
        logging.error("No valid embeddings were generated for the corrected text.")
        return corrected_text

    faiss.normalize_L2(corrected_embeddings)
    similarities, _ = index.search(corrected_embeddings, 1)
    similarities = similarities.flatten()

    # Filter chunks
    filtered_chunks = []
    non_header_index = 0
    for chunk, is_header in corrected_chunks:
        if is_header:
            filtered_chunks.append(chunk)
        else:
            if similarities[non_header_index] >= threshold:
                filtered_chunks.append(chunk)
            else:
                logging.debug(f"Chunk filtered out. Similarity: {similarities[non_header_index]:.4f}")
            non_header_index += 1

    # Join filtered chunks
    filtered_text = '\n'.join(filtered_chunks)

    # Log metrics
    log_filtering_results(filtered_text, raw_text)
    logging.info(f"Similarities - Mean: {np.mean(similarities):.4f}, Median: {np.median(similarities):.4f}, Min: {min(similarities):.4f}, Max: {max(similarities):.4f}")
    logging.info(f"Chunks before filtering: {len(corrected_text_chunks):,}, after filtering: {len(filtered_chunks):,}")
    logging.info(f"Percentage of chunks retained: {(len(filtered_chunks) / len(corrected_text_chunks)) * 100:.2f}%")

    return filtered_text

async def process_chunk(chunk: str, prev_context: str, chunk_index: int, total_chunks: int, check_if_valid_english: bool, reformat_as_markdown: bool, suppress_headers_and_page_numbers: bool) -> Tuple[str, str]:
    logging.info(f"Processing chunk {chunk_index}/{total_chunks} (length: {len(chunk):,} characters)")
    
    # Step 1: OCR Correction
    ocr_correction_prompt = f"""Correct OCR-induced errors in the text, ensuring it flows coherently with the previous context. Follow these guidelines:

1. Fix OCR-induced typos and errors:
   - Correct words split across line breaks
   - Fix common OCR errors (e.g., 'rn' misread as 'm')
   - Use context and common sense to correct errors
   - Only fix clear errors, don't alter the content unnecessarily
   - Do not add extra periods or any unnecessary punctuation

2. Maintain original structure:
   - Keep all headings and subheadings intact

3. Preserve original content:
   - Keep all important information from the original text
   - Do not add any new information not present in the original text
   - Remove unnecessary line breaks within sentences or paragraphs
   - Maintain paragraph breaks
   
4. Maintain coherence:
   - Ensure the content connects smoothly with the previous context
   - Handle text that starts or ends mid-sentence appropriately

IMPORTANT: Respond ONLY with the corrected text. Preserve all original formatting, including line breaks. Do not include any introduction, explanation, or metadata.

Previous context:
{prev_context[-300:]}

Current chunk to process:
{chunk}

Corrected text:
"""
    
    ocr_corrected_chunk = await generate_completion(ocr_correction_prompt, max_tokens=len(chunk) + 500)
    
    processed_chunk = ocr_corrected_chunk

    # Step 2: Markdown Formatting (if requested)
    if reformat_as_markdown:
        markdown_prompt = f"""Reformat the following text as markdown, improving readability while preserving the original structure. Follow these guidelines:
1. Preserve all original headings, converting them to appropriate markdown heading levels (# for main titles, ## for subtitles, etc.)
   - Ensure each heading is on its own line
   - Add a blank line before and after each heading
2. Maintain the original paragraph structure. Remove all breaks within a word that should be a single word (for example, "cor- rect" should be "correct")
3. Format lists properly (unordered or ordered) if they exist in the original text
4. Use emphasis (*italic*) and strong emphasis (**bold**) where appropriate, based on the original formatting
5. Preserve all original content and meaning
6. Do not add any extra punctuation or modify the existing punctuation
7. {"Identify but do not remove headers, footers, or page numbers. Instead, format them distinctly, e.g., as blockquotes." if not suppress_headers_and_page_numbers else "Carefully remove headers, footers, and page numbers while preserving all other content."}

Text to reformat:

{ocr_corrected_chunk}

Reformatted markdown:
"""
        processed_chunk = await generate_completion(markdown_prompt, max_tokens=len(ocr_corrected_chunk) + 500)
    new_context = processed_chunk[-1000:]  # Use the last 1000 characters as context for the next chunk
    logging.info(f"Chunk {chunk_index}/{total_chunks} processed. Output length: {len(processed_chunk):,} characters")
    return processed_chunk, new_context

async def process_chunks(chunks: List[str], check_if_valid_english: bool, reformat_as_markdown: bool, suppress_headers_and_page_numbers: bool) -> List[str]:
    total_chunks = len(chunks)
    async def process_chunk_with_context(chunk: str, prev_context: str, index: int) -> Tuple[int, str, str]:
        processed_chunk, new_context = await process_chunk(chunk, prev_context, index, total_chunks, check_if_valid_english, reformat_as_markdown, suppress_headers_and_page_numbers)
        return index, processed_chunk, new_context
    if USE_LOCAL_LLM:
        logging.info("Using local LLM. Processing chunks sequentially...")
        context = ""
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunk, context = await process_chunk(chunk, context, i, total_chunks, check_if_valid_english, reformat_as_markdown, suppress_headers_and_page_numbers)
            processed_chunks.append(processed_chunk)
    else:
        logging.info("Using API-based LLM. Processing chunks concurrently while maintaining order...")
        tasks = [process_chunk_with_context(chunk, "", i) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        # Sort results by index to maintain order
        sorted_results = sorted(results, key=lambda x: x[0])
        processed_chunks = [chunk for _, chunk, _ in sorted_results]
    logging.info(f"All {total_chunks} chunks processed successfully")
    return processed_chunks

async def process_document(list_of_extracted_text_strings: List[str], check_if_valid_english: bool = False, reformat_as_markdown: bool = True, suppress_headers_and_page_numbers: bool = True) -> str:
    logging.info(f"Starting document processing. Total pages: {len(list_of_extracted_text_strings):,}")
    full_text = "\n\n".join(list_of_extracted_text_strings)
    logging.info(f"Size of full text before processing: {len(full_text):,} characters")
    chunk_size, overlap = 8000, 1000
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        if end < len(full_text):
            end = full_text.rfind(' ', start, end) + 1
        chunks.append(full_text[start:end])
        start = end - overlap
    logging.info(f"Document split into {len(chunks):,} chunks. Chunk size: {chunk_size:,}, Overlap: {overlap:,}")
    processed_chunks = await process_chunks(chunks, check_if_valid_english, reformat_as_markdown, suppress_headers_and_page_numbers)
    final_text = "".join(processed_chunks)
    logging.info(f"Size of text after combining chunks: {len(final_text):,} characters")
    logging.info(f"Document processing complete. Final text length: {len(final_text):,} characters")
    return final_text

async def smart_post_process_output(text: str, chunk_size: int = 5000) -> str:
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    async def process_chunk(chunk, index):
        prompt = f"""Review and refine the following text chunk ({index+1}/{len(chunks)}), focusing on improving its formatting and readability. Follow these guidelines:

        1. Ensure proper Markdown formatting:
           - Headers should be on their own lines with appropriate spacing before and after.
           - Maintain consistent indentation for list items and code blocks.

        2. Fix any formatting inconsistencies:
           - Remove unnecessary line breaks within paragraphs.
           - Ensure there's a blank line between paragraphs and different sections.

        3. Correct punctuation and spacing:
           - Remove any double periods.
           - Ensure proper spacing after periods, commas, and other punctuation marks.

        4. Preserve the original content and meaning of the text.

        5. Do not add any new content or remove any existing content.

        6. Ensure smooth transitions between chunks.

        IMPORTANT: Provide ONLY the refined text chunk without any introduction, explanation, or metadata.

        Text chunk to process:

        {chunk}

        Refined text chunk:
        """

        refined_chunk = await generate_completion(prompt, max_tokens=len(chunk) + 500)
        
        if 0:
            # Apply some basic checks as a fallback
            refined_chunk = re.sub(r'\.{2,}', '.', refined_chunk)  # Remove any remaining double periods
            refined_chunk = re.sub(r'(?<=[.!?])(?=[A-Z])', ' ', refined_chunk)  # Ensure space after sentence-ending punctuation
            
        return index, refined_chunk.strip()

    # Process all chunks in parallel
    results = await asyncio.gather(*[process_chunk(chunk, i) for i, chunk in enumerate(chunks)])
    
    # Sort results by index and join
    processed_chunks = [chunk for _, chunk in sorted(results, key=lambda x: x[0])]
    result = "\n\n".join(processed_chunks)

    # Final pass to ensure consistency across the entire document
    final_prompt = f"""Review the entire processed document for overall consistency and make any final adjustments:

    1. Ensure consistent formatting throughout the document.
    2. Fix any remaining punctuation or spacing issues.
    3. Do not add any new content or remove any existing content.
    4. Do not add any introductions, explanations, or metadata.

    IMPORTANT: Provide ONLY the final refined document without any introduction or explanation.

    Here's a sample of the document (it may be truncated):
    {result[:10000]}

    Final refined document:
    """

    final_result = await generate_completion(final_prompt, max_tokens=len(result) + 500)

    return final_result.strip()

async def smart_remove_duplicates(text: str, chunk_size: int = 5000, overlap: int = 500) -> str:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            end = text.rfind('\n', start + chunk_size - overlap, end) + 1
        chunks.append(text[start:end])
        start = end - overlap

    async def process_chunk(chunk, index, prev_chunk, next_chunk):
        context = f"{prev_chunk[-200:]}\n\n{chunk}\n\n{next_chunk[:200]}"
        prompt = f"""Review the following text chunk and its surrounding context. Remove ONLY obviously duplicated content that appears to have been accidentally included twice. Follow these strict guidelines:

        1. Remove only exact or near-exact repeated paragraphs or sections within the main chunk.
        2. Consider the context (before and after the main chunk) to identify duplicates that span chunk boundaries.
        3. Do not remove content that is simply similar but conveys different information.
        4. Preserve all unique content, even if it seems redundant.
        5. Ensure the text flows smoothly after removing duplicates.
        6. Do not add any new content or explanations.
        7. If no obvious duplicates are found, return the main chunk unchanged.
        8. When removing a duplicate, include a comment: [DUPLICATE REMOVED] to mark where content was deleted.

        IMPORTANT: Process and return ONLY the main chunk (between the --- markers). Do not modify the context sections.

        Context and text chunk to process:

        {context}

        De-duplicated main chunk (only return the part between the --- markers):
        """

        de_duped_chunk = await generate_completion(prompt, max_tokens=len(chunk) + 500)
        
        # Extract only the main chunk from the response
        main_chunk_start = de_duped_chunk.find("---")
        main_chunk_end = de_duped_chunk.rfind("---")
        if main_chunk_start != -1 and main_chunk_end != -1:
            de_duped_chunk = de_duped_chunk[main_chunk_start+3:main_chunk_end].strip()
        
        return index, de_duped_chunk

    # Process all chunks in parallel
    tasks = []
    for i, chunk in enumerate(chunks):
        prev_chunk = chunks[i-1] if i > 0 else ""
        next_chunk = chunks[i+1] if i < len(chunks) - 1 else ""
        tasks.append(process_chunk(chunk, i, prev_chunk, next_chunk))
    
    results = await asyncio.gather(*tasks)
    
    # Sort results by index and join
    processed_chunks = [chunk for _, chunk in sorted(results, key=lambda x: x[0])]
    
    # Remove overlap
    final_chunks = []
    for i, chunk in enumerate(processed_chunks):
        if i < len(processed_chunks) - 1:
            next_chunk = processed_chunks[i+1]
            overlap_end = next_chunk.find('\n', overlap) + 1
            chunk = chunk[:-overlap] + next_chunk[:overlap_end]
        final_chunks.append(chunk)
    
    result = "".join(final_chunks)

    return result.strip()

def check_deduplication_safety(original_text: str, de_duped_text: str, max_removal_percentage: float = 20.0):
    original_length = len(original_text)
    de_duped_length = len(de_duped_text)
    removed_percentage = ((original_length - de_duped_length) / original_length) * 100 if original_length > 0 else 0
    
    if removed_percentage > max_removal_percentage:
        logging.warning(f"Deduplication removed {removed_percentage:.2f}% of content, which exceeds the safety threshold of {max_removal_percentage}%.")
        logging.warning("Reverting to original text to prevent excessive content loss.")
        return original_text
    return de_duped_text

async def assess_output_quality(original_text, processed_text):
    # Limit the input text to avoid exceeding token limits
    max_chars = 8000
    original_sample = original_text[:max_chars]
    processed_sample = processed_text[:max_chars]
    
    prompt = f"""Compare the following samples of original OCR text with the processed output and assess the quality of the processing. Consider the following factors:
1. Accuracy of error correction
2. Improvement in readability
3. Preservation of original content and meaning
4. Appropriate use of markdown formatting (if applicable)
5. Removal of hallucinations or irrelevant content

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

    response = await generate_completion(prompt, max_tokens=300)
    
    try:
        lines = response.strip().split('\n')
        score_line = next(line for line in lines if line.startswith('SCORE:'))
        score = int(score_line.split(':')[1].strip())
        explanation = '\n'.join(line for line in lines if line.startswith('EXPLANATION:')).replace('EXPLANATION:', '').strip()
        logging.info(f"Quality assessment: Score {score}/100")
        logging.info(f"Explanation: {explanation}")
        return score, explanation
    except Exception as e:
        logging.error(f"Error parsing quality assessment response: {e}")
        logging.error(f"Raw response: {response}")
        return None, None

async def main():
    try:
        # Suppress HTTP request logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        input_pdf_file_path = '160301289-Warren-Buffett-Katharine-Graham-Letter.pdf'
        max_test_pages = 0
        skip_first_n_pages = 0
        starting_hallucination_similarity_threshold = 0.40
        check_if_valid_english = False
        reformat_as_markdown = True
        suppress_headers_and_page_numbers = True
        sentence_embeddings_db_path = "./sentence_embeddings.sqlite"
        test_filtering_hallucinations = False
        
        # Download the model if using local LLM
        if USE_LOCAL_LLM:
            _, download_status = await download_models()
            logging.info(f"Model download status: {download_status}")
            logging.info(f"Using Local LLM with Model: {DEFAULT_LOCAL_MODEL_NAME}")
        else:
            logging.info(f"Using API for completions: {API_PROVIDER}")
            logging.info(f"Using OpenAI model for embeddings: {OPENAI_EMBEDDING_MODEL}")

        base_name = os.path.splitext(input_pdf_file_path)[0]
        output_extension = '.md' if reformat_as_markdown else '.txt'
        
        if not test_filtering_hallucinations:
            list_of_scanned_images = convert_pdf_to_images(input_pdf_file_path, max_test_pages, skip_first_n_pages)
            logging.info(f"Tesseract version: {pytesseract.get_tesseract_version()}")
            logging.info("Extracting text from converted pages...")
            with ThreadPoolExecutor() as executor:
                list_of_extracted_text_strings = list(executor.map(ocr_image, list_of_scanned_images))
            logging.info("Done extracting text from converted pages.")
            raw_ocr_output = "\n".join(list_of_extracted_text_strings)
            raw_ocr_output_file_path = f"{base_name}__raw_ocr_output.txt"
            with open(raw_ocr_output_file_path, "w") as f:
                f.write(raw_ocr_output)
            
            logging.info("Processing document...")
            final_text = await process_document(list_of_extracted_text_strings, check_if_valid_english, reformat_as_markdown, suppress_headers_and_page_numbers)            
            
            # Save the pre-filtered output
            pre_filtered_output_file_path = base_name + '_pre_filtered' + output_extension
            with open(pre_filtered_output_file_path, 'w') as f:
                f.write(final_text)
            logging.info(f"Pre-filtered LLM corrected text written to: {pre_filtered_output_file_path}")
            logging.info(f"First 500 characters of pre-filtered processed text:\n{final_text[:500]}...")
        else:  # For debugging
            raw_ocr_output_file_path = f"{base_name}__raw_ocr_output.txt"        
            pre_filtered_output_file_path = base_name + '_pre_filtered' + output_extension
            with open(pre_filtered_output_file_path, 'r') as f:
                final_text = f.read()
            with open(raw_ocr_output_file_path, 'r') as f:
                raw_ocr_output = f.read()

        # New step: Remove duplicates
        logging.info('Removing duplicate content...')
        de_duped_text = await smart_remove_duplicates(final_text)
        logging.info('Done removing duplicates.')

        de_duped_text = check_deduplication_safety(final_text, de_duped_text)

        # Save the de-duplicated output
        deduped_output_file_path = base_name + '_deduped' + output_extension
        with open(deduped_output_file_path, 'w') as f:
            f.write(de_duped_text)
        logging.info(f"De-duplicated text written to: {deduped_output_file_path}")
        
        logging.info('Now filtering out hallucinations from de-duplicated text...')
        filtered_output = await filter_hallucinations(de_duped_text, raw_ocr_output, starting_hallucination_similarity_threshold, input_pdf_file_path, sentence_embeddings_db_path)
        logging.info('Done filtering out hallucinations.')
        
        logging.info('Post-processing the filtered output...')
        post_processed_output = await smart_post_process_output(filtered_output)
        logging.info('Done post-processing.')
        
        final_output_file_path = base_name + '_post_filtered' + output_extension
        with open(final_output_file_path, 'w') as f:
            f.write(post_processed_output)
        logging.info(f"Post-filtered text written to: {final_output_file_path}")
        logging.info(f"First 500 characters of post-filtered text:\n{post_processed_output[:500]}...")

        logging.info(f"Done processing {input_pdf_file_path}.")
        logging.info("\nSee output files:")
        logging.info(f" Raw OCR: {raw_ocr_output_file_path}")
        logging.info(f" Pre-filtered LLM Corrected: {pre_filtered_output_file_path}")
        logging.info(f" De-duplicated: {deduped_output_file_path}")
        logging.info(f" Post-filtered LLM Corrected: {final_output_file_path}")

        # Perform a final quality check
        quality_score, explanation = await assess_output_quality(raw_ocr_output, filtered_output)
        if quality_score is not None:
            logging.info(f"Final quality score: {quality_score}/100")
            logging.info(f"Explanation: {explanation}")
        else:
            logging.warning("Unable to determine final quality score.")
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
        logging.error(traceback.format_exc())

if __name__ == '__main__':
    asyncio.run(main())