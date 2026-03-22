# Changelog

All notable changes to the [LLM-Aided OCR](https://github.com/Dicklesworthstone/llm_aided_ocr) project are documented in this file.

This project has no formal releases or semantic version tags. History is organized by logical milestones derived from exhaustive review of the commit record. Every commit hash links to its GitHub page.

---

## 2026-02-21 / 2026-02-22 -- Licensing and branding

### Licensing

- Replace the plain MIT license with an **MIT + OpenAI/Anthropic Rider** that restricts use by OpenAI, Anthropic, and their affiliates without express written permission from Jeffrey Emanuel ([`99086ee`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/99086eea99c36e81845f13879f8ff8a5f8c5a608))
- Update all README references to reflect the new "MIT License (with OpenAI/Anthropic Rider)" language ([`841913d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/841913d4089246926cf217070d352542a9f69d06))

### Social / repository metadata

- Add GitHub social preview image (`gh_og_share_image.png`, 1280x640) for consistent link previews when sharing the repository URL ([`0554334`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/05543348707bdde91d05e40a5a24b76eeb1b732e))

---

## 2026-01-14 / 2026-01-18 / 2026-01-21 -- Stability fixes and project hygiene

### Crash and type-safety fixes

Six distinct crash paths -- reported across issues #11, #20, #21, and #24 -- were eliminated in two commits.

- **Return type consistency**: `generate_completion_from_local_llm()` previously returned either a `dict` or a `str` depending on code path; it now always returns `Optional[str]`, extracting `generated_text` from the dict before returning ([`f5bb024`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/f5bb0249ba5d3c4aec5ff52c0efc10daed92a80c))
- **Negative `max_tokens` prevention**: wrap token arithmetic in `max(1, ...)` across all three provider paths (OpenAI, Claude, local LLM) to prevent passing zero or negative values ([`f5bb024`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/f5bb0249ba5d3c4aec5ff52c0efc10daed92a80c))
- **Chunking condition**: fix the guard that decides whether input needs to be chunked so it fires reliably ([`f5bb024`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/f5bb0249ba5d3c4aec5ff52c0efc10daed92a80c))
- **None propagation in `process_chunk()`**: add None check after `generate_completion()` with fallback to original text instead of crashing on `len(None)` or `None[-1000:]` ([`f5bb024`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/f5bb0249ba5d3c4aec5ff52c0efc10daed92a80c))
- **None propagation in `assess_output_quality()`**: guard against `response.strip().split()` on None ([`f5bb024`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/f5bb0249ba5d3c4aec5ff52c0efc10daed92a80c))
- **Empty results in Claude chunking fallback**: when all chunks fail during Claude API processing, return `None` instead of empty string, consistent with OpenAI and local LLM paths ([`6c1937d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/6c1937dd4ff836e0762d57621fa4c568d36ba3f3))

### Dependency maintenance

- Add `UPGRADE_LOG.md` documenting dependency review status; project uses unpinned `requirements.txt` ([`80cdd7d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/80cdd7d4eea0e9ca5f9823800d28b37662e0a879))

### License

- Add initial MIT License file ([`282c85b`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/282c85b0309a484f8e03e8f83c4b15a2b4c94ced))

---

## 2025-02-27 -- README refresh

### Documentation

- Add commercial project links (YoutubeTranscriptOptimizer.com, FixMyDocuments.com) to README footer ([`d8c140c`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d8c140c9fc2d2ff42e95751403de1aab00215607))
- Fix typo and em-dash formatting in README footer text ([`cc0eb67`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/cc0eb6735b7f582aaaf53b234a806b39f5bb0917))

---

## 2024-08-20 -- Docker support (community contribution, PR #16)

### Containerization

- Add `docker/Dockerfile` with NVIDIA CUDA 12.1 + cuDNN base image, Tesseract installation, Miniconda environment, and full project setup ([`a2f064d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a2f064d53619535c269ca395fd06593aa3886e54))
- Add `docker/docker-compose.yml` with GPU passthrough and SSH service for remote access ([`a2f064d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a2f064d53619535c269ca395fd06593aa3886e54))
- Merged via PR #16 from [hotwa](https://github.com/hotwa) ([`c79f6b7`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/c79f6b709a8db62903f55647bec4816823b86b9e))

---

## 2024-08-09 -- Community fixes and polish

### Tokenizer robustness (community contribution, PR #9)

- `get_tokenizer()` now uses `.lower().startswith()` for model name matching, so names like `"Claude-3-haiku..."` or `"GPT-4o-mini"` are detected correctly; previously only exact lowercase prefixes worked. Contributed by [Backendmagier](https://github.com/Backendmagier) ([`fcee75b`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fcee75b48aff2fb9c04e3301efb19d7917a73a50), merged at [`a3fd34d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a3fd34d4acc2044a15ecdc855cac8c205e0cdfe2))

### Prompt and documentation cleanup

- Fix typo in markdown formatting prompt: "Remove and obviously" changed to "Remove any obviously" ([`850505c`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/850505c9ed2bd57edbe829b6a734fa31a0bc36c4))
- README wording updates ([`537c2d1`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/537c2d16ac15421d2e39e583511ff4c2e24b1e54), [`6a9d6e3`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/6a9d6e3be7bfbd194e0a37bdf227fdc2c4526077))

---

## 2024-08-07 / 2024-08-09 -- v2: complete rewrite and stabilization

The original 236-line Llama-2-only script (`tesseract_with_llama2_corrections.py`) was replaced with a fully async, multi-provider system. After the initial rewrite ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e)), a rapid stabilization pass ([`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e) and follow-ups) renamed files, pruned unused features, and refined chunking. Together these commits represent the single largest change in the project's history.

### LLM provider support

- **Anthropic Claude** via `AsyncAnthropic` streaming with automatic chunking fallback when prompts exceed model limits ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **OpenAI** via `AsyncOpenAI` with configurable model selection (default `gpt-4o-mini`) ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **OpenRouter** via HTTP (added in rewrite, removed during stabilization) ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e), removed in [`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))
- **Local LLM backend** upgraded from Llama 2 7B/13B (GGML) to **Llama 3.1 8B** (GGUF) with automatic GPU-to-CPU fallback ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- Unified dispatch via `generate_completion()` that routes to local, Claude, or OpenAI based on `.env` configuration ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- Default API provider changed from OpenAI to Claude during stabilization ([`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))

### Processing pipeline

- **Two-step correction**: (1) LLM-based OCR error correction preserving structure, then (2) optional markdown formatting with headings, lists, and emphasis -- each step uses its own LLM prompt ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **Async chunk processing** via `asyncio` for concurrent API calls while maintaining chunk ordering ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **Smart text chunking** with sentence-boundary splitting, configurable overlap, and chunk-overlap adjustment to maintain cross-boundary context ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **Improved chunking** during stabilization: paragraph-first splitting with sentence-level fallback, chunk size raised from 2000 to 4000 characters ([`c595280`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/c5952809176edf4a4e9ca614dfc90ae9f458f410))
- **Duplicate content removal** within the markdown formatting step ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **Header and page number suppression** as a configurable option ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **Quality assessment** (`assess_output_quality()`) comparing final processed text against raw OCR via LLM scoring ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))

### Token management

- `estimate_tokens()` with model-specific tokenizers: `tiktoken` for GPT models, HuggingFace `AutoTokenizer` for Claude and Llama ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- `approximate_tokens()` heuristic fallback handling digits, acronyms, long words, and punctuation with 10% buffer ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- Dynamic `max_tokens` adjustment based on prompt length and model limits; `TOKEN_BUFFER` (500) and `TOKEN_CUSHION` (100, raised to 300 during stabilization) constants ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e), [`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))
- OpenAI max tokens raised from 4096 to 12000; default completion max_tokens raised from 1000 to 5000 ([`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))

### Image preprocessing

- OpenCV-based `preprocess_image()`: grayscale conversion, Otsu binary thresholding, morphological dilation for enhanced OCR clarity ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))

### GPU support

- GPU detection via `nvgpu` with automatic fallback to CPU when unavailable or when GPU model loading fails ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))

### Configuration and CLI

- `.env`-based configuration via `python-decouple` for API keys, model names, provider selection, and token limits ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- CLI wrapper with argparse for PDF path, page range, hallucination threshold, and formatting options; initially `tesseract-llm-cli.py`, renamed to `llm-aided-ocr-cli.py` during stabilization ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e), [`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))

### Project structure

- Main script renamed: `tesseract_with_llm_corrections.py` to `llm_aided_ocr.py` ([`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))
- `.gitignore` added with comprehensive Python exclusions ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- `.python-version` set to Python 3.12 ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- LLM-corrected example output added (`160301289-Warren-Buffett-Katharine-Graham-Letter_llm_corrected.md`) ([`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))
- README fully rewritten to document the new multi-provider architecture ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e), [`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))

### Dependencies

- Added: `anthropic`, `openai`, `tiktoken`, `httpx`, `faiss-cpu`, `filelock`, `transformers`, `opencv-python-headless`, `langdetect`, `nltk`, `numpy`, `nvgpu`, `python-decouple`, `ruff` ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- Removed during stabilization: `httpx`, `langdetect`, `faiss-cpu`, `nltk` (embedding/hallucination features pruned) ([`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))
- Carried forward from v1: `llama-cpp-python`, `pdf2image`, `pytesseract`, `pillow` ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))

### Removed from v1

- Entire `tesseract_with_llama2_corrections.py` codebase ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- v1 hallucination-filtering system (LangChain embeddings + cosine similarity + SQLite cache) ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- FAISS-based embedding infrastructure and `generate_embedding()` function (added in rewrite, removed during stabilization) ([`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))
- NLTK sentence tokenizer, replaced with regex-based splitting ([`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))

### Stabilization quick-fixes

Rapid-fire fixes immediately following the refactor, addressing README cleanup, chunking refinement, and configuration adjustments:
[`80d828b`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/80d828bc13aeeae31d1f547c058f3d9727c434f5),
[`fa02759`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fa02759a695ca0dacea9eeaf8469ff8b4bfaf4ea),
[`a46e344`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a46e344a157d469c3b2af07546f15f911eb51c77),
[`c595280`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/c5952809176edf4a4e9ca614dfc90ae9f458f410),
[`409a1d1`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/409a1d1402d6725ec2ec308af6321df99055b02f)

---

## 2023-08-01 / 2023-08-02 -- v1 maturation: hallucination filtering and embedding cache

Rapid iteration that grew the script from 60 lines to 236 lines, adding a hallucination-detection system and example documents. All changes to `tesseract_with_llama2_corrections.py`.

### OCR and extraction

- **Multiprocessing OCR** via `Pool().map()` for parallel page extraction across all converted images ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e))
- Full-document processing support: `max_test_pages=0` now processes all pages instead of requiring a fixed limit ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e))
- Page-length thresholds lowered (min chars from 30 to 10, min words from 20 to 5) for more inclusive extraction ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e))

### Text correction and formatting

- **Markdown reformatting step** as an optional second LLM pass after OCR correction ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e))
- **English-language validation** via LLM yes/no query before attempting correction ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e))
- **LLM intro-text stripping** (`remove_intro()`) to drop spurious preamble added by the model ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e))
- Improved exception handling in `remove_intro()` to prevent crashes on unexpected LLM output format ([`4c2fe3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/4c2fe3de0e39d21e4f9e5472bca7228601fa386b))

### Hallucination detection

- **Cosine similarity filtering** using LangChain `LlamaCppEmbeddings` + scikit-learn to compare corrected sentences against original OCR, with a configurable similarity threshold ([`4c2fe3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/4c2fe3de0e39d21e4f9e5472bca7228601fa386b))
- **SQLite-backed embedding cache** with SHA-3 file hashing for incremental re-runs without recomputing embeddings ([`4392c4a`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/4392c4a588a221c63148aa4b661fa769d8b15eaf))
- Adaptive threshold search: starts at a base threshold and increments until filtered text length drops below the original, then rolls back one step ([`4c2fe3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/4c2fe3de0e39d21e4f9e5472bca7228601fa386b))

### Model and output

- LLM model upgraded from Llama 2 7B (GGML q4_0) to Llama 2 13B-chat (GGML q5_K_S) for better correction quality ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e), [`5719a9a`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/5719a9aede6b0666f6f08d239cac7b1550298b79))
- Automatic output naming based on input PDF basename with `.md` or `.txt` extension ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e))
- Three-file output: raw OCR, LLM-corrected, and hallucination-filtered ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e))
- Progress bars via `tqdm` for embedding computation and filtering ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e))

### Example documents

- Warren Buffett / Katharine Graham letter PDF with raw OCR output and filtered markdown result ([`84bcbb6`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/84bcbb67d346ee3bbfbde90fa2ce9f753043ada6), [`e572391`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/e5723917db9f2c891bf28097bc8491451f2c1143), [`a98be94`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a98be946cbaba0abeb596d2cb7593e41ca758bd9))

### Project structure

- Added `if __name__ == '__main__':` guard for clean imports ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e))
- Dependencies: added `langchain`, `scikit-learn`, `tqdm`, `sqlite3`; removed `pandas` ([`89a3492`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/89a3492559d107b8794b24f9bc7ad3d421539c31), [`900283a`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/900283aebe99fa9179195a669ea0afc04e4de417))
- Extensive README documentation covering the full pipeline, hallucination filtering, and configuration ([`9106c6b`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/9106c6bf1b7df7af4245d119382bb866ecd9872d) through [`fdab1da`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fdab1da15ebf04479baabea5e43d5b2cc83d83f1))

---

## 2023-07-26 -- Initial release (v1)

60-line proof of concept: Tesseract OCR with Llama 2 7B (GGML) for typo correction.

### Core capabilities

- `tesseract_with_llama2_corrections.py`: PDF-to-image conversion via `pdf2image`, text extraction via `pytesseract`, and LLM-based typo correction via `llama_cpp` with Llama 2 7B GGML ([`315faed`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/315faed5d55cbc4511f05c93eda74ce2f2c05815))
- Simple prompt-based correction: wraps extracted text in a single prompt asking the LLM to fix OCR typos using common sense reasoning ([`315faed`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/315faed5d55cbc4511f05c93eda74ce2f2c05815))
- Basic page validation: skips pages with fewer than 30 characters or 20 words ([`315faed`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/315faed5d55cbc4511f05c93eda74ce2f2c05815))
- Configurable page range with `max_test_pages` and `skip_first_n_pages` ([`315faed`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/315faed5d55cbc4511f05c93eda74ce2f2c05815))

### Dependencies

- `requirements.txt`: `pdf2image`, `pytesseract`, `llama-cpp-python`, `pandas`, `opencv-python`, `pillow` ([`315faed`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/315faed5d55cbc4511f05c93eda74ce2f2c05815))

### Documentation

- Initial `README.md` ([`0cfc2ed`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/0cfc2edce229ec0d1e2d775bbf68a59c2a1ada19))
