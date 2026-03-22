# Changelog

All notable changes to the [LLM-Aided OCR](https://github.com/Dicklesworthstone/llm_aided_ocr) project are documented in this file.

This project has no formal releases or tags. History is organized by logical milestones derived from the commit record. Every commit hash links to its GitHub page.

---

## 2026-02-22 -- License and branding updates

### Changed
- Update README license text to reference "MIT License (with OpenAI/Anthropic Rider)" ([`841913d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/841913d4089246926cf217070d352542a9f69d06))

### Changed (license)
- Replace plain MIT license with MIT + OpenAI/Anthropic Rider restricting use by OpenAI, Anthropic, and affiliates without express permission from Jeffrey Emanuel ([`99086ee`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/99086eea99c36e81845f13879f8ff8a5f8c5a608))

### Added
- GitHub social preview image (1280x640 `gh_og_share_image.png`) ([`0554334`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/05543348707bdde91d05e40a5a24b76eeb1b732e))

---

## 2026-01-21 -- MIT license added

### Added
- Initial MIT License file ([`282c85b`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/282c85b0309a484f8e03e8f83c4b15a2b4c94ced))

---

## 2026-01-18 -- Dependency maintenance

### Added
- `UPGRADE_LOG.md` documenting dependency review status; project uses unpinned `requirements.txt` ([`80cdd7d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/80cdd7d4eea0e9ca5f9823800d28b37662e0a879))

---

## 2026-01-14 -- Bug-fix batch: return-type safety and None handling

Two fixes that together eliminate several crash paths reported across issues #11, #20, #21, and #24.

### Fixed
- **Return type inconsistency in `generate_completion_from_local_llm()`**: function previously returned either a `dict` or a `str` depending on code path; now always returns `Optional[str]`, extracting `generated_text` from the dict before returning ([`f5bb024`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/f5bb0249ba5d3c4aec5ff52c0efc10daed92a80c))
- **Negative `max_tokens` crash**: wrap token arithmetic in `max(1, ...)` across all three provider paths (OpenAI, Claude, local LLM) to prevent passing zero or negative values ([`f5bb024`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/f5bb0249ba5d3c4aec5ff52c0efc10daed92a80c))
- **Chunking condition**: fix the guard that decides whether input needs to be chunked so it fires reliably ([`f5bb024`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/f5bb0249ba5d3c4aec5ff52c0efc10daed92a80c))
- **None propagation in `process_chunk()`**: add None check after `generate_completion()` with fallback to original text instead of crashing on `len(None)` or `None[-1000:]` ([`f5bb024`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/f5bb0249ba5d3c4aec5ff52c0efc10daed92a80c))
- **None propagation in `assess_output_quality()`**: guard against `response.strip().split()` on None ([`f5bb024`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/f5bb0249ba5d3c4aec5ff52c0efc10daed92a80c))
- **Empty results in Claude chunking fallback**: when all chunks fail, return `None` instead of empty string, consistent with OpenAI and local LLM paths ([`6c1937d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/6c1937dd4ff836e0762d57621fa4c568d36ba3f3))

---

## 2025-02-27 -- README refresh

### Changed
- Add commercial project links (YoutubeTranscriptOptimizer.com, FixMyDocuments.com) to README footer ([`d8c140c`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d8c140c9fc2d2ff42e95751403de1aab00215607))
- Fix typo and formatting in README footer text ([`cc0eb67`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/cc0eb6735b7f582aaaf53b234a806b39f5bb0917))

---

## 2024-08-20 -- Docker support (community PR #16)

### Added
- `docker/Dockerfile` with NVIDIA CUDA 12.1 + cuDNN base image, Tesseract, Miniconda, and full project setup ([`a2f064d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a2f064d53619535c269ca395fd06593aa3886e54))
- `docker/docker-compose.yml` with GPU passthrough and SSH service ([`a2f064d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a2f064d53619535c269ca395fd06593aa3886e54))
- Merged via PR #16 from [hotwa](https://github.com/hotwa) ([`c79f6b7`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/c79f6b709a8db62903f55647bec4816823b86b9e))

---

## 2024-08-09 -- Tokenizer fix (community PR #9) and typo fix

### Fixed
- **Case-sensitive tokenizer model detection**: `get_tokenizer()` now uses `.lower().startswith()` so model names like `"Claude-3-haiku..."` or `"GPT-4o-mini"` are matched correctly; previously only lowercase prefixes worked. Contributed by [Backendmagier](https://github.com/Backendmagier) via PR #9 ([`fcee75b`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fcee75b48aff2fb9c04e3301efb19d7917a73a50), merged at [`a3fd34d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a3fd34d4acc2044a15ecdc855cac8c205e0cdfe2))
- Fix typo in markdown formatting prompt: "Remove and obviously" changed to "Remove any obviously" ([`850505c`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/850505c9ed2bd57edbe829b6a734fa31a0bc36c4))

### Changed
- README wording updates ([`537c2d1`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/537c2d16ac15421d2e39e583511ff4c2e24b1e54), [`6a9d6e3`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/6a9d6e3be7bfbd194e0a37bdf227fdc2c4526077))

---

## 2024-08-09 -- v2 stabilization: rename, refactor, cleanup

Immediate follow-up to the v2 rewrite. The project reached its current file layout here.

### Changed
- **Rename script**: `tesseract_with_llm_corrections.py` renamed to `llm_aided_ocr.py` ([`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))
- **Rename CLI**: `tesseract-llm-cli.py` renamed to `llm-aided-ocr-cli.py` ([`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))
- **Default API provider** changed from `OPENAI` to `CLAUDE` ([`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))
- **OpenAI max tokens** raised from 4096 to 12000 ([`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))
- **TOKEN_CUSHION** raised from 100 to 300 ([`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))
- **Default completion max_tokens** raised from 1000 to 5000 ([`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))

### Removed
- OpenRouter provider support and `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` configuration ([`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))
- `generate_embedding()` function and FAISS-based embedding infrastructure ([`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))
- NLTK sentence tokenizer (`sent_tokenize`); replaced with regex-based splitting ([`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))
- Dependencies: `httpx`, `langdetect`, `faiss`, `nltk` ([`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))
- Example pre-filtered markdown output file ([`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))

### Added
- LLM-corrected example output (`160301289-Warren-Buffett-Katharine-Graham-Letter_llm_corrected.md`) ([`d008f3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/d008f3dbb25094c98ed850d4bc60c7e5487f527e))

Quick-fix commits immediately after the refactor:
- [`fa02759`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fa02759a695ca0dacea9eeaf8469ff8b4bfaf4ea), [`a46e344`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a46e344a157d469c3b2af07546f15f911eb51c77), [`c595280`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/c5952809176edf4a4e9ca614dfc90ae9f458f410), [`409a1d1`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/409a1d1402d6725ec2ec308af6321df99055b02f)

---

## 2024-08-07 -- v2 rewrite: multi-provider async architecture

Complete rewrite of the project. The original 236-line Llama-2-only script was replaced with a 1057-line async system supporting multiple LLM providers. This is the single largest change in the project's history.

### Added
- **Anthropic Claude support** (`generate_completion_from_claude()` via `AsyncAnthropic`) ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **OpenAI API support** (`generate_completion_from_openai()` via `AsyncOpenAI`) ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **OpenRouter support** (`generate_completion_from_openrouter()`) -- later removed in v2 stabilization ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **Async processing pipeline** with `asyncio` for concurrent chunk processing ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **Smart text chunking** with sentence-boundary splitting and configurable overlap ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **Token management**: `estimate_tokens()` with model-specific tokenizers (tiktoken for GPT, HuggingFace AutoTokenizer for Claude/Llama), plus `approximate_tokens()` fallback ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **Two-step OCR correction pipeline**: (1) OCR error correction, (2) optional markdown formatting -- each step uses separate LLM prompts ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **Duplicate content removal** in the markdown formatting step ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **Quality assessment** (`assess_output_quality()`) comparing final vs raw OCR via LLM scoring ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **Image preprocessing** with OpenCV: grayscale conversion, Otsu thresholding, dilation ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **GPU detection** via `nvgpu` with graceful fallback ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **Header/page-number suppression** option ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **`.env`-based configuration** via `python-decouple` ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **CLI wrapper** (`tesseract-llm-cli.py`, later renamed to `llm-aided-ocr-cli.py`) with argparse for PDF path, page range, thresholds ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- **Comprehensive logging** with structured format and HTTP log suppression ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- `.gitignore` ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- `.python-version` (Python 3.12) ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))

### Removed
- `tesseract_with_llama2_corrections.py` (the entire v1 codebase) ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- v1 hallucination-filtering system based on LangChain embeddings + cosine similarity ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- v1 SQLite-backed sentence embedding cache ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))

### Changed
- README fully rewritten to document the new architecture ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- Requirements overhauled: added `anthropic`, `openai`, `tiktoken`, `httpx`, `faiss-cpu`, `filelock`, `transformers`, `cv2`, `langdetect`, `nltk`; removed `fuzzywuzzy`, `langchain` ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))
- Local LLM backend upgraded from Llama 2 (7B/13B GGML) to Llama 3.1 (8B GGUF) ([`fe28bfe`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/fe28bfe4a8f929dbcc80202290c200926cb4865e))

---

## 2023-08-01 to 2023-08-02 -- v1 maturation: hallucination filtering and embedding cache

Rapid iteration that grew the script from 60 lines to 236 lines and added the hallucination-detection system.

### Added
- **Multiprocessing OCR** via `Pool().map()` for parallel page extraction ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e))
- **Markdown reformatting step** as optional LLM post-processing pass ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e))
- **English-language validation** via LLM query before correction ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e))
- **LLM intro-text stripping** (`remove_intro()`) to drop spurious preamble added by the model ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e))
- **Hallucination filtering** using LangChain `LlamaCppEmbeddings` + scikit-learn cosine similarity with configurable threshold ([`4c2fe3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/4c2fe3de0e39d21e4f9e5472bca7228601fa386b))
- **SQLite-backed embedding cache** with SHA-3 file hashing for incremental re-runs ([`4392c4a`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/4392c4a588a221c63148aa4b661fa769d8b15eaf))
- **Example PDF** (Warren Buffett / Katharine Graham letter) with raw OCR output and filtered markdown ([`84bcbb6`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/84bcbb67d346ee3bbfbde90fa2ce9f753043ada6), [`e572391`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/e5723917db9f2c891bf28097bc8491451f2c1143))
- File output: automatic naming based on input PDF basename with `.md` or `.txt` extension ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e))
- Progress bars via `tqdm` ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e))

### Changed
- LLM model upgraded from Llama 2 7B (GGML q4_0) to Llama 2 13B-chat (GGML q4_0, later q5_K_S) ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e), [`5719a9a`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/5719a9aede6b0666f6f08d239cac7b1550298b79))
- Page-length thresholds lowered (min chars from 30 to 10, min words from 20 to 5) for more inclusive extraction ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e))
- Added `if __name__ == '__main__':` guard for clean imports ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e))
- Full-document processing support (`max_test_pages=0` processes all pages) ([`a43f0c0`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/a43f0c0b8814c8584f35da09f01a7342ff556a2e))

### Fixed
- Exception handling in `remove_intro()` to prevent crashes on unexpected LLM output format ([`4c2fe3d`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/4c2fe3de0e39d21e4f9e5472bca7228601fa386b))

---

## 2023-07-26 -- Initial release (v1)

60-line proof of concept: Tesseract OCR with Llama 2 7B (GGML) for typo correction.

### Added
- `tesseract_with_llama2_corrections.py`: PDF-to-image conversion via `pdf2image`, OCR via `pytesseract`, LLM-based typo correction via `llama_cpp` with Llama 2 7B ([`315faed`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/315faed5d55cbc4511f05c93eda74ce2f2c05815))
- `requirements.txt` with initial dependencies: `pdf2image`, `pytesseract`, `llama-cpp-python` ([`315faed`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/315faed5d55cbc4511f05c93eda74ce2f2c05815))
- `README.md` ([`0cfc2ed`](https://github.com/Dicklesworthstone/llm_aided_ocr/commit/0cfc2edce229ec0d1e2d775bbf68a59c2a1ada19))
