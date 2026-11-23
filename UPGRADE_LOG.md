# Dependency Upgrade Log

**Date:** 2026-01-18  |  **Project:** llm_aided_ocr  |  **Language:** Python (requirements.txt)

## Summary
- **Updated:** N/A (uses requirements.txt, not pyproject.toml)
- **Skipped:** 0
- **Failed:** 0
- **Needs attention:** 0

## Notes

This project uses `requirements.txt` without pinned versions (just package names).
Dependencies are pulled at latest compatible versions at install time.

Current dependencies:
- anthropic, openai (LLM providers)
- transformers, llama-cpp-python
- opencv-python-headless, pillow, pdf2image, pytesseract
- numpy, scikit-learn
- tiktoken, langdetect

---

**Date:** 2026-11-22  |  **Project:** llm_aided_ocr  |  **Language:** Python (requirements.txt)

## Summary
- **Updated:** nothing to pin; `requirements.txt` stays unpinned.
- **Verified:** a fresh Python 3.12 environment with the newest release of every requirement
  passes the test suite (33 passed).
- **Needs attention:** ruff 0.16 widened its default lint rules, so `ruff check` failed on
  unchanged code with the newest ruff (26 findings, none from the rules this code was checked
  against before). Added `ruff.toml` pinning the earlier default selection (`E4`, `E7`, `E9`,
  `F`); `ruff check` passes again on ruff 0.14 through 0.17.6.
- **Not covered:** `llama-cpp-python` (optional local-LLM backend, a source build) was not
  installed for this check; the code imports it lazily. `ruff format --check` reports 3 files
  that it would reformat on every ruff version tried, so formatting is not a gate here.

Versions tested: anthropic 2.12.0, filelock 4.3.0, numpy 2.5.5, openai 3.57.0,
opencv-python-headless 5.1.0.95, pdf2image 1.17.0, pillow 13.0.0, pytesseract 0.3.13,
pytest 9.2.0, pytest-asyncio 1.5.0, python-decouple 3.8, ruff 0.17.6, tiktoken 0.14.0,
tqdm 4.70.1, transformers 5.24.0.

The current requirements are those in `requirements.txt`: scikit-learn and langdetect,
listed in the entry above, are no longer dependencies.
