# LLM-Aided OCR Project

## Introduction

The LLM-Aided OCR Project is an advanced system designed to significantly enhance the quality of Optical Character Recognition (OCR) output. By leveraging cutting-edge natural language processing techniques and large language models (LLMs), this project transforms raw OCR text into highly accurate, well-formatted, and readable documents.

## Example Outputs

To see what the LLM-Aided OCR Project can do, check out these example outputs:

- [Original PDF](https://github.com/Dicklesworthstone/llm_aided_ocr/blob/main/160301289-Warren-Buffett-Katharine-Graham-Letter.pdf) 
- [Raw OCR Output](https://github.com/Dicklesworthstone/llm_aided_ocr/blob/main/160301289-Warren-Buffett-Katharine-Graham-Letter__raw_ocr_output.txt)
- [LLM-Corrected Markdown Output](https://github.com/Dicklesworthstone/llm_aided_ocr/blob/main/160301289-Warren-Buffett-Katharine-Graham-Letter_llm_corrected.md)

## Features

- PDF to image conversion
- OCR using Tesseract
- Advanced error correction using LLMs (local or API-based)
- Smart text chunking for efficient processing
- Markdown formatting option
- Header and page number suppression (optional)
- Quality assessment of the final output
- Support for both local LLMs and cloud-based API providers (OpenAI, Anthropic)
- Asynchronous processing for improved performance
- Detailed logging for process tracking and debugging
- GPU acceleration for local LLM inference

## Detailed Technical Overview

### PDF Processing and OCR

1. **PDF to Image Conversion**
   - Function: `convert_pdf_to_images()`
   - Uses `pdf2image` library to convert PDF pages into images
   - Supports processing a subset of pages with `max_pages` and `skip_first_n_pages` parameters

2. **OCR Processing**
   - Function: `ocr_image()`
   - Utilizes `pytesseract` for text extraction
   - Includes image preprocessing with `preprocess_image()` function:
     - Converts image to grayscale
     - Applies binary thresholding using Otsu's method
     - Performs dilation to enhance text clarity

### Text Processing Pipeline

1. **Chunk Creation**
   - The `process_document()` function splits the full text into manageable chunks
   - Uses sentence boundaries for natural splits
   - Implements an overlap between chunks to maintain context

2. **Error Correction and Formatting**
   - Core function: `process_chunk()`
   - Two-step process:
     a. OCR Correction:
        - Uses LLM to fix OCR-induced errors
        - Maintains original structure and content
     b. Markdown Formatting (optional):
        - Converts text to proper markdown format
        - Handles headings, lists, emphasis, and more

3. **Duplicate Content Removal**
   - Implemented within the markdown formatting step
   - Identifies and removes exact or near-exact repeated paragraphs
   - Preserves unique content and ensures text flow

4. **Header and Page Number Suppression (Optional)**
   - Can be configured to remove or distinctly format headers, footers, and page numbers

### LLM Integration

1. **Flexible LLM Support**
   - Supports both local LLMs and cloud-based API providers (OpenAI, Anthropic)
   - Configurable through environment variables

2. **Local LLM Handling**
   - Function: `generate_completion_from_local_llm()`
   - Uses `llama_cpp` library for local LLM inference
   - Supports custom grammars for structured output

3. **API-based LLM Handling**
   - Functions: `generate_completion_from_claude()` and `generate_completion_from_openai()`
   - Implements proper error handling and retry logic
   - Manages token limits and adjusts request sizes dynamically

4. **Asynchronous Processing**
   - Uses `asyncio` for concurrent processing of chunks when using API-based LLMs
   - Maintains order of processed chunks for coherent final output

### Token Management

1. **Token Estimation**
   - Function: `estimate_tokens()`
   - Uses model-specific tokenizers when available
   - Falls back to `approximate_tokens()` for quick estimation

2. **Dynamic Token Adjustment**
   - Adjusts `max_tokens` parameter based on prompt length and model limits
   - Implements `TOKEN_BUFFER` and `TOKEN_CUSHION` for safe token management

### Quality Assessment

1. **Output Quality Evaluation**
   - Function: `assess_output_quality()`
   - Compares original OCR text with processed output
   - Uses LLM to provide a quality score and explanation

### Logging and Error Handling

- Comprehensive logging throughout the codebase
- Detailed error messages and stack traces for debugging
- Suppresses HTTP request logs to reduce noise

## Configuration and Customization

The project uses a `.env` file for easy configuration. Key settings include:

- LLM selection (local or API-based)
- API provider selection
- Model selection for different providers
- Token limits and buffer sizes
- Markdown formatting options

## Output and File Handling

1. **Raw OCR Output**: Saved as `{base_name}__raw_ocr_output.txt`
2. **LLM Corrected Output**: Saved as `{base_name}_llm_corrected.md` or `.txt`

The script generates detailed logs of the entire process, including timing information and quality assessments.

## Requirements

- Python 3.12+
- Tesseract OCR engine
- PDF2Image library
- PyTesseract
- OpenAI API (optional)
- Anthropic API (optional)
- Local LLM support (optional, requires compatible GGUF model)

## Installation

1. Install Pyenv and Python 3.12 (if needed):

```bash
# Install Pyenv and python 3.12 if needed and then use it to create venv:
if ! command -v pyenv &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git

    git clone https://github.com/pyenv/pyenv.git ~/.pyenv
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
    echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
    source ~/.zshrc
fi
cd ~/.pyenv && git pull && cd -
pyenv install 3.12
```

2. Set up the project:

```bash
# Use pyenv to create virtual environment:
git clone https://github.com/Dicklesworthstone/llm_aided_ocr    
cd llm_aided_ocr          
pyenv local 3.12
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install wheel
python -m pip install --upgrade setuptools wheel
pip install -r requirements.txt
```

3. Install Tesseract OCR engine (if not already installed):
   - For Ubuntu: `sudo apt-get install tesseract-ocr`
   - For macOS: `brew install tesseract`
   - For Windows: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

4. Set up your environment variables in a `.env` file:
   ```
   USE_LOCAL_LLM=False
   API_PROVIDER=OPENAI
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

## Usage

### Using the Command-Line Interface (`llm-aided-ocr-cli.py`)

```bash
# Process a PDF file (default 300 DPI, Markdown output, strip running headers/page numbers)
python llm-aided-ocr-cli.py 160301289-Warren-Buffett-Katharine-Graham-Letter.pdf

# Process specific page ranges or override provider/model/output directory/concurrency/chunk size
python llm-aided-ocr-cli.py document.pdf \
  --provider openai \
  --model gpt-4o-mini \
  --dpi 300 \
  --max-pages 5 \
  --chunk-size 6000 \
  --max-concurrency 4 \
  --output-dir ./out \
  --report-json ./out/conservation_report.json

# Process an already-extracted raw OCR text file without re-running Tesseract (skipping final LLM quality check)
python llm-aided-ocr-cli.py \
  --raw-ocr-input 160301289-Warren-Buffett-Katharine-Graham-Letter__raw_ocr_output.txt \
  --no-quality-check \
  --output-dir ./out

# Run 100% offline deterministic OCR cleanup (page-number stripping, footnote relocation, dehyphenation, and paragraph reflow) without any LLM
python llm-aided-ocr-cli.py \
  --raw-ocr-input 160301289-Warren-Buffett-Katharine-Graham-Letter__raw_ocr_output.txt \
  --no-llm \
  --output-dir ./out

# Run a standalone deterministic content conservation audit and save JSON report
python llm-aided-ocr-cli.py --verify-only \
  160301289-Warren-Buffett-Katharine-Graham-Letter__raw_ocr_output.txt \
  160301289-Warren-Buffett-Katharine-Graham-Letter_llm_corrected.md \
  --report-json ./conservation_report.json
```

### Running `llm_aided_ocr.py` Directly

1. Place your PDF file in the project directory (or pass parameters to `main()`).
2. Run the script:
   ```bash
   python llm_aided_ocr.py
   ```
3. The script will generate `{base_name}__raw_ocr_output.txt` and `{base_name}_llm_corrected.md`, followed by a multi-sample and deterministic content conservation quality assessment.

## How It Works

The LLM-Aided OCR project employs a multi-step process to transform raw OCR output into high-quality, readable text:

1. **PDF Conversion**: Converts input PDF into images at configurable DPI (`PDF_DPI=300` by default) using `pdf2image`.
2. **Document Image Preprocessing**: Applies grayscale conversion, rotationally-invariant horizontal line-strip contour median deskewing (`estimate_skew_angle` / `deskew_image`), Contrast-Limited Adaptive Histogram Equalization (CLAHE), Gaussian denoising, and Otsu adaptive binarization before running Tesseract OCR (toggleable via `--no-preprocess`).
3. **Structural Page Cleaning & Footnote Preservation**: Deterministically strips OCR-mangled standalone page numbers (`-2-`, `-1O-`, `-LL-`, `~Li-`, `~L5-`) and repeating running headers while anchoring bottom-of-page author footnotes (`* ...`, `† ...`, `® ...`) so split sentences across page breaks join cleanly without losing footnotes.
4. **Lossless Text Chunking**: Splits text into paragraph- and sentence-aligned chunks (`chunk_text` and `split_long_sentence`) with cached offline-safe tokenization (`tiktoken` `cl100k_base` fallback).
5. **Two-Pass LLM Error Correction & Markdown Formatting**: Each chunk is corrected for OCR typos and formatted as Markdown using read-only `<previous_context>` tags, bounded async concurrency (`MAX_CONCURRENT_REQUESTS`), and exponential backoff retries.
6. **Deterministic Seam Stitching (`stitch_chunks`)**: Aligns word sequences across chunk boundaries to strip any echoed overlap phrases or spurious mid-sentence heading markers.
7. **Deterministic Conservation & Semantic Similarity Assessment**: Evaluates start, middle, and end segments alongside `verify_content_conservation()` (numeric/financial token recall, footnote retention, seam integrity, word-count conservation, and word/char-3-gram lexical cosine similarity `compute_lexical_similarity()`) plus `compute_embedding_similarity()` via `OPENAI_EMBEDDING_MODEL`.


## Configuration

The project uses a `.env` file (or environment variables) for configuration. Key settings include:

- `USE_LOCAL_LLM`: Set to `True` to use a local GGUF LLM via `llama-cpp-python`, `False` for API-based LLMs.
- `API_PROVIDER`: Choose between `"OPENAI"` or `"CLAUDE"`.
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`: API keys for respective services.
- `CLAUDE_MODEL_STRING`, `OPENAI_COMPLETION_MODEL`: Specify the model to use for each provider.
- `OPENAI_MAX_TOKENS`, `CLAUDE_MAX_TOKENS`: Context/output token budgets for API providers.
- `LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS`: Context size for local LLMs (default `4096`).
- `PDF_DPI`: PDF rendering resolution in DPI (default `300`).
- `MAX_CONCURRENT_REQUESTS`: Maximum concurrent API requests during chunk processing (default `4`).
- `API_MAX_RETRIES`, `API_RETRY_BASE_DELAY`: Retry settings for transient API errors.

## Output Files

The script generates two primary output files:

1. `{base_name}__raw_ocr_output.txt`: Raw OCR output from Tesseract.
2. `{base_name}_llm_corrected.md`: Final LLM-corrected and stitched Markdown text.

## Limitations and Future Improvements

- The system's performance is heavily dependent on the quality of the LLM used.
- Processing very large documents can be time-consuming and may require significant computational resources.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License (with OpenAI/Anthropic Rider).

---

Thanks for your interest in my open-source project! I hope you find it useful. You might also find my commercial web apps useful, and I would really appreciate it if you checked them out:

**[YoutubeTranscriptOptimizer.com](https://youtubetranscriptoptimizer.com)** makes it really quick and easy to paste in a YouTube video URL and have it automatically generate not just a really accurate direct transcription, but also a super polished and beautifully formatted written document that can be used independently of the video.

The document basically sticks to the same material as discussed in the video, but it sounds much more like a real piece of writing and not just a transcript. It also lets you optionally generate quizzes based on the contents of the document, which can be either multiple choice or short-answer quizzes, and the multiple choice quizzes get turned into interactive HTML files that can be hosted and easily shared, where you can actually take the quiz and it will grade your answers and score the quiz for you.

**[FixMyDocuments.com](https://fixmydocuments.com/)** lets you submit any kind of document— PDFs (including scanned PDFs that require OCR), MS Word and Powerpoint files, images, audio files (mp3, m4a, etc.) —and turn them into highly optimized versions in nice markdown formatting, from which HTML and PDF versions are automatically generated. Once converted, you can also edit them directly in the site using the built-in markdown editor, where it saves a running revision history and regenerates the PDF/HTML versions.

In addition to just getting the optimized version of the document, you can also generate many other kinds of "derived documents" from the original: interactive multiple-choice quizzes that you can actually take and get graded on; slick looking presentation slides as PDF or HTML (using LaTeX and Reveal.js), an in-depth summary, a concept mind map (using Mermaid diagrams) and outline, custom lesson plans where you can select your target audience, a readability analysis and grade-level versions of your original document (good for simplifying concepts for students), Anki Flashcards that you can import directly into the Anki app or use on the site in a nice interface, and more.
