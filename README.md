# LLM-Aided OCR Project: In-Depth Technical Overview

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
- Duplicate content removal
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

1. Place your PDF file in the project directory.

2. Update the `input_pdf_file_path` variable in the `main()` function with your PDF filename.

3. Run the script:
   ```
   python llm_aided_ocr.py
   ```

4. The script will generate several output files, including the final post-processed text.

## How It Works

The LLM-Aided OCR project employs a multi-step process to transform raw OCR output into high-quality, readable text:

1. **PDF Conversion**: Converts input PDF into images using `pdf2image`.

2. **OCR**: Applies Tesseract OCR to extract text from images.

3. **Text Chunking**: Splits the raw OCR output into manageable chunks for processing.

4. **Error Correction**: Each chunk undergoes LLM-based processing to correct OCR errors and improve readability.

5. **Markdown Formatting** (Optional): Reformats the corrected text into clean, consistent Markdown.

6. **Quality Assessment**: An LLM-based evaluation compares the final output quality to the original OCR text.

## Code Optimization

- **Concurrent Processing**: When using API-based models, chunks are processed concurrently to improve speed.
- **Context Preservation**: Each chunk includes a small overlap with the previous chunk to maintain context.
- **Adaptive Token Management**: The system dynamically adjusts the number of tokens used for LLM requests based on input size and model constraints.

## Configuration

The project uses a `.env` file for configuration. Key settings include:

- `USE_LOCAL_LLM`: Set to `True` to use a local LLM, `False` for API-based LLMs.
- `API_PROVIDER`: Choose between "OPENAI" or "CLAUDE".
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`: API keys for respective services.
- `CLAUDE_MODEL_STRING`, `OPENAI_COMPLETION_MODEL`: Specify the model to use for each provider.
- `LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS`: Set the context size for local LLMs.

## Output Files

The script generates several output files:

1. `{base_name}__raw_ocr_output.txt`: Raw OCR output from Tesseract.
2. `{base_name}_llm_corrected.md`: Final LLM-corrected and formatted text.

## Limitations and Future Improvements

- The system's performance is heavily dependent on the quality of the LLM used.
- Processing very large documents can be time-consuming and may require significant computational resources.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License.
