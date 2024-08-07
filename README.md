# LLM-Aided OCR Project

## Introduction

The LLM-Aided OCR Project is a sophisticated system designed to dramatically improve the quality of Optical Character Recognition (OCR) output. It employs a combination of advanced natural language processing techniques, machine learning, and intelligent text processing to transform raw OCR text into highly accurate, well-formatted, and readable documents.

The project addresses common OCR issues such as:

- Misrecognized characters and words
- Incorrect line breaks and paragraph structures
- Hallucinated content
- Formatting inconsistencies
- Duplicated content

By leveraging the power of large language models (LLMs) and embedding-based similarity checks, this system can produce high-quality results even when the initial OCR output contains numerous errors.

## Features

- PDF to image conversion
- OCR using Tesseract
- Advanced error correction using LLMs (local or API-based)
- Smart text chunking for efficient processing
- Markdown formatting
- Header and page number suppression (optional)
- Hallucination filtering using FAISS and embedding-based similarity checks
- Duplicate content removal
- Smart post-processing for consistency and readability
- Quality assessment of the final output
- Support for both local LLMs and cloud-based API providers (OpenAI, Anthropic, OpenRouter)
- Asynchronous processing for improved performance
- Detailed logging for process tracking and debugging

## Requirements

- Python 3.7+
- Tesseract OCR engine
- PDF2Image library
- PyTesseract
- FAISS (Facebook AI Similarity Search)
- Numpy
- OpenAI API (optional)
- Anthropic API (optional)
- OpenRouter API (optional)
- Local LLM support (optional, requires compatible GGUF model)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Dicklesworthstone/llm_aided_ocr.git
   cd llm_aided_ocr
   ```

2. Install the required Python packages:
   ```
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
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```

## Usage

1. Place your PDF file in the project directory.

2. Update the `input_pdf_file_path` variable in the `main()` function with your PDF filename.

3. Run the script:
   ```
   python llm_aided_ocr.py
   ```

4. The script will generate several output files, including the final post-processed text.

## Example Output

To demonstrate the effectiveness of the LLM-Aided OCR project, we've processed a sample document and provided links to the various stages of output. This allows you to see the transformation from the original PDF to the final, polished markdown document.

### Sample Document: Warren Buffett's Letter to Katharine Graham

1. **Original Source PDF**: 
   [Warren-Buffett-Katharine-Graham-Letter.pdf](https://github.com/Dicklesworthstone/llm_aided_ocr/blob/main/160301289-Warren-Buffett-Katharine-Graham-Letter.pdf)
   
   This is the original PDF document that we'll be processing through our OCR pipeline.

2. **Raw Tesseract OCR Output**: 
   [Raw OCR Output](https://github.com/Dicklesworthstone/llm_aided_ocr/blob/main/160301289-Warren-Buffett-Katharine-Graham-Letter__raw_ocr_output.txt)
   
   This file shows the unprocessed output from the Tesseract OCR engine. You'll notice various errors, formatting issues, and potential misrecognitions in this raw text.

3. **Initial LLM-Corrected Output**: 
   [Pre-filtered LLM Output](https://github.com/Dicklesworthstone/llm_aided_ocr/blob/main/160301289-Warren-Buffett-Katharine-Graham-Letter_pre_filtered.md)
   
   This is the output after the initial LLM-based error correction and formatting, but before the hallucination filtering step. You may notice significant improvements in readability and formatting compared to the raw OCR output, but there might also be some hallucinated content.

4. **Final Filtered Markdown Output**: 
   [Post-filtered LLM Output](https://github.com/Dicklesworthstone/llm_aided_ocr/blob/main/160301289-Warren-Buffett-Katharine-Graham-Letter_post_filtered.md)
   
   This is the final output after all processing steps, including hallucination filtering. Compare this to the pre-filtered output to see how much hallucinated content has been removed while preserving the accurate information from the original document.

By comparing these different stages of output, you can observe:

1. The initial quality of the OCR from Tesseract.
2. The dramatic improvements made by the LLM in terms of readability and formatting.
3. The extent of hallucinations introduced by the LLM (by comparing the pre-filtered and post-filtered outputs).
4. The effectiveness of our hallucination filtering process in producing a final document that closely matches the content of the original PDF.

This example demonstrates the power of combining traditional OCR techniques with advanced LLM processing and our sophisticated filtering mechanisms to produce high-quality, accurate text from complex PDF documents.

## How It Works

The LLM-Aided OCR project employs a sophisticated multi-step process to transform raw OCR output into high-quality, readable text. Here's a detailed breakdown of each step:

### 1. PDF Conversion

The process begins with converting the input PDF into a series of images using the `pdf2image` library.

- **Function**: `convert_pdf_to_images(input_pdf_file_path, max_pages, skip_first_n_pages)`
- **Implementation Details**:
  - Uses `convert_from_path` from `pdf2image` to transform PDF pages into PIL Image objects.
  - Supports processing a subset of pages through `max_pages` and `skip_first_n_pages` parameters.
  - Handles potential PDF encryption or corruption issues.
  - Returns a list of Image objects, each representing a page from the PDF.

### 2. OCR (Optical Character Recognition)

Tesseract OCR is applied to each image to extract the raw text content.

- **Function**: `ocr_image(image)`
- **Implementation Details**:
  - Utilizes `pytesseract.image_to_string()` to perform OCR on each image.
  - Applies image preprocessing using OpenCV:
    - Converts image to grayscale.
    - Applies thresholding to create a binary image.
    - Uses dilation to enhance text features.
  - Returns the extracted text as a string for each page.

### 3. Text Chunking

The raw OCR output is intelligently split into manageable chunks for efficient processing.

- **Function**: `smart_split_into_chunks(text, chunk_size=1000)`
- **Implementation Details**:
  - Uses an LLM to analyze the document structure and split it into logical chunks.
  - Preserves document structure, including headers and paragraphs.
  - Employs a prompt-based approach, asking the LLM to:
    1. Analyze the text structure.
    2. Split it into chunks of approximately 1000 characters.
    3. Identify headers and content sections.
    4. Output the result as a JSON array of objects with 'text' and 'is_header' properties.
  - Handles potential LLM failures by falling back to a simple character-based split.
  - Returns a list of tuples, each containing chunk text and a boolean indicating if it's a header.

### 4. Error Correction

Each chunk undergoes LLM-based processing to correct OCR errors and improve readability.

- **Function**: `process_chunk(chunk, prev_context, chunk_index, total_chunks, check_if_valid_english, reformat_as_markdown, suppress_headers_and_page_numbers)`
- **Implementation Details**:
  - Utilizes a carefully engineered prompt for the LLM, instructing it to:
    1. Fix OCR-induced typos and errors.
    2. Correct words split across line breaks.
    3. Use context to infer correct words and phrases.
    4. Maintain original structure and content.
    5. Ensure coherence with previous context.
  - Handles context by providing the last 300 characters of the previous chunk.
  - Supports different LLM providers (OpenAI, Anthropic, OpenRouter) or local LLMs.
  - Implements token limit management to avoid exceeding API constraints.
  - Returns the corrected text along with new context for the next chunk.

### 5. Markdown Formatting (Optional)

The corrected text is reformatted into clean, consistent Markdown.

- **Function**: Part of `process_chunk()` when `reformat_as_markdown=True`
- **Implementation Details**:
  - Uses a separate LLM prompt specifically for Markdown formatting.
  - Instructions include:
    1. Converting headings to appropriate Markdown syntax (e.g., #, ##, ###).
    2. Properly formatting lists (ordered and unordered).
    3. Applying emphasis (*italic*) and strong emphasis (**bold**) where appropriate.
    4. Preserving code blocks and other special formatting.
    5. Ensuring consistent paragraph breaks and line spacing.
  - Handles edge cases like mixed formatting styles and complex nested structures.

### 6. Duplicate Removal

An LLM-based approach identifies and removes duplicated content across the entire document.

- **Function**: `smart_remove_duplicates(text, chunk_size=5000)`
- **Implementation Details**:
  - Splits the entire processed text into large chunks (5000 characters each).
  - For each chunk, uses an LLM with a prompt designed to:
    1. Identify repeated paragraphs, sentences, or phrases.
    2. Remove duplicates while preserving unique information.
    3. Ensure the text flows smoothly after removal.
  - Processes chunks in parallel for efficiency.
  - Performs a final pass over the entire document to catch duplicates that might span chunk boundaries.
  - Uses carefully crafted prompts that instruct the LLM to preserve document structure and avoid removing similar but distinct content.

### 7. Hallucination Filtering

FAISS (Facebook AI Similarity Search) and embedding-based similarity checks are used to filter out hallucinated content.

- **Function**: `filter_hallucinations(corrected_text, raw_text, threshold, pdf_file_path, db_path)`
- **Implementation Details**:
  - Generates embeddings for both the corrected text and the original raw OCR output using `calculate_embeddings_batch()`.
  - Utilizes FAISS to create an efficient similarity search index for the raw text embeddings.
  - For each chunk of corrected text:
    1. Generates its embedding.
    2. Searches for the most similar chunk in the raw text using FAISS.
    3. If the similarity score is below the threshold, the chunk is considered a potential hallucination and is filtered out.
  - Implements adaptive thresholding based on document characteristics.
  - Handles edge cases where legitimate content might be mistakenly identified as hallucinations.
  - Returns the filtered text with hallucinations removed.

### 8. Post-processing

A final LLM pass ensures consistency and improves overall quality across the entire document.

- **Function**: `smart_post_process_output(text, chunk_size=5000)`
- **Implementation Details**:
  - Splits the filtered text into large chunks for efficient processing.
  - Uses an LLM with a sophisticated prompt designed to:
    1. Ensure consistent formatting throughout the document.
    2. Improve readability and flow between sections.
    3. Correct any remaining grammatical or structural issues.
    4. Maintain the document's overall style and tone.
  - Handles document-wide concerns that might not be apparent when processing smaller chunks.
  - Implements a final pass to ensure consistency across chunk boundaries.
  - Returns the final, polished version of the document.

### 9. Quality Assessment

An LLM-based evaluation compares the final output quality to the original OCR text.

- **Function**: `assess_output_quality(original_text, processed_text)`
- **Implementation Details**:
  - Samples portions of both the original OCR text and the final processed output.
  - Utilizes an LLM with a prompt designed to:
    1. Compare the two texts for accuracy, readability, and content preservation.
    2. Assess the improvement in formatting and structure.
    3. Identify any potential issues or areas for further improvement.
    4. Provide a numerical score (0-100) and a detailed explanation.
  - Handles potential biases in the assessment by using specific criteria and examples.
  - Returns both a quantitative score and a qualitative explanation of the improvements and any remaining issues.

### Additional Implementation Notes

- **Asynchronous Processing**: The project extensively uses Python's `asyncio` for concurrent processing of chunks, significantly improving performance for large documents.
- **Error Handling and Logging**: Comprehensive error handling and logging are implemented throughout the pipeline to catch and report issues at each stage.
- **Adaptive Token Management**: The system dynamically adjusts the number of tokens used for LLM requests based on input size and model constraints, ensuring efficient use of API resources.
- **Fallback Mechanisms**: In case of LLM API failures or unexpected outputs, the system includes fallback options to ensure some level of improvement over the raw OCR output.
- **Configuration Management**: A flexible configuration system using environment variables allows easy switching between different LLM providers and adjustment of processing parameters.

This multi-stage, LLM-driven approach allows the system to handle a wide variety of document types and OCR challenges, producing high-quality output even from initially poor-quality OCR results.

## Advanced Techniques

### Smart Chunking
The system uses an LLM to intelligently split the text into logical chunks, preserving document structure and ensuring headers are handled correctly.

### Embedding-based Hallucination Filtering
By comparing embeddings of the processed text with the original OCR output, the system can identify and remove content that doesn't have a strong similarity to the source material.

### Asynchronous Processing
The project uses Python's `asyncio` to process multiple chunks concurrently, significantly improving performance for large documents.

### Adaptive Token Management
The system dynamically adjusts the number of tokens used for LLM requests based on the input size and model constraints, ensuring efficient use of resources.

### Multi-stage Processing
By applying multiple passes of LLM-based processing (error correction, de-duplication, post-processing), the system can handle complex documents and produce high-quality results.

### Fallback Mechanisms
If the LLM-based processing fails or produces unexpected results, the system has fallback mechanisms to ensure some level of improvement over the raw OCR output.

## Configuration

The project uses a `.env` file for configuration. Key settings include:

- `USE_LOCAL_LLM`: Set to `True` to use a local LLM, `False` for API-based LLMs.
- `API_PROVIDER`: Choose between "OPENAI", "CLAUDE", or "OPENROUTER".
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`: API keys for respective services.
- `CLAUDE_MODEL_STRING`, `OPENAI_COMPLETION_MODEL`, `OPENROUTER_MODEL`: Specify the model to use for each provider.
- `LOCAL_LLM_CONTEXT_SIZE_IN_TOKENS`: Set the context size for local LLMs.

## Output Files

The script generates several output files:

1. `{base_name}__raw_ocr_output.txt`: Raw OCR output from Tesseract.
2. `{base_name}_pre_filtered.md`: Initial LLM-corrected text before hallucination filtering.
3. `{base_name}_deduped.md`: Text after removing duplicates.
4. `{base_name}_post_filtered.md`: Final output after all processing steps.

## Quality Assessment

The system includes an LLM-based quality assessment step that compares the final output to the original OCR text. It provides a score out of 100 and an explanation of the improvements made.

## Limitations and Future Improvements

- The system's performance is heavily dependent on the quality of the LLM used.
- Processing very large documents can be time-consuming and may require significant computational resources.
- The hallucination filtering step may occasionally remove valid content if it's significantly different from the majority of the text.

Future improvements could include:

- Integration with more OCR engines for comparison and improved initial text recognition.
- Implementation of a user interface for easier configuration and monitoring.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License.