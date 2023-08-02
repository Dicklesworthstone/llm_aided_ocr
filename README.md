# Use Llama2 to Improve the Accuracy of Tesseract OCR:

This project aims to improve the quality of Optical Character Recognition (OCR) outputs using Large Language Models (LLMs). 

## Purpose
The purpose of the project is to convert scanned PDFs into readable text files. It does this by first leveraging OCR and then enhancing the quality of the OCR output by using an LLM to correct errors and format the text for readability. 

## Example Output
If you just want to preview the output, you can look at the [source PDF](https://github.com/Dicklesworthstone/llama2_aided_tesseract/blob/main/160301289-Warren-Buffett-Katharine-Graham-Letter.pdf) and the [raw output of the Tesseract OCR process](https://github.com/Dicklesworthstone/llama2_aided_tesseract/blob/main/160301289-Warren-Buffett-Katharine-Graham-Letter__raw_ocr_output.txt), and compare this to the [final markdown output after filtering out hallucinations](https://github.com/Dicklesworthstone/llama2_aided_tesseract/blob/main/160301289-Warren-Buffett-Katharine-Graham-Letter_filtered.md) (and also the [initial output from the LLM corrections](https://github.com/Dicklesworthstone/llama2_aided_tesseract/blob/main/160301289-Warren-Buffett-Katharine-Graham-Letter.md) to see just how much the LLM hallucinates-- it's a often a lot!).

## Set up instructions:

```
sudo apt-get install -y tesseract-ocr libtesseract-dev poppler-utils
git lfs install
git clone https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML

python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install wheel
pip install -r requirements.txt
```

## How It Works
The project begins by converting a given PDF into images using the `pdf2image` library. It then applies OCR to each image using `pytesseract`, with parallel processing enabled via the multiprocessing library for speed. The OCR'ed text is subsequently passed through the Llama2 13B Chat model, which aids in correcting OCR errors and enhancing the formatting of the text. The program offers options to verify if the OCR output is valid English and to reformat the text using markdown. The final text is written to an output file. Furthermore, the project has a function to filter potential hallucinations from the LLM corrected text using sentence embeddings and cosine similarity to compare with the original OCR text. 

## Usage
1. Set the `input_pdf_file_path` variable to the path of the PDF file you want to process.
2. Set the `max_test_pages` variable to the number of pages you want to process. If you set it to 0, the program will process all pages of the PDF file.
3. Set the `skip_first_n_pages` variable to the number of pages you want to skip. If you set it to 0, the program will process all pages.
4. Set the `check_if_valid_english` variable to True if you want to check if the extracted text is valid English. 
5. Set the `reformat_as_markdown` variable to True if you want to reformat the corrected extracted text using markdown formatting.
6. Run the program. 

The program will create 3 output files: 
1. Raw OCR output
2. LLM corrected output
3. LLM corrected output with hallucinations filtered out

## Functions
Here are some of the important functions and what they do:

- `convert_pdf_to_images_func`: Converts a PDF to images.
- `check_extracted_pages_func`: Checks if the extracted text is long enough to be a page.
- `remove_intro`: Removes the introduction from the LLM output text.
- `is_valid_english`: Checks if the given text is valid English.
- `process_text_with_llm_func`: Processes the text with the LLM.
- `calculate_sentence_embedding`: Computes the sentence embedding for a given text.
- `calculate_similarity`: Computes the cosine similarity between the embeddings of the corrected and original sentences.
- `filter_hallucinations`: Filters out hallucinations from the corrected text.
- `ocr_image`: Performs OCR on the given image.

## Note
This project uses cosine similarity between sentence embeddings to filter out potential hallucinations from the LLM corrected text. This is a simple heuristic and may not be perfect. You may need to adjust the threshold for cosine similarity or use a different method depending on your needs.
