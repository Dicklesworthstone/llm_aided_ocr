# Use Llama2 to Improve the Accuracy of Tesseract OCR:

This project focuses on improving the quality of Optical Character Recognition (OCR) outputs using Large Language Models (LLMs). 

## Purpose
The purpose of the project is to convert scanned PDFs into readable text files by leveraging OCR and then enhance the quality of the OCR output by correcting errors and formatting the text for readability using an LLM. 

## Example Output
If you just want to preview the output, you can look at the [source PDF](https://github.com/Dicklesworthstone/llama2_aided_tesseract/blob/main/160301289-Warren-Buffett-Katharine-Graham-Letter.pdf) and the [raw output of the Tesseract OCR process](https://github.com/Dicklesworthstone/llama2_aided_tesseract/blob/main/160301289-Warren-Buffett-Katharine-Graham-Letter__raw_ocr_output.txt), and compare this to the [final markdown output after filtering out hallucinations](https://github.com/Dicklesworthstone/llama2_aided_tesseract/blob/main/160301289-Warren-Buffett-Katharine-Graham-Letter_filtered.md).

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
The project starts by converting a given PDF into images using `pdf2image` library. Then, it applies OCR to each image using `pytesseract`; to speed up processing, this is done in parallel using the multiprocessing library. The OCR'ed text is then passed through the Llama2 13B Chat model, which helps to correct OCR errors and improve the formatting of the text. It also provides an option to check if the OCR output is valid English and to reformat the text using markdown. The final text is then written to an output file. Additionally, the project includes a function to filter out potential hallucinations from the LLM corrected text, using fuzzy string matching with the original OCR text. 

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

- `convert_pdf_to_images_func`: Converts PDF to images.
- `check_extracted_pages_func`: Checks if the extracted text is long enough to be a page.
- `remove_intro`: Removes the intro from the LLM output text.
- `is_valid_english`: Checks if the given text is valid English.
- `process_text_with_llm_func`: Processes the text with the LLM.
- `filter_hallucinations`: Filters out hallucinations from the corrected text.
- `ocr_image`: Performs OCR on the given image.

## Note
This project uses the fuzzy string matching to filter out potential hallucinations from the LLM corrected text. This is a simple heuristic and may not be perfect. You may need to adjust the threshold for fuzzy matching or use a different method depending on your needs.
