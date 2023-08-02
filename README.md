# Use Llama2 to Improve the Accuracy of Tesseract OCR:

This project aims to improve the quality of Optical Character Recognition (OCR) outputs using Large Language Models (LLMs). 

## Purpose
The purpose of the project is to convert scanned PDFs into readable text files. It does this by first leveraging OCR and then enhancing the quality of the OCR output by using an LLM to correct errors and format the text for readability. 

## Example Output
If you just want to preview the output, you can look at the [source PDF](https://github.com/Dicklesworthstone/llama2_aided_tesseract/blob/main/160301289-Warren-Buffett-Katharine-Graham-Letter.pdf) and the [raw output of the Tesseract OCR process](https://github.com/Dicklesworthstone/llama2_aided_tesseract/blob/main/160301289-Warren-Buffett-Katharine-Graham-Letter__raw_ocr_output.txt), and compare this to the [final markdown output after filtering out hallucinations](https://github.com/Dicklesworthstone/llama2_aided_tesseract/blob/main/160301289-Warren-Buffett-Katharine-Graham-Letter_filtered.md) (and also compare to the [initial output from the LLM corrections](https://github.com/Dicklesworthstone/llama2_aided_tesseract/blob/main/160301289-Warren-Buffett-Katharine-Graham-Letter.md) to see just how much the LLM hallucinates-- it's a often a lot!).

## Set up instructions:

```
sudo apt-get install -y tesseract-ocr libtesseract-dev poppler-utils

git clone https://github.com/Dicklesworthstone/llama2_aided_tesseract
cd llama2_aided_tesseract
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install wheel
pip install -r requirements.txt

git lfs install
git clone https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML
```
*Warning*: The last command above will download ~108gb worth of data for the model weights, so make sure you have enough free storage!


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

*Note*: This script is pretty slow, particularly on longer PDFs. The sample PDF mentioned above in the "Example Output" section took around an hour to completely process on a fairly powerful machine, but where everything was done using a CPU rather than a GPU.

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

## Hallucination Filtering

The process of filtering out hallucinations is pretty intricate. Here is a detailed breakdown of how it works:

The primary function for this process is `filter_hallucinations`, which takes as arguments the text corrected by the LLM (`corrected_text`), the original text extracted by OCR (`raw_text`), a starting threshold (`threshold`) for filtering, the path to the PDF file (`pdf_file_path`), and the path to the SQLite database (`db_path`) where embeddings are stored for reuse.

The function begins by defining a `threshold_increment` of 0.02. This is the amount by which the threshold for filtering out hallucinations is incremented in each iteration.

If `db_path` and `pdf_file_path` are provided, the function first checks if an SQLite database file exists at the `db_path`. If it does not, the function creates one. It then computes a SHA3-256 hash of the PDF file to uniquely identify it. This hash is used as a key to store and retrieve embeddings from the database.

The function connects to the SQLite database and creates a table named 'embeddings' if it does not already exist. This table has three columns: 'file_hash', 'original_embeddings', and 'corrected_embeddings'. The function then tries to fetch the embeddings corresponding to the file hash from the database. If it does not find them, it sets `original_embeddings` and `corrected_embeddings` to `None`. 

Next, the function creates a `LlamaCppEmbeddings` object, which is used to compute embeddings for the sentences. It then splits the `raw_text` and `corrected_text` into sentences. 

If `original_embeddings` is `None`, the function calculates the embeddings for the original sentences and stores them in a dictionary where the keys are the sentences and the values are the embeddings. It does the same for `corrected_embeddings`.

Once it has the embeddings, the function saves them to the database using the PDF file's hash as the key. 

The function then enters a loop where it filters out sentences from the corrected text based on the cosine similarity of their embeddings to the original sentences. It starts with the initial `threshold` and in each iteration, it increments the threshold by `threshold_increment`. 

In each iteration, the function first initializes an empty list `filtered_sentences`. It then goes through each sentence in the corrected text. For each sentence, it gets its embedding from `corrected_embeddings` and calculates its cosine similarity with the embeddings of the original sentences. If the maximum similarity is greater than or equal to the current threshold, the sentence is added to `filtered_sentences`. 

The function then joins the `filtered_sentences` into a string `filtered_corrected_text`. If the length of this text is less than the length of the original text minus 30 (an arbitrary value to allow for some difference), the function breaks the loop. If not, it increments the threshold and continues to the next iteration.

Finally, the function returns the `filtered_corrected_text`, `original_embeddings`, and `corrected_embeddings`.

There are two additional helper functions used in this process:

- `calculate_sentence_embedding`: This function takes as arguments a `LlamaCppEmbeddings` object and a text string. It attempts to calculate the embedding of the text. If it encounters an exception that the text has too many tokens, it trims the text by 5% and tries again until it succeeds.

- `calculate_similarity`: This function takes a tuple of two embeddings as argument and returns their cosine similarity.

This mechanism of filtering hallucinations is a heuristic approach and it might not be perfect. The `threshold` and `threshold_increment` values might need to be tuned according to the specific requirements of the use case.

## Contributions 
I made this mostly for fun and to learn more about using LLMs for more "utilitarian" tasks. That being said, I think this could eventually become a pretty useful tool for doing OCR on challenging files that result in tons of errors when using regular OCR without any "smart" corrections. If you're interested in helping, please submit a PR!
