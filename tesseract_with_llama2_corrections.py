from pdf2image import convert_from_path
import pytesseract
from llama_cpp import Llama


# sudo apt-get install -y tesseract-ocr libtesseract-dev poppler-utils
# git lfs install
# git clone https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML
# python3 -m venv venv
# source venv/bin/activate
# python3 -m pip install --upgrade pip
# python3 -m pip install wheel
# pip install -r requirements.txt

llm = Llama(model_path="./Llama-2-7B-GGML/llama-2-7b.ggmlv3.q4_0.bin", n_ctx=2048)

print(f"Tesseract version: {pytesseract.get_tesseract_version()}")

input_pdf_file_path = 'royalnavyhistor03clow.pdf'
max_test_pages = 30

def convert_pdf_to_images_func(input_pdf_file_path, max_test_pages):
    print(f"Now converting first {max_test_pages} pages of PDF file {input_pdf_file_path} to images...")
    # Use first_page and last_page to limit the pages that get converted
    list_of_scanned_images = convert_from_path(input_pdf_file_path, first_page=1, last_page=max_test_pages)
    print(f"Done converting pages from PDF file {input_pdf_file_path} to images.")
    return list_of_scanned_images

def check_extracted_pages_func(extracted_text_string):
    #first check if it's long enough to be a page:
    if len(extracted_text_string) < 30:
        return False
    #now check if it has enough words to be a page:
    if len(extracted_text_string.split()) < 20:
        return False
    return extracted_text_string

def process_text_with_llm_func(extracted_text_string):
    # prompt_text_1 = f"Q: Is this valid English text? (y/n): ```{extracted_text_string}``` A: _|_"
    max_tokens = 2*len(extracted_text_string) + 50
    prompt_text_2 = f"Q: Correct any typos caused by bad OCR in this text, using common sense reasoning, responding only with the corrected text: ```{extracted_text_string}``` A: _|_"
    # llm_output_1 = llm(prompt_text_1, max_tokens=max_tokens, stop=["Q:", "_|_"], echo=True)
    # llm_output_1_text = llm_output_1["choices"][0]["text"].replace(prompt_text_1, "")
    # print(f"llm_output_1_text: {llm_output_1_text}")
    llm_output_1_text = ''
    llm_output_2 = llm(prompt_text_2, max_tokens=max_tokens, stop=["Q:", "_|_"], echo=True)
    llm_output_2_text = llm_output_2["choices"][0]["text"].replace(prompt_text_2, "")
    print(f"llm_output_2_text: {llm_output_2_text}")    
    return llm_output_1_text, llm_output_2_text

list_of_extracted_text_strings = []
list_of_scanned_images = convert_pdf_to_images_func(input_pdf_file_path, max_test_pages)

skip_first_n_pages = 20
print(f"Extracting text from converted pages...")
for ii, current_image in enumerate(list_of_scanned_images):
    if ii < skip_first_n_pages:
        continue
    text = pytesseract.image_to_string(current_image)
    extracted_text_string = check_extracted_pages_func(text)
    if extracted_text_string:
        print(f"\nText from page {ii + 1}:")
        llm_output_1_text, llm_output_2_text = process_text_with_llm_func(extracted_text_string)
        list_of_extracted_text_strings.append(llm_output_2_text)
        print(text)
