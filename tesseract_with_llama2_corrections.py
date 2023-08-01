from pdf2image import convert_from_path
import pytesseract
from llama_cpp import Llama
from fuzzywuzzy import fuzz
from multiprocessing import Pool
import os

# sudo apt-get install -y tesseract-ocr libtesseract-dev poppler-utils
# git lfs install
# git clone https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML


def convert_pdf_to_images_func(input_pdf_file_path, max_test_pages):
    if max_test_pages == 0:
        print(f"Now converting all pages of PDF file {input_pdf_file_path} to images...")
        list_of_scanned_images = convert_from_path(input_pdf_file_path, first_page=1)
    else:
        print(f"Now converting first {max_test_pages} pages of PDF file {input_pdf_file_path} to images...")
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

def remove_intro(llm_output_2_text):
    # Strip leading and trailing whitespace before splitting the lines
    lines = llm_output_2_text.strip().splitlines()
    # Skip the first line and the following blank line
    lines = lines[2:] if lines[1].strip() == '' else lines[1:]
    return '\n'.join(lines)

def is_valid_english(llm_output_1_text):
    # Use the lower() function to make the string comparison case-insensitive.
    # Use the strip() function to remove any leading or trailing white space.
    # Then check if the string starts with 'yes'.
    return llm_output_1_text.lower().strip().startswith('yes')

def process_text_with_llm_func(extracted_text_string, check_if_valid_english=False, reformat_as_markdown=True):
    max_tokens = 2*len(extracted_text_string) + 50
    prompt_text_1 = f"Q: Is this valid English text? (y/n): ```{extracted_text_string}``` A: _|_"
    prompt_text_2 = f"Q: Correct any typos caused by bad OCR in this text, using common sense reasoning, responding only with the corrected text: ```{extracted_text_string}``` A: _|_"
    if check_if_valid_english:
        llm_output_1 = llm(prompt_text_1, max_tokens=max_tokens, stop=["Q:", "_|_"], echo=True)
        llm_output_1_text = llm_output_1["choices"][0]["text"].replace(prompt_text_1, "")
        valid_english = is_valid_english(llm_output_1_text)
    else:
        valid_english = False
    if valid_english or not check_if_valid_english:
        llm_output_2 = llm(prompt_text_2, max_tokens=max_tokens, stop=["Q:", "_|_"], echo=True)
        llm_output_2_text = llm_output_2["choices"][0]["text"].replace(prompt_text_2, "")
        corrected_extracted_text_string = remove_intro(llm_output_2_text)
    if reformat_as_markdown:
        prompt_test_3 = f"Q: Reformat this text to be more readable using markdown formatting and using common sense reasoning; respond only with the reformatted text: ```{corrected_extracted_text_string}``` A: _|_"    
        llm_output_3 = llm(corrected_extracted_text_string, max_tokens=max_tokens, stop=["Q:", "_|_"], echo=True)
        corrected_extracted_text_string = llm_output_3["choices"][0]["text"].replace(prompt_test_3, "")
    return corrected_extracted_text_string

def filter_hallucinations(corrected_text, raw_text, threshold=90):
    corrected_sentences = corrected_text.split('. ')
    raw_sentences = raw_text.split('. ')
    filtered_sentences = []
    for corrected_sentence in corrected_sentences:
        matches = [fuzz.ratio(corrected_sentence, raw_sentence) for raw_sentence in raw_sentences]
        best_match = max(matches) if matches else 0
        if best_match >= threshold:
            filtered_sentences.append(corrected_sentence)
    return '. '.join(filtered_sentences)

def ocr_image(image):
    return pytesseract.image_to_string(image)
    

if __name__ == '__main__':
    input_pdf_file_path = '160301289-Warren-Buffett-Katharine-Graham-Letter.pdf'
    max_test_pages = 0 # set to 0 to convert all pages of the PDF file using Tesseract
    skip_first_n_pages = 0 # set to 0 to process all pages with the LLM
    check_if_valid_english = False # set to True to check if the extracted text is valid English
    reformat_as_markdown = True # set to True to reformat the corrected extracted text using markdown formatting
    
    list_of_scanned_images = convert_pdf_to_images_func(input_pdf_file_path, max_test_pages)
    model_file_path = "./Llama-2-13B-chat-GGML/llama-2-13b-chat.ggmlv3.q4_0.bin"
    print(f"Loading Llama model from {model_file_path}...")
    llm = Llama(model_path=model_file_path, n_ctx=2048)
    print(f"Tesseract version: {pytesseract.get_tesseract_version()}")

    print(f"Extracting text from converted pages...")
    with Pool() as p:
        list_of_extracted_text_strings = p.map(ocr_image, list_of_scanned_images)
    print(f"Done extracting text from converted pages. \n")

    raw_ocr_output = "\n".join(list_of_extracted_text_strings)
    base_name = os.path.splitext(input_pdf_file_path)[0]
    raw_ocr_output_file_path = f"{base_name}__raw_ocr_output.txt"
    with open(raw_ocr_output_file_path, "w") as f:
        f.write(raw_ocr_output)

    # process the OCR output
    list_of_corrected_text_strings = []
    for ii, text in enumerate(list_of_extracted_text_strings):
        if ii < skip_first_n_pages:
            continue
        extracted_text_string = check_extracted_pages_func(text)
        if extracted_text_string:
            print(f"Processing page {ii + 1} with LLM...")
            corrected_extracted_text_string = process_text_with_llm_func(extracted_text_string, check_if_valid_english, reformat_as_markdown)
            print(f"Corrected text from page {ii + 1}:")
            print(corrected_extracted_text_string)
            print('_'*80)
            list_of_corrected_text_strings.append(corrected_extracted_text_string)

    # join the list of strings into a single string with a newline after each page
    final_text = "\n".join(list_of_corrected_text_strings)

    # get the base name of the input file (without the extension)
    base_name = os.path.splitext(input_pdf_file_path)[0]
    # choose the extension based on the reformat_as_markdown flag
    output_extension = '.md' if reformat_as_markdown else '.txt'
    # create the output file path
    output_file_path = base_name + output_extension

    # write the final text to the output file
    with open(output_file_path, 'w') as f:
        f.write(final_text)

    print(f"LLM corrected text written to: {output_file_path}")
    
    print('Now filtering out hallucinations from corrected text...')
    # filter out hallucinations from the corrected output
    filtered_output = filter_hallucinations(final_text, raw_ocr_output)
    print('Done filtering out hallucinations.')
    
    final_output_file_path = base_name + '_filtered' + output_extension
    with open(final_output_file_path, 'w') as f:
        f.write(filtered_output)
        
    print(f"Filtered text written to: {final_output_file_path}")
    print(f"Done processing {input_pdf_file_path}. See output files: Raw OCR: {raw_ocr_output_file_path} \n LLM Corrected: {output_file_path} \n LLM Corrected with Hallucinations Filtered: {final_output_file_path}\n")
    
