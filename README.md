# Use Llama2 to Improve the Accuracy of Tesseract OCR:

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

