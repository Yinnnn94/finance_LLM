import os

import fitz
import pytesseract
from pdf2image import convert_from_path
from pytesseract import Output

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()

pytesseract.pytesseract.tesseract_cmd = os.getenv(
    "TESSERACT_PATH",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)


def extract_text_from_pdf(pdf_path):
    extracted_text = []

    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text = page.get_text("text")

            if text.strip():
                extracted_text.append(f"Page {page_num + 1}:\n{text}")
            else:
                print(f"Page {page_num + 1} of {pdf_path} appears scanned, using OCR.")
                image = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1)[0]
                ocr_text = pytesseract.image_to_string(image, output_type=Output.STRING, lang="chi_tra")
                extracted_text.append(f"Page {page_num + 1} (OCR):\n{ocr_text}")

    return "\n".join(extracted_text)


def extract_text_from_folder(folder_path):
    pdf_files = [file_name for file_name in os.listdir(folder_path) if file_name.lower().endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"Extracting text from {pdf_file}...")

        text_output = extract_text_from_pdf(pdf_path)
        output_path = os.path.join(folder_path, f"{os.path.splitext(pdf_file)[0]}.txt")

        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(text_output)

        print(f"Text extracted from {pdf_file} and saved to {output_path}")


if __name__ == "__main__":
    input_dir = os.getenv("OCR_INPUT_DIR")
    if not input_dir:
        raise ValueError("Set OCR_INPUT_DIR before running OCR preprocessing.")

    extract_text_from_folder(input_dir)
