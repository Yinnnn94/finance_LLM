import os
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from pytesseract import Output

# 設置 Tesseract 的執行檔路徑
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def extract_text_from_pdf(pdf_path):
    # 用來儲存提取的文本
    extracted_text = []

    # 打開 PDF
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]

            # 檢查頁面是否有可提取的文字
            text = page.get_text("text")
            if text.strip():
                # 如果有文字，直接加入提取結果
                extracted_text.append(f"Page {page_num + 1}:\n{text}")
            else:
                # 沒有文字的話，假設它是掃描頁面，並使用 OCR
                print(f"Page {page_num + 1} of {pdf_path} appears to be a scanned image, using OCR.")
                # count += 1
                # 將 PDF 頁面轉換為圖像
                image = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1)[0]

                # 使用 pytesseract 進行 OCR
                ocr_text = pytesseract.image_to_string(image, output_type=Output.STRING, lang='chi_tra')
                extracted_text.append(f"Page {page_num + 1} (OCR):\n{ocr_text}")

    # 將提取的文字合併並返回
    return "\n".join(extracted_text)


def extract_text_from_folder(folder_path):
    # 找到資料夾中的所有 PDF 文件
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"Extracting text from {pdf_file}...\n")

        # 提取每個 PDF 的文字
        text_output = extract_text_from_pdf(pdf_path)

        # 可以選擇儲存到單獨的文字檔案
        output_path = os.path.join(folder_path, f"{os.path.splitext(pdf_file)[0]}.txt")
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(text_output)

        print(f"Text extracted from {pdf_file} and saved to {output_path}\n")


# 使用範例

folder_path = "C:\\Users\\User\\PycharmProjects\\Usan\\reference\\finance" # 替換為資料夾的路徑
extract_text_from_folder(folder_path)
