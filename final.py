import os
import pdfplumber
import json
from Model.insurance import InsuranceQuery
from tqdm import tqdm
def load_data(source_path):
    masked_file_ls = os.listdir(source_path)
    corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in tqdm(masked_file_ls)}
    return corpus_dict

def read_pdf(pdf_loc, page_infos: list = None):
    pdf = pdfplumber.open(pdf_loc)
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for page in pages:
        text = page.extract_text()
        if text:
            pdf_text += text
    pdf.close()
    return pdf_text

# 主程式：加載問題並進行檢索
print('load_model')
answer_dict = {"answers": []}

with open('dataset\preliminary\questions_preliminary.json', 'r', encoding='utf8') as f:
    qs_ref = json.load(f)

# 加載保險資料庫資料
source_path_insurance = os.path.join('reference', 'insurance')
corpus_dict_insurance = load_data(source_path_insurance)
insurance_qs = InsuranceQuery()
# 搜尋問題
for q_dict in qs_ref['questions']:
    print(q_dict)
    if q_dict['category'] == 'insurance':
        retrieved = insurance_qs.BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_insurance)
        answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
        print('-' * 100)
    else:
        pass

# 將答案字典保存為json文件
with open('output_insurance.json', 'w', encoding='utf8') as f:
    json.dump(answer_dict, f, ensure_ascii=False, indent=4)

