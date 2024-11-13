import os
import json
from tqdm import tqdm
import jieba  # 用於中文文本分詞
import pdfplumber
from rank_bm25 import BM25Okapi
import re


def expand_query(query):
    # 示例：使用簡單的同義詞字典進行查詢擴展
    synonyms = {
        "保險": ["險", "保險產品", "投保"],
        "理賠": ["賠償", "索賠"],
        '貸款': ["借款", '信貸', '貸款協議'],
        "財務報表": ["財報", "財務狀況表"],
        "效力": ["效", "有效", "有效力"],
        "要保": ["要保人"],
        "被保": ["被保人"]
    }

    expanded_query = set(jieba.lcut(query))
    additional_terms = set()  # 新的集合來存儲擴展的詞

    for word in expanded_query:
        if word in synonyms:
            additional_terms.update(synonyms[word])  # 將同義詞添加到新的集合中

    # 合併原查詢的詞和擴展的詞
    expanded_query.update(additional_terms)
    return ' '.join(expanded_query)


def load_data(source_path):
    masked_file_ls = os.listdir(source_path)
    corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in
                   tqdm(masked_file_ls)}
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


def split_text_into_chunks(text, max_chunk_size=200):
    # 將文本按標點符號分割，並進行分段
    sentences = re.split(r'[，。\n]', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # 如果當前段落加上新的句子超過最大長度，則保存當前段落並重新初始化
        if len(current_chunk) + len(sentence) > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = ""
        current_chunk += sentence

    # 加入最後的 chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks  # 回傳list


def BM25_retrieve(qs, source, corpus_dict):
    # 過濾出符合條件的文檔
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    # 分段並分詞
    all_chunks = []  # 所有文檔的分段集合
    chunk_to_doc_map = []  # 用於記錄每個 chunk 對應的文檔 ID
    for idx, doc in enumerate(filtered_corpus):
        chunks = split_text_into_chunks(doc)  # 將文檔切割為 chunks
        tokenized_chunks = [list(jieba.cut_for_search(chunk)) for chunk in chunks]  # 將每個 chunk 分詞
        all_chunks.extend(tokenized_chunks)  # 將所有分詞後的 chunk 加入集合
        chunk_to_doc_map.extend([source[idx]] * len(tokenized_chunks))  # 記錄每個 chunk 的來源文檔

    # 使用 BM25 建立檢索模型
    bm25 = BM25Okapi(all_chunks)

    # 對查詢進行分詞並檢索
    tokenized_query = list(jieba.cut_for_search(qs))
    ans = bm25.get_top_n(tokenized_query, all_chunks, n=1)  # 檢索最相關的分段（chunk）
    best_chunk = ans[0]

    # 找回與最佳匹配分段相對應的文檔
    best_chunk_index = all_chunks.index(best_chunk)
    best_doc_id = chunk_to_doc_map[best_chunk_index]

    return best_doc_id  # 回傳文檔 ID


# 主程式：加載問題並進行檢索
print('load_model')
answer_dict = {"answers": []}

with open('dataset\preliminary\questions_preliminary.json', 'r', encoding='utf8') as f:
    qs_ref = json.load(f)

# 加載資料庫資料
source_path_insurance = os.path.join('reference', 'insurance')
corpus_dict_insurance = load_data(source_path_insurance)

# 搜尋問題
for q_dict in qs_ref['questions']:
    print(q_dict)
    if q_dict['category'] == 'insurance':
        retrieved = BM25_retrieve(q_dict['query'], q_dict['source'], corpus_dict_insurance)
        answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
        print('-' * 100)
    else:
        pass

# 將答案字典保存為json文件
with open('output_insurance.json', 'w', encoding='utf8') as f:
    json.dump(answer_dict, f, ensure_ascii=False, indent=4)