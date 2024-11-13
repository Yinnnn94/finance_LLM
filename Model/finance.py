import json
import os
from openai import OpenAI
import csv
import jieba
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import re
from typing import List, Dict, Tuple
import numpy as np


class KeywordSearch:
    def __init__(self):
        self.api_key = "YOUR_API_KEY" # 請替換為您的OpenAI API金鑰
        self.log_file = 'query_log_with_e5.csv'
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='', encoding='utf-8-sig') as file:
                writer = csv.writer(file)
                writer.writerow(["QID", "Query", "Keywords", "Pred QID"])

    def get_keywords(self, query: str) -> list:
        """從查詢中提取多個財務關鍵詞"""
        messages = [
            {"role": "system", "content": """你的任務是從輸入的問句中找出最重要的三個財務相關關鍵詞。
                               規則：
                               1. 不要包含公司名稱
                               2. 不要包含年份、日期(請注意確保不要出現)
                               3. 關注財務指標、會計項目、財務術語
                               4. 用逗號分隔三個關鍵詞
                               5. 若找不到三個關鍵詞，則儘可能找出關鍵詞
                               6. 請直接輸出關鍵詞，不要加入任何解釋或標點符號（除了分隔的逗號）
                               例如輸出格式：營業收入,毛利率,營業利益"""},
            {"role": "user", "content": query}
        ]

        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        )

        keywords = completion.choices[0].message.content.strip().split(',')
        print(f"\nQuery: {query}")
        print(f"Extracted Keywords: {keywords}")
        return keywords

    def find_matching_docs(self, keywords: list, source_docs: list, corpus_dict: dict) -> list:
        """找出包含關鍵詞的文件，並計算匹配度"""
        doc_scores = {}

        for doc_id in source_docs:
            if doc_id not in corpus_dict:
                continue

            doc_text = corpus_dict[doc_id]
            score = 0
            matched_keywords = []

            for keyword in keywords:
                if keyword.strip() in doc_text:  # 添加strip()去除可能的空白
                    score += 1
                    matched_keywords.append(keyword)

            if score > 0:
                doc_scores[doc_id] = {
                    'score': score,
                    'matched_keywords': matched_keywords
                }

        # 根據匹配的關鍵詞數量排序
        sorted_docs = sorted(doc_scores.items(),
                             key=lambda x: x[1]['score'],
                             reverse=True)

        matches = [doc_id for doc_id, _ in sorted_docs]

        if matches:
            print(f"Found matches: {matches}")
            print("Matching details:")
            for doc_id, details in sorted_docs:
                print(f"Doc {doc_id}: Matched {details['score']} keywords: {details['matched_keywords']}")
        else:
            print("No direct keyword matches found.")

        return matches

    def log_query(self, qid: int, query: str, keywords: list, pred_qid: int):
        """記錄查詢記錄到CSV"""
        with open(self.log_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([qid, query, ','.join(keywords), pred_qid])


def load_data(source_path: str) -> dict:
    """載入文件"""
    corpus_dict = {}
    for file in os.listdir(source_path):
        if file.endswith('.txt'):
            try:
                file_id = int(file.replace('.txt', ''))
                with open(os.path.join(source_path, file), 'r', encoding='utf-8') as f:
                    corpus_dict[file_id] = f.read()
            except ValueError:
                continue
    return corpus_dict


def split_text_into_chunks(text: str, max_chunk_size: int = 200) -> list:
    """將文本按標點符號分割並進行分段"""
    sentences = re.split(r'[，。\n]', text)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = ""
        current_chunk += sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


class EmbeddingRetriever:
    def __init__(self, model_name="intfloat/multilingual-e5-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_embedding(self, text: str) -> torch.Tensor:
        """獲取文本的embedding向量"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu()

    def retrieve(self, query: str, keywords: list, source: list, corpus_dict: dict) -> int:
        """改進的檢索方法，同時考慮問句和關鍵詞"""
        try:
            # 獲取問句的embedding
            query_embedding = self.get_embedding(query)

            # 獲取關鍵詞的embedding
            keywords_text = ' '.join(keywords)
            keywords_embedding = self.get_embedding(keywords_text)

            best_doc_id, highest_score = None, float("-inf")

            for doc_id in source:
                if doc_id not in corpus_dict:
                    continue

                doc_text = corpus_dict[doc_id]
                doc_chunks = split_text_into_chunks(doc_text)

                # 批次處理chunks以提高效率
                chunk_embeddings = []
                batch_size = 8  # 可以根據記憶體調整
                for i in range(0, len(doc_chunks), batch_size):
                    batch_chunks = doc_chunks[i:i + batch_size]
                    batch_embeddings = [self.get_embedding(chunk) for chunk in batch_chunks]
                    chunk_embeddings.extend(batch_embeddings)

                chunk_embeddings_tensor = torch.stack(chunk_embeddings)

                # 計算問句相似度
                query_scores = cosine_similarity(query_embedding.reshape(1, -1),
                                                 chunk_embeddings_tensor.numpy())
                query_max_score = query_scores.max()

                # 計算關鍵詞相似度
                keywords_scores = cosine_similarity(keywords_embedding.reshape(1, -1),
                                                    chunk_embeddings_tensor.numpy())
                keywords_max_score = keywords_scores.max()

                # 組合分數 (可以調整權重)
                combined_score = 0.5 * query_max_score + 0.5 * keywords_max_score

                if combined_score > highest_score:
                    highest_score = combined_score
                    best_doc_id = doc_id

            return best_doc_id

        except Exception as e:
            print(f"Error in retrieve: {str(e)}")
            return source[0] if source else None


def process_queries(args):
    """處理查詢的主函數"""
    try:
        search_system = KeywordSearch()
        corpus_dict = load_data(os.path.join(args.source_path, 'finance'))
        retriever = EmbeddingRetriever()

        with open(args.question_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)

        answer_dict = {"answers": []}

        for q in tqdm(questions['questions']):
            if q['category'] == 'finance':
                try:
                    # 獲取關鍵詞列表
                    keywords = search_system.get_keywords(q['query'])

                    # 找出匹配的文檔
                    matches = search_system.find_matching_docs(keywords, q['source'], corpus_dict)

                    if len(matches) == 1:
                        print('單一最佳匹配，直接使用')
                        retrieved = matches[0]
                    elif len(matches) > 1:
                        # 使用改進的檢索方法
                        retrieved = retriever.retrieve(q['query'], keywords, matches, corpus_dict)
                    else:
                        # 無匹配時對所有候選文檔進行檢索
                        retrieved = retriever.retrieve(q['query'], keywords, q['source'], corpus_dict)

                    if retrieved is not None:
                        answer_dict['answers'].append({"qid": q['qid'], "retrieve": retrieved})
                        print(f"Processed Query ID {q['qid']} - Retrieved Doc: {retrieved}")

                        search_system.log_query(
                            qid=q['qid'],
                            query=q['query'],
                            keywords=keywords,
                            pred_qid=retrieved
                        )
                except Exception as e:
                    print(f"Error processing question {q['qid']}: {str(e)}")
                    continue

        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(answer_dict, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"Error in process_queries: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Prompt-based keyword search system')
    parser.add_argument('--question_path', type=str, required=True, help="Path to question JSON file")
    parser.add_argument('--source_path', type=str, required=True, help="Path to source documents")
    parser.add_argument('--output_path', type=str, required=True, help="Output path for results JSON")

    args = parser.parse_args()
    process_queries(args)