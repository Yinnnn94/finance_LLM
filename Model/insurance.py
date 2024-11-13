import jieba  # 用於中文文本分詞
from rank_bm25 import BM25Okapi
import re

class InsuranceQuery:
    def __init__(self, synonyms=None):
        # 如果未提供自訂同義詞字典，使用預設字典
        self.synonyms = synonyms if synonyms else {
            "保險": ["險", "保險產品", "投保"],
            "理賠": ["賠償", "索賠"],
            '貸款': ["借款", '信貸', '貸款協議'],
            "財務報表": ["財報", "財務狀況表"],
            "效力": ["效", "有效", "有效力"],
            "要保": ["要保人"],
            "被保": ["被保人"]
        }

    def expand_query(self, query):
        """使用同義詞字典進行查詢擴展"""
        expanded_query = set(jieba.lcut(query))
        additional_terms = set()
        
        for word in expanded_query:
            if word in self.synonyms:
                additional_terms.update(self.synonyms[word])
        
        expanded_query.update(additional_terms)
        return ' '.join(expanded_query)

    def split_text_into_chunks(self, text, max_chunk_size=200):
        """將文本分割成指定長度的 chunks"""
        sentences = re.split(r'[，。\n]', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            current_chunk += sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def BM25_retrieve(self, qs, source, corpus_dict):
        """使用 BM25 進行檢索，返回最匹配的文檔 ID"""
        # 過濾出符合條件的文檔
        filtered_corpus = [corpus_dict[int(file)] for file in source]
        
        all_chunks = []
        chunk_to_doc_map = []
        
        for idx, doc in enumerate(filtered_corpus):
            chunks = self.split_text_into_chunks(doc)
            tokenized_chunks = [list(jieba.cut_for_search(chunk)) for chunk in chunks]
            all_chunks.extend(tokenized_chunks)
            chunk_to_doc_map.extend([source[idx]] * len(tokenized_chunks))
        
        bm25 = BM25Okapi(all_chunks)
        tokenized_query = list(jieba.cut_for_search(qs))
        
        ans = bm25.get_top_n(tokenized_query, all_chunks, n=1)
        best_chunk = ans[0]
        
        best_chunk_index = all_chunks.index(best_chunk)
        best_doc_id = chunk_to_doc_map[best_chunk_index]
        
        return best_doc_id
