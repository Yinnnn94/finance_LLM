import os
import json
import argparse
import torch
import pickle
from tqdm import tqdm
import jieba
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_e5_model():
    """載入 E5 模型和 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, tokenizer, device

def get_embedding(text, model, tokenizer, device):
    """獲取文本的 embedding"""
    if not text.startswith('passage: ') and not text.startswith('query: '):
        text = 'passage: ' + text
        
    encoded = tokenizer(text, 
                       max_length=512, 
                       padding=True, 
                       truncation=True, 
                       return_tensors='pt')
    
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    with torch.no_grad():
        output = model(**encoded)
        embedding = output.last_hidden_state.mean(dim=1)
        
    return embedding.cpu().numpy()

def load_or_create_faq_embeddings(faq_data, model, tokenizer, device, cache_file="faq_embeddings.pkl"):
    """載入或創建 FAQ embeddings"""
    if os.path.exists(cache_file):
        print("載入已儲存的 FAQ embeddings...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print("創建新的 FAQ embeddings...")
    embeddings_dict = {}
    for key, text in tqdm(faq_data.items(), desc="處理FAQ embeddings"):
        embedding = get_embedding(text, model, tokenizer, device)
        embeddings_dict[key] = embedding
    
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    
    return embeddings_dict

def semantic_retrieve(query, source, embeddings_dict, model, tokenizer, device):
    """使用語義檢索查找最相關文檔"""
    try:
        query = 'query: ' + query
        query_embedding = get_embedding(query, model, tokenizer, device)
        
        source_ids = [int(s) for s in source]
        source_embeddings = np.vstack([embeddings_dict[sid] for sid in source_ids])
        
        similarities = cosine_similarity(query_embedding, source_embeddings)[0]
        max_sim_idx = np.argmax(similarities)
        retrieved_id = source_ids[max_sim_idx]
        
        return retrieved_id
        
    except Exception as e:
        print(f"檢索過程中發生錯誤: {str(e)}")
        return source[0]

def main():
    parser = argparse.ArgumentParser(description='FAQ檢索系統')
    parser.add_argument('--question_path', type=str, required=True, help='問題檔案路徑')
    parser.add_argument('--source_path', type=str, required=True, help='參考資料路徑')
    parser.add_argument('--output_path', type=str, required=True, help='輸出檔案路徑')
    args = parser.parse_args()

    answer_dict = {"answers": []}

    # 讀取問題
    print("讀取問題...")
    with open(args.question_path, 'r', encoding='utf-8') as f:
        qs_ref = json.load(f)

    # 載入 FAQ 資料
    print("載入 FAQ 資料...")
    with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'r', encoding='utf-8') as f_s:
        key_to_source_dict = json.load(f_s)
        key_to_source_dict = {int(key): str(value) for key, value in key_to_source_dict.items()}

    # 載入 E5 模型
    print("載入 E5 模型...")
    model, tokenizer, device = load_e5_model()
    embeddings_dict_faq = load_or_create_faq_embeddings(
        key_to_source_dict, model, tokenizer, device,
        cache_file="faq_embeddings.pkl"
    )
    
    # 只取出FAQ類別的問題
    faq_questions = [q for q in qs_ref['questions'] if q['category'] == 'faq']
    
    # 處理每個FAQ問題
    for q_dict in tqdm(faq_questions, desc="處理FAQ問題"):
        try:
            retrieved = semantic_retrieve(q_dict['query'], q_dict['source'], 
                                       embeddings_dict_faq, model, tokenizer, device)
            
            answer_dict['answers'].append({
                "qid": q_dict['qid'],
                "retrieve": retrieved
            })
                
        except Exception as e:
            print(f"處理問題 {q_dict['qid']} 時發生錯誤: {str(e)}")
            answer_dict['answers'].append({
                "qid": q_dict['qid'],
                "retrieve": int(q_dict['source'][0])
            })
    
    # 儲存結果
    print("儲存結果...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)
    
    print("完成！")

if __name__ == "__main__":
    main()