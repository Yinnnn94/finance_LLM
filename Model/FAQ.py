import argparse
import json
import os
import pickle

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_e5_model():
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device


def get_embedding(text, model, tokenizer, device):
    if not text.startswith("passage: ") and not text.startswith("query: "):
        text = "passage: " + text

    encoded = tokenizer(
        text,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)
        embedding = output.last_hidden_state.mean(dim=1)

    return embedding.cpu().numpy()


def load_or_create_faq_embeddings(faq_data, model, tokenizer, device, cache_file="faq_embeddings.pkl"):
    if os.path.exists(cache_file):
        print("Loading cached FAQ embeddings...")
        with open(cache_file, "rb") as file:
            return pickle.load(file)

    print("Creating FAQ embeddings...")
    embeddings_dict = {}
    for key, text in tqdm(faq_data.items(), desc="Embedding FAQ"):
        embeddings_dict[key] = get_embedding(text, model, tokenizer, device)

    with open(cache_file, "wb") as file:
        pickle.dump(embeddings_dict, file)

    return embeddings_dict


def semantic_retrieve(query, source, embeddings_dict, model, tokenizer, device):
    try:
        query_embedding = get_embedding("query: " + query, model, tokenizer, device)
        source_ids = [int(source_id) for source_id in source]
        source_embeddings = np.vstack([embeddings_dict[source_id] for source_id in source_ids])

        similarities = cosine_similarity(query_embedding, source_embeddings)[0]
        max_sim_idx = np.argmax(similarities)
        return source_ids[max_sim_idx]
    except Exception as exc:
        print(f"FAQ retrieval error: {exc}")
        return int(source[0])


def main():
    parser = argparse.ArgumentParser(description="FAQ semantic retrieval system")
    parser.add_argument("--question_path", type=str, required=True, help="Path to question JSON file")
    parser.add_argument("--source_path", type=str, required=True, help="Path to source documents")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for results JSON")
    args = parser.parse_args()

    with open(args.question_path, "r", encoding="utf-8") as file:
        qs_ref = json.load(file)

    faq_path = os.path.join(args.source_path, "faq", "pid_map_content.json")
    with open(faq_path, "r", encoding="utf-8") as file:
        key_to_source_dict = json.load(file)
        key_to_source_dict = {int(key): str(value) for key, value in key_to_source_dict.items()}

    model, tokenizer, device = load_e5_model()
    embeddings_dict_faq = load_or_create_faq_embeddings(key_to_source_dict, model, tokenizer, device)

    answer_dict = {"answers": []}
    faq_questions = [question for question in qs_ref["questions"] if question.get("category") == "faq"]

    for question in tqdm(faq_questions, desc="Processing FAQ"):
        retrieved = semantic_retrieve(
            question["query"],
            question["source"],
            embeddings_dict_faq,
            model,
            tokenizer,
            device,
        )
        answer_dict["answers"].append({"qid": question["qid"], "retrieve": retrieved})

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(answer_dict, file, ensure_ascii=False, indent=4)

    print("FAQ retrieval completed.")


if __name__ == "__main__":
    main()
