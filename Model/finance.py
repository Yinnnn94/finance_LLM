import argparse
import csv
import json
import os
import re

import torch
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


class KeywordSearch:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.log_file = "query_log_with_e5.csv"

        if not os.path.exists(self.log_file):
            with open(self.log_file, mode="w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file)
                writer.writerow(["QID", "Query", "Keywords", "Pred QID"])

    def get_keywords(self, query: str) -> list[str]:
        if self.client is None:
            raise ValueError("OPENAI_API_KEY is required for finance keyword extraction.")

        messages = [
            {
                "role": "system",
                "content": (
                    "Extract concise Traditional Chinese financial keywords from the user query. "
                    "Return only comma-separated keywords, with no explanation."
                ),
            },
            {"role": "user", "content": query},
        ]

        completion = self.client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=messages,
            temperature=0,
        )

        keywords = completion.choices[0].message.content.strip().split(",")
        keywords = [keyword.strip() for keyword in keywords if keyword.strip()]
        print(f"\nQuery: {query}")
        print(f"Extracted Keywords: {keywords}")
        return keywords

    def find_matching_docs(self, keywords: list[str], source_docs: list[int], corpus_dict: dict[int, str]) -> list[int]:
        doc_scores = {}

        for doc_id in source_docs:
            doc_id = int(doc_id)
            if doc_id not in corpus_dict:
                continue

            doc_text = corpus_dict[doc_id]
            matched_keywords = [keyword for keyword in keywords if keyword in doc_text]

            if matched_keywords:
                doc_scores[doc_id] = {
                    "score": len(matched_keywords),
                    "matched_keywords": matched_keywords,
                }

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        matches = [doc_id for doc_id, _ in sorted_docs]

        if matches:
            print(f"Found matches: {matches}")
        else:
            print("No direct keyword matches found.")

        return matches

    def log_query(self, qid: int, query: str, keywords: list[str], pred_qid: int):
        with open(self.log_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([qid, query, ",".join(keywords), pred_qid])


def load_data(source_path: str) -> dict[int, str]:
    corpus_dict = {}

    for file_name in os.listdir(source_path):
        if not file_name.endswith(".txt"):
            continue

        try:
            file_id = int(file_name.replace(".txt", ""))
        except ValueError:
            continue

        with open(os.path.join(source_path, file_name), "r", encoding="utf-8") as file:
            corpus_dict[file_id] = file.read()

    return corpus_dict


def split_text_into_chunks(text: str, max_chunk_size: int = 200) -> list[str]:
    sentences = re.split(r"[。！？\n]", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = ""
        current_chunk += sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


class EmbeddingRetriever:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_embedding(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu()

    def retrieve(self, query: str, keywords: list[str], source: list[int], corpus_dict: dict[int, str]) -> int | None:
        try:
            query_embedding = self.get_embedding(query)
            keywords_embedding = self.get_embedding(" ".join(keywords) if keywords else query)

            best_doc_id = None
            highest_score = float("-inf")

            for doc_id in source:
                doc_id = int(doc_id)
                if doc_id not in corpus_dict:
                    continue

                doc_chunks = split_text_into_chunks(corpus_dict[doc_id])
                if not doc_chunks:
                    continue

                chunk_embeddings = [self.get_embedding(chunk) for chunk in doc_chunks]
                chunk_embeddings_tensor = torch.stack(chunk_embeddings)

                query_score = cosine_similarity(query_embedding.reshape(1, -1), chunk_embeddings_tensor.numpy()).max()
                keyword_score = cosine_similarity(keywords_embedding.reshape(1, -1), chunk_embeddings_tensor.numpy()).max()
                combined_score = 0.5 * query_score + 0.5 * keyword_score

                if combined_score > highest_score:
                    highest_score = combined_score
                    best_doc_id = doc_id

            return best_doc_id
        except Exception as exc:
            print(f"Error in retrieve: {exc}")
            return int(source[0]) if source else None


def process_queries(args):
    search_system = KeywordSearch()
    corpus_dict = load_data(os.path.join(args.source_path, "finance"))
    retriever = EmbeddingRetriever()

    with open(args.question_path, "r", encoding="utf-8") as file:
        questions = json.load(file)

    answer_dict = {"answers": []}

    for question in tqdm(questions["questions"]):
        if question.get("category") != "finance":
            continue

        try:
            keywords = search_system.get_keywords(question["query"])
            matches = search_system.find_matching_docs(keywords, question["source"], corpus_dict)

            if len(matches) == 1:
                retrieved = matches[0]
            elif len(matches) > 1:
                retrieved = retriever.retrieve(question["query"], keywords, matches, corpus_dict)
            else:
                retrieved = retriever.retrieve(question["query"], keywords, question["source"], corpus_dict)

            if retrieved is not None:
                answer_dict["answers"].append({"qid": question["qid"], "retrieve": retrieved})
                search_system.log_query(question["qid"], question["query"], keywords, retrieved)
                print(f"Processed Query ID {question['qid']} - Retrieved Doc: {retrieved}")
        except Exception as exc:
            print(f"Error processing question {question.get('qid')}: {exc}")

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(answer_dict, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finance retrieval system")
    parser.add_argument("--question_path", type=str, required=True, help="Path to question JSON file")
    parser.add_argument("--source_path", type=str, required=True, help="Path to source documents")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for results JSON")
    process_queries(parser.parse_args())
