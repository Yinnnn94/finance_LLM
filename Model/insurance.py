import argparse
import json
import os
import re

import jieba
import pdfplumber
from rank_bm25 import BM25Okapi
from tqdm import tqdm


def expand_query(query):
    synonyms = {}
    expanded_query = set(jieba.lcut(query))

    for word in list(expanded_query):
        if word in synonyms:
            expanded_query.update(synonyms[word])

    return " ".join(expanded_query)


def read_pdf(pdf_loc, page_infos=None):
    with pdfplumber.open(pdf_loc) as pdf:
        pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
        pdf_text = ""
        for page in pages:
            text = page.extract_text()
            if text:
                pdf_text += text
    return pdf_text


def load_data(source_path):
    file_names = [file_name for file_name in os.listdir(source_path) if file_name.lower().endswith(".pdf")]
    corpus_dict = {}

    for file_name in tqdm(file_names, desc="Loading insurance PDFs"):
        file_id = int(file_name.replace(".pdf", ""))
        corpus_dict[file_id] = read_pdf(os.path.join(source_path, file_name))

    return corpus_dict


def split_text_into_chunks(text, max_chunk_size=200):
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


def bm25_retrieve(query, source, corpus_dict):
    filtered_corpus = [corpus_dict[int(file)] for file in source if int(file) in corpus_dict]
    if not filtered_corpus:
        return int(source[0])

    all_chunks = []
    chunk_to_doc_map = []

    for idx, doc in enumerate(filtered_corpus):
        chunks = split_text_into_chunks(doc)
        tokenized_chunks = [list(jieba.cut_for_search(chunk)) for chunk in chunks]
        all_chunks.extend(tokenized_chunks)
        chunk_to_doc_map.extend([source[idx]] * len(tokenized_chunks))

    if not all_chunks:
        return int(source[0])

    bm25 = BM25Okapi(all_chunks)
    tokenized_query = list(jieba.cut_for_search(expand_query(query)))
    best_chunk = bm25.get_top_n(tokenized_query, all_chunks, n=1)[0]
    best_chunk_index = all_chunks.index(best_chunk)
    return int(chunk_to_doc_map[best_chunk_index])


def process_queries(args):
    with open(args.question_path, "r", encoding="utf-8") as file:
        qs_ref = json.load(file)

    source_path_insurance = os.path.join(args.source_path, "insurance")
    corpus_dict_insurance = load_data(source_path_insurance)
    answer_dict = {"answers": []}

    for question in tqdm(qs_ref["questions"], desc="Processing insurance"):
        if question.get("category") != "insurance":
            continue

        retrieved = bm25_retrieve(question["query"], question["source"], corpus_dict_insurance)
        answer_dict["answers"].append({"qid": question["qid"], "retrieve": retrieved})

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(answer_dict, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insurance BM25 retrieval system")
    parser.add_argument("--question_path", type=str, required=True, help="Path to question JSON file")
    parser.add_argument("--source_path", type=str, required=True, help="Path to source documents")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for results JSON")
    process_queries(parser.parse_args())
