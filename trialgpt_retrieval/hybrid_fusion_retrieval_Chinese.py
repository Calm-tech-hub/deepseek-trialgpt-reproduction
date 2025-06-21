__author__ = "qiao"

"""
Conduct the first stage retrieval by the hybrid retriever 
"""

from beir.datasets.data_loader import GenericDataLoader
import faiss
import json
from nltk import word_tokenize
import numpy as np
import os
from rank_bm25 import BM25Okapi
import sys
import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

ARTICLE_ENCODER_PATH = "/data/kmxu/TrialGPT/models/MedCPT-Article-Encoder"
QUERY_ENCODER_PATH = "/data/kmxu/TrialGPT/models/MedCPT-Query-Encoder"

def get_bm25_corpus_index():
    corpus_path = "trialgpt_retrieval/bm25_corpus_ovarian11.json"

    # if already cached then load, otherwise build
    if os.path.exists(corpus_path):
        corpus_data = json.load(open(corpus_path))
        tokenized_corpus = corpus_data["tokenized_corpus"]
        corpus_ids = corpus_data["corpus_ids"]
    else:
        tokenized_corpus = []
        corpus_ids = []

        with open("/data/kmxu/TrialGPT/dataset/sigir/processed_ovarian_cancer_data.json", "r") as f:
            corpus = json.load(f)
            for idx, entry in enumerate(corpus):
                corpus_ids.append(str(idx))
                
                # Just use the text field for tokens
                tokens = word_tokenize(entry["文本"].lower())
                tokenized_corpus.append(tokens)

        corpus_data = {
            "tokenized_corpus": tokenized_corpus,
            "corpus_ids": corpus_ids,
        }

        with open(corpus_path, "w") as f:
            json.dump(corpus_data, f, indent=4)
    
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, corpus_ids

def get_medcpt_corpus_index():
    corpus_path = "trialgpt_retrieval/ovarian_embeds.npy"
    ids_path = "trialgpt_retrieval/ovarian_ids.json"

    if os.path.exists(corpus_path):
        embeds = np.load(corpus_path)
        corpus_ids = json.load(open(ids_path))
    else:
        embeds = []
        corpus_ids = []

        model = AutoModel.from_pretrained(ARTICLE_ENCODER_PATH, trust_remote_code=True, use_safetensors=False).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(ARTICLE_ENCODER_PATH, trust_remote_code=True)

        with open("/data/kmxu/TrialGPT/dataset/sigir/processed_ovarian_cancer_data.json", "r") as f:
            corpus = json.load(f)
            print("Encoding the corpus")
            for idx, entry in enumerate(tqdm.tqdm(corpus)):
                corpus_ids.append(str(idx))
                text = entry["文本"]

                with torch.no_grad():
                    encoded = tokenizer(
                        [["", text]], # Empty title since we only have text
                        truncation=True,
                        padding=True,
                        return_tensors='pt',
                        max_length=512,
                    ).to("cuda")
                    
                    embed = model(**encoded).last_hidden_state[:, 0, :]
                    embeds.append(embed[0].cpu().numpy())

        embeds = np.array(embeds)
        np.save(corpus_path, embeds)
        with open(ids_path, "w") as f:
            json.dump(corpus_ids, f, indent=4)

    index = faiss.IndexFlatIP(768)
    index.add(embeds)
    
    return index, corpus_ids

if __name__ == "__main__":
    # query type 
    q_type = sys.argv[1]
    
    # different k for fusion
    k = int(sys.argv[2])

    # bm25 weight
    bm25_wt = int(sys.argv[3])

    # medcpt weight 
    medcpt_wt = int(sys.argv[4])

    # how many to rank
    N = 2000

    # loading the queries
    id2queries = json.load(open("/data/kmxu/TrialGPT/results/retrieval_keywords_deepseek/deepseek-r1/community_sigir.json"))

    # loading the indices
    bm25, bm25_ids = get_bm25_corpus_index()
    medcpt, medcpt_ids = get_medcpt_corpus_index()

    # loading the query encoder for MedCPT
    model = AutoModel.from_pretrained(QUERY_ENCODER_PATH).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(QUERY_ENCODER_PATH)
    
    output_path = f"results/qid2ids_results_{q_type}_ovarian_k{k}_bm25wt{bm25_wt}_medcptwt{medcpt_wt}_N{N}.json"
    
    qid2ids = {}

    with open("dataset/sigir/queries.jsonl", "r") as f:
        for line in tqdm.tqdm(f.readlines()):
            entry = json.loads(line)
            qid = entry["_id"]

            if q_type in ["raw", "human_summary"]:
                conditions = [id2queries[qid]["summary"]]
            elif "deepseek" in q_type:
                conditions = id2queries[qid]["conditions"]
            elif "Clinician" in q_type:
                conditions = id2queries[qid].get("conditions", [])

            if len(conditions) == 0:
                id2score = {}
            else:
                # a list of id lists for the bm25 retriever
                bm25_condition_top_ids = []

                for condition in conditions:
                    tokens = word_tokenize(condition.lower())
                    top_ids = bm25.get_top_n(tokens, bm25_ids, n=N)
                    bm25_condition_top_ids.append(top_ids)
                
                # doing MedCPT retrieval
                with torch.no_grad():
                    encoded = tokenizer(
                        conditions,
                        truncation=True,
                        padding=True,
                        return_tensors='pt',
                        max_length=256,
                    ).to("cuda")

                    embeds = model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()
                    scores, inds = medcpt.search(embeds, k=N)

                medcpt_condition_top_ids = []
                for ind_list in inds:
                    top_ids = [medcpt_ids[ind] for ind in ind_list]
                    medcpt_condition_top_ids.append(top_ids)

                id2score = {}

                for condition_idx, (bm25_top_ids, medcpt_top_ids) in enumerate(zip(bm25_condition_top_ids, medcpt_condition_top_ids)):
                    if bm25_wt > 0:
                        for rank, doc_id in enumerate(bm25_top_ids):
                            if doc_id not in id2score:
                                id2score[doc_id] = 0
                            id2score[doc_id] += (1 / (rank + k)) * (1 / (condition_idx + 1))
                    
                    if medcpt_wt > 0:
                        for rank, doc_id in enumerate(medcpt_top_ids):
                            if doc_id not in id2score:
                                id2score[doc_id] = 0
                            id2score[doc_id] += (1 / (rank + k)) * (1 / (condition_idx + 1))

            id2score = sorted(id2score.items(), key=lambda x: -x[1])
            top_ids = [doc_id for doc_id, _ in id2score[:N]]
            qid2ids[qid] = top_ids

    with open(output_path, "w") as f:
        json.dump(qid2ids, f, indent=4)