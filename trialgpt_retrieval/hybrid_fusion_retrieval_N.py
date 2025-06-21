import json
import numpy as np
import torch
import tqdm
import sys
import os
from transformers import AutoTokenizer, AutoModel
import faiss
from nltk import word_tokenize
from beir.datasets.data_loader import GenericDataLoader
from rank_bm25 import BM25Okapi

# 全局变量
ARTICLE_ENCODER_PATH = "/data/kmxu/TrialGPT/models/MedCPT-Article-Encoder"
QUERY_ENCODER_PATH = "/data/kmxu/TrialGPT/models/MedCPT-Query-Encoder"

def load_data(corpus):
    """加载所有必要的数据，只执行一次"""
    print("加载数据中...")
    
    # 加载qrels和查询
    _, _, qrels = GenericDataLoader(data_folder=f"dataset/{corpus}/").load(split="test")
    id2queries = json.load(open(f"/data/kmxu/TrialGPT/results/retrieval_keywords_gpt/gpt-4-turbo/{corpus}.json"))
    
    # 加载BM25索引
    bm25, bm25_nctids = get_bm25_corpus_index(corpus)
    
    # 加载MedCPT索引
    medcpt, medcpt_nctids = get_medcpt_corpus_index(corpus)
    
    # 加载查询编码器
    query_model = AutoModel.from_pretrained(QUERY_ENCODER_PATH).to("cuda")
    query_tokenizer = AutoTokenizer.from_pretrained(QUERY_ENCODER_PATH)
    
    # 加载查询数据
    queries = []
    qids = []
    with open(f"dataset/{corpus}/queries.jsonl", "r") as f:
        for line in f.readlines():
            entry = json.loads(line)
            queries.append(entry["text"])
            qids.append(entry["_id"])
    
    print("数据加载完成")
    return {
        "qrels": qrels,
        "id2queries": id2queries,
        "bm25": bm25,
        "bm25_nctids": bm25_nctids,
        "medcpt": medcpt,
        "medcpt_nctids": medcpt_nctids,
        "query_model": query_model,
        "query_tokenizer": query_tokenizer,
        "queries": queries,
        "qids": qids,
        "corpus": corpus
    }

def get_bm25_corpus_index(corpus):
    """获取BM25索引，已有的代码保持不变"""
    corpus_path = os.path.join(f"trialgpt_retrieval/bm25_corpus_{corpus}.json")
    if os.path.exists(corpus_path):
        corpus_data = json.load(open(corpus_path))
        tokenized_corpus = corpus_data["tokenized_corpus"]
        corpus_nctids = corpus_data["corpus_nctids"]
    else:
        tokenized_corpus = []
        corpus_nctids = []
        with open(f"dataset/{corpus}/corpus.jsonl", "r") as f:
            for line in f.readlines():
                entry = json.loads(line)
                corpus_nctids.append(entry["_id"])
                tokens = word_tokenize(entry["title"].lower()) * 3
                for disease in entry["metadata"]["diseases_list"]:
                    tokens += word_tokenize(disease.lower()) * 2
                tokens += word_tokenize(entry["text"].lower())
                tokenized_corpus.append(tokens)
        corpus_data = {
            "tokenized_corpus": tokenized_corpus,
            "corpus_nctids": corpus_nctids,
        }
        with open(corpus_path, "w") as f:
            json.dump(corpus_data, f, indent=4)
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, corpus_nctids

def get_medcpt_corpus_index(corpus):
    """获取MedCPT索引，已有的代码保持不变"""
    corpus_path = f"trialgpt_retrieval/{corpus}_embeds.npy" 
    nctids_path = f"trialgpt_retrieval/{corpus}_nctids.json"
    if os.path.exists(corpus_path):
        embeds = np.load(corpus_path)
        corpus_nctids = json.load(open(nctids_path)) 
    else:
        embeds = []
        corpus_nctids = []
        model = AutoModel.from_pretrained(ARTICLE_ENCODER_PATH, trust_remote_code=True, use_safetensors=False).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(ARTICLE_ENCODER_PATH, trust_remote_code=True)
        with open(f"dataset/{corpus}/corpus.jsonl", "r") as f:
            print("Encoding the corpus")
            for line in tqdm.tqdm(f.readlines()):
                entry = json.loads(line)
                corpus_nctids.append(entry["_id"])
                title = entry["title"]
                text = entry["text"]
                with torch.no_grad():
                    encoded = tokenizer(
                        [[title, text]], 
                        truncation=True, 
                        padding=True, 
                        return_tensors='pt', 
                        max_length=512,
                    ).to("cuda")
                    embed = model(**encoded).last_hidden_state[:, 0, :]
                    embeds.append(embed[0].cpu().numpy())
        embeds = np.array(embeds)
        np.save(corpus_path, embeds)
        with open(nctids_path, "w") as f:
            json.dump(corpus_nctids, f, indent=4)
    index = faiss.IndexFlatIP(768)
    index.add(embeds)
    return index, corpus_nctids

def get_query_embeddings(data, q_type):
    """获取所有查询的嵌入向量，只执行一次"""
    print("计算查询嵌入向量...")
    model = data["query_model"]
    tokenizer = data["query_tokenizer"]
    id2queries = data["id2queries"]
    qids = data["qids"]
    
    query_embeddings = {}
    all_conditions = []
    qid_to_condition_indices = {}
    
    for qid in qids:
        if qid not in data["qrels"]:
            continue
        print(q_type)
        # 获取查询条件
        if q_type in ["raw", "human_summary"]:
            conditions = [id2queries[qid]["summary"]]
        elif "deepseek" in q_type:
            conditions = id2queries[qid]["conditions"]
        elif "gpt" in q_type:
            conditions = id2queries[qid]["conditions"]
        elif "Clinician" in q_type:
            conditions = id2queries[qid].get("conditions", [])
            
        if not conditions:
            continue
            
        start_idx = len(all_conditions)
        all_conditions.extend(conditions)
        end_idx = len(all_conditions)
        qid_to_condition_indices[qid] = (start_idx, end_idx)
    
    # 批量编码所有条件
    if all_conditions:
        with torch.no_grad():
            encoded = tokenizer(
                all_conditions, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=256,
            ).to("cuda")
            embeds = model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()
        
        # 分配给每个查询
        for qid, (start, end) in qid_to_condition_indices.items():
            query_embeddings[qid] = embeds[start:end]
    
    print(f"计算完成，共处理 {len(query_embeddings)} 个查询")
    return query_embeddings

def get_bm25_results(data, q_type, max_N=500):
    """获取BM25的检索结果，只执行一次，计算最大N值的结果"""
    print(f"计算BM25检索结果 (N={max_N})...")
    bm25 = data["bm25"]
    bm25_nctids = data["bm25_nctids"]
    id2queries = data["id2queries"]
    qids = data["qids"]
    
    bm25_results = {}
    
    for qid in qids:
        if qid not in data["qrels"]:
            continue
            
        # 获取查询条件
        if q_type in ["raw", "human_summary"]:
            conditions = [id2queries[qid]["summary"]]
        elif "deepseek" in q_type:
            conditions = id2queries[qid]["conditions"]
        elif "gpt" in q_type:
            conditions = id2queries[qid]["conditions"]
        elif "Clinician" in q_type:
            conditions = id2queries[qid].get("conditions", [])
            
        if not conditions:
            continue
            
        # 对每个条件执行BM25检索
        condition_results = []
        for condition in conditions:
            tokens = word_tokenize(condition.lower())
            top_nctids = bm25.get_top_n(tokens, bm25_nctids, n=max_N)
            condition_results.append(top_nctids)
            
        bm25_results[qid] = condition_results
    
    print("BM25检索完成")
    return bm25_results

def get_medcpt_results(data, query_embeddings, max_N=500):
    """获取MedCPT的检索结果，只执行一次，计算最大N值的结果"""
    print(f"计算MedCPT检索结果 (N={max_N})...")
    medcpt = data["medcpt"]
    medcpt_nctids = data["medcpt_nctids"]
    qids = data["qids"]
    
    medcpt_results = {}
    
    for qid in qids:
        if qid not in data["qrels"] or qid not in query_embeddings:
            continue
            
        # 获取查询嵌入
        embeds = query_embeddings[qid]
        
        # 执行检索
        scores, inds = medcpt.search(embeds, k=max_N)
        
        # 转换索引为NCT ID
        condition_results = []
        for ind_list in inds:
            top_nctids = [medcpt_nctids[ind] for ind in ind_list]
            condition_results.append(top_nctids)
            
        medcpt_results[qid] = condition_results
    
    print("MedCPT检索完成")
    return medcpt_results

def evaluate_n_values(data, q_type, k, bm25_wt, medcpt_wt, min_N=1, max_N=500, step=1):
    """评估不同N值的检索性能，使用预计算的结果"""
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 输出路径
    if bm25_wt == 0:
        base_output_path = f"results/qid2nctids_results_gpt/{data['corpus']}_k{k}_bm25wt{bm25_wt}_medcptwt{medcpt_wt}_medcpt"
    elif medcpt_wt == 0:
        base_output_path = f"results/qid2nctids_results_gpt/{data['corpus']}_k{k}_bm25wt{bm25_wt}_medcptwt{medcpt_wt}_bm25_wt"
    else:
        base_output_path = f"results/qid2nctids_results_gpt/{data['corpus']}_k{k}_bm25wt{bm25_wt}_medcptwt{medcpt_wt}_hybrid"
    
    # 加载数据
    qrels = data["qrels"]
    qids = data["qids"]
    
    # 预计算查询嵌入
    query_embeddings = get_query_embeddings(data, q_type)
    
    # 预计算BM25和MedCPT的检索结果（使用最大N值）
    bm25_results = get_bm25_results(data, q_type, max_N)
    medcpt_results = get_medcpt_results(data, query_embeddings, max_N)
    
    # 记录所有N值的结果
    all_recalls = {}
    
    # 对每个N值进行评估
    print(f"\n开始评估N值从 {min_N} 到 {max_N}，步长为 {step}")
    
    for N in range(min_N, max_N + 1, step):
        print(f"\n评估 N = {N}...")
        
        qid2nctids = {}
        recalls = []
        
        for qid in qids:
            if qid not in qrels or qid not in bm25_results or qid not in medcpt_results:
                continue
                
            truth_sum = sum(qrels[qid].values())
            
            # 获取该查询的条件结果
            bm25_condition_results = bm25_results[qid]
            medcpt_condition_results = medcpt_results[qid]
            
            # 融合分数
            nctid2score = {}
            for condition_idx, (bm25_top_nctids, medcpt_top_nctids) in enumerate(zip(bm25_condition_results, medcpt_condition_results)):
                # 只使用前N个结果
                bm25_top_nctids = bm25_top_nctids[:N]
                medcpt_top_nctids = medcpt_top_nctids[:N]
                
                if bm25_wt > 0:
                    for rank, nctid in enumerate(bm25_top_nctids):
                        if nctid not in nctid2score:
                            nctid2score[nctid] = 0
                        nctid2score[nctid] += (1 / (rank + k)) * (1 / (condition_idx + 1))
                
                if medcpt_wt > 0:
                    for rank, nctid in enumerate(medcpt_top_nctids):
                        if nctid not in nctid2score:
                            nctid2score[nctid] = 0
                        nctid2score[nctid] += (1 / (rank + k)) * (1 / (condition_idx + 1))
            
            # 排序并获取top-N结果
            nctid2score = sorted(nctid2score.items(), key=lambda x: -x[1])
            top_nctids = [nctid for nctid, _ in nctid2score[:N]]
            qid2nctids[qid] = top_nctids
            
            # 计算召回率
            actual_sum = sum([qrels[qid].get(nctid, 0) for nctid in top_nctids])
            recalls.append(actual_sum / truth_sum)
        
        # 计算平均召回率
        average_recall = sum(recalls) / len(recalls)
        all_recalls[N] = average_recall
        
        print(f"N = {N}: 平均召回率 = {average_recall:.4f}")
        
        # # 保存结果（每10个N值保存一次，避免文件过多）
        # if N % 10 == 0 or N == max_N or N == min_N:
        #     output_path = f"{base_output_path}_N{N}.json"
        #     with open(output_path, "w") as f:
        #         json.dump(qid2nctids, f, indent=4)
            
        # 保存指标
        metrics_path = f"{base_output_path}_metrics.json"
        # metrics = {
        #     "N": N,
        #     "average_recall": average_recall,
        #     "parameters": {
        #     "corpus": data["corpus"],
        #     "query_type": q_type,
        #     "k": k,
        #     "bm25_weight": bm25_wt,
        #     "medcpt_weight": medcpt_wt
        #     }
        # }
            
        # 以追加模式写入，每个N值的结果作为单独的JSON对象


        with open(metrics_path, "a") as f:
            if os.path.exists(metrics_path) == 0:
                    f.write("[\n")
            f.write(f"{average_recall}")
            f.write(",\n")
            # json.dump(metrics, f, indent=4)
    
    # 关闭metrics文件
    with open(f"{base_output_path}_metrics.json", "a") as f:
        f.write("\n]")
    
    
    return all_recalls

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("用法: python retrieval_optimized.py <corpus> <q_type> <k> <bm25_wt> <medcpt_wt> [min_N] [max_N] [step]")
        sys.exit(1)
    
    corpus = sys.argv[1]
    q_type = sys.argv[2]
    k = int(sys.argv[3])
    bm25_wt = int(sys.argv[4])
    medcpt_wt = int(sys.argv[5])
    
    # 默认N值范围
    min_N = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    max_N = int(sys.argv[7]) if len(sys.argv) > 7 else 500
    step = int(sys.argv[8]) if len(sys.argv) > 8 else 1
    
    # 加载数据
    data = load_data(corpus)
    
    # 评估不同N值
    recalls = evaluate_n_values(data, q_type, k, bm25_wt, medcpt_wt, min_N, max_N, step)
    
    # 打印结果摘要
    print("\n=== 召回率评估结果摘要 ===")
    print(f"Corpus: {corpus}")
    print(f"Query Type: {q_type}")
    print(f"k: {k}, BM25 Weight: {bm25_wt}, MedCPT Weight: {medcpt_wt}")
    print(f"N范围: {min_N} 到 {max_N}, 步长: {step}")
    
    # 打印几个关键点的召回率
    key_points = [1, 10, 50, 100, 200, 500]
    for n in key_points:
        if n in recalls:
            print(f"Recall@{n}: {recalls[n]:.4f}")
