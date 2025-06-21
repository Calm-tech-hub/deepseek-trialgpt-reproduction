import subprocess

q_type = "deepseek/deepseek-r1/community"
corpus = "trec_2022" # "sigir", "trec_2021", or "trec_2022"
k = 5
q = 1
bm25_wt = 1
medcpt_wt = 0
N = 1
for N in range(65, 1001, 5):
    command = [
        "python", "/data/kmxu/TrialGPT/trialgpt_retrieval/hybrid_fusion_retrieval.py",
        corpus,
        q_type,
        str(k),
        str(bm25_wt), 
        str(medcpt_wt),
        str(N),
    ]
    result = subprocess.run(command)

print(result.stdout)