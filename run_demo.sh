
#retrieval
python trialgpt_retrieval/keyword_generation.py sigir gpt-4-turbo

python trialgpt_retrieval/hybrid_fusion_retrieval.py sigir deepseek/deepseek-r1/community 20 1 1

#matching
python trialgpt_matching/run_matching.py sigir deepseek/deepseek-r1/community

#ranking
python trialgpt_ranking/run_aggregation.py sigir deepseek/deepseek-r1/community /data/kmxu/TrialGPT/results/matching_results_sigir_deepseek/deepseek-r1/community.json

python3 trialgpt_ranking/rank_results.py /data/kmxu/TrialGPT/results/matching_results_sigir_deepseek/deepseek-r1/community.json /data/kmxu/TrialGPT/results/aggregation_results_sigir_deepseek/deepseek-r1/community.json

