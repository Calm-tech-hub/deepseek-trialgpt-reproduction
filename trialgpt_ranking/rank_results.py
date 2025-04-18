__author__ = "qiao"

"""
Rank the trials given the matching and aggregation results
"""

import json
import sys
import re

eps = 1e-9

# def get_matching_score(matching):
# 	# count only the valid ones
# 	included = 0
# 	not_inc = 0
# 	na_inc = 0
# 	no_info_inc = 0

# 	excluded = 0
# 	not_exc = 0
# 	na_exc = 0
# 	no_info_exc = 0
	
# 	# first count inclusions
# 	for criteria, info in matching["inclusion"].items():
		
# 		if len(info) != 3:
# 			continue

# 		if info[2] == "included":
# 			included += 1	
# 		elif info[2] == "not included":
# 			not_inc += 1
# 		elif info[2] == "not applicable":
# 			na_inc += 1
# 		elif info[2] == "not enough information":
# 			no_info_inc += 1
	
# 	# then count exclusions
# 	for criteria, info in matching["exclusion"].items():

# 		if len(info) != 3:
# 			continue

# 		if info[2] == "excluded":
# 			excluded += 1	
# 		elif info[2] == "not excluded":
# 			not_exc += 1
# 		elif info[2] == "not applicable":
# 			na_exc += 1
# 		elif info[2] == "not enough information":
# 			no_info_exc += 1

# 	# get the matching score
# 	score = 0
	
# 	score += included / (included + not_inc + no_info_inc + eps)
	
# 	if not_inc > 0:
# 		score -= 1
	
# 	if excluded > 0:
# 		score -= 1
	
# 	return score 

def get_matching_score(matching):
    # Initialize counters
    included = not_inc = na_inc = no_info_inc = 0
    excluded = not_exc = na_exc = no_info_exc = 0
    
    # Safely process inclusions
    if "inclusion" in matching:
        for criteria, info in matching["inclusion"].items():
            if len(info) != 3:
                continue
            # ...existing inclusion counting code...
    
    # Safely process exclusions
    if "exclusion" in matching:
        for criteria, info in matching["exclusion"].items():
            if len(info) != 3:
                continue
            # ...existing exclusion counting code...
    
    # Calculate score
    score = 0
    score += included / (included + not_inc + no_info_inc + eps)
    if not_inc > 0:
        score -= 1
    if excluded > 0:
        score -= 1
    
    return score


def get_agg_score(assessment):
	try:
		rel_score = float(assessment["relevance_score_R"])
		eli_score = float(assessment["eligibility_score_E"])
	except:
		rel_score = 0
		eli_score = 0
	
	score = (rel_score + eli_score) / 100

	return score 

def clean_trial_results(trial_results):
    """Enhanced cleaning with better JSON parsing."""
    cleaned = {}
    for key in ["inclusion", "exclusion"]:
        try:
            content = trial_results[key]
            if isinstance(content, str):
                # 提取 JSON 部分
                if "<think>" in content:
                    content = content.split("</think>")[-1].strip()
                if "```json" in content:
                    content = content.split("```json")[-1].split("```")[0].strip()
                # 修复 JSON 格式
                content = fix_json_format(content)
                content = json.loads(content)
            cleaned[key] = content
        except Exception as e:
            print(f"Error processing {key}: {e}")
            continue
    return cleaned

def fix_json_format(json_str):
    """修复常见的 JSON 格式问题"""
    # 处理缺少引号的属性名
    json_str = re.sub(r'(\s*)(\w+)(:)', r'\1"\2"\3', json_str)
    # 处理单引号
    json_str = json_str.replace("'", '"')
    return json_str


if __name__ == "__main__":
    # args are the results paths
    matching_results_path = sys.argv[1]
    agg_results_path = sys.argv[2]
    output_path = "/data/kmxu/TrialGPT/results/ranking_results_sigir_deepseek/deepseek-r1/ranking.json"

    # loading the results
    matching_results = json.load(open(matching_results_path))
    agg_results = json.load(open(agg_results_path))
    
    final_ranking = {}

    # loop over the patients
    for patient_id, label2trial2results in matching_results.items():

        trial2score = {}

        for _, trial2results in label2trial2results.items():
            for trial_id, results in trial2results.items():
                results = clean_trial_results(results)
                matching_score = get_matching_score(results)
                
                if patient_id not in agg_results or trial_id not in agg_results[patient_id]:
                    print(f"Patient {patient_id} Trial {trial_id} not in the aggregation results.")
                    agg_score = 0
                else:
                    agg_score = get_agg_score(agg_results[patient_id][trial_id])

                trial_score = matching_score + agg_score
                
                trial2score[trial_id] = trial_score

        sorted_trial2score = sorted(trial2score.items(), key=lambda x: -x[1])
        
        final_ranking[patient_id] = sorted_trial2score

        print()
        print(f"Patient ID: {patient_id}")
        print("Clinical trial ranking:")
        
        for trial, score in sorted_trial2score:
            print(trial, score)

        print("===")
        print()

    # Save the final ranking to a JSON file
    with open(output_path, "w") as f:
        json.dump(final_ranking, f, indent=4)
