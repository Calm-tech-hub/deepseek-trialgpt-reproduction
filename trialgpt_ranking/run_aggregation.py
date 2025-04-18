__author__ = "qiao"

"""
Using GPT to aggregate the scores by itself.
"""

from beir.datasets.data_loader import GenericDataLoader
import json
from nltk.tokenize import sent_tokenize
import os
import sys
import time

from TrialGPT import trialgpt_aggregation

def clean_trial_results(trial_results):
    """Clean the trial results by extracting only the JSON data."""
    cleaned = {}
    for key in ["inclusion", "exclusion"]:
        if key not in trial_results:
            continue
            
        # Extract JSON content
        content = trial_results[key]
        if isinstance(content, str):
            # Remove thinking process
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()
            # Extract JSON from markdown code block if present
            if "```json" in content:
                content = content.split("```json")[-1].split("```")[0].strip()
            # Parse the JSON string
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for {key}")
                continue
        cleaned[key] = content
    return cleaned

if __name__ == "__main__":
	corpus = sys.argv[1] 
	model = sys.argv[2]

	# the path of the matching results
	matching_results_path = sys.argv[3]
	results = json.load(open(matching_results_path))

	# loading the trial2info dict
	trial2info = json.load(open("dataset/trial_info.json"))
	
	# loading the patient info
	_, queries, _ = GenericDataLoader(data_folder=f"dataset/{corpus}/").load(split="test")
	
	# output file path
	output_path = f"results/aggregation_results_{corpus}_{model}.json"

	if os.path.exists(output_path):
		output = json.load(open(output_path))
	else:
		output = {}

	# patient-level
	for patient_id, info in results.items():
		# get the patient note
		patient = queries[patient_id]
		sents = sent_tokenize(patient)
		sents.append("The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")
		sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
		patient = "\n".join(sents)

		if patient_id not in output:
			output[patient_id] = {}
		
		# label-level, 3 label / patient
		for label, trials in info.items():
				
			# trial-level
			for trial_id, trial_results in trials.items():
				# print(f"processing {patient_id} {trial_id}")
				trial_results = clean_trial_results(trial_results)
				# print(trial_results)
				# already cached results
				if trial_id in output[patient_id]:
					continue

				if type(trial_results) is not dict:
					output[patient_id][trial_id] = "matching result error"

					with open(output_path, "w") as f:
						json.dump(output, f, indent=4)

					continue

				# specific trial information
				trial_info = trial2info[trial_id]	

				# try:
				result = trialgpt_aggregation(patient, trial_results, trial_info, model)
				print(result)
				output[patient_id][trial_id] = result 

				with open(output_path, "w") as f:
					json.dump(output, f, indent=4)

				# except:
				# 	print("fail")
				# 	continue
