__author__ = "qiao"

"""
generate the search keywords for each patient
"""

import json
import os
from openai import AzureOpenAI
from openai import OpenAI
import re
import sys

# client = AzureOpenAI(
# 	api_version="2023-09-01-preview",
# 	azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
# 	api_key=os.getenv("OPENAI_API_KEY"),
# )

client = OpenAI(
    base_url="https://api.ppinfra.com/v3/openai",
    api_key="sk_8jOsRemta_Wn35bEZt3ZNEkwrQWtJ-Fw8NvIKVwcNes",
	http_client=None
)

# def get_keyword_generation_messages(note):
# 	system = '你是一个乐于助人的助手，你的任务是帮助为给定的患者描述搜索相关的临床试验。请首先总结患者的主要医疗问题。然后生成最多32个关键条件，用于搜索与该患者相关的临床试验。关键条件列表应按优先级排序。请仅输出一个JSON字典，格式为Dict{{"summary": Str(summary), "conditions": List[Str(condition)]}}。'

# 	prompt =  f"以下是患者描述：\n{note}\n\nJSON输出："

# 	messages = [
# 		{"role": "system", "content": system},
# 		{"role": "user", "content": prompt}
# 	]
	
# 	return messages

def get_keyword_generation_messages(note):
	system = 'You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description. Please first summarize the main medical problems of the patient. Then generate up to 32 key conditions for searching relevant clinical trials for this patient. The key condition list should be ranked by priority. Please output only a JSON dict formatted as Dict{{"summary": Str(summary), "conditions": List[Str(condition)]}}.'

	prompt =  f"Here is the patient description: \n{note}\n\nJSON output:"

	messages = [
		{"role": "system", "content": system},
		{"role": "user", "content": prompt}
	]
	
	return messages


if __name__ == "__main__":
	# the corpus: trec_2021, trec_2022, or sigir
	corpus = sys.argv[1]

	# the model index to use
	model = sys.argv[2]

	outputs = {}
	
	with open(f"dataset/{corpus}/queries.jsonl", "r") as f:
		for line in f.readlines():
			entry = json.loads(line)
			messages = get_keyword_generation_messages(entry["text"])

			response = client.chat.completions.create(
				model=model,
				messages=messages,
				temperature=0,
			)
			# print(f"response: {response}")

			output = response.choices[0].message.content
			# print(output)
			think_index = output.find("<think>")

			if think_index != -1:
				end_think_index = output.find("</think>")
				if end_think_index != -1:
					output = output[:think_index] + output[end_think_index + len("</think>"):]


			# print(f"output: {output}")
			match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
			if match:
				output = match.group(1).strip()
			# print(f"output: {output}")
			
			outputs[entry["_id"]] = json.loads(output)

			with open(f"results/retrieval_keywords_{model}/{corpus}.json", "w") as f:
				# print(f"results/retrieval_keywords_{model}_{corpus}.json")
				json.dump(outputs, f, indent=4)
