import json
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv # 导入 find_dotenv 帮助定位
import re
import sys
import tiktoken

dotenv_path_found = find_dotenv() # 如果CWD没有，则按标准方式查找（从脚本目录向上）

print(f"DEBUG: 找到 .env 文件路径: {dotenv_path_found}")
loaded_successfully = load_dotenv(dotenv_path=dotenv_path_found, verbose=True, override=True)

# 1. 从环境变量加载 API 密钥和基础 URL
api_key = os.getenv("OPENAI_API_KEY")
base_url_from_env = os.getenv("OPENAI_BASE_URL")

default_base_url = "OPENAI_BASE_URL" # 您常用的 URL 作为默认值
base_url = base_url_from_env if base_url_from_env else default_base_url
print(f"使用的 Base URL: {base_url}")

if base_url == default_base_url and not base_url_from_env :
    print(f"(提示: OPENAI_BASE_URL 未在 .env 文件或环境变量中指定, 当前使用的是代码中的默认值 '{default_base_url}'。)")

client = OpenAI(
	# api_version="2023-09-01-preview",
	base_url=base_url,
	api_key=api_key,
)


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
	encoding = tiktoken.encoding_for_model(model)
	with open(f"dataset/{corpus}/queries.jsonl", "r") as f:
		for line in f.readlines():
			entry = json.loads(line)
			messages = get_keyword_generation_messages(entry["text"])
			
			input_tokens = sum([len(encoding.encode(message["content"])) for message in messages])



			response = client.chat.completions.create(
				model=model,
				messages=messages,
				temperature=0
			)

			output = response.choices[0].message.content

			# match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
			# if match:
			# 	output = match.group(1).strip()
			print(output)
			if output == None:
				continue
			output = output.strip("`").strip("json")
   
			
			outputs[entry["_id"]] = json.loads(output)

			with open(f"results/retrieval_keywords_gpt/{model}/{corpus}.json", "w") as f:
				json.dump(outputs, f, indent=4)