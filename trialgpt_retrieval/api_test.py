import os
from dotenv import load_dotenv, find_dotenv # 导入 find_dotenv 帮助定位
import openai
from openai import OpenAI
import httpx



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

# ... (后续的 OpenAI 客户端初始化、API 调用和错误处理代码保持不变) ...
# 2. 配置 API 客户端
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=httpx.Timeout(300.0, connect=60.0),
    max_retries=1,
)

# 3. 准备 API 请求的消息体
messages = [
    {"role": "user", "content": "你好，你好，你能做什么？请用中文回答。"}
]

# 4. 发送请求并处理响应
# try:
print("正在尝试调用 OpenAI API...")
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=3500,
    temperature=0,
)

assistant_reply = response.choices[0].message.content
print("模型回复:", assistant_reply)

