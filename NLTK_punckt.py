import nltk
import os

# 指定具体的 NLTK 数据路径
nltk_data_dir = "/data/kmxu/TrialGPT/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# 下载必需的 NLTK 数据
print(f"正在下载 NLTK punkt 分词器到 {nltk_data_dir}...")
nltk.download('punkt', download_dir=nltk_data_dir)
print("下载完成！")

# 验证下载
try:
    nltk.data.find('tokenizers/punkt')
    print("验证成功：punkt 分词器可以正常使用")
except LookupError:
    print("错误：punkt 分词器未能正确安装")