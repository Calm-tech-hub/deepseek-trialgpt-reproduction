import json
import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. 读取数据 ----------------------
# 定义文件路径列表
file_paths = [
    "/data/kmxu/TrialGPT/paper_data/deepseek-v1/trec_2022/community_trec_2022_k5_bm25wt1_medcptwt1_hybrid_metrics.json",
    "//data/kmxu/TrialGPT/paper_data/deepseek-r3-turbo/trec_2022/trec_2022_k5_bm25wt1_medcptwt1_hybrid_metrics.json",
    "/data/kmxu/TrialGPT/results/qid2nctids_results_gpt/trec_2022_k5_bm25wt1_medcptwt1_hybrid_metrics.json" 
]

# 定义标签列表，用于图例显示
labels = ["Deepseek-r1", "Deepseek-v3", "GPT-4"]

# 定义颜色列表，为每条曲线设置不同颜色
colors = ["#AF4C4C", "#4C72B0", "#55A868"]  # 分别对应三条曲线的颜色

# 存储所有数据集
all_y_values = []

# 读取所有数据文件
for file_path in file_paths:
    with open(file_path, "r") as f:
        y_values = json.load(f)
        
        # 定义要读取的行号列表（从0开始计数）
        # 例如：读取第1行(索引0)、第6行(索引5)、第11行(索引10)等
        line_numbers = [i for i in range(0, len(y_values), 1)]  # 从0开始，每隔5个索引
        
        # 根据行号列表读取数据
        filtered_y_values = [y_values[i] for i in line_numbers if i < len(y_values)]
        
        all_y_values.append(filtered_y_values)

# ---------------------- 2. 构建横轴（深度映射） ----------------------
# 使用筛选后的数据长度
num_points = len(all_y_values[0])
x_depth = np.linspace(0, 1000, num_points)  # 深度从0到500，均匀分布num_points个点

# ---------------------- 3. 绘图配置 ----------------------
plt.figure(figsize=(8, 5))  # 调整图形尺寸以容纳多条曲线和图例

# 绘制多条折线
for i, (y_values, label, color) in enumerate(zip(all_y_values, labels, colors)):
    plt.plot(
        x_depth, 
        y_values, 
        color=color,
        linewidth=2,
        linestyle="-",
        label=label  # 添加标签用于图例
    )

# ---------------------- 4. 坐标轴与标签配置 ----------------------
# 横轴：范围、刻度、标签
plt.xlim(0, 1000)
plt.xticks([0, 200, 400, 600, 800, 1000], fontsize=10)
plt.xlabel("Depth", fontsize=12, fontweight="bold")

# plt.xlim(0, 500)
# plt.xticks([0, 100, 200, 300, 400, 500], fontsize=10)
# plt.xlabel("Depth", fontsize=12, fontweight="bold")

# 纵轴：范围、刻度、标签
plt.ylim(0, 1.0)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=10)
plt.ylabel("Corpus = TREC 2022", fontsize=12, fontweight="bold")
# plt.ylabel("Corpus = SIGIR", fontsize=12, fontweight="bold")

# 标题
plt.title("Retriever Comparison", fontsize=14, fontweight="bold")

# 添加图例
plt.legend(fontsize=10, loc="best")  # 自动选择最佳位置放置图例



# ---------------------- 5. 细节优化 ----------------------
# 去除顶部/右侧边框
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 调整布局
plt.tight_layout()

# 保存或显示
plt.savefig("/data/kmxu/TrialGPT/paper_pictures/trec_2022.png", dpi=300, bbox_inches="tight")
