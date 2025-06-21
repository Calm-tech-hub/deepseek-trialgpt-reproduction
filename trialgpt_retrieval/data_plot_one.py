import json
import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. 读取数据 ----------------------
file_path = "/data/kmxu/TrialGPT/paper_data/trec_2022/community_trec_2022_k5_bm25wt1_medcptwt0_bm25_wt_metrics.json"  # 替换为你的 JSON 文件路径
with open(file_path, "r") as f:
    y_values = json.load(f)  # 读取为列表，如 [0.0277..., 0.0680..., ...]

# ---------------------- 2. 构建横轴（深度映射） ----------------------
num_points = len(y_values)  # 共15个数据点
x_depth = np.linspace(0, 1000, num_points)  # 深度从0到500，均匀分布15个点

# ---------------------- 3. 绘图配置 ----------------------
plt.figure(figsize=(6, 4))  # 调整图形尺寸以更好地展示数据

# 绘制折线：颜色参考示例图的绿色，线条平滑
plt.plot(
    x_depth, 
    y_values, 
    color="#AF4C4C",  # 接近示例图的绿色
    linewidth=2,       # 线条粗细
    linestyle="-"      # 实线
)

# ---------------------- 4. 坐标轴与标签配置 ----------------------
# 横轴：范围、刻度、标签
plt.xlim(0, 1000)
plt.xticks([0, 200, 400, 600, 800, 1000], fontsize=9)  # 刻度文字大小
plt.xlabel("Depth", fontsize=10, fontweight="bold")

# 纵轴：范围、刻度、标签
plt.ylim(0, 1.0)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=9)
plt.ylabel("Corpus = TREC 2022", fontsize=10, fontweight="bold")

# 标题
plt.title("Retriever = BM25", fontsize=11, fontweight="bold")

# # 网格（可选，示例图隐约有网格）
# plt.grid(
#     True, 
#     linestyle="--", 
#     color="gray", 
#     alpha=0.3  # 透明度调低，避免干扰
# )

# ---------------------- 5. 细节优化 ----------------------
# 去除顶部/右侧边框（更接近论文图风格）
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 调整布局，避免标签重叠
plt.tight_layout()

# 保存或显示
plt.savefig("/data/kmxu/TrialGPT/paper_pictures/trec_2022/trec_2022_bm25.png", dpi=300, bbox_inches="tight")  # 高清保存
# plt.show()