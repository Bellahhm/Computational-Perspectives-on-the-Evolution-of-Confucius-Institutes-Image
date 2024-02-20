import pandas as pd
from top2vec import Top2Vec
import umap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from tqdm import tqdm

# Load the pre-trained Top2Vec model from file
model = Top2Vec.load("D:/pythonProject1/数据尝试/狗屎/新建文件夹/top2vector_model_xin.pkl")

# 获取主题的大小
topic_sizes, _ = model.get_topic_sizes()

# 获取主题的数量
num_topics = model.get_num_topics()
print(f"Total number of topics: {num_topics}")

# 初始化主题进度条
with tqdm(total=num_topics, desc="Extracting Topics") as topic_pbar:
    # 打印每个主题的大小
    for topic_id, size in enumerate(topic_sizes):
        print(f"Topic {topic_id}: Size - {size}")
        topic_pbar.update(1)

# Assume you have a list of document IDs (doc_ids)
doc_ids = model.document_ids

# Get the document-topic distribution
document_topics = model.get_documents_topics(doc_ids)

# Extract the topic information for each document (assuming it's in the first position of the tuple)
document_topic_labels = document_topics[0]

# Set UMAP visualization parameters
umap_args_for_plot = {
    "n_neighbors": 8,
    "n_components": 2,
    "metric": "cosine",
    'min_dist': 0.7,
    'spread': 1
}

# Use UMAP for dimensionality reduction of document vectors
umap_plot_mapper = umap.UMAP(**umap_args_for_plot).fit(model.document_vectors)

# Define 20 distinct colors for topics 0-19
distinct_colors = [
    '#FF0000', '#FF6600', '#FFCC00', '#33FF00', '#FFFF14',
    '#580F41', '#3300FF', '#9900FF', '#0066FF','#01153E', 
    '#FFC0CB', '#008000', '#00FFCC'
]

# Append gray color to the list
distinct_colors.append('#E6E6FA')

# Create a custom colormap
cmap = ListedColormap(distinct_colors)  # The last color is gray
norm = Normalize(vmin=0, vmax=13)
# Set colors to include only the first 20 topics
colors = np.where((document_topic_labels >= 0) & (document_topic_labels <= 12), document_topic_labels, 13)

# Plot the scatter plot with different colors for different topics
scatter = plt.scatter(
    umap_plot_mapper.embedding_[:, 0],
    umap_plot_mapper.embedding_[:, 1],
    c=colors,
    cmap=cmap,
    norm=norm,
    marker='o',
    s=0.5,
    alpha=1,
    linewidths=0.5,
)

# Add a colorbar to indicate different topics
plt.colorbar(scatter, label='Topic')
plt.savefig("D:/pythonProject1/数据尝试/狗屎/umap_visualization test.pdf", dpi=300, bbox_inches='tight')


plt.show()
