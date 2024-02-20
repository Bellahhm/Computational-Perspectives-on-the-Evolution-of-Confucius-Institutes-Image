import pandas as pd
from top2vec import Top2Vec
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt

# 读取Excel文件
df = pd.read_excel('/CI_total_no_empty_Confuciusinstitute.xlsx')  


# # 删除没有full article的行
# df = df.dropna(subset=['content'])

# 初始化主进度条
with tqdm(total=len(df), desc="Processing Documents") as pbar:
    documents = []

    # 处理文档并添加文档进度条
    for index, row in df.iterrows():
        title = str(row['title'])
        article = str(row['content'])
        document = title + '\n' + article
        documents.append(document)
        pbar.update(1)

# Initialize Top2Vec model with umap_args and hdbscan_args as dictionaries
model = Top2Vec(documents, speed="learn", embedding_model='doc2vec', keep_documents=True, workers=8)

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


# 保存模型到指定目录
model.save('/top2vector_model_xin.pkl')

# a = model.search_documents_by_topic(1, reduced=True)
# print(a)
