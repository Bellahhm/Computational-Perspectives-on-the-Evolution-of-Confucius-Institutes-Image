
from top2vec import Top2Vec
import os
import pandas as pd

# Load the model
model = Top2Vec.load("D:/pythonProject1/数据尝试/狗屎/新建文件夹/top2vector_model_xin.pkl")

# Get similar words
words, word_scores = model.similar_words(keywords=["Confuciusin"], keywords_neg=[], num_words=100)

# Create a DataFrame
df = pd.DataFrame({"Word": words, "Score": word_scores})

# Save to Excel
excel_path = "D:/pythonProject1/数据尝试/狗屎/新建文件夹/similar words to space_CI.xlsx"
df.to_excel(excel_path, index=False)

# Print path to the saved Excel file
print(f"Results saved to: {excel_path}")
