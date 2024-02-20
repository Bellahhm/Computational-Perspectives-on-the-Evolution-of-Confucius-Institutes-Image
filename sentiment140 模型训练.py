
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# 读取数据集
data = pd.read_csv(r"\sentiment dataset\Sentiment140 dataset with 1.6 million tweets\training.1600000.processed.noemoticon.csv", 
                    encoding="latin1", header=None, names=['target', 'ids', 'date', 'flag', 'user', 'text'])

# 选择目标列和文本列
data = data[['target', 'text']]

# 将标签映射为二进制类别（0 = negative, 1 = neutral/positive）
data['target'] = data['target'].map({0: 0, 2: 1, 4: 1})

# 划分训练集和测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(data['text'], data['target'], test_size=0.2, random_state=42)

# 使用Tokenizer将文本转换为序列
max_words = 10000  # 设定词汇表大小
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

# 将文本转换为序列
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# 对序列进行填充，使它们具有相同的长度
max_length = 200  # 设定文本长度
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=60, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dropout(0.5),  # Add dropout layer
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
num_epochs = 5
history = model.fit(train_padded, train_labels, epochs=num_epochs, validation_data=(test_padded, test_labels))

# 评估模型
test_loss, test_accuracy = model.evaluate(test_padded, test_labels)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Plot accuracy and loss over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig('accuracy_plot.png', dpi=300)  # 设置 DPI 为 300
plt.show()
