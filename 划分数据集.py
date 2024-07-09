import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
data = pd.read_csv('data.csv')
texts = data.iloc[:, 1].values
labels = data.iloc[:, -1].values.astype(int)
# 划分训练集和测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
# 创建训练集和测试集的DataFrames
# def re_stops(sentences):#去除停用词
#     tokens =list(jieba.lcut(sentences))
#     filtered_tokens = [token for token in tokens if token.lower() not in stopwords and token not in punc]
#     return filtered_tokens
# for text in train_texts:
#     datas.append(*re_stops(text))# 或者你可以使用其他分词方法
train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})
# 保存训练集和测试集为CSV文件
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
print("划分数据集成功！")