import pandas as pd
import jieba
import pickle
data = pd.read_csv('data.csv')
texts = data.iloc[:, 1].values
# print(texts)
punc=r'~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
def stopwordslist(filepath):
    stop_words = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stop_words
stopwords=stopwordslist('chinesestopwords.txt')
# 划分训练集和测试集
vocab = set()
def re_stops(sentences):#去除停用词
    tokens =list(jieba.lcut(sentences))
    filtered_tokens = [token for token in tokens if token not in stopwords and token not in punc]
    return filtered_tokens
# print(re_stops("中国反腐风刮到阿根廷，这个美到让人瘫痪的女总统，因为8个本子摊上大事了"))
for text in texts:
    # print(len(re_stops(text)))
    for text in re_stops(text):# 或者你可以使用其他分词方法
        vocab.add(text)
word2idx = {word: idx for idx, word in enumerate(vocab)}
word2idx["<PAD>"]=18707
word2idx["<UNK>"]=18708
# print(word2idx)
# # 将word2idx保存为.pkl文件
with open('word2idx.pkl', 'wb') as f:
    pickle.dump(word2idx, f)
    print("字典索引保存成功")#长度为18708

