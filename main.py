import torch
import torch.nn as nn
import jieba
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pickle
# device ="cuda"
batch_size = 64
sequence_length =20
punc=r'~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}▶'
with open('word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')
pad_id=word2idx["<PAD>"]
def stopwordslist(filepath):
    stop_words = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stop_words
stopwords=stopwordslist('chinesestopwords.txt')
def re_stops(sentences):#去除停用词
    tokens =list(jieba.lcut(sentences))
    filtered_tokens = [token for token in tokens if token not in stopwords and token not in punc]
    return filtered_tokens
def tokenizer(data):
    tokens=re_stops(data)
    temp = [word2idx.get(j, pad_id) for j in tokens]
    if (len(temp) < sequence_length):
            # 应该padding。
        for _ in range(sequence_length - len(temp)):
            temp.append(pad_id)
    else:
        temp = temp[:sequence_length]
    # inputs.append(temp)
    temp=torch.tensor([temp])
    return temp
class RNNCNNClassifier(nn.Module):
    def __init__(self, embedding_dim=200, hidden_dim = 128, num_classes = 2, num_filters = 100, filter_sizes = [3, 4, 5]):
        super(RNNCNNClassifier, self).__init__()
        self.embedding = nn.Embedding(len(word2idx), embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_filters, kernel_size=fs) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters + hidden_dim, num_classes)
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_output, _ = self.rnn(embedded)
        rnn_features = rnn_output[:, -1, :]
        cnn_features = []
        x = rnn_output.permute(0, 2, 1)
        for conv_layer in self.conv_layers:
            conv_output = conv_layer(x)
            pooled_output = torch.max(conv_output, dim=2)[0]
            cnn_features.append(pooled_output)

        cnn_features = torch.cat(cnn_features, dim=1)
        features = torch.cat([rnn_features, cnn_features], dim=1)

        logits = self.fc(features)
        return logits
model = RNNCNNClassifier()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
def answer(text):
    test = tokenizer(text)
    with torch.no_grad():
        output=model(test)
        predicted_labels = torch.argmax(output, dim=1)
        return predicted_labels
        print(f"Predicted Label: {predicted_labels.item()}")
        print(type((predicted_labels.item())))