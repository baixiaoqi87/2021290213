import torch
import torch.nn as nn
import jieba
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
import os
device ="cuda"
batch_size = 64
sequence_length =20
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
# for text in train_texts:
#     datas.append(*re_stops(text))# 或者你可以使用其他分词方法
def tokenizer(data):
    inputs = []
    for text in data["text"]:
        tokens =re_stops(text)  # 或者你可以使用其他分词方法
        # print(tokens)
        temp = [word2idx.get(j, pad_id) for j in tokens]
        if (len(tokens) < sequence_length):
            # 应该padding。
            for _ in range(sequence_length - len(tokens)):
                temp.append(pad_id)
        else:
            temp = temp[:sequence_length]
        inputs.append(temp)
    # print(inputs)
    return inputs
def collate_fn(batch):
    # print([b[0] for b in batch])
    data = torch.tensor([b[0] for b in batch])
    # print(type(batch[0][1]))
    # print([b[1] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    return [data, labels]

class NewsDataset(Dataset):
    def __init__(self, dataframe):
        datas=pd.read_csv(dataframe+'_data.csv')
        self.texts = tokenizer(datas)
        # print(len(self.texts))
        # print(type(self.texts))
        self.labels = datas['label']
        # print(type(self.labels))
        # print(len(self.labels))
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label
train_loader = DataLoader(NewsDataset("train"), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
test_loader = DataLoader(NewsDataset("test"), batch_size=batch_size, shuffle=False,collate_fn=collate_fn)
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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

import matplotlib.pyplot as plt

train_losses = []
train_accs = []
test_accs = []
from tqdm import tqdm
import copy

best_test_acc = 0
best_model = None

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for texts, labels in tqdm(train_loader):
        texts = texts.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(texts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        epoch_acc += (predicted == labels).sum().item()
    train_losses.append(epoch_loss / len(train_loader))
    train_accs.append(epoch_acc / len(train_df))

    model.eval()
    test_acc = 0
    with torch.no_grad():
        for texts, labels in test_loader:
            texts = torch.LongTensor(texts).to(device)
            labels = torch.LongTensor(labels).to(device)
            logits = model(texts)
            _, predicted = torch.max(logits, 1)
            test_acc += (predicted == labels).sum().item()
    test_accs.append(test_acc / len(test_df))

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, Test Acc: {test_accs[-1]:.4f}")

    if test_accs[-1] > best_test_acc:
        best_test_acc = test_accs[-1]
        best_model = copy.deepcopy(model)

# 保存最好的模型
torch.save(best_model.state_dict(), "best_model.pth")

# 绘制训练过程中的损失和准确率
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accs, label="Train Acc")
plt.plot(range(1, num_epochs + 1), test_accs, label="Test Acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("training_results.png")
plt.show()
# from tqdm import tqdm
# best_accuracy = 0.0  # 跟踪最高准确率
# best_model_state = None  # 保存最高准确率时的模型状态字典
# for epoch in range(num_epochs):
#     model.train()
#     for texts, labels in tqdm(train_loader):
#         texts = texts.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
#         logits = model(texts)
#         loss = criterion(logits, labels)
#         loss.backward()
#         optimizer.step()
#     model.eval()
#     total_correct = 0
#     total_samples = 0
#     with torch.no_grad():
#         for texts, labels in tqdm(test_loader):
#             texts = torch.LongTensor(texts).to(device)
#             labels = torch.LongTensor(labels).to(device)
#             logits = model(texts)
#             _, predicted = torch.max(logits, dim=1)
#             total_correct += (predicted == labels).sum().item()
#             total_samples += labels.size(0)
#
#     accuracy = total_correct / total_samples
#     print(f"Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {accuracy:.4f}")
#     # 更新最高准确率，并保存模型
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_model_state = model.state_dict()
#
# # 保存最高准确率时的模型
# torch.save(best_model_state, 'best2_model.pth')
# print("模型保存成功，准确率为"+str( round(best_accuracy, 3)))
# model.load_state_dict(torch.load('best2_model.pth'))
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for texts, labels in test_loader:
#         texts = torch.LongTensor(texts).to(device)
#         labels = torch.LongTensor(labels).to(device)
#         outputs = model(texts)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# test_accuracy = 100 * correct / total
# print(f'Test Accuracy: {test_accuracy:.2f}%')

