# %%
# code by Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data

'''
本质上是使用卷积池化层来提取文本的特征信息
然后再做flatten打平操作，最后接上分类头，预测分类结果
'''
# class TextCNN(nn.Module): # 原作者实现版本，不太合理，尤其是在卷积层部分
#     def __init__(self):
#         super(TextCNN, self).__init__()
#         self.num_filters_total = num_filters * len(filter_sizes)
#         self.W = nn.Embedding(vocab_size, embedding_size)
#         self.Weight = nn.Linear(self.num_filters_total, num_classes, bias=False)
#         self.Bias = nn.Parameter(torch.ones([num_classes]))
#         # (size, embedding_size) 表示窗口大小，在文本中也就是同时捕获相邻两个词的相关性
#         self.filter_list = nn.ModuleList([nn.Conv2d(1, num_filters, (size, embedding_size)) for size in filter_sizes])
#
#     def forward(self, X):
#         embedded_chars = self.W(X) # [batch_size, sequence_length, Embedding_size]
#         embedded_chars = embedded_chars.unsqueeze(1) # (6,1,3,2) # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
#
#         pooled_outputs = []
#         for i, conv in enumerate(self.filter_list):
#             # conv : [input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]
#             h = F.relu(conv(embedded_chars)) # (6,3,2,1)
#             # mp : ((filter_height, filter_width))
#             mp = nn.MaxPool2d((2, 1))
#             # pooled : [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3)]
#             pooled = mp(h).permute(0, 3, 2, 1) # (6,1,1,3)
#             pooled_outputs.append(pooled)
#
#         h_pool = torch.cat(pooled_outputs, len(filter_sizes)) # [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3) * 3]
#         h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total]) # [batch_size(=6), output_height * output_width * (output_channel * 3)]
#         model = self.Weight(h_pool_flat) + self.Bias # [batch_size, num_classes]
#         return model

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        output_channels = 3
        self.W = nn.Embedding(vocab_size, embedding_size) # 嵌入层
        self.conv = nn.Sequential(
            nn.Conv2d(1, output_channels, (2, embedding_size)),
            nn.ReLU(),
            nn.MaxPool2d((2,1))
        )
        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, X):
        batch_size = X.shape[0]
        ebd = self.W(X)
        ebd = ebd.unsqueeze(1)
        convd = self.conv(ebd)
        flatten = torch.reshape(convd,(batch_size, -1))
        output = self.fc(flatten)
        return output


def make_data(sentences, labels, word_dict):
    inputs = []
    for sen in sentences:
        inputs.append([word_dict[n] for n in sen.split()])
    targets = []
    for out in labels:
        targets.append(out)
    return inputs, targets

if __name__ == '__main__':
    # 3 words sentences (=sequence_length is 3)
    sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
    labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

    embedding_size = 2 # embedding size
    sequence_length = len(sentences[0]) # sequence length
    num_classes = len(set(labels)) # number of classes
    filter_sizes = [2, 2, 2] # n-gram windows
    num_filters = 3 # number of filters
    batch_size = 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    vocab_size = len(word_dict)

    model = TextCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    inputs, targets = make_data(sentences, labels, word_dict)
    inputs, targets = torch.LongTensor(inputs), torch.LongTensor(targets)
    dataset = Data.TensorDataset(inputs, targets)
    dataloader = Data.DataLoader(dataset,batch_size= batch_size,shuffle= True)

    # Training
    print(f'训练设备:{device},begin Training......')
    for epoch in range(5000):
        for inputs,targets in dataloader:
            optimizer.zero_grad()
            output = model(inputs)

            # output : [batch_size, num_classes], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, targets)
            if (epoch + 1) % 1000 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

            loss.backward()
            optimizer.step()

    # 测试
    test_text = 'i love you'
    tests = np.array([[word_dict[n] for n in test_text.split()]])
    test_batch = torch.LongTensor(tests)

    # Predict
    predict = model(test_batch).data.max(dim=1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_text,"is Bad Mean...")
    else:
        print(test_text,"is Good Mean!!")