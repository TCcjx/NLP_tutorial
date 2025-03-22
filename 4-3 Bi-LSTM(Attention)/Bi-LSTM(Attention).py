import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


'''
任务: 句子语义二分类任务
Bi-LSTM:实际上是计算final_hidden_state 与 outputs之间的注意力分数，
再计算最终的注意力输出结果作为分类的依据，最后接上线性分类层实现句义二分类任务
'''
class BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional= True)
        self.out = nn.Linear(n_hidden * 2, num_classes)

    def attention_net(self, lstm_out, final_state):
        hidden = final_state.view(-1, n_hidden * 2, 1) #  # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights =  torch.bmm(lstm_out, hidden).squeeze(2) # (batch_size, seq_len)
        soft_attn_weights = F.softmax(attn_weights, 1) # (batch_size, seq_len)

        # (batch_size, n_hidden * 2, seq_len) * (batch_size, seq_len, 1) 计算出来注意力分数后的结果
        context = torch.bmm(lstm_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy() # context:(batch_size, n_hidden * 2)  soft_attn_weights:(bath_size, seq_len)

    def forward(self, x):
        input = self.embedding(x) # (batch_size, sequence_length, embedding_size)
        input = input.permute(1, 0, 2) # input维度交换 (sequence_length, batch_size, embedding_size)

        hidden_state = torch.zeros(1*2, len(x), n_hidden) # n_hidden 隐藏态
        cell_state = torch.zeros(1*2, len(x), n_hidden)

        output, (final_hidden_state, final_cell_state) = self.lstm(input,(hidden_state, cell_state))
        output = output.permute(1, 0, 2) # output(batch_size, seq_len, n_hidden * 2)
        attn_output, attention = self.attention_net(output, final_hidden_state) # lstm输出 与 最后一个时刻的隐藏态 计算注意力输出
        return self.out(attn_output), attention # attention 注意力分数表


# 句子语义分类任务
if __name__ == '__main__':
    embedding_dim  = 2 # embedding size
    n_hidden = 5 # number of hidden units in one cell
    num_classes = 2 # 0 or 1

    sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
    labels = [1, 1, 1, 0, 0, 0]

    word_list = ' '.join(sentences).split()
    word_list = list(set(word_list)) # 词表
    word2idx = {w: i for i, w in enumerate(word_list)} # 词到数字的索引表
    vocab_size = len(word2idx)

    model = BiLSTM_Attention()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    inputs = torch.LongTensor(np.array([[word2idx[n] for n in sen.split()] for sen in sentences]))
    targets = torch.LongTensor(np.array([out for out in labels]))

    # Training
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        output, attention = model(inputs)
        loss = criterion(output, targets)
        if (epoch + 1) % 200 == 0 :
            print('Epoch:','%05d'%(epoch + 1), 'loss:','{:.5f}'.format(loss.item()))
        loss.backward()
        optimizer.step() # 梯度更新

    # Test
    test_text = 'sorry hate you'
    tests = np.array([[word2idx[n] for n in test_text.split()]])
    test_batch = torch.LongTensor(tests)

    # predict
    predict, _ = model(test_batch)
    predict = predict.max(1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_text,'is bad mean')
    else:
        print(test_text,'is good mean')

    fig = plt.figure(figsize=(6, 3))  # [batch_size, n_step]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    ax.set_xticklabels([''] + ['first_word', 'second_word', 'third_word'], fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels([''] + ['batch_1', 'batch_2', 'batch_3', 'batch_4', 'batch_5', 'batch_6'],
                       fontdict={'fontsize': 14})
    plt.show()

