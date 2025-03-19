import torch
from fontTools.misc.iterTools import batched
from sympy.core.random import shuffle
from torch import nn
import numpy as np
import torch.utils.data as Data
import torch.optim as optim
import numpy as np


# 准备数据
def make_data():
    inputs = []
    targets = []

    words = sentence.split()
    for i, word in enumerate(words[:-1]):
        input = [word2idx[n] for n in words[:(i+1)]]
        target = word2idx[words[i+1]]
        input = input + (max_len - len(input)) * [0]
        inputs.append(np.eye(n_class)[input])
        targets.append(target)
    return torch.FloatTensor(np.array(inputs)), torch.LongTensor(np.array(targets))



class BiLSTM(nn.Module):

    def __init__(self):
        super().__init__()
        # 双向LSTM
        self.lstm = nn.LSTM(input_size= n_class , hidden_size= n_hidden, bidirectional=True)
        self.w = nn.Linear(2 * n_hidden, n_class )

    def forward(self, x):
        input = x.transpose(0,1)

        # len(x) = batch_size
        hidden_state = torch.zeros(2, len(x), n_hidden).cuda()
        cell_state = torch.zeros(2, len(x), n_hidden).cuda()

        outputs,(hn,cn) = self.lstm(input, (hidden_state, cell_state))
        # 用hn做分类特征信息
        # hn = hn.reshape(-1 ,2 * n_hidden)
        hn = outputs[-1].reshape(-1, 2 * n_hidden)
        model = self.w(hn)
        return model

'''
用前面的字符串预测后面的单词
都填充到最大长度
'''
if __name__ == '__main__':
    n_hidden = 20
    sentence = (
        'Lorem ipsum dolor sit amet consectetur adipisicing elit '
        'sed do eiusmod tempor incididunt ut labore et dolore magna '
        'aliqua Ut enim ad minim veniam quis nostrud exercitation'
    )


    # 0 作为 空白字符填充符
    word2idx = {w:(i+1) for i, w in enumerate(list(set(sentence.split())))}
    word2idx['<unkownn>'] = 0
    idx2word = {(i+1):w for i, w in enumerate(list(set(sentence.split())))}
    idx2word[0] = '<unkownn>'
    n_class = len(word2idx)
    max_len = len(sentence.split()) # 序列字符串长度

    inputs, targets = make_data()
    dataset = Data.TensorDataset(inputs, targets)
    dataloader = Data.DataLoader(dataset, batch_size= 5, shuffle = True)

    model = BiLSTM()
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= 0.001)
    for epoch in range(2000):
        for input_batch,target_batch in dataloader:
            input_batch, target_batch = input_batch.cuda(), target_batch.cuda()
            optimizer.zero_grad()
            output = model(input_batch)
            loss = criterion(output, target_batch)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 100 == 0:
            print('epoch:','%05d'%(epoch+1),'loss:','{:.4f}'.format(loss.item()))

    inputs = inputs.cuda()
    predict = model(inputs).data.max(1, keepdim=True)[1]
    print(sentence)
    print([idx2word[n.item()] for n in predict.squeeze()])

