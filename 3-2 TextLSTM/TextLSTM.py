import time

import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as Data

def make_batch(seq_data): # 处理批次数据，将单词进行序列化
    inputs = []
    targets = []
    for sen in seq_data:
        inputs.append([np.eye(n_class)[word2idx[i]] for i in sen[:-1]])
        targets.append(word2idx[sen[-1]])
    return inputs, targets

class TextLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(n_class, hidden_size=n_hidden)
        self.w = nn.Linear(n_hidden, n_class, bias= False)
        self.b = nn.Parameter(torch.ones(n_class))

    def forward(self, x):  # x: [bs, sq, fs]
        input = x.transpose(0, 1)

        hidden_state = torch.zeros(1, len(x), n_hidden).to(device)
        cell_state = torch.zeros(1,len(x), n_hidden).to(device)

        outputs, (hn, cn) = self.lstm(input, (hidden_state, cell_state))
        output = hn.squeeze(0)
        model = self.w(output) + self.b
        return model

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(f'训练设备:{device}')
    start_time = time.time()

    # LSTM 单元数
    n_step = 3
    n_hidden = 128

    char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
    word2idx = {n:i for i, n in enumerate(char_arr)}
    idx2word = {i:n for i, n in enumerate(char_arr)}
    n_class = len(word2idx) # 26个类别

    seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']
    inputs, targets = make_batch(seq_data=seq_data)
    inputs, targets = torch.FloatTensor(np.array(inputs)), torch.LongTensor(np.array(targets))
    dataset = Data.TensorDataset(inputs, targets)
    dataloader = Data.DataLoader(dataset, 3, shuffle=True)

    model = TextLSTM()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= 0.0001)

    for epoch in range(1000):
        for input_batch, target_batch in dataloader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            output = model(input_batch)
            loss = criterion(output, target_batch)
            if (epoch + 1) % 10 == 0:
                print('Epoch:','%04d' % epoch, 'loss:','{:.6f}'.format(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    input = [sen[:-1] for sen in seq_data]
    end_time = time.time()
    print(f'训练时间：{end_time - start_time}')


    inputs = inputs.to(device)
    predict = model(inputs).data.max(1,keepdim=True)[1]
    print(input,'->',[idx2word[n.item()] for n in predict.squeeze()])






