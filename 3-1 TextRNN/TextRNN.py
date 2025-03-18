import torch
from torch import nn
import numpy as np
import torch.optim as optim
import torch.utils.data as data
from torchgen.packaged.autograd.load_derivatives import load_derivatives


def make_data(sentences):
    inputs = []
    targets = []
    for sen in sentences:
        word = sen.split() # 切成列表
        inputs.append([np.eye(n_class)[word2idx[x]] for x in word[:-1]]) # 转换成one-hot编码
        targets.append(word2idx[word[-1]]) # 最后一个词作为标签target
    return inputs, targets

class TextRNN(nn.Module):

    def  __init__(self):
        super(TextRNN, self).__init__()
        self.rnn = nn.RNN(input_size= n_class, hidden_size= n_hidden)
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, hidden, x): # bs * sq * fs
        x = x.transpose(0, 1)
        output, hidden = self.rnn(x, hidden)
        model = self.fc(hidden[0]) # 直接用最后一个隐藏态作为分类依据
        return model


if __name__ == '__main__':

    sentences = [ "i like dog", "i love coffee", "i hate milk"]
    word_list = ' '.join(sentences).split()
    vocab = list(set(word_list))
    word2idx = {w: i for i, w in enumerate(vocab)} # 词和索引的对应表
    idx2word = {i: w for w,i in word2idx.items()}
    n_class = len(vocab) # 词的类别数

    # TextRNN Parameters
    batch_size = 2
    n_step = 2 # 两个词元
    n_hidden = 5 # 隐藏态大小

    # 数据集准备
    inputs, targets = make_data(sentences)
    inputs = torch.Tensor(inputs)
    targets = torch.LongTensor(targets)
    dataset = data.TensorDataset(inputs, targets)
    dataloader = data.DataLoader(dataset, batch_size,shuffle=True)

    model = TextRNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= 0.001)

    # Training
    for epoch in range(500):
        for x, y in dataloader:
            # x.shape[0]
            hidden = torch.zeros(1, x.shape[0], n_hidden)
            pred = model(hidden, x)
            loss = criterion(pred, y)
            if (epoch + 1) % 100 == 0:
                print('Epoch:','%04d'% (epoch+1), 'loss=', '{:.6f}'.format(loss.item()))
            optimizer.zero_grad() # 梯度清零
            loss.backward() # 反向传播计算梯度
            optimizer.step() # 更新模型参数

    # Test
    input = [sen.split()[:2] for sen in sentences]
    hidden = torch.zeros(1, len(input), n_hidden) # batch_size维度
    predict = model(hidden, inputs).data.max(1, keepdim= True)[1]
    print(input, '->' , [idx2word[n.item()] for n in predict.squeeze()])