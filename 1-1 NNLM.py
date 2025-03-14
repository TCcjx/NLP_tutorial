import torch
import torch.nn as nn
import torch.optim as optim

def make_batch():
    input_batch = []
    target_batch = []

    for sentence in sentences:
        word = sentence.split()
        input = [word_dict[n] for n in word[:-1]] # create (1~n-1) as input
        target = word_dict[word[-1]] # create (n) as target,usually call this 'casual language model'

        input_batch.append(input)
        target_batch.append(target)

    return input_batch,target_batch

class NNLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.C = nn.Embedding(n_class, m)
        self.H = nn.Linear(n_step * m, n_hidden, bias=False)
        self.d = nn.Parameter(torch.ones(n_hidden)) # 偏置项
        self.U = nn.Linear(n_hidden, n_class, bias=False)
        self.W = nn.Linear(n_step * m, n_class, bias=False)
        self.b = nn.Parameter(torch.ones(n_class))

    def forward(self, x):
        x = self.C(x) # 嵌入表示 X:[batch_size, sequence_length, m]
        x = x.view(-1, n_step * m) # 进行一个形状变换 (batch_size, sequence_length * m)
        tanh = torch.tanh(self.d + self.H(x)) # (batch_size, n_hidden)
        output = self.b + self.W(x) + self.U(tanh) # (batch_size, n_class)
        return output

if __name__ == '__main__':
    n_step = 2 # number of steps, n-1 in paper, sequence length 每句的长度
    n_hidden = 2 # number of hidden size, h in paper, 隐藏层的维度大小
    m = 2 # 嵌入的维度大小

    sentences = ['I like dog', ' I love coffee', ' I hate milk',' I love you',' he love cat']

    word_list = ''.join(sentences).split()
    word_list = list(set(word_list)) # 去除重复的单词
    word_dict = {w:i for i,w in enumerate(word_list)} # 单词到数字的映射表
    number_dict = {i:w for i,w in enumerate(word_list)} # 数字到单词的映射表
    n_class = len(word_dict) # number of vocabulary 词表的类别数目

    model = NNLM()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),lr = 0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # Training
    for epoch in range(500):
        optimizer.zero_grad() # 梯度清零
        output = model(input_batch)

        loss = loss_func(output, target_batch)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch:','%04d' % (epoch + 1), 'cost =','{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # predict
    predict = model(input_batch).data.max(1, keepdim = True)[1]
    # print('predict:',predict)

    # Test
    print([sen.split()[:2] for sen in sentences],'->',[number_dict[n.item()] for n in predict.squeeze()])