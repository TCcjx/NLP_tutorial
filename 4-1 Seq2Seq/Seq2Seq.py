import numpy as np
import torch
from torch import nn
import torch.utils.data as Data

def make_batch(): # 准备训练数据
    input_batch, output_batch, target_batch = [], [], []

    for seq in seq_data:
        for i in range(2):
            # 如果长度比n_step短，则进行空白填充
            seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))

        input = [num_dic[n] for n in seq[0]]
        output = [num_dic[n] for n in ('S' + seq[1])]
        target = [num_dic[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(n_class)[input])
        output_batch.append(np.eye(n_class)[output])
        target_batch.append(target)

    return torch.FloatTensor(np.array(input_batch)), torch.FloatTensor(np.array(output_batch)), torch.LongTensor(np.array(target_batch))

# Seq2Seq Model
class Seq2Seq(nn.Module):
    '''
    Seq2Seq Model
    优点：输入和输出都可以是不定长的序列信息
    问题：那为什么前面的make_batch()函数又对输入序列进行填充到相同维度了呢？
    答：因为要批量成一个np矩阵，所以维度大小需要填充到一致，如果过长的话，需要做截断处理
    '''
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.enc_cell = nn.RNN(input_size = n_class, hidden_size = n_hidden)
        self.dec_cell = nn.RNN(input_size = n_class, hidden_size = n_hidden)
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, enc_input, enc_hidden, dec_input):
        enc_input = enc_input.transpose(0, 1)
        dec_input = dec_input.transpose(0, 1)
        _, enc_states = self.enc_cell(enc_input, enc_hidden)
        outputs, _ = self.dec_cell(dec_input, enc_states)

        model = self.fc(outputs) # 分类头
        return model

def make_testbatch(input_word): # 准备测试数据

    input_w = input_word + 'P' * (n_step - len(input_word))
    input = [num_dic[n] for n in input_w]
    output = [num_dic[n] for n in 'S' + 'P' * n_step]

    input_batch = np.eye(n_class)[input] # (batch_size, sequence_length, embedding_size)
    output_batch = np.eye(n_class)[output] # (batch_size, sequence_length+1 , embedding_size)

    return torch.FloatTensor(input_batch).unsqueeze(0), torch.FloatTensor(output_batch).unsqueeze(0)

class TranslateDataset(Data.Dataset):

    def __init__(self, input_all, output_all, target_all):
        self.input_all = input_all
        self.output_all = output_all
        self.target_all = target_all

    def __getitem__(self, index):
        return self.input_all[index], self.output_all[index], self.target_all[index]

    def __len__(self):
        return len(self.input_all)


if __name__ == '__main__':
    n_hidden = 128

    '''
    S:解码开始符号
    E:解码结束符号
    P:空白填充符号
    '''
    char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
    num_dic = {n: i for i, n in enumerate(char_arr)}
    seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low'],['cjx', 'syj'],['wqlt', 'czr']]

    n_step = max([max(len(i), len(j)) for i, j in seq_data])
    n_class = len(num_dic)
    batch_size = 3


    input_batch, output_batch, target_batch = make_batch()
    dataset = TranslateDataset(input_batch, output_batch, target_batch)
    dataloader = Data.DataLoader(dataset, batch_size, shuffle= True)
    model = Seq2Seq()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)

    for epoch in range(1000):
        for input_batch, output_batch, target_batch in dataloader:
            hidden_0 = torch.zeros((1, len(input_batch), n_hidden)) # RNN 隐藏态
            optimizer.zero_grad()
            output = model(input_batch, hidden_0, output_batch)
            # output : [max_len + 1(=6), batch_size(=1), n_class)
            output = output.transpose(0, 1)
            loss = 0
            target_batch = target_batch.long()
            for i in range(0, len(target_batch)):
                loss += criterion(output[i], target_batch[i]) # one-hot编码计算损失
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            print('Epoch','%05d'%(epoch+1),'loss:','{:.5f}'.format(loss.item()))



    # Test
    def translate(word):
        input_batch, output_batch = make_testbatch(word)

        hidden = torch.zeros(1, 1, n_hidden) # 每次输入一个批次的数据
        output = model(input_batch, hidden , output_batch)


        predict = output.max(2, keepdim=True)[1]
        print(predict.shape)
        decoded = [char_arr[i] for i in predict]
        end = decoded.index('E')
        translated = ''.join(decoded[:end])

        return translated.replace('P','')
        # return translated

    print('test')
    print('man ->', translate('man'))
    print('mans ->', translate('mans'))
    print('king ->', translate('king'))
    print('black ->', translate('black'))
    print('girl ->',translate('girl'))
    print('up ->', translate('up'))
    print('high ->',translate('high'))
    print('cjx ->',translate('cjx'))
    print('wqlt ->',translate('wqlt'))