import torch as th
import numpy as np
import random
import datetime
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader



# 加载数据集
class DateDataset(Dataset):
    def __init__(self, n):
        self.date_cn = [] # 中文日期
        self.date_en = [] # 英文日期
        for _ in range(n):
            year = random.randint(1950, 2050) # 包含a,b上下界
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            date = datetime.date(year, month, day)
            # 格式化日期
            self.date_cn.append(date.strftime("%y-%m-%d")) # eg:25-03-25
            self.date_en.append(date.strftime("%d/%b/%Y")) # eg:25/Mar/2025
        # 创建一个词汇表，包含0-9的数字、'-'、'/'和英文日期中的月份缩写
        self.vocab = set([str(i) for i in range(0, 10)] + ['-','/'] + [i.split('/')[1] for i in self.date_en])
        self.word2index = {v:i for i, v in enumerate(sorted(list(self.vocab)),start=3)}
        self.word2index['<PAD>'] = PAD_token
        self.word2index['<SOS>'] = SOS_token
        self.word2index['<EOS>'] = EOS_token
        self.index2word = {value:key for key,value in self.word2index.items()}
        # 将开始、结束、填充符号添加到词汇表中
        self.vocab.add('<PAD>')
        self.vocab.add('<SOS>')
        self.vocab.add('<EOS>')
        # 初始化输入和目标列表
        self.input, self.target = [],[]
        for cn, en in zip(self.date_cn, self.date_en):
            self.input.append([self.word2index[v] for v in cn])
            self.target.append(
                [self.word2index['<SOS>'],] +
                [self.word2index[v] for v in en[:3]] +
                [self.word2index[en[3:6]]] +
                [self.word2index[v] for v in en[6:]] +
                [self.word2index['<EOS>']]
            )
        self.input, self.target = np.array(self.input), np.array(self.target)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.target[index], len(self.target[index])

    '''
    @property 是一个装饰器，用于将类中的方法定义为属性。
    这意味着你可以像访问类的普通属性一样访问这些方法，而无需显式调用它们。
    '''
    @property
    def num_word(self):
        return len(self.vocab)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding =  nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        output, hidden = self.rnn(x)
        return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(th.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = th.bmm(weights, keys)
        return context, weights


class AttentionDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttentionDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = Attention(hidden_size)
        self.rnn = nn.RNN(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = th.empty(batch_size, 1, dtype=th.long).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []
        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)
            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = th.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = th.cat(attentions, dim=1)
        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_rnn = th.cat((embedded, context), dim=2)
        output, hidden = self.rnn(input_rnn, hidden)
        output = self.out(output)
        return output, hidden, attn_weights

class Model(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.encoder = EncoderRNN(vocab_size , hidden_size)
        self.AttnDecoder = AttentionDecoderRNN(hidden_size, vocab_size)

    def forward(self, x, target_tensor = None):
        encoder_outputs, encoder_hidden = self.encoder(x)
        decoder_outputs, _, _ = self.AttnDecoder(encoder_outputs, encoder_hidden, target_tensor)
        decoder_outputs = decoder_outputs.view(-1, decoder_outputs.size(-1))
        return decoder_outputs

def evaluate(model, x):
    x = th.tensor(np.array([x])) # 转换为 tensor
    model.eval() # 评估模式
    decoder_outputs = model(x)
    _, topi = decoder_outputs.topk(1)
    decoded_id = topi.squeeze()
    decoded_words = []
    for i in decoded_id:
        decoded_words.append(dataset.index2word[i.item()])
    return ''.join(decoded_words)

if __name__ == '__main__':
    PAD_token = 0
    SOS_token = 1
    EOS_token = 2

    n_epochs = 10
    batch_size = 32
    MAX_LENGTH = 11
    hidden_size = 128
    learning_rate = 0.001
    dataset = DateDataset(1000)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    
    model = Model(dataset.num_word, hidden_size)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # print(dataset.num_word)
    for i in range(n_epochs):
        total_loss = 0
        for input_tensor, target_tensor, target_length in dataloader:
            optimizer.zero_grad()
            decoder_outputs = model(input_tensor,target_tensor)
            loss = criterion(decoder_outputs, target_tensor.view(-1).long())
            loss.backward()
            optimizer.step()
            total_loss += loss
        total_loss /= len(dataloader)
        if (i+1) % 10 == 0:
            print(f'epoch: {i+1}, loss:{total_loss}')
    # test
    print('testing...')
    for idx in range(5):
        predict = evaluate(model,dataset[idx][0])
        print(f'input:{dataset.date_cn[idx]}->target:{dataset.date_en[idx]},predict:{predict}',end='')
        if '<SOS>' + dataset.date_en[idx] + '<EOS>' == predict:
            print(' right')
        else:
            print(' wrong')