import re
import math
import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


text = (
    'Hello, how are you? I am Romeo.\n' # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n' # J
    'Nice meet you too. How are you today?\n' # R
    'Great. My baseball team won the competition.\n' # J
    'Oh Congratulations, Juliet\n' # R
    'Thank you Romeo\n' # J
    'Where are you going today?\n' # R
    'I am going shopping. What about you?\n' # J
    'I am going to visit my grandmother. she is not very well' # R
)
sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n') # filter '.', ',', '?', '!'
# print(sentences)
word_list = list(set(' '.join(sentences).split()))
word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2,'[MASK]': 3} # word2idx
for i, w in enumerate(word_list):
    word2idx[w] = i + 4
idx2word = {i:w for w, i in word2idx.items()}
vocab_size = len(word2idx)

token_list = list()
for sentence in sentences:
    arr = [word2idx[s] for s in sentence.split()] # 将每个句子转换为token_id进行存储
    token_list.append(arr)

# BERT parameters
max_len = 30 # 表示同一个batch里面所有句子都由30个token组成，不够的补PAD
batch_size = 6 # 批次数
max_pred = 5 # max tokens of prediction
n_layers = 6 # encoder layer number
n_heads = 12 # 多头数目
d_model = 768 # 嵌入维度
d_ff = 768 * 4 # feedforward层扩大的中间维度数
d_k = d_v = 64 # dimension K(=Q), V
n_segments = 2

def make_data():
    batch = []
    positive = negative = 0 # 控制NSP的标签数据，一样一半
    while positive != batch_size/2 or negative != batch_size/2: # sample random index in sentences
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences))
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM n_pred(代表需要mask的数量)
        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15))) # 15% of tokens in one sentence
        # 候选的掩码位置索引
        cand_maked_pos = [i for i, token in enumerate(input_ids) if token != word2idx['[CLS]'] and token != word2idx['[SEP]']]
        shuffle(cand_maked_pos) # 随机打乱索引
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos) # 需要掩码的位置
            masked_tokens.append(input_ids[pos]) # 掩码的词元id
            if random() < 0.8:
                input_ids[pos] = word2idx['[MASK]'] # make mask
            elif random() > 0.9:# 另外10%不做处理
                index = randint(0, vocab_size - 1) # random index in vocabulary
                while index < 4: # 防止掩码填充[PAD]\[CLS]\[SEP]\[MASK]等符号
                    index = randint(0, vocab_size - 1)
                input_ids[pos] = index # replace 随机选择一个token进行替换
        # Zero Paddings
        n_pad = max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad) # 这里填充0，是否合理？

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # Is Next
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])
            # IsNotNext
            negative += 1
    return batch

# 预处理完成
batch = make_data()
# print(batch[0][0])
# print(batch[0][1])
# print(batch[0][2])
# print(batch[0][3])
input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)
input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
    torch.LongTensor(input_ids), torch.LongTensor(segment_ids),\
    torch.LongTensor(masked_tokens),\
    torch.LongTensor(masked_pos), torch.LongTensor(isNext)

class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos, isNext):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.masked_tokens = masked_tokens
        self.isNext = isNext
        self.masked_pos = masked_pos

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.segment_ids[idx],\
                self.masked_tokens[idx], self.masked_pos[idx], self.isNext[idx]

# 数据加载器
dataloader = Data.DataLoader(MyDataSet(input_ids, segment_ids, masked_tokens,masked_pos, isNext), batch_size, True)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token.
    # .eq(0)会逐元素检查seq_q中元素为0的值，并返回相同形状的数组,0为True,1为False
    pad_attn_mask = seq_q.eq(0).unsqueeze(1) # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]

def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Embedding(nn.Module): # 实现token_id -> 词元embedding
    def __init__(self):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model) # token embedding
        self.pos_embed = nn.Embedding(max_len, d_model) # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model) # segment(token type)embedding
        self.norm = nn.LayerNorm(d_model) # 在每个维度上做层归一化

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x) # [seq_len] -> [batch_size, seq_len]
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q,K.transpose(-1, -2)) / np.sqrt(d_k) # 嵌入维度
        scores.masked_fill_(attn_mask, -1e9) # = 1 的位置填充 无穷大负数
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module): # 多头注意力机制实现
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
    def forward(self, Q, K, V, attn_mask):
        # Q、K、V:[batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1,n_heads,1,1)# attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context =  context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)  # context: [batch_size, seq_len, n_heads * d_v]
        output = nn.Linear(n_heads * d_v, d_model)(context)
        # 先进行残差连接，再进行layernorm
        return nn.LayerNorm(d_model)(output + residual) # output:[batch_size, seq_len, d_model]

class PowiseFeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, X):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        # -> (batch_size, seq_len, d_model
        return self.fc2(gelu(self.fc1(X)))

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PowiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # enc_inputs to same Q,K,V
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # enc_outputs: [batch_size, seq_len, d_model]
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs

class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding() # 嵌入表示处理
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(d_model, 2)
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids) # [batch_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids) # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output = layer(output, enc_self_attn_mask)
        # it will be decided by first token(CLS)
        h_pooled = self.fc(output[:, 0]) # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2] predict isNext

        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model) # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size, max_pred, d_model] (max_pred < max_len)故可以做采样
        h_masked = self.activ2(self.linear(h_masked))  # [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked)  # [batch_size, max_pred, vocab_size]
        return logits_lm, logits_clsf

model = BERT()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.0001)


if __name__ == '__main__':
    # train
    print('start training...')
    for epoch in range(100):
        for input_ids, segment_ids, masked_tokens, masked_pos, isNext in dataloader:
            logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
            loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1))
            loss_clfs = criterion(logits_clsf, isNext)
            loss = loss_lm + loss_clfs # 总的loss
            if (epoch + 1) % 10 == 0:
                print('Epoch:','%04d'% (epoch + 1),'loss:','{:.4f}'.format(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # 执行梯度更新
    print('training done!')

    # test
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[2]
    print(text)
    print([idx2word[w] for w in input_ids if idx2word[w] != '[PAD]'])
    logits_lm, logits_clsf = model(torch.LongTensor([input_ids]), torch.LongTensor([segment_ids]),\
                                   torch.LongTensor([masked_pos]))
    # print(logits_lm.shape) # (1, 5, 40)
    # print(logits_clsf.shape) # (1, 2)
    logits_lm = logits_lm.max(dim=2)[1][0].numpy()
    print('masked tokens list : ',[pos for pos in masked_tokens if pos != 0])
    print('predict masked tokens list : ', [pos for pos in logits_lm if pos != 0])
    print('预测文字对比:')
    print('原文:',end='')
    pos = 0
    str_list = ''
    for i in input_ids:
        if i != 0 and i != 3:
            str_list += idx2word[i] + ' '
        elif i == 3 and masked_tokens[pos] != 0:
            str_list += idx2word[masked_tokens[pos]] + ' '
            pos += 1
    print(str_list)
    print('predict:',end='')
    str_list = ''
    pos = 0
    for i in input_ids:
        if i != 0 and i != 3:
            str_list += idx2word[i] + ' '
        elif i == 3 and masked_tokens[pos] != 0:
            str_list += idx2word[logits_lm[pos]] + ' '
            pos += 1
    print(str_list)

    logits_clsf = logits_clsf.max(1)[1].numpy()
    print('isNext:',True if isNext else False)
    print('predict isNext:',True if logits_clsf else False)

