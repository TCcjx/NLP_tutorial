import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

def make_batch():
    input_batch = [np.eye(n_class)[[word2idx[n] for n in sentence[0].split()]]]
    output_batch = [np.eye(n_class)[[word2idx[n] for n in sentence[1].split()]]]
    target_batch = [word2idx[n] for n in sentence[2].split()]
    return torch.FloatTensor(np.array(input_batch)), torch.FloatTensor(np.array(output_batch)), torch.LongTensor(np.array(target_batch))

class AttentionSeq2Seq(nn.Module):
    def __init__(self):
        super(AttentionSeq2Seq, self).__init__()
        self.enc_cell = nn.RNN(n_class, n_hidden)
        self.dec_cell = nn.RNN(n_class, n_hidden)

        # Attention 实现
        self.attn = nn.Linear(n_hidden, n_hidden) # 线性投影变换
        self.out = nn.Linear(n_hidden * 2, n_class)
    def forward(self, enc_inputs, hidden, dec_inputs):
        enc_inputs = enc_inputs.transpose(0, 1) # (5,1,11)
        dec_inputs = dec_inputs.transpose(0, 1) # (5,1,11)

        # 编码器输出和最后一个隐藏态
        # enc_outpus:(5,1,128)
        # enc_hidden:(1,1,128)
        enc_outputs, enc_hidden = self.enc_cell(enc_inputs, hidden)

        trained_attn = [] # 注意力分数总表
        hidden = enc_hidden # (1, 1, 128)
        n_step = len(dec_inputs)
        model = torch.empty([n_step, 1, n_class])

        for i in range(n_step):
            dec_output, hidden = self.dec_cell(dec_inputs[i].unsqueeze(0), hidden)
            attn_weights = self.get_att_weight(dec_output, enc_outputs)  # 计算解码器单个输出 和 编码器全部输入的注意力分数
            trained_attn.append(attn_weights.squeeze().data.numpy())

            context = attn_weights.bmm(enc_outputs.transpose(0, 1)) # 注意力分数 * 编码器输出 = 计算出注意力分数得到的上下文向量
            dec_output = dec_output.squeeze(0) # (1, 128)
            context = context.squeeze(1) # (1, 128)
            model[i] = self.out(torch.cat((dec_output, context), 1))

        return model.transpose(0, 1).squeeze(0), trained_attn # 注意力分数

    def get_att_weight(self, dec_output, enc_outputs):
        n_step = len(enc_outputs)
        attn_scores = torch.zeros(n_step) # 注意力分数表

        for i in range(n_step):  # 计算当前解码器输出 和 每一个编码器输出的注意力相关分数
            attn_scores[i] = self.get_att_score(dec_output, enc_outputs[i])

        return F.softmax(attn_scores,dim=0).view(1, 1, -1)

    def get_att_score(self, dec_output, enc_output):
        score = self.attn(enc_output)
        return torch.dot(dec_output.view(-1), score.view(-1))




if __name__ == '__main__':
    n_step = 5 # 单元数目
    n_hidden = 128 # 嵌入维度大小

    sentence = ['I mochte ein syj P', 'S i want a beer', 'i want a beer E']
    word_list = " ".join(sentence).split()
    word_list = list(set(word_list)) # 去重
    word2idx = {w:i for i,w in enumerate(word_list)}
    idx2word = {i:w for i,w in enumerate(word_list)}
    n_class = len(word2idx)

    # RNN隐藏态 (num_layers * num_direction, batch_size, n_hidden)
    hidden = torch.zeros(1, 1, n_hidden) # hidden: (1, 1, 128)
    # input_batch:(1, 5, 11)
    # output_batch:(1, 5, 11)
    # target_batch: (5, )
    input_batch, output_batch, target_batch = make_batch()

    model = AttentionSeq2Seq()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train
    for epoch in range(2000):
        optimizer.zero_grad()
        output, _ = model(input_batch, hidden, output_batch)

        loss = criterion(output, target_batch.squeeze(0))
        if (epoch + 1) % 400 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # Test
    test_batch = [np.eye(n_class)[[word2idx[n] for n in 'SPPPP']]]
    test_batch = torch.FloatTensor(np.array(test_batch))
    predict, attn = model(input_batch, hidden, test_batch)
    predict = predict.max(1,keepdim=True)[1]
    print(sentence[0],'->',[idx2word[n.item()] for n in predict.squeeze()])

    # show Attention
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels([''] + sentence[0].split(), fontdict={'fontsize': 14})
    ax.set_yticklabels([''] + sentence[2].split(), fontdict={'fontsize': 14})
    plt.show()



