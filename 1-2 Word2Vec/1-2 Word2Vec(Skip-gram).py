import numpy as np
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt


'''
训练word2vec词向量的方法有两种：
理解： 中心词 和 上下文词的意义
1、CBOW 用上下文词来预测中心词
2、skip-gram 用中心词来预测上下文词
'''

def random_batch():
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=False)

    for i in random_index:
        random_inputs.append(np.eye(voc_size)[skip_grams[i][0]])  # target 中心词
        random_labels.append(skip_grams[i][1])  # context word 上下文词

    return random_inputs, random_labels

# Model
class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        # W and WT is not Traspose relationship
        self.W = nn.Linear(voc_size, embedding_size, bias=False) # voc_size > embedding_size Weight
        self.WT = nn.Linear(embedding_size, voc_size, bias=False) # embedding_size > voc_size Weight

    def forward(self, X):
        '''
        仔细思考一下，这样训练是不是就可以实现用W层的输出，
        来embedding表示嵌入层维度大小
        '''
        # X : [batch_size, voc_size]
        hidden_layer = self.W(X) # hidden_layer : [batch_size, embedding_size]
        output_layer = self.WT(hidden_layer) # output_layer : [batch_size, voc_size]
        return output_layer

if __name__ == '__main__':
    batch_size = 2 # mini_batch size
    embedding_size = 2 # embedding size

    sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",\
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]
    word_sequence = " ".join(sentences).split()
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    voc_size = len(word_list) # 词表大小

    # Make skip gram of one size window
    skip_grams = []
    for i in range(1, len(word_sequence) - 1):
        target = word_dict[word_sequence[i]] # 去掉的单词作为target
        context = [word_dict[word_sequence[i-1]], word_dict[word_sequence[i+1]]] # 上下文词
        for w in context:
            skip_grams.append([target, w]) # 将context,target追加到skip_grams

    model = Word2Vec()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(5000):
        input_batch, target_batch = random_batch()

        # print(f'input_batch:{input_batch}')
        # print(f'target_batch:{target_batch}')
        input_batch = torch.Tensor(np.array(input_batch))
        target_batch = torch.LongTensor(target_batch)

        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 ==0:
            print('Epoch:','%04d'%(epoch + 1),'cost = ','{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    for i, label in enumerate(word_list):
        W, WT = model.parameters()
        x, y = W[0][i].item(), W[1][i].item() # 每一列就是单词的Embedding表示
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()