import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
 
dtype = torch.FloatTensor
 
# 数据预处理
sentences = ['i like dog', 'i love coffee', 'i hate milk']
word_list = " ".join(sentences).split()  # ['i', 'like', 'dog', 'i', 'love', 'coffee', 'i', 'hate', 'milk']
word_list = list(set(word_list))  # 去除重复的单词
word_dict = {w: i for i, w in
             enumerate(word_list)}  # {'hate': 0, 'dog': 1, 'milk': 2, 'love': 3, 'like': 4, 'i': 5, 'coffee': 6}
number_dict = {i: w for i, w in
               enumerate(word_list)}  # {0: 'like', 1: 'dog', 2: 'coffee', 3: 'hate', 4: 'i', 5: 'love', 6: 'milk'}
n_class = len(word_dict)  # 词典|V|的大小，也是最后分类的类别，这里是7
 
# NNLM(Neural Network Language Model) Parameter，模型的参数
n_step = len(sentences[0].split()) - 1  # 文中用n_step个词预测下一个词，在本程序中其值为2
n_hidden = 2  # 隐藏层神经元的数量
m = 2  # 词向量的维度
 
 
# 实现一个mini-batch迭代器
def make_batch(sentences):
    input_batch = []
    target_batch = []
 
    for sen in sentences:
        word = sen.split()  # ['i', 'like', 'dog']
        input = [word_dict[n] for n in word[:-1]]  # 列表对应的数字序列，一句话中最后一个词是要用来预测的，不作为输入
        target = word_dict[word[-1]]  # 每句话的最后一个词作为目标值
 
        input_batch.append(input)
        target_batch.append(target)
 
    return input_batch, target_batch  # ([[5, 2], [5, 0], [5, 6]], [4, 3, 1])
 
 
input_batch, target_batch = make_batch(sentences)
input_batch = torch.LongTensor(input_batch)
target_batch = torch.LongTensor(target_batch)
 
dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset=dataset, batch_size=16, shuffle=True)
 
 
# 定义模型
class NNLM(nn.Module):
    def __init__(self):
        """
        C: 词向量，大小为|V|*m的矩阵
        H: 隐藏层的weight
        W: 输入层到输出层的weight
        d: 隐藏层的bias
        U: 输出层的weight
        b: 输出层的bias
        1. 首先将输入的 n-1 个单词索引转为词向量，然后将这 n-1 个词向量进行 concat，形成一个 (n-1)*w 的向量，用 X 表示
        2. 将 X 送入隐藏层进行计算，hidden = tanh(d + X * H)
        3. 输出层共有|V|个节点，每个节点yi表示预测下一个单词i的概率，y的计算公式为y = b + X * W + hidden * U
        n_step: 文中用n_step个词预测下一个词，在本程序中其值为2
        n_hidden： 隐藏层（中间那一层）神经元的数量
        m: 词向量的维度
        """
        super(NNLM, self).__init__()
        self.C = nn.Embedding(n_class, m)  # 词向量随机赋值，代替了先使用one-hot，然后使用matrix C映射到词向量这一步
        self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype))
        self.W = nn.Parameter(torch.randn(n_step * m, n_class).type(dtype))
        self.d = nn.Parameter(torch.randn(n_hidden).type(dtype))
        self.U = nn.Parameter(torch.randn(n_hidden, n_class).type(dtype))
        self.b = nn.Parameter(torch.randn(n_class).type(dtype))
 
    def forward(self, X):
        """
        X: [batch_size, n_step]
        """
        X = self.C(X)  # [batch_size, n_step] => [batch_size, n_step, m]
        X = X.view(-1, n_step * m)  # [batch_size, n_step * m]
        hidden_out = torch.tanh(self.d + torch.mm(X, self.H))  # [batch_size, n_hidden], torch.mm就是矩阵的相乘
        output = self.b + torch.mm(X, self.W) + torch.mm(hidden_out, self.U)  # [batch_size, n_class]
        return output
 
 
# 实例化模型，优化器，损失函数
model = NNLM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
 
# train
for epoch in range(5000):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_x)
 
        loss = criterion(output, batch_y)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()
 
# Test
predict = model(input_batch).data.max(1, keepdim=True)[1]
# squeeze()：对张量的维度进行减少的操作，原来：tensor([[2],[6],[3]])，squeeze()操作后变成tensor([2, 6, 3])
print([sen.split()[:n_step] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])