import torch
from torch import nn
import numpy as np

# 通过下面的例子,也就理解了rnn函数的基本调用
# nn.RNN()
rnn = nn.RNN(input_size=10, hidden_size=20)
print(rnn._parameters.keys())
print(rnn.weight_ih_l0.shape) # (20, 10)
print(rnn.weight_hh_l0.shape) # (20, 20)
print(rnn.bias_ih_l0.shape) # (20,)
print(rnn.bias_hh_l0.shape) # (20,)


rnn1 = nn.RNN(input_size=100, hidden_size=20, bidirectional=True)
x = torch.randn(10, 3, 100)
out, h_t = rnn1(x, torch.zeros(2, 3, 20))
print(out.shape) # (10, 3, 40)
print(h_t.shape) # (2, 3, 20)

print('-' * 20)
rnn2 = nn.RNN(input_size=100, hidden_size=20, num_layers=1)
x = torch.randn(10, 3, 100)
out, h_t = rnn2(x, torch.zeros(1, 3, 20))
print(out.shape) # (10, 3, 20)
print(h_t.shape) # (1, 3, 20)
print('-' * 20)


print('-' * 20)
rnn2 = nn.RNN(input_size=100, hidden_size=20, num_layers=4)
x = torch.randn(10, 3, 100)
out, h_t = rnn2(x, torch.zeros(4, 3, 20))
print(out.shape) # (10, 3, 20)
print(h_t.shape) # (4, 3, 20)
print('-' * 20)



rnn3 = nn.RNN(input_size=100, hidden_size=20, num_layers=4,bidirectional=True)
x = torch.randn(10, 3, 100)
out, h_t = rnn3(x, torch.zeros(8, 3, 20))
print(out.shape) # (10, 3, 40)
print(h_t.shape) # (8, 3, 20)

# nn.RNNCell()
rnncell = nn.RNNCell(input_size=10, hidden_size=20)
print(rnncell._parameters.keys())
print(rnncell.weight_ih.shape) # (20,100)
print(rnncell.weight_hh.shape) # (20, 20)
print(rnncell.bias_ih.shape) # (20,)
print(rnncell.bias_hh.shape) # (20,)

# 单个RNNCell的运算方式
rnn = nn.RNNCell(10, 20)
input = torch.randn(6, 3, 10)
hx = torch.randn(3, 20)
output = []
print(input[0].shape) # (3,10)
for i in range(6):
    hx = rnn(input[i], hx)
    output.append(hx)
print(output[0].shape) # (3,20)

# 用RNNCell来循环实现RNN的执行过程（串行）
cell1 = nn.RNNCell(100, 20)
x = torch.randn(10, 3, 100)
h_t = torch.zeros((3,20))
output = []
for i in range(10):
    # 上一个RNNCell的输出再作为下一个RNNCell的输入
    h_t = cell1(x[i],h_t)
    output.append(h_t.detach().numpy())
output = torch.Tensor(np.array(output))
print(output.shape) # torch.Size([10, 3, 20])
print(h_t.shape) # 最后一个隐藏态的输出 torch.Size([3, 20])

# 上面的操作等价于下面的RNN()
rnn = nn.RNN(100,20,num_layers=1)
x_t = torch.zeros(1,3,20)
output1, h_t1 = rnn(x, x_t)
print(output1.shape)  # torch.Size([10, 3, 20])
print(h_t1.shape) # torch.Size([1, 3, 20])

# 用RNNCell实现两层RNN
cell1 = nn.RNNCell(100,30)
cell2 = nn.RNNCell(30, 20)
x = torch.randn(10, 3, 100)
h1 = torch.zeros(3, 30)
h2 = torch.zeros(3, 20)
for xt in x:
    h1 = cell1(xt, h1)
    h2 = cell2(h1, h2)
print(h2.shape) # (3,20)




