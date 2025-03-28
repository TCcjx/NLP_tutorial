'''
这里来记录一下torch.gather() 这个接口函数的用法
out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2


需要的参数就是源张量、index维度、index
input 和 index 必须具有相同的维度数。对于所有维度， index.size(d) <= input.size(d) 对于所有 d ！= dim.out 将具有与 index 相同的形状
input 和 index 不会相互进行广播
------------------------------------------------------
Parameters  参数
input (Tensor) – the source tensor
input （Tensor） – 源张量

dim (int) – the axis along which to index
dim （int） – 索引沿其轴

index (LongTensor) – the indices of elements to gather
index （LongTensor） – 要收集的元素的索引
------------------------------------------------------
Keyword Arguments  关键字参数
sparse_grad (bool, optional) – If True, gradient w.r.t. input will be a sparse tensor.
sparse_grad （bool， 可选 ） – 如果为 True，则梯度 w.r.t. 输入将是一个稀疏张量。

out (Tensor, optional) – the destination tensor
out （Tensor， optional） – 目标张量
'''
import torch
# 二维的情况举例
t = torch.tensor([[1, 2], [3, 4]])
output = torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))
print(output)
'''
采样的四个索引:
[0][index[0][0]] = [0][0] -> 1
[0][index[0][1]] = [0][0] -> 1
[1][index[1][0]] = [1][1] -> 4
[1][index[1][1]] = [1][0] -> 3
output:
tensor([[1, 1],
        [4, 3]])
为什么是上面的结果呢？
因为dim=1,也就是相当于我们需要用index索引来替换output
'''

# 换到0维度，进行gather
t = torch.tensor([[1, 2], [3, 4]])
output = torch.gather(t, 0, torch.tensor([[0, 0], [1, 0]]))
print(output)
"""
[index[0][0]][0] -> 1
[index[0][1]][1] -> 2
[index[1][0]][0] -> 3
[index[1][1]][1] -> 2
"""

# 三维的情况同理
'''
out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
'''

# 下面这篇博客的有对这个torch函数接口的说明，本人绝对有一定参考价值
# https://wmathor.com/index.php/archives/1457/