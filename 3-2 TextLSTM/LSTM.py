import torch
from torch import nn


# input_size hidden_size num_layers
rnn = nn.LSTM(input_size= 10, hidden_size= 30, num_layers= 2)
input = torch.randn(5, 3, 10) # H_in = input_size
h0 = torch.randn(2, 3, 30) # H_out = hidden_size
c0 = torch.randn(2, 3, 30) # H_cell = hidden_size
output, (hn, cn) = rnn(input, (h0, c0))
print(output.shape) # (5, 3, 30)
print(hn.shape)  # (2, 3, 30)
print(cn.shape)  # (2, 3, 30)
# print(hn == output[-1,:,:])

# input_size hidden_size num_layers
rnn = nn.LSTM(input_size= 10, hidden_size= 30, num_layers= 1, bidirectional= True)
input = torch.randn(5, 3, 10) # H_in = input_size
h0 = torch.randn(2, 3, 30) # H_out = hidden_size
c0 = torch.randn(2, 3, 30) # H_cell = hidden_size
output, (hn, cn) = rnn(input, (h0, c0))
print(output.shape) # (5, 3, 60)
print(hn.shape)  # (2, 3, 30)
print(cn.shape)  # (2, 3, 30)
# print(hn == output[-1,:,:])


# input_size hidden_size num_layers
rnn = nn.LSTM(input_size= 10, hidden_size= 30, num_layers= 1, bidirectional= False)
input = torch.randn(5, 3, 10) # H_in = input_size
h0 = torch.randn(1, 3, 30) # H_out = hidden_size
c0 = torch.randn(1, 3, 30) # H_cell = hidden_size
output, (hn, cn) = rnn(input, (h0, c0))
print(output.shape) # (5, 3, 30)
print(hn.shape)  # (1, 3, 30)
print(cn.shape)  # (1, 3, 30)
print(hn == output[-1,:,:])