import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

vocab_list = ['[BOS]', '[EOS]' ,'[PAD]' ] + list("abcdefghijklmnopqrstuvwxyz") # 词表
char_table = list("abcdefghijklmnopqrstuvwxyz") # 字母表
bos_token = '[BOS]'
eos_token = '[EOS]'
pad_token = '[PAD]'

# 数据处理函数
def process_data(source, target):
    max_length = 12
    '''
    由于generative生成的字符长度为 3 ~ 10
    所以跳过了截断处理
    '''
    # 这里有一个bug，当输入长度超过12的时候，输入输出是对不上的，因为保持源输入和目标输入长度一致，导致会多截取目标输入的字符
    if len(source) > max_length: # 大于最大长度做截断处理
        source = source[:max_length]
    if len(target) > max_length - 1: # 这里存疑  ？？？
        target = target[:max_length - 1]

    source_id = [vocab_list.index(p) for p in source]
    target_id = [vocab_list.index(p) for p in target]
    target_id = [vocab_list.index('[BOS]')] + target_id + [vocab_list.index('[EOS]')]

    # 源字符串掩码 和 目标字符串掩码
    source_m = np.array([1] * max_length)
    target_m = np.array([1] * (max_length + 1))
    if len(source_id) < max_length: # 如果源字符串长度小于max_length，则进行填充，使其长度变为max_length
        pad_len = max_length - len(source_id)
        source_id += [vocab_list.index('[PAD]')] * pad_len
        source_m[-pad_len: ] = 0 # 将填充字符位置的掩码值设置为0
    if len(target_id) < max_length + 1: # 同理，如果目标字符串掩码的长度小于max_length + 1,则进行填充，使其长度变为max_length + 1
        pad_len = max_length - len(target_id) + 1
        target_id += [vocab_list.index('[PAD]')] * pad_len
        target_m[-pad_len: ] = 0 # 将填充字符位置的掩码值设置为0
    return source_id, source_m, target_id, target_m # 源字符串id,源字符串掩码值,目标字符串id,目标字符串掩码值


class MyDataset(Dataset): # 继承Dataset类
    def __init__(self, source_path, target_path):
        super().__init__()
        self.source_list = []
        self.target_list = []
        with open(source_path) as f:
            content = f.readlines() # 读取每一行数据,并放在列表里
            for str in content:
                self.source_list.append(str.strip()) # 去除字符串两端的空格和换行符
        with open(target_path) as f:
            content = f.readlines()
            for str in content:
                self.target_list.append(str.strip())
    def __getitem__(self, index):
        source_id, source_m, target_id, target_m =  process_data(self.source_list[index], self.target_list[index])
        return (torch.tensor(source_id, dtype= torch.long), torch.tensor(source_m, dtype=torch.long),\
            torch.tensor(target_id, dtype=torch.long), torch.tensor(target_m, dtype= torch.long)) # 转换成tensor类型返回

    def __len__(self): # 返回字符串列表长度
        return len(self.source_list)

source_path = 'source.txt'
target_path = 'target.txt'

if __name__ == '__main__':
    test_data = MyDataset(source_path, target_path)
    print(len(test_data))
    a = test_data[8]
    pass
