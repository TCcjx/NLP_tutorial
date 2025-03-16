import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

vocab_list = ['[BOS]', '[EOS]' ,'[PAD]' ] + list("abcdefghijklmnopqrstuvwxyz")
char_table = list("abcdefghijklmnopqrstuvwxyz")
bos_token = '[BOS]'
eos_token = '[EOS]'
pad_token = '[PAD]'

def process_data(source, target):
    max_length = 12
    if len(source) > max_length:
        source = source[:max_length]
    if len(target) > max_length - 1: # 这里存疑  ？？？
        target = target[:max_length - 1]
    source_id = [vocab_list.index(p) for p in source]
    target_id = [vocab_list.index(p) for p in target]
    target_id = [vocab_list.index('[BOS]')] + target_id + [vocab_list.index('[EOS]')]
    source_m = np.array([1] * max_length)
    target_m = np.array([1] * (max_length + 1))
    if len(source_id) < max_length:
        pad_len = max_length - len(source_id)
        source_id += [vocab_list.index('[PAD]')] * pad_len
        source_m[-pad_len: ] = 0
    if len(target_id) < max_length + 1:
        pad_len = max_length - len(target_id) + 1
        target_id += [vocab_list.index('[PAD]')] * pad_len
        target_m[-pad_len: ] = 0
    return source_id, source_m, target_id, target_m


class MyDataset(Dataset):
    def __init__(self, source_path, target_path):
        super().__init__()
        self.source_list = []
        self.target_list = []
        with open(source_path) as f:
            content = f.readlines()
            for str in content:
                self.source_list.append(str.strip())
        with open(target_path) as f:
            content = f.readlines()
            for str in content:
                self.target_list.append(str.strip())
    def __getitem__(self, index):
        source_id, source_m, target_id, target_m =  process_data(self.source_list[index], self.target_list[index])
        return torch.tensor(source_id, dtype= torch.long), torch.tensor(source_m, dtype=torch.long),\
            torch.tensor(target_id, dtype=torch.long), torch.tensor(target_m, dtype= torch.long)

    def __len__(self):
        return len(self.source_list)

source_path = 'source.txt'
target_path = 'target.txt'

if __name__ == '__main__':
    test_data = MyDataset(source_path, target_path)
    a = test_data[8]
    pass
