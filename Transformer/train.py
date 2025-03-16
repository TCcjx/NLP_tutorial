from model import Transformer
from torch.utils.data import Dataset,DataLoader
from data import MyDataset
from torch import nn
from tqdm import tqdm
import torch


my_model = Transformer().cuda()
dataset = MyDataset("source.txt", "target.txt")
dataloader = DataLoader(dataset, 32, shuffle=True)
loss_func = nn.CrossEntropyLoss(ignore_index=2)
optimizer = torch.optim.AdamW(params=my_model.parameters(), lr=0.0005)

for epoch in range(10):
    t = tqdm(dataloader)
    for input_id, input_m, output_id, output_m in t:
        target = output_id[:,1:].cuda()
        output = my_model(input_id.cuda(), input_m.cuda(), output_id[:, :-1].cuda(), output_m[:, :-1].cuda())
        loss = loss_func(output.reshape(-1, 29), target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(my_model.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()
        t.set_description(str({loss.item()}))


torch.save(my_model.state_dict(), 'model.pth')


