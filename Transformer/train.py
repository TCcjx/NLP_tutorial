from model import Transformer
from torch.utils.data import Dataset,DataLoader
from data import MyDataset
from torch import nn
from tqdm import tqdm
import torch


epochs = 10
my_model = Transformer().cuda() # 加载模型
dataset = MyDataset("source.txt", "target.txt") # 加载数据集
dataloader = DataLoader(dataset, 32, shuffle=True) # 封装数据
loss_func = nn.CrossEntropyLoss(ignore_index=2) # 损失函数
optimizer = torch.optim.AdamW(params=my_model.parameters(), lr=0.0005) # 优化函数
for epoch in range(epochs): # 训练批次数
    t = tqdm(dataloader) # 进度条显示

    for input_id, input_m, output_id, output_m in t: # 加载批次训练数据
        target = output_id[:,1:].cuda() # 标签数据
        output = my_model(input_id.cuda(), input_m.cuda(), output_id[:, :-1].cuda(), output_m[:, :-1].cuda())
        loss = loss_func(output.reshape(-1,29), target.reshape(-1)) # 词表大小29,所以分类类别也就是29
        loss.backward() # 反向传播计算梯度值
        torch.nn.utils.clip_grad_norm_(my_model.parameters(), 1) # 防止梯度爆炸
        optimizer.step() # 参数更新
        optimizer.zero_grad() # 梯度信息清零
        t.set_description(str(loss.item())) # 打印梯度值

torch.save(my_model.state_dict(), 'model.pth') # 保存模型


