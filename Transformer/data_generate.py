import random


'''
vocab_lst: 词表
char_lst: 字母表
bos_token: 开始字符
eos_token: 结束字符
pad_token: 填充字符
'''

vocab_lst = ["[BOS]", "[EOS]", "[PAD]", 'a', 'b', 'c', 'd','e', 'f', 'g', 'h','i', 'j', 'k', 'l','m', 'n',
 'o', 'p','q', 'r', 's', 't','u', 'v', 'w', 'x','y', 'z']
char_lst = ['a', 'b', 'c', 'd','e', 'f', 'g', 'h','i', 'j', 'k', 'l','m', 'n',
 'o', 'p','q', 'r', 's', 't','u', 'v', 'w', 'x','y', 'z']
bos_token = "[BOS]"
eos_token = "[EOS]"
pad_token = "[PAD]"

# 对于一个英语字符串，输出加密后的字符串，加密方式如下：
# 对于每一个字符，使其ASCII码值循环减5，然后将整个字符串逆序
# 例如：abcfg -> baxwv

source_path = "source.txt"
target_path = "target.txt"
with open(source_path, 'w') as f:
    pass
with open(target_path, 'w') as f:
    pass

for _ in range(10000):
    source_str = ""
    target_str = ""
    for idx in range(random.randint(3, 10)): # 生成 3~10长度的 字符串
        i = random.randint(0, 25)
        source_str += char_lst[i] # 随机源字符串
        target_str += char_lst[(i + 26 - 5) % 26] # 随机目标字符串，是源字符串前移五个位置得到的结果
    # target_str = target_str[::-1] # 对目标字符串进行翻转逆序处理
    reversed(target_str)
    with open(source_path, 'a') as f:
        f.write(source_str + '\n')
    with open(target_path, 'a') as f:
        f.write(target_str + '\n')