from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import csv
import os


model_name = 'D:\Search\pretrained_weights\\bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

text = '真他妈地烦！！！！！'

# Tokenization
tokens = tokenizer.tokenize(text)
input_ids0 = tokenizer.convert_tokens_to_ids(tokens)

# 添加特殊标记[CLS]和[SEP]
input_ids0 = [tokenizer.cls_token_id] + input_ids0 + [tokenizer.sep_token_id]

# Padding到最大长度50
max_length = 50

input_ids1 =input_ids0 + [tokenizer.pad_token_id] * (max_length - len(input_ids0))


# 将输入转换为tensor
input_tensor0 = torch.tensor([input_ids0])
input_tensor1 = torch.tensor([input_ids1])

# 获取BERT模型的隐藏表示
with torch.no_grad():
    outputs0 = model(input_tensor0)
    outputs1 = model(input_tensor1)

# 获取最后一层的隐藏状态
last_hidden_states0 = outputs0.last_hidden_state
last_hidden_states1 = outputs1.last_hidden_state
print(last_hidden_states0)
print(last_hidden_states1)

# 事实证明padding会对最终的embedding造成影响，所以是先bert还是先padding