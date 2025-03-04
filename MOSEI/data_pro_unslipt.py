import numpy as np
import pickle
import pandas as pd
import os
import csv
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
# 标签及模式保存
csv_path = "D:\Search\MSA\MOSEI\MOSEI_RAW\label.csv"
# todo
label_data = pd.read_csv(csv_path)
label_A = label_data['label_A'].values
label_V = label_data['label_V'].values
label_T = label_data['label_T'].values
label_M = label_data['label'].values
mode = label_data['mode'].values
text_raw = label_data['text'].values
# print(label_A[0])

from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
def tokenize(text, model, tokenizer, max_length=50):
    # Tokenization
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # 添加特殊标记 [CLS] 和 [SEP]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    # 进行 Padding 或截断
    attention_mask = [1] * len(input_ids)
    if len(input_ids) < max_length:
        padding_length = max_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
    else:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]

    # 转换为 tensor
    input_tensor = torch.tensor([input_ids])
    attention_mask_tensor = torch.tensor([attention_mask])

    # 获取 BERT 模型的隐藏表示
    with torch.no_grad():
        outputs = model(input_tensor, attention_mask=attention_mask_tensor)

    # 获取最后一层的隐藏状态
    last_hidden_states = outputs.last_hidden_state.squeeze()

    return last_hidden_states


def get_text_features(text_raw):
    embedding = []
    model_name = 'D:\Search\pretrained_weights\\bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # 确保模型处于评估模式，避免 dropout
    model.eval()

    for text in tqdm(text_raw):
        embedding.append(tokenize(text, model, tokenizer))

    return embedding
# 获取音频地址

def creat_filelist(input_path, csv_path, data_type):
    file_list = []
    with open(csv_path, encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            audio_path = os.path.join(input_path, row[0], f'{row[1]}.{data_type}')
            file_list.append(audio_path)
    return file_list

file_list = creat_filelist("D:\Search\MSA\MOSEI\AudioFeature\\audioRaw", csv_path, "wav")
file_list2 = creat_filelist("D:\Search\MSA\MOSEI\MOSEI_RAW\Raw", csv_path, "mp4")
# print(file_list)
text = get_text_features(text_raw)
data = {
    'audio': file_list,
    'video': file_list2,
    'text': text,
    'label_A': label_A,
    'label_V': label_V,
    'label_T': label_T,
    'label_M': label_M,
    'mode': mode
}


with open(r'D:\Search\MSA\MOSEI\MOSEI_unsplit.pkl', 'ab') as f:
    pickle.dump(data, f)



