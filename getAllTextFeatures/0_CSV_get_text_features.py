from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from tqdm import tqdm
import csv
import os


def tokenize(text, model, tokenizer, pad_first = False):
    # Tokenization
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # 添加特殊标记[CLS]和[SEP]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    # Padding到最大长度50
    max_length = 50
    if len(input_ids) < max_length:
        if pad_first:
            input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
    else:
        input_ids = input_ids[:max_length]

    # 将输入转换为tensor
    input_tensor = torch.tensor([input_ids])

    # 获取BERT模型的隐藏表示
    with torch.no_grad():
        outputs = model(input_tensor)

    # 获取最后一层的隐藏状态
    last_hidden_states = outputs.last_hidden_state.squeeze()

    if not pad_first:
        pad_sizes = (0,0,0, max_length - last_hidden_states.size(0))
        fill_value = 0.0
        token_embeddings = torch.nn.functional.pad(last_hidden_states, pad_sizes, "constant", fill_value)
        # print(token_embeddings.size())
    else:
        token_embeddings = last_hidden_states

    return token_embeddings



def get_text_features(input_dir, output_dir):
    text_id = []
    text_clip_id = []
    text_id_clip_id = []
    feature_T = []
    raw_text = []

    model_name = 'D:\Search\pretrained_weights\\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    file = os.path.join(input_dir, 'label.csv')
    with open(file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            print(f"{i}/2281")
            text_id.append(row['video_id'])
            text_clip_id.append(row['clip_id'])
            text_id_clip_id.append(f"{row['video_id']}_{row['clip_id']}")
            raw_text.append(row['text'])
            embedding = tokenize(row['text'], model, tokenizer)
            feature_T.append(embedding)
    # 保存
    save_path = os.path.join(output_dir, 'textFeature.npz')
    np.savez(save_path, text_id=text_id, text_clip_id=text_clip_id, text_id_clip_id=text_id_clip_id,
             feature_T=feature_T, raw_text=raw_text)

if __name__ == '__main__':
    input_dir = 'D:\Search\MSA\data\SIMS_raw'
    output_dir = 'D:\Search\MSA\data\TextFeature'
    os.makedirs(output_dir, exist_ok=True)
    get_text_features(input_dir, output_dir)
