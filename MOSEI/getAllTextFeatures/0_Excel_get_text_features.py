import xlrd
from transformers import BertTokenizer
import numpy as np
import os

tokenizer = BertTokenizer.from_pretrained('D:\Search\pretrained_weights\\bert-base-chinese')

PAD, CLS = '[PAD]', '[CLS]'
path = 'D:\Search\MSA\data\label\\rawText.xls'
text_csv = xlrd.open_workbook(path).sheet_by_name('Sheet1')
text_id = []
text_clip_id = []
text_id_clip_id = []
raw_text = []
text_ids = []
text_mask = []
pad_size = 39
text_id_num = text_csv.cell(0, 0).value
for i in range(text_csv.nrows):
    if i == 0: continue
    raw = text_csv.cell(i, 2).value
    token = tokenizer.tokenize(raw)
    token = [CLS] + token
    ids = tokenizer.convert_tokens_to_ids(token)
    if len(token) < pad_size:
        mask = [1] * len(ids) + [0] * (pad_size - len(token))
        ids += ([0] * (pad_size - len(token)))
    else:
        mask = [1] * pad_size
        ids = ids[:pad_size]
    raw_text.append(raw)
    text_ids.append(ids)
    text_mask.append(mask)
    if (text_csv.cell(i, 0).value != ''):
        text_id_num = text_csv.cell(i, 0).value
    text_clip_id_num = text_csv.cell(i, 1).value
    text_id.append(text_id_num)
    text_clip_id.append(text_clip_id_num)
    text_id_clip_id.append(str(text_id_num) + str('-') + str(text_clip_id_num))

save_path = os.path.join('D:\Search\MSA\SIMS\TextFeature\\textFeature.npz')
np.savez(save_path, text_id=text_id, text_clip_id=text_clip_id, text_id_clip_id=text_id_clip_id,
         raw_text=raw_text, text_ids=text_ids, text_mask=text_mask)


