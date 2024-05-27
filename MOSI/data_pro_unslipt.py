import numpy as np
import pickle
import pandas as pd
import os
import csv

# 标签及模式保存
csv_path = "D:\Search\MSA\MOSI\MOSI_RAW\label.csv"
# todo
label_data = pd.read_csv(csv_path)
label_A = label_data['label_A'].values
label_V = label_data['label_V'].values
label_T = label_data['label_T'].values
label_M = label_data['label'].values
mode = label_data['mode'].values
# print(label_A[0])

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

file_list = creat_filelist("D:\Search\MSA\MOSI\AudioFeature\\audioRaw", csv_path, "wav")
file_list2 = creat_filelist("D:\Search\MSA\MOSI\MOSI_RAW\Raw", csv_path, "mp4")
# print(file_list)

data = {
    'audio': file_list,
    'video':file_list2,
    'label_A': label_A,
    'label_V': label_V,
    'label_T': label_T,
    'label_M': label_M,
    'mode': mode
}


with open(r'D:\Search\MSA\MOSI\MOSI.pkl', 'ab') as f:
    pickle.dump(data, f)



