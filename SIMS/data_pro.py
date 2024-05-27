import numpy as np
import pickle
import pandas as pd
import os
import csv

# 标签及模式保存
csv_path = "D:\Search\MSA\SIMS\SIMS_raw\\label.csv"
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

file_list = creat_filelist("D:\Search\MSA\SIMS\AudioFeature\\audioRaw", csv_path, "wav")
file_list2 = creat_filelist("D:\Search\MSA\SIMS\SIMS_raw\Raw", csv_path, "mp4")
# print(file_list)

data = {
    'file_list': file_list,
    'file_list2':file_list2,
    'label_A': label_A,
    'label_V': label_V,
    'label_T': label_T,
    'label_M': label_M,
    'mode': mode
}

audio_train = []
audio_valid = []
audio_test = []

video_train = []
video_valid = []
video_test = []

labelA_train = []
labelA_valid = []
labelA_test = []

labelT_train = []
labelT_valid = []
labelT_test = []

labelV_train = []
labelV_valid = []
labelV_test = []

labelM_train = []
labelM_valid = []
labelM_test = []

for index in range(len(label_data['label_A'])):
    if (data['mode'][index]) == 'train':
        audio_train.append(data['file_list'][index])
        labelM_train.append(data['label_M'][index])
        labelA_train.append(data['label_A'][index])
        labelV_train.append(data['label_V'][index])
        labelT_train.append(data['label_T'][index])
        video_train.append(data['file_list2'][index])

    elif(data['mode'][index]) == 'valid':
        audio_valid.append(data['file_list'][index])
        labelM_valid.append(data['label_M'][index])
        labelA_valid.append(data['label_A'][index])
        labelV_valid.append(data['label_V'][index])
        labelT_valid.append(data['label_T'][index])
        video_valid.append(data['file_list2'][index])

    elif(data['mode'][index]) =='test':
        audio_test.append(data['file_list'][index])
        labelM_test.append(data['label_M'][index])
        labelA_test.append(data['label_A'][index])
        labelV_test.append(data['label_V'][index])
        labelT_test.append(data['label_T'][index])
        video_test.append(data['file_list2'][index])

traindata = {
    "audio":np.array(audio_train),
    "video":np.array(video_train),
    "label_A":np.array(labelA_train),
    "label_V":np.array(labelV_train),
    "label_T":np.array(labelT_train),
    "label":np.array(labelM_train)
}

validdata = {
    "audio":np.array(audio_valid),
    "video": np.array(video_valid),
    "label_A": np.array(labelA_valid),
    "label_V": np.array(labelV_valid),
    "label_T": np.array(labelT_valid),
    "label":np.array(labelM_valid)
}

testdata = {
    "audio":np.array(audio_test),
    "video": np.array(video_test),
    "label_A": np.array(labelA_test),
    "label_V": np.array(labelV_test),
    "label_T": np.array(labelT_test),
    "label":np.array(labelM_test)
}

data = {
    'train': traindata,
    'valid': validdata,
    'test': testdata
}

with open(r'D:\Search\MSA\SIMS\audio_video.pkl', 'ab') as f:
    pickle.dump(data, f)



