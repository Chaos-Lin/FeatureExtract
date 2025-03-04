import pickle
import numpy as np

import numpy as np
import pickle
import pandas as pd

def __init_MESIC():

    # 读取text的特征
    text_data = np.load('D:\Search\MSA\SIMS\TextFeature\\textFeature.npz')
    text = text_data['text_mask']
    # 读取video的特征
    video_data = np.load('D:\\Search\\MSA\\SIMS\\video_hid_Feature.npz')
    feature_V = video_data['feature_V']

    # 读取label
    path = "D:\Search\MSA\SIMS\SIMS_raw\\label.csv"
    # todo
    label_data = pd.read_csv(path)
    label_M = label_data['label'].values
    mode = label_data['mode'].values

    # 写入pkl文件
    vision_train = []
    vision_valid = []
    vision_test = []

    label_M_train = []
    label_M_valid = []
    label_M_test = []

    text_mask_train = []
    text_mask_valid = []
    text_mask_test = []

    for index in range(len(text_data['text_mask'])):
        if (mode[index]) == 'train':
            text_mask_train.append(text[index])
            vision_train.append(feature_V[index])
            label_M_train.append(label_M[index])

        elif (mode[index]) == 'valid':
            text_mask_valid.append(text[index])
            vision_valid.append(feature_V[index])
            label_M_valid.append(label_M[index])
        if (mode[index]) == 'test':
            text_mask_test.append(text[index])
            vision_test.append(feature_V[index])
            label_M_test.append(label_M[index])

    validdata = {
        'text': np.array(text_mask_valid),
        'vision': np.array(vision_valid),
        'label_M': np.array(label_M_valid),
    }

    testdata = {
        'text': np.array(text_mask_test),
        'vision': np.array(vision_test),
        'label_M': np.array(label_M_test),
    }
    traindata = {
        'text': np.array(text_mask_train),
        'vision': np.array(vision_train),
        'label_M': np.array(label_M_train),
    }

    data = {
        'train': traindata,
        'valid': validdata,
        'test': testdata
    }

    with open(r'D:\Search\MSA\SIMS\SIMS_video.pkl', 'ab') as f:
        pickle.dump(data, f)
