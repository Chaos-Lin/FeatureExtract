import numpy as np
import pickle
import pandas as pd

def __init_MESIC():
    # 读取audio的特征
    audio_data = np.load('D:\Search\MSA\MOSEI\AudioFeature\\audioFeature.npz')
    feature_A = audio_data['feature_A']

    # 读取text的特征
    text_data = np.load('D:\Search\MSA\MOSEI\TextFeature\\textFeature.npz')
    raw_text = text_data['raw_text']
    tokens = text_data["tokens"]
    feature_T = text_data['feature_T']

    # 读取video的特征
    video_data = np.load('D:\Search\MSA\MOSEI\VideoFeature\\videoFeature.npz')
    feature_V = video_data['feature_V']
    # video_id = video_data['video_id']
    # video_clip_id = video_data['video_clip_id']
    # video_id_clip_id = video_data['video_id_clip_id']

    # 读取label
    path = "D:\Search\MSA\MOSEI\MOSEI_raw\\label.csv"
    # todo
    label_data = pd.read_csv(path)
    label_M = label_data['label'].values
    label_T = label_data['label_T'].values
    label_V = label_data['label_V'].values
    label_A = label_data['label_A'].values
    mode = label_data['mode'].values
    video_id = label_data['video_id']
    video_clip_id = label_data['clip_id']


    # 写入pkl文件
    mesicData = {
            'video_id': video_id,
            'video_clip_id': video_clip_id,
            # 'video_id_clip_id': video_id_clip_id,
            'text_raw': raw_text,
            "tokens": tokens,
            'text_mask': feature_T,
            'audio': feature_A,
            'vision': feature_V,
            'label_M': label_M,
            'label_T': label_T,
            'label_V': label_V,
            'label_A': label_A,
            'mode':mode
        }
    with open(r'D:\Search\MSA\MOSEI\unaligned_unsplit_data.pkl', 'ab') as f:
        pickle.dump(mesicData, f)

__init_MESIC()
