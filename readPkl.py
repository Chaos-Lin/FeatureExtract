import pickle



# # 确认数据格式
import numpy as np

# audioFeature = np.load("D:\SHU\大论文\老人数据集\AudioFeature\\audioFeature.npz")
# print(audioFeature['audio_id'])  # 'video_0001'
# print(audioFeature['audio_clip_id'])  # '0001'
# print(audioFeature['audio_id_clip_id'])  # 'video_00010001'
# for i in range(100):
#     print(audioFeature['feature_A'][i][0])
#
#
#
# import numpy as np
#
# text = np.load('D:\SHU\大论文\老人数据集\TextFeature\\textFeature.npz')
# print(text['text_id'])
# print(len(text['text_id']))
# print(text['raw_text'])
#


# with open("D:\Search\MSA\Dataset\MOSI/unaligned_50.pkl", 'rb') as file:
with open("D:\Search\MSA\Dataset\SIMS/unaligned.pkl", 'rb') as file:
    data = pickle.load(file)
print(data.keys())
print(data['train'].keys())
# print(data['train']['raw_text'][0])
print(data['train']['text'].shape)
print(data['train']['audio'].shape)
print(data['train']['vision'].shape)


# MOSI是先pad再bert
# SIMS是先bert再pad

# print(data['train']['text_bert'][0])
# 应该是转化成id
# print((data['train']['regression_labels']).shape)
#
# # "F:\MIntRec-main\MIA-datasets\MOSI\unaligned_50.pkl"
# with open("F:/dataset\household\mesic.pkl", 'rb') as file:
#     data = pickle.load(file)
# print(data.keys())
# print((data['train']['label_M']).shape)

with open("D:\Search\MSA\data\\unaligned_unsplit_data.pkl", 'rb') as file:
    data = pickle.load(file)
    print(data['text_mask'].shape)
    print(data['audio'].shape)
    print(data['vision'].shape)


with open("D:\Search\MSA\data/unaligned.pkl", 'rb') as file:
    data = pickle.load(file)
print(data.keys())
print(data['train'].keys())
# print(data['train']['raw_text'][0])
print(data['train']['text_mask'].shape)
print(data['train']['audio'].shape)
print(data['train']['vision'].shape)
print(data['train']['video_id'][0])
print(data['train']['video_clip_id'][0])
print(data['train']['video_id_clip_id'][0])
print(data['train']['text_raw'][0])
print(data['train']['label_M'][0])
print(data['train']['label_A'][0])
print(data['train']['label_T'][0])
print(data['train']['label_V'][0])
