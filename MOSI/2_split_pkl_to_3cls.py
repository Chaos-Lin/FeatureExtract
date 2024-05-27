import pickle
import numpy as np

with open(r'D:\Search\MSA\data\unaligned_unsplit_data.pkl', 'rb') as f:
    data = pickle.load(f)
    data['audio'] = np.array(data['audio'])
    data['vision'] = np.array(data['vision'])
    data['label_M'] = np.array(data['label_M'])
    data['video_id'] = np.array(data['video_id'])
    data['video_clip_id'] = np.array(data['video_clip_id'])
    data['video_id_clip_id'] = np.array(data['video_id_clip_id'])
    data['text_ids'] = np.array(data['text_ids'])
    data['text_mask'] = np.array(data['text_mask'])
    data['text_raw'] = np.array(data['text_raw'])
    np.random.seed(666)
    index = [i for i in range(len(data['text_mask']))]
    # 师姐这里用的shuffle，我就按照给定的mode来分就好了
    np.random.shuffle(index)
    data['audio'] = data['audio'][index]
    data['vision'] = data['vision'][index]
    data['label_M'] = data['label_M'][index]
    data['video_id'] = data['video_id'][index]
    data['video_clip_id'] = data['video_clip_id'][index]
    data['video_id_clip_id'] = data['video_id_clip_id'][index]
    data['text_ids'] = data['text_ids'][index]
    data['text_mask'] = data['text_mask'][index]
    data['text_raw'] = data['text_raw'][index]

    audio_train = data['audio'][:578]
    audio_valid = data['audio'][578:868]
    audio_test = data['audio'][868:]

    vision_train = data['vision'][:578]
    vision_valid = data['vision'][578:868]
    vision_test = data['vision'][868:]

    label_M_train = data['label_M'][:578]
    label_M_valid = data['label_M'][578:868]
    label_M_test = data['label_M'][868:]

    video_id_train = data['video_id'][:578]
    video_id_valid = data['video_id'][578:868]
    video_id_test = data['video_id'][868:]

    video_clip_id_train = data['video_clip_id'][:578]
    video_clip_id_valid = data['video_clip_id'][578:868]
    video_clip_id_test = data['video_clip_id'][868:]

    video_id_clip_id_train = data['video_id_clip_id'][:578]
    video_id_clip_id_valid = data['video_id_clip_id'][578:868]
    video_id_clip_id_test = data['video_id_clip_id'][868:]

    text_raw_train = data['text_raw'][:578]
    text_raw_valid = data['text_raw'][578:868]
    text_raw_test = data['text_raw'][868:]

    text_ids_train = data['text_ids'][:578]
    text_ids_valid = data['text_ids'][578:868]
    text_ids_test = data['text_ids'][868:]

    text_mask_train = data['text_mask'][:578]
    text_mask_valid = data['text_mask'][578:868]
    text_mask_test = data['text_mask'][868:]

    validdata = {
        'ids': video_id_clip_id_valid,
        'video_id': video_id_valid,
        'video_clip_id': video_clip_id_valid,
        'video_id_clip_id': video_id_clip_id_valid,
        'text_raw': text_raw_valid,
        'text_ids': text_ids_valid,
        'text_mask': text_mask_valid,
        'audio': audio_valid,
        'vision': vision_valid,
        'label_M': label_M_valid
    }

    testdata = {
        'ids': video_clip_id_test,
        'video_id': video_id_test,
        'video_clip_id': video_clip_id_test,
        'video_id_clip_id': video_id_clip_id_test,
        'text_raw': text_raw_test,
        'text_ids': text_ids_test,
        'text_mask': text_mask_test,
        'audio': audio_test,
        'vision': vision_test,
        'label_M': label_M_test
    }
    traindata = {
        'ids': video_id_clip_id_train,
        'video_id': video_id_train,
        'video_clip_id': video_clip_id_train,
        'video_id_clip_id': video_id_clip_id_train,
        'text_raw': text_raw_train,
        'text_ids': text_ids_train,
        'text_mask': text_mask_train,
        'audio': audio_train,
        'vision': vision_train,
        'label_M': label_M_train
    }

    data = {
        'train': traindata,
        'valid': validdata,
        'test': testdata
    }

    with open(r'D:\Search\MSA\MOSI\unaligned.pkl', 'ab') as f:
        pickle.dump(data, f)

# with open(r'D:\SHU\大论文\老人数据集\mesic.pkl', 'rb') as f:
#     data = pickle.load(f)
#     print(len(data))
#     print(data['train'].keys())
