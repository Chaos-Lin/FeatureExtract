import pickle
import numpy as np

with open(r'D:\Search\MSA\MOSEI\unaligned_unsplit_data.pkl', 'rb') as f:
    data = pickle.load(f)
    data['audio'] = np.array(data['audio'])
    data['vision'] = np.array(data['vision'])

    data['label_M'] = np.array(data['label_M'])
    data['label_V'] = np.array(data['label_V'])
    data['label_A'] = np.array(data['label_A'])
    data['label_T'] = np.array(data['label_T'])
    data['video_id'] = np.array(data['video_id'])
    data['video_clip_id'] = np.array(data['video_clip_id'])
    # data['video_id_clip_id'] = np.array(data['video_id_clip_id'])
    data['text_mask'] = np.array(data['text_mask'])
    # data['text_raw'] = np.array(data['text_raw'])
    # data['tokens'] = np.array(data["tokens"])

    audio_train = []
    audio_valid = []
    audio_test = []

    vision_train = []
    vision_valid = []
    vision_test = []

    label_M_train = []
    label_M_valid = []
    label_M_test = []

    label_V_train = []
    label_V_valid = []
    label_V_test = []

    label_A_train = []
    label_A_valid = []
    label_A_test = []

    label_T_train = []
    label_T_valid = []
    label_T_test = []

    video_id_train = []
    video_id_valid = []
    video_id_test = []

    video_clip_id_train = []
    video_clip_id_valid = []
    video_clip_id_test = []

    video_id_clip_id_train = []
    video_id_clip_id_valid = []
    video_id_clip_id_test = []

    text_raw_train = []
    text_raw_valid = []
    text_raw_test = []

    text_mask_train = []
    text_mask_valid = []
    text_mask_test = []

    tokens_train = []
    tokens_valid = []
    tokens_test = []


    for index in range(len(data['text_mask'])):
        if (data['mode'][index])== 'train':
            # text_raw_train.append(data['text_raw'][index])
            text_mask_train.append(data['text_mask'][index])
            vision_train.append(data['vision'][index])
            audio_train.append(data['audio'][index])
            video_id_train.append(data['video_id'][index])
            # video_clip_id_train.append(data['video_clip_id'][index])
            # video_id_clip_id_train.append(data['video_id_clip_id'][index])
            label_T_train.append(data['label_T'][index])
            label_A_train.append(data['label_A'][index])
            label_V_train.append(data['label_V'][index])
            label_M_train.append(data['label_M'][index])
            # tokens_train.append(data['tokens'][index])

        elif  (data['mode'][index])== 'valid':
            # text_raw_valid.append(data['text_raw'][index])
            text_mask_valid.append(data['text_mask'][index])
            vision_valid.append(data['vision'][index])
            audio_valid.append(data['audio'][index])
            video_id_valid.append(data['video_id'][index])
            # video_clip_id_valid.append(data['video_clip_id'][index])
            # video_id_clip_id_valid.append(data['video_id_clip_id'][index])
            label_T_valid.append(data['label_T'][index])
            label_A_valid.append(data['label_A'][index])
            label_V_valid.append(data['label_V'][index])
            label_M_valid.append(data['label_M'][index])
            # tokens_valid.append(data['tokens'][index])

        elif (data['mode'][index]) == 'test':
            # text_raw_test.append(data['text_raw'][index])
            text_mask_test.append(data['text_mask'][index])
            vision_test.append(data['vision'][index])
            audio_test.append(data['audio'][index])
            video_id_test.append(data['video_id'][index])
            # video_clip_id_test.append(data['video_clip_id'][index])
            # video_id_clip_id_test.append(data['video_id_clip_id'][index])
            label_T_test.append(data['label_T'][index])
            label_A_test.append(data['label_A'][index])
            label_V_test.append(data['label_V'][index])
            label_M_test.append(data['label_M'][index])
            # tokens_test.append(data['tokens'][index])

    validdata = {
        'ids': np.array(video_id_clip_id_valid),
        'video_id': np.array(video_id_valid),
        # 'video_clip_id': np.array(video_clip_id_valid),
        # 'video_id_clip_id': np.array(video_id_clip_id_valid),
        # 'text_raw': np.array(text_raw_valid),
        'text': np.array(text_mask_valid),
        'audio': np.array(audio_valid),
        'vision': np.array(vision_valid),
        'label_M': np.array(label_M_valid),
        'label_A': np.array(label_A_valid),
        'label_T': np.array(label_T_valid),
        'label_V': np.array(label_V_valid),
        # 'tokens':np.array(tokens_valid)
    }

    testdata = {
        'ids': np.array(video_clip_id_test),
        'video_id': np.array(video_id_test),
        # 'video_clip_id': np.array(video_clip_id_test),
        # 'video_id_clip_id': np.array(video_id_clip_id_test),
        # 'text_raw': np.array(text_raw_test),
        'text': np.array(text_mask_test),
        'audio': np.array(audio_test),
        'vision': np.array(vision_test),
        'label_M': np.array(label_M_test),
        'label_A': np.array(label_A_test),
        'label_T': np.array(label_T_test),
        'label_V': np.array(label_V_test),
        # 'tokens':np.array(tokens_test)
    }
    traindata = {
        'ids': np.array(video_id_clip_id_train),
        'video_id': np.array(video_id_train),
        # 'video_clip_id': np.array(video_clip_id_train),
        # 'video_id_clip_id': np.array(video_id_clip_id_train),
        # 'text_raw': np.array(text_raw_train),
        'text': np.array(text_mask_train),
        'audio': np.array(audio_train),
        'vision': np.array(vision_train),
        'label_M': np.array(label_M_train),
        'label_A': np.array(label_A_train),
        'label_T': np.array(label_T_train),
        'label_V': np.array(label_V_train),
        # 'tokens':np.array(tokens_train)
    }

    data = {
        'train': traindata,
        'valid': validdata,
        'test': testdata
    }

    with open(r'D:\Search\MSA\MOSEI\unaligned.pkl', 'ab') as f:
        pickle.dump(data, f)
