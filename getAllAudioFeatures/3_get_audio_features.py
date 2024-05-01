# 第3步提取特征
import os
import argparse
import librosa
import struct
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class getFeatures():
    def __init__(self, input_dir, output_dir):
        self.data_dir = input_dir
        self.output_dir = output_dir
        self.padding_mode = 'zeros'
        self.padding_location = 'back'


    # 文本转录
    def __transText(self, wavFile):
        audio, rate = torchaudio.load(wavFile) # 加载音频
        resampler = torchaudio.transforms.Resample(rate, 16_000)
        audio = resampler(audio).squeeze().numpy()
        processor = Wav2Vec2Processor.from_pretrained("D:\\Search\\pretrained_weights\\wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("D:\\Search\\pretrained_weights\\wav2vec2-base-960h")
        input1 = processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
        input2 = input1.input_values.squeeze()
        with torch.no_grad():
            logit = model(input2).logits
        prediction = torch.argmax(logit, dim=-1)
        transcription = processor.batch_decode(prediction)[0]
        return transcription

    """音频嵌入"""

    def __getAudioEmbedding(self, audio_path):
        y, sr = librosa.load(audio_path)
        # print('sr=',sr)
        # using librosa to get audio features (f0, mfcc, cqt)
        hop_length = 512  # hop_length smaller, seq_len larger,  hop_length ：S列之间的音频样本数,帧移
        f0 = librosa.feature.zero_crossing_rate(y, hop_length=hop_length).T  # (seq_len, 1),计算音频时间序列的过零率。
        cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length).T  # (seq_len, 12)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, htk=True).T  # (seq_len, 20)
        return np.concatenate([f0, mfcc, cqt], axis=-1)  # (seq_len, 33)
    # 也就是说最后的是这三个部分的cat
    # MFCC 主要捕获了音频信号的语音特征，用于语音识别、语音情感分析等任务。

    def __padding(self, feature, MAX_LEN):
        """
        mode:
            zero: padding with 0
            normal: padding with normal distribution
        location: front / back
        """
        assert self.padding_mode in ['zeros', 'normal']
        assert self.padding_location in ['front', 'back']

        length = feature.shape[0]
        if length >= MAX_LEN:
            return feature[:MAX_LEN, :]

        if self.padding_mode == "zeros":
            pad = np.zeros([MAX_LEN - length, feature.shape[-1]])
        elif self.padding_mode == "normal":
            mean, std = feature.mean(), feature.std()
            pad = np.random.normal(mean, std, (MAX_LEN - length, feature.shape[1]))

        feature = np.concatenate([pad, feature], axis=0) if (self.padding_location == "front") else \
            np.concatenate((feature, pad), axis=0)
        return feature

    def __paddingSequence(self, sequences):
        feature_dim = sequences[0].shape[-1]
        # print('feature_dim',feature_dim)
        lens = [s.shape[0] for s in sequences]
        # print('lens',lens)
        # confirm length using (mean + std)
        final_length = int(np.mean(lens) + 3 * np.std(lens))
        # final_length = 400
        # print('final_length',final_length)
        # padding sequences to final_length
        final_sequence = np.zeros([len(sequences), final_length, feature_dim])
        for i, s in enumerate(sequences):
            final_sequence[i] = self.__padding(s, final_length)
        # for i in range(len(sequences)):
        #     final_sequence[i] = self.__padding(sequences[i], final_length)

        return final_sequence

    def results(self):
        audio_id = []
        audio_clip_id = []
        audio_id_clip_id = []
        features_A = []
        file_list = os.listdir(self.data_dir)
        for video_dir in file_list:
            # 枚举单个音频
            single_audio_files = os.listdir(os.path.join(self.data_dir, video_dir))
            for single_audio in tqdm(single_audio_files):
                audio_id.append(video_dir)
                audio_clip_id.append(single_audio.split('.')[0])
                audio_id_clip_id.append(str(video_dir) + "_" + str(single_audio.split('.')[0]))
                # print(single_audio)  ### 0001.wav
                audio_path = os.path.join(self.data_dir, video_dir, single_audio)
                embedding_A = self.__getAudioEmbedding(audio_path)
                features_A.append(embedding_A)
                # text = self.__transText(audio_path)
                # features_AtoT.append(text)

        # 补长
        # print('features_A',len(features_A))
        # print('print(features_A[0])',len(features_A[0]))
        # print('print(features_A[0][0])', len(features_A[0][0]))
        # print('print(features_A[1])',len(features_A[1]))
        # print('print(features_A[1][0])', len(features_A[1][0]))
        # print('print(features_A[27])', len(features_A[27]))
        feature_A = self.__paddingSequence(features_A)

        # 保存
        save_path = os.path.join(self.output_dir, 'audioFeature.npz')
        np.savez(save_path, audio_id=audio_id, audio_clip_id=audio_clip_id, audio_id_clip_id=audio_id_clip_id,
                 feature_A=feature_A)

        print('Features are saved in %s!' % save_path)





if __name__ == "__main__":
    input_dir= "D:\Search\MSA\data\AudioFeature\\audioRaw"
    output_dir= 'D:\Search\MSA\data\AudioFeature'
    os.makedirs(output_dir, exist_ok=True)
    gf = getFeatures(input_dir,output_dir)
    gf.results()