# 第3步提取特征
import os
import argparse
import librosa
import soundfile as sf
import struct
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer, Wav2Vec2Model
import csv

class getFeatures():
    def __init__(self, input_dir, output_dir, csv_path, mode='libraso'):
        self.data_dir = input_dir
        self.output_dir = output_dir
        self.padding_mode = 'zeros'
        self.padding_location = 'back'
        self.csv_path = csv_path
        self.mode = mode

        if mode == 'wav2vec2':
            self.model_id = "D:\\Search\\pretrained_weights\\wav2vec2-base-960h"
            self.CTC = Wav2Vec2Processor.from_pretrained(self.model_id)
            self.model = Wav2Vec2Model.from_pretrained(self.model_id)
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            print(self.device)



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

    def __getAudioFeatures(self, wavFile):
        audio, sample_rate = sf.read(wavFile)
        target_sample_rate = 16000
        audio_resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
        input_values = self.processor(audio_resampled,
                                 sampling_rate=target_sample_rate,
                                 return_tensors="pt").input_values.permute(1,0).to(self.device)
        result = self.model(input_values)
        return result["last_hidden_state"].squeeze()
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
        features_A = []

        with open(self.csv_path, newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            count = 1
            for row in csvreader:
                print(f"{count}/2198")
                count +=1
                audio_id.append(row[0])
                audio_clip_id.append(row[1])
                audio_path = os.path.join(self.data_dir, row[0], f'{row[1]}.wav')
                if self.mode == "libraso":
                    embedding_A = self.__getAudioEmbedding(audio_path)
                elif self.mode == "wav2vec2":
                    embedding_A = self.__getAudioFeatures(audio_path)
                features_A.append(embedding_A)

        # 保存
        if self.mode == "libraso":
            features_A = self.__paddingSequence(features_A)
            features_A = np.array(features_A)
        save_path = os.path.join(self.output_dir, 'audioFeature.npz')
        np.savez(save_path, audio_id=audio_id, audio_clip_id=audio_clip_id,
                 feature_A=features_A)

        print('Features are saved in %s!' % save_path)





if __name__ == "__main__":
    input_dir= "D:\Search\MSA\MOSI\AudioFeature\\audioRaw"
    csv_path="D:\Search\MSA\MOSI\MOSI_RAW\label.csv"
    output_dir= 'D:\Search\MSA\MOSI\AudioFeature'
    os.makedirs(output_dir, exist_ok=True)
    gf = getFeatures(input_dir,output_dir,csv_path,"libraso")
    gf.results()