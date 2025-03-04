import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor
import av
import pickle

class getFeatures():
    def __init__(self, data_file, output_dir):
        # self.data_dir = input_dir
        self.output_dir = output_dir
        self.processor = AutoProcessor.from_pretrained("D:\Search\LLM\\xclip-base-patch32")
        self.data_file = data_file
        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)
            self.data = data['video']

    def read_video_pyav(self, container, indices):

        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def sample_frame_indices(self, clip_len, frame_sample_rate, seg_len):
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices

    def __getVideoEmbedding(self, video_path):

        container = av.open(video_path)
        indices = self.sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        video = self.read_video_pyav(container, indices)
        inputs = self.processor(videos=list(video), return_tensors="pt")
        features = inputs["pixel_values"].squeeze(0)
        return np.array(features)

    def results(self):
        features_V = []  # 存储视频特征
        for video in tqdm(self.data):
            embedding_V = self.__getVideoEmbedding(video)
            features_V.append(embedding_V)
        save_path = os.path.join(self.output_dir, 'video_hid_Feature.npz')
        np.savez(save_path,feature_V=features_V)
        # 打印保存路径
        print('Features are saved in %s!' % save_path)


if __name__ == "__main__":
    input_dir = "D:\\Search\\MSA\\SIMS\\SIMS_unsplit.pkl"
    output_dir = "D:\\Search\\MSA\\SIMS"
    os.makedirs(output_dir, exist_ok=True)
    gf = getFeatures(input_dir, output_dir)
    gf.results()