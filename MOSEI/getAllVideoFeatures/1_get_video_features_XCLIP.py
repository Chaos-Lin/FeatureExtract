import av
import torch
import numpy as np

from transformers import AutoProcessor, AutoModel
# from huggingface_hub import hf_hub_download

np.random.seed(0)







# 输入是图片
import os
import argparse
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm


class getFeatures():
    def __init__(self, input_dir, output_dir, model_id):
        self.data_dir = input_dir
        self.output_dir = output_dir
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)


    """视频嵌入"""

    def getVideoFeature(self, file_path):
        container = av.open(file_path)

        # sample 8 frames
        indices = self.sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        video = self.read_video_pyav(container, indices)

        # processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
        # model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")


        inputs = self.processor(videos=list(video), return_tensors="pt")

        video_features = self.model.get_video_features(**inputs)

        return video_features

    def read_video_pyav(self, container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
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
        '''
        Sample a given number of frame indices from the video.
        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.
        Returns:
            indices (`List[int]`): List of sampled frame indices
        '''
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices

    def results(self):
        features_V = []  # 存储视频特征

        # 遍历视频数据目录中的所有视频文件夹
        file_list = os.listdir(self.data_dir)
        for video_dir in file_list:
            # 构建当前视频文件夹的路径
            video_dirs = os.path.join(self.data_dir, video_dir)
            # 获取当前视频文件夹下的所有视频帧文件夹
            dirs_list = os.listdir(video_dirs)
            # 遍历当前视频文件夹下的所有视频帧文件夹
            for clip in tqdm(dirs_list):
                file_path = os.path.join(self.data_dir, video_dir, clip)
                features_V.append(self.getVideoFeature(file_path).squeeze().cpu().detach().numpy())
        # 将视频 ID、视频帧 ID、视频 ID 和视频帧 ID 组合、以及填充后的视频特征保存到 .npz 文件中
        save_path = os.path.join(self.output_dir, 'videoFeature.npz')
        np.savez(save_path,feature_V=features_V)
        # 打印保存路径
        print('Features are saved in %s!' % save_path)


if __name__ == "__main__":
    input_dir = "D:\Search\MSA\MOSEI\MOSEI_raw\Raw"
    output_dir = "D:\\Search\\MSA\\MOSEI\\VideoFeature"
    model_id = "D:\Search\LLM\\xclip-base-patch32"
    os.makedirs(output_dir, exist_ok=True)
    gf = getFeatures(input_dir, output_dir, model_id)
    gf.results()

    # video clip consists of 300 frames (10 seconds at 30 FPS)
    # file_path = hf_hub_download(
    #     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
    # )


