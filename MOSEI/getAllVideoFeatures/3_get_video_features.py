
# 输入是图片
import os
import argparse
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm


class getFeatures():
    def __init__(self, input_dir, output_dir):
        self.data_dir = input_dir
        self.output_dir = output_dir
        self.padding_mode = 'zeros'
        self.padding_location = 'back'

    """视频嵌入"""

    def __getVideoEmbedding(self, csv_path, pool_size=5):
        df = pd.read_csv(csv_path)
        features, local_features = [], []
        for i in range(len(df)):
            local_features.append(np.array(df.loc[i][df.columns[5:]]))
            if (i + 1) % pool_size == 0:
                features.append(np.array(local_features).mean(axis=0))
                local_features = []
        if len(local_features) != 0:
            features.append(np.array(local_features).mean(axis=0))
        return np.array(features)

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
        # feature_dim = sequences[0].shape[-1]
        #         # lens = [s.shape[0] for s in sequences]
        #         # # confirm length using (mean + std)
        #         # final_length = int(np.mean(lens) + 3 * np.std(lens))
        #         # # padding sequences to final_length
        #         # final_sequence = np.zeros([len(sequences), final_length, feature_dim])
        #         # for i, s in enumerate(sequences):
        #         #     final_sequence[i] = self.__padding(s, final_length)
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
        for i in range(len(sequences)):
            final_sequence[i] = self.__padding(sequences[i], final_length)

        return final_sequence
    def results(self):
        # 初始化存储视频特征的列表和视频信息的列表
        video_id = []  # 存储视频 ID
        video_clip_id = []  # 存储视频帧 ID
        video_id_clip_id = []  # 存储视频 ID 和视频帧 ID 的组合
        features_V = []  # 存储视频特征

        # 遍历视频数据目录中的所有视频文件夹
        file_list = os.listdir(self.data_dir)
        for video_dir in file_list:
            # 构建当前视频文件夹的路径
            image_dirs = os.path.join(self.data_dir, video_dir)
            # 获取当前视频文件夹下的所有视频帧文件夹
            dirs_list = os.listdir(image_dirs)
            # 遍历当前视频文件夹下的所有视频帧文件夹
            for image_dir in tqdm(dirs_list):
                # 将视频 ID 添加到视频 ID 列表中
                video_id.append(video_dir)
                # 将视频帧 ID 添加到视频帧 ID 列表中
                video_clip_id.append(image_dir)
                # 将视频 ID 和视频帧 ID 的组合添加到视频 ID 和视频帧 ID 组合列表中
                video_id_clip_id.append(str(video_dir) + str(image_dir))
                # 构建当前视频帧的 CSV 文件路径
                csv_path = os.path.join(self.data_dir, video_dir, image_dir, image_dir + '.csv')
                # 如果当前视频帧的 CSV 文件不存在
                if not os.path.exists(csv_path):
                    # 将空特征向量添加到视频特征列表中
                    embedding_V = np.array([[0] * 709])
                    features_V.append(embedding_V)
                else:
                    # 否则，从 CSV 文件中提取视频特征，并添加到视频特征列表中
                    embedding_V = self.__getVideoEmbedding(csv_path, pool_size=5)
                    features_V.append(embedding_V)

        # 对视频特征进行填充，使其具有相同的长度
        feature_V = self.__paddingSequence(features_V)

        # 将视频 ID、视频帧 ID、视频 ID 和视频帧 ID 组合、以及填充后的视频特征保存到 .npz 文件中
        save_path = os.path.join(self.output_dir, 'videoFeature.npz')
        np.savez(save_path, video_id=video_id, video_clip_id=video_clip_id, video_id_clip_id=video_id_clip_id,
                 feature_V=feature_V)
        # 打印保存路径
        print('Features are saved in %s!' % save_path)


if __name__ == "__main__":
    input_dir = "D:\\Search\\MSA\\SIMS\\VideoFeature\\openface_feature"
    output_dir = "D:\\Search\\MSA\\SIMS\\VideoFeature"
    os.makedirs(output_dir, exist_ok=True)
    gf = getFeatures(input_dir, output_dir)
    gf.results()