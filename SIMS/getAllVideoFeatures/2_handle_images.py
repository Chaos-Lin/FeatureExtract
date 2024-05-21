# 第二步：openCV提取特征
import os
from tqdm import tqdm
import logging

logging.basicConfig(filename='extraction.log', level=logging.INFO,
                    format='%(message)s')
def handleImages(data_dir ,output_folder, openface2Path):
    # 获取数据目录下的所有视频文件夹列表
    file_list = os.listdir(data_dir) # [video_0001, video_0002, ...]

    # 遍历每个视频文件夹
    for video_dir in tqdm(file_list):
        logging.info(f"Processing video {video_dir} of {len(file_list)}")
        # 构建当前视频文件夹完整路径
        image_dirs = os.path.join(data_dir, video_dir)
        # 获取图像文件夹列表
        dirs_list = os.listdir(image_dirs)  # [0001, 0002, ...]
        # 遍历当前视频文件夹下的每个图像文件夹
        for image_dir in tqdm(dirs_list):  # 0001
            logging.info(f"Processing image {image_dir} of {len(dirs_list)}")
            # 构建输出目录的完整路径，用于保存 OpenFace2 提取的特征
            output_dir = os.path.join(output_folder, video_dir, image_dir)
            # 构建输入图像文件夹的完整路径
            input_dir = os.path.join(image_dirs, image_dir)
            # 如果输出目录不存在，则创建
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # 构建执行 OpenFace2 的命令 选择抽取的特征
            cmd = openface2Path + ' -fdir ' + input_dir + ' -out_dir ' + output_dir + ' -2Dfp -pose -aus -gaze'
            # 执行命令
            os.system(cmd)


if __name__ == "__main__":
    input_dir = "D:\\Search\\MSA\\SIMS\\VideoFeature\\video_frames"
    openface2_path = "D:\\Tool\\OpenFace_2.2.0_win_x64_with_models\\FeatureExtraction.exe"
    output_dir = "D:\\Search\\MSA\\SIMS\\VideoFeature\\openface_feature"
    os.makedirs(output_dir, exist_ok=True)
    handleImages(input_dir, output_dir, openface2_path)