import os
import subprocess
from tqdm import tqdm

# 将所有音频从视频中抽取出来
def extract_audio(input_path, output_path):
    try:
        # 构建命令字符串
        str_cmd = f'ffmpeg -i "{input_path}" -c:v libx264 -c:a aac -strict experimental -async 1 "{output_path}"'

        # 执行命令
        result = subprocess.run(str_cmd, shell=True, capture_output=True, text=True, check=True)

        # 打印命令执行结果
        print(f"Successfully processed video {input_path} to {output_path}")
    except subprocess.CalledProcessError as e:
        # 如果提取失败，打印日志信息
        print(f"Failed to extract audio from {input_path}: {e}")


def batch_extract_audio(data_dir, output_dir):
    video_dirs = os.listdir(data_dir)
    num_videos_dirs = len(video_dirs)
    for i, video_dir in enumerate(video_dirs):
        print(f'{i}/{num_videos_dirs}')
        if video_dir == 'Annotation.xls': continue
        # 枚举单个视频
        for single_video in tqdm(os.listdir(os.path.join(data_dir, video_dir))):

            input_path = os.path.join(data_dir, video_dir, single_video)
            output_path = os.path.join(output_dir, video_dir, single_video)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if os.path.exists(input_path) and os.access(input_path, os.R_OK):
                extract_audio(input_path, output_path)
            else:
                # 如果文件不存在或无法读取，打印日志信息
                print(f"Input video {input_path} does not exist or is not readable")

if __name__ == '__main__':
    data_dir = r'D:\Search\MSA\MOSEI\MOSEI_RAW\Raw'
    output_dir = 'D:\Search\MSA\MOSEI\VideoFeature'  # 存放音频的总体路径
    os.makedirs(output_dir, exist_ok=True)
    batch_extract_audio(data_dir, output_dir)