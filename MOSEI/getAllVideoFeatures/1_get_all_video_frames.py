import os
import subprocess
from tqdm import tqdm

# 将视频切分成帧

def extract_frames(input_path, output_path):
    str_cmd = f'ffmpeg -i "{input_path}" "{output_path}"'
    result = subprocess.run(str_cmd, shell=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error occurred while extracting frames from {input_path}")
        print(result.stderr.decode())


def split_videos_and_extract_frames(data_dir, output_dir):
    videos_dirs = os.listdir(data_dir)
    num_videos_dirs = len(videos_dirs)
    for i, videos_dir in enumerate(videos_dirs):
        print(f'{i} / {num_videos_dirs}')
        video_dir = os.path.join(data_dir, videos_dir)
        if not os.path.isdir(video_dir) or videos_dir == 'Annotation.xls':
            continue
        output_video_dir = os.path.join(output_dir, videos_dir)

        # Create output directory if it does not exist
        if not os.path.exists(output_video_dir):
            os.makedirs(output_video_dir)

        for single_video in tqdm(os.listdir(video_dir)):
            input_path = os.path.join(video_dir, single_video)
            output_path_pre = os.path.join(output_video_dir, single_video.replace('.mp4', ''))
            output_path_template = os.path.join(output_path_pre, 'image-%4d.jpg')

            # Create output directory if it does not exist
            if not os.path.exists(output_path_pre):
                os.makedirs(output_path_pre)

            # Check if input video exists and is readable
            if os.path.exists(input_path) and os.access(input_path, os.R_OK):
                extract_frames(input_path, output_path_template)
            else:
                print(f"Input video {input_path} does not exist or is not readable")


data_dir = 'D:\Search\MSA\SIMS\SIMS_raw\Raw'
output_dir = 'D:\Search\MSA\SIMS\VideoFeature\\video_frames' #存放视频帧的总路径
os.makedirs(output_dir, exist_ok=True)
split_videos_and_extract_frames(data_dir, output_dir)
