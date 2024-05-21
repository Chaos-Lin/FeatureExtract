import os
from glob import glob
from tqdm import tqdm
import subprocess


def fetch_voices(input_dir, output_dir):
    print("Start Fetching human voices...")
    # 验证输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return

    file_list = os.listdir(input_dir)
    num_files = len(file_list)
    for i, video_dir in enumerate(file_list):
        print(f"{i}/{num_files}")
        video_dir_1 = os.path.join(input_dir, video_dir)
        audio_paths = sorted(glob(os.path.join(video_dir_1, '*.wav')))
        # print(len(audio_paths))
        for audio_path in tqdm(audio_paths):
            output_dir_p = os.path.join(output_dir, video_dir)
            # 创建输出目录
            os.makedirs(output_dir_p, exist_ok=True)
            """调用spleeter执行人声分离"""
            try:
                cmd = ['spleeter', 'separate', '-o', output_dir_p, audio_path]
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                # 如果出现错误，记录日志信息
                print(f"Error occurred while processing {audio_path}: {e}")

if __name__ == "__main__":
    input_path = r'D:\Search\MSA\SIMS\AudioFeature\audioRaw'
    output_path = r'D:\Search\MSA\SIMS\AudioFeature\audioPeople'
    os.makedirs(output_path, exist_ok=True)
    fetch_voices(input_path, output_path)
