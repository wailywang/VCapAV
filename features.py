#!/usr/bin/env python
"""
Script to extract and save CLIP and CLAP features from videos.

Usage:
$: python extract_save_features.py --gpu=0 --data_dir=/SMIIPdata2/ASVspoof5/data/wangyx/scp/train_videos --output_dir=/path/to/save_features

Author: "Yikang Wang"
Email: "wwm1995@alps-lab.org"
"""

import os
import torch
import cv2
import argparse
from PIL import Image
from moviepy.editor import VideoFileClip
import tempfile
import librosa
import laion_clap
import clip
from tqdm import tqdm

# 提取 CLIP 特征
def extract_clip_features(video_path, model, preprocess, device, batch_size=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_features = []
    batch_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_input = preprocess(image).unsqueeze(0).to(device)
        batch_frames.append(image_input)

        if len(batch_frames) == batch_size:
            batch_frames = torch.cat(batch_frames, dim=0)
            with torch.no_grad():
                image_features = model.encode_image(batch_frames)
            frame_features.append(image_features)
            batch_frames = []

    if batch_frames:
        batch_frames = torch.cat(batch_frames, dim=0)
        with torch.no_grad():
            image_features = model.encode_image(batch_frames)
        frame_features.append(image_features)

    cap.release()
    torch.cuda.synchronize()

    frame_features = torch.cat(frame_features, axis=0).cpu().numpy()
    return frame_features

# 提取 CLAP 特征
def extract_audio_features_from_video(video_path, model):
    video = VideoFileClip(video_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        video.audio.write_audiofile(temp_audio_file.name, codec='pcm_s16le')
        audio_data, _ = librosa.load(temp_audio_file.name, sr=48000)
    audio_embed = model.get_audio_embedding_from_data(x=audio_data.reshape(1, -1), use_tensor=False)
    return audio_embed

# 拼接 CLIP 和 CLAP 特征（不再包含fbank特征）
def extract_all_features(video_path, clip_model, clap_model, preprocess, device, batch_size=30):
    clip_features = extract_clip_features(video_path, clip_model, preprocess, device, batch_size=batch_size)
    clap_features = extract_audio_features_from_video(video_path, clap_model)

    if not isinstance(clip_features, np.ndarray):
        clip_features = clip_features.cpu().numpy()
    if not isinstance(clap_features, np.ndarray):
        clap_features = clap_features.cpu().numpy()

    # 确保 clip_features 和 clap_features 在第 0 维的大小相同
    if clip_features.shape[0] != clap_features.shape[0]:
        if clap_features.shape[0] == 1:
            # 如果 clap_features 只有一个特征，重复它以匹配 clip_features 的大小
            clap_features = np.repeat(clap_features, clip_features.shape[0], axis=0)
        else:
            raise ValueError(f"无法匹配 clip_features 和 clap_features 的大小：{clip_features.shape[0]} vs {clap_features.shape[0]}")

    # 拼接特征
    combined_features = np.concatenate((clip_features, clap_features), axis=1)
    combined_features_tensor = torch.tensor(combined_features, device=device).float()
    return combined_features_tensor

# 保存提取的特征
def save_features(features, output_path):
    torch.save(features, output_path)

# 提取并保存特征
def extract_and_save_features(video_paths, clip_model, clap_model, preprocess, device, output_dir, batch_size=30):
    for video_path in tqdm(video_paths):
        features = extract_all_features(video_path, clip_model, clap_model, preprocess, device, batch_size=batch_size)
        video_name = os.path.basename(video_path).split('.')[0]
        output_path = os.path.join(output_dir, f'{video_name}_features.pt')
        save_features(features, output_path)
        print(f"Saved features for {video_name} at {output_path}")

def get_video_paths(data_dir):
    """
    获取视频文件的路径列表。
    :param data_dir: 包含 wav.scp 和 utt2label 的目录，内部有 videos 文件夹。
    :return: 一个包含所有视频文件路径的列表。
    """
    video_dir = os.path.join(data_dir, "videos")
    video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
    return video_paths

def main(args):
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    # 初始化 CLIP 和 CLAP 模型
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clap_model = laion_clap.CLAP_Module(enable_fusion=False)
    clap_model.load_ckpt('/SMIIPdata2/ASVspoof5/clipclap/clap_htsat_tiny.pt')
    clap_model.to(device)
    clap_model.eval()

    # 获取视频路径
    video_paths = get_video_paths(args.data_dir)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 提取并保存特征
    extract_and_save_features(video_paths, clip_model, clap_model, preprocess, device, args.output_dir, batch_size=args.batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and save CLIP and CLAP features from videos.')

    # 参数
    parser.add_argument('--gpu', type=str, default='0', help='GPU device ID')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing wav.scp, utt2label, and videos folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save extracted features')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size for feature extraction')

    args = parser.parse_args()
    main(args)
