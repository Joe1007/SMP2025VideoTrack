import os
import openl3
import numpy as np
import torch
from moviepy.editor import VideoFileClip
import tensorflow as tf
import soundfile as sf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# ==== 基本參數設定 ====
video_root = "../raw_data/video_file"
save_root = "./audio_feature"
os.makedirs(save_root, exist_ok=True)

# ==== 音訊擷取 ====
def extract_audio_from_mp4(mp4_path, output_wav_path, target_sr=48000):
    video = VideoFileClip(mp4_path)
    audio = video.audio
    audio.write_audiofile(output_wav_path, fps=target_sr, verbose=False, logger=None)

# ==== OpenL3 特徵提取 + Mean Pooling ====
def extract_openl3_features(wav_path, content_type="music", input_repr="mel256", embedding_size=512):
    audio, sr = sf.read(wav_path)
    emb, _ = openl3.get_audio_embedding(
        audio=audio,
        sr=sr,
        input_repr=input_repr,
        content_type=content_type,
        embedding_size=embedding_size,
        center=True,
        hop_size=0.1,
        verbose=False
    )
    pooled = np.mean(emb, axis=0)  # shape: (512,)
    return pooled

# ==== 處理 train/test 分割 ====
def process_split(split_name):
    split_path = os.path.join(video_root, split_name)
    if not os.path.exists(split_path):
        return

    for user_folder in os.listdir(split_path):
        user_path = os.path.join(split_path, user_folder)
        if not os.path.isdir(user_path):
            continue

        for filename in os.listdir(user_path):
            if not filename.endswith(".mp4"):
                continue

            mp4_path = os.path.join(user_path, filename)
            file_prefix = f"{user_folder}_{os.path.splitext(filename)[0]}"
            save_dir = os.path.join(save_root, split_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{file_prefix}.pt")

            if os.path.exists(save_path):
                # print(f"[SKIP] {save_path} already exists.")
                continue

            print(f"[PROCESSING] {mp4_path}")
            try:
                temp_wav = "temp.wav"
                extract_audio_from_mp4(mp4_path, temp_wav)
                features = extract_openl3_features(temp_wav)
                features_tensor = torch.from_numpy(features.astype(np.float32))  # shape: (512,)
                torch.save(features_tensor, save_path)
                print(f"[SAVED] {save_path}")
            except Exception as e:
                print(f"[ERROR] {mp4_path}: {e}")
            finally:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)

# ==== 執行 ====
process_split("train")
# process_split("test")
