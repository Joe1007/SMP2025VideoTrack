import os
import sys
import torch
import numpy as np
import cv2
from tqdm import tqdm
import clip


# 你要的參數
TARGET_NUM_FRAMES = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_frames(video_path, target_num_frames=TARGET_NUM_FRAMES):
    """平均取樣 target_num_frames 幀"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError(f"No frames found in {video_path}")

    indices = np.linspace(0, total_frames - 1, target_num_frames).astype(int)

    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    cap.release()

    # 如果影片很短 (< target_num_frames)，最後一幀補滿
    if len(frames) < target_num_frames:
        pad_len = target_num_frames - len(frames)
        pad = [frames[-1]] * pad_len
        frames.extend(pad)

    return frames  # list of np.ndarray (H, W, 3)

def preprocess_frames(frames, preprocess):
    """Resize + Normalize frames using CLIP preprocess"""
    processed = [preprocess(Image.fromarray(f)) for f in frames]
    return torch.stack(processed)  # shape (N, 3, H, W)

def encode_video(video_path, model, preprocess):
    frames = extract_frames(video_path)
    frames_tensor = preprocess_frames(frames, preprocess)  # (128, 3, 224, 224)
    frames_tensor = frames_tensor.to(DEVICE)

    with torch.no_grad():
        features = model.encode_image(frames_tensor)  # (128, feature_dim)
        features = features / features.norm(dim=-1, keepdim=True)  # normalize

    return features.cpu()  # shape (128, feature_dim)

def save_features(features, save_path):
    torch.save(features, save_path)

def process_dataset(root_dir, output_dir):
    model, preprocess = clip.load("ViT-L/14", device=DEVICE)


    subsets = ["train", "test"]
    for subset in subsets:
        subset_path = os.path.join(root_dir, subset)
        if not os.path.isdir(subset_path):
            continue

        for user_id in os.listdir(subset_path):
            user_dir = os.path.join(subset_path, user_id)
            if not os.path.isdir(user_dir):
                continue

            os.makedirs(os.path.join(output_dir, subset), exist_ok=True)

            video_files = [f for f in os.listdir(user_dir) if f.endswith(".mp4")]
            for video_file in tqdm(video_files, desc=f"{subset}/{user_id}"):
                video_path = os.path.join(user_dir, video_file)

                try:
                    features = encode_video(video_path, model, preprocess)
                    video_name = os.path.splitext(video_file)[0]
                    save_name = f"{user_id}_{video_name}.pt"
                    save_path = os.path.join(output_dir, subset, save_name)
                    save_features(features, save_path)
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clip_video_encode.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]   # 你的 video_file 目錄
    print(input_dir)
    output_dir = sys.argv[2]  # 要存 features 的目錄
    print(output_dir)

    from PIL import Image
    process_dataset(input_dir, output_dir)


# python extract_clip_features02.py ../raw_data/video_file ./video_feature
