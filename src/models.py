import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from torchvision.models.video import r3d_18
from sklearn.preprocessing import LabelEncoder
import librosa
from kymatio.torch import Scattering1D
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import requests
import tarfile
import scipy.io
import sys
import cv2
from moviepy import VideoFileClip
import time
import random
import mediapipe as mp
from tqdm import tqdm

sys.path.append(os.getcwd()+'/MediaPipePyTorch/')
from blazepose import *
from blazepose_landmark import *


# --------- 1. Audio Model Config ---------
def extract_audio_features(audio_path, out_npy_path, sr=24000, n_mfcc=40, hop_length=512, use_scattering=False, scattering_J=6, scattering_Q=8, augment_func=None, specaug=False):
    y, _ = librosa.load(audio_path, sr=sr)
    if use_scattering:
        target_len = 2 ** scattering_J * scattering_Q
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        y = y[:target_len]
        x = torch.from_numpy(y).float().unsqueeze(0)
        scattering = Scattering1D(J=scattering_J, shape=x.shape[-1], Q=scattering_Q)
        Sx = scattering(x)
        features = Sx.squeeze(0).numpy()
    else:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        features = mfcc
    np.save(out_npy_path, features)

def pad_or_trim_feat(feat, target_shape):
    # feat: (n_features, n_frames)
    out = np.zeros(target_shape, dtype=feat.dtype)
    n0 = min(target_shape[0], feat.shape[0])
    n1 = min(target_shape[1], feat.shape[1])
    out[:n0, :n1] = feat[:n0, :n1]
    return out

def score_to_label(score):
    if score < 0.25:
        return "Bad"
    elif score < 0.50:
        return "Average"
    elif score < 0.75:
        return "Good"
    else:
        return "Great"

def prepare_features(features_dir):
    mfcc_files = sorted(glob.glob(os.path.join(features_dir, '*_mfcc.npy')))
    scatter_files = sorted(glob.glob(os.path.join(features_dir, '*_scatter.npy')))
    # Normalize base names and DataFrame File values
    mfcc_basenames = [os.path.splitext(os.path.basename(f))[0].replace('_mfcc','').strip() for f in mfcc_files]
    scatter_basenames = [os.path.splitext(os.path.basename(f))[0].replace('_scatter','').strip() for f in scatter_files]
    features = []
    # Set your desired shapes here
    target_mfcc_shape = (40, 100)
    target_scatter_shape = (128, 100)
    for mfcc_file in mfcc_files:
        base = os.path.splitext(os.path.basename(mfcc_file))[0].replace('_mfcc','').strip()
        scatter_file = os.path.join(features_dir, f"{base}_scatter" + mfcc_file[mfcc_file.find('_mfcc')+5:])
        if not os.path.exists(scatter_file):
            continue
        mfcc_feat = np.load(mfcc_file)
        scatter_feat = np.load(scatter_file)
        mfcc_feat = pad_or_trim_feat(mfcc_feat, target_mfcc_shape)
        scatter_feat = pad_or_trim_feat(scatter_feat, target_scatter_shape)
        combined_feat = np.concatenate([mfcc_feat.flatten(), scatter_feat.flatten()])
        print(f"{base}: MFCC shape {mfcc_feat.shape}, Scatter shape {scatter_feat.shape}, Combined length {combined_feat.shape[0]}")
        features.append(combined_feat)
    if not features:
        raise ValueError('No features found. Please check that your features directory contains the correct .npy files and that your File column matches the audio file basenames.')
    features = np.stack(features)
    return features

class MultiLabelAudioRegressor(nn.Module):
    def __init__(self, input_dim, num_tasks=6, dropout=0.5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([
            nn.Linear(64, 1) for _ in range(num_tasks)
        ])
    def forward(self, x):
        x = self.shared(x)
        return torch.cat([head(x) for head in self.heads], dim=1)  # [batch, num_tasks]

def get_predictions_regression(model, data, device, categories):
        model.eval()
        print("\nTest set predictions (predicted | true):")
        pred_list = []
        with torch.no_grad():
            data = data.to(device)
            outputs = model(data)
            preds = outputs.cpu().numpy()
            for i in range(preds.shape[0]):
                pred = preds[i]
                pred_str = ', '.join([f"{cat}: {score_to_label(p)} ({p:.2f})" for cat, p in zip(categories, pred)])
                pred_list.append(pred_str)
        return pred_list


# --------- 2. Pose Estimation Model Config ---------
frames_per_clip = 16
batch_size = 8
posture_epochs = 30
lr = 1e-3
weight_decay = 1e-4
patience = 30
val_split = 0.2
posture_issues = [
    "forward_head", "flat_back", "sway_back", "rounded_shoulders",
    "weak_abdominals", "bent_knees", "raised_chest", "bent_neck"
]
activities = ['beatboxing','busking', 'playing clarinet', 'playing drums', 'playing flute', 'playing guitar', 'playing piano', 'playing recorder', 'playing saxophone', 'playing trombone', 'playing violin', 'recording music', 'singing', 'whistling']
num_labels = len(posture_issues)

MPII_TO_BLAZEPOSE = {
    0: 28, 1: 26, 2: 24, 3: 23, 4: 25, 5: 27, 6: 0, 7: 12, 8: 0, 9: 0,
    10: 16, 11: 14, 12: 12, 13: 11, 14: 13, 15: 15,
}
MAPPED_INDICES = list(MPII_TO_BLAZEPOSE.values())

def extract_frames(out_path, max_frames=300, frame_interval=1, compression_quality=85):
    """
    Extract frames from video with memory optimization.
    Args:
        out_path: Path to video file
        max_frames: Maximum number of frames to extract
        frame_interval: Extract every nth frame
        compression_quality: JPEG compression quality (0-100)
    """
    cap = cv2.VideoCapture(out_path)
    if not cap.isOpened():
        print(f"Error opening video file: {out_path}")
        return None
        
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame interval to get desired number of frames
    if total_frames > max_frames:
        frame_interval = max(1, total_frames // max_frames)
    
    frames_data = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Resize frame to reduce memory usage
            frame = cv2.resize(frame, (640, 480))  # Adjust size as needed
            
            # Convert to RGB and compress
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, compression_quality])
            compressed_frame = buffer.tobytes()
            
            # Convert to tensor with reduced precision
            tensor_frame = torch.from_numpy(np.frombuffer(compressed_frame, np.uint8))
            frames_data.append(tensor_frame)
            
            # Free up memory
            del frame_rgb
            del frame
            del buffer
            del compressed_frame
            
        frame_count += 1
        
        # Break if we have enough frames
        if len(frames_data) >= max_frames:
            break
    
    cap.release()
    
    if not frames_data:
        print(f"No frames extracted for {out_path}")
        return None
    
    # Stack frames and free up memory
    frames_tensor = torch.stack(frames_data)
    del frames_data
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return frames_tensor

def process_videos_batch(filtered_df, save_dir, batch_size=10):
    """
    Process videos in batches to manage memory usage.
    """
    import gc
    from tqdm import tqdm
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Process videos in batches
    for i in tqdm(range(0, len(filtered_df), batch_size), desc="Processing video batches"):
        batch_df = filtered_df.iloc[i:i+batch_size]
        
        for _, row in batch_df.iterrows():
            video_id = str(row['youtube_id'])
            label = str(row['label'])
            out_path = os.path.join(os.getcwd(), 'kinetics', 'train', label, video_id+".mp4")
            save_path = os.path.join(save_dir, f"{label}_{video_id}.pt")
            
            if not os.path.exists(out_path):
                print(f"Video file not found: {out_path}, skipping.")
                continue
                
            try:
                # Extract frames with compression
                vid_frames = extract_frames(out_path, compression_quality=85)
                if vid_frames is None or vid_frames.numel() == 0:
                    print(f"No frames extracted for {video_id}, skipping.")
                    continue
                    
                # Save with compression
                torch.save(vid_frames, save_path, _use_new_zipfile_serialization=True)
                print(f"Saved {save_path}")
                
                # Free up memory
                del vid_frames
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing {video_id}: {e}, skipping.")
                continue
        
        # Force garbage collection after each batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def load_videos_batch(save_dir, batch_size=10):
    """
    Load videos in batches to manage memory usage.
    """
    import gc
    from tqdm import tqdm
    
    video_files = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    videos_np = []
    
    for i in tqdm(range(0, len(video_files), batch_size), desc="Loading video batches"):
        batch_files = video_files[i:i+batch_size]
        batch_tensors = []
        
        for fname in batch_files:
            try:
                tensor = torch.load(os.path.join(save_dir, fname))
                batch_tensors.append(tensor)
            except Exception as e:
                print(f"Error loading {fname}: {e}, skipping.")
                continue
        
        if batch_tensors:
            videos_np.extend(batch_tensors)
            
        # Free up memory
        del batch_tensors
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return np.array(videos_np, dtype=object)

def get_blazepose_keypoints_from_model(image, model, device):
    # image: BGR numpy array
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # (1, 3, H, W)
    img = img.to(device)
    with torch.no_grad():
        keypoints = model(img).cpu().numpy()[0]  # (33, 3)
    return keypoints

def extract_pose_keypoints_from_videos(video_dir, out_dir, blazepose_model, device):
    for activity in activities:
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4') or f.endswith('.avi')]
        blazepose_model.eval()
        for video_file in tqdm(video_files, desc='Extracting pose from videos'):
            video_path = os.path.join(video_dir, video_file)
            out_npy_path = os.path.join(out_dir, os.path.splitext(video_file)[0] + '_pose.npy')
            cap = cv2.VideoCapture(video_path)
            keypoints_list = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                keypoints = get_blazepose_keypoints_from_model(frame, blazepose_model, device)
                keypoints_list.append(keypoints)
            cap.release()
            keypoints_arr = np.stack(keypoints_list) if keypoints_list else np.zeros((1, 33, 3), dtype=np.float32)
            np.save(out_npy_path, keypoints_arr)
            print(f"Saved pose keypoints to {out_npy_path}")
    print('All videos processed.')

def analyze_forward_head(pose):
    head = pose[0, :2]
    left_shoulder = pose[11, :2]
    right_shoulder = pose[12, :2]
    shoulder_center = (left_shoulder + right_shoulder) / 2
    return int(abs(head[0] - shoulder_center[0]) > 0.08)

def analyze_flat_back(pose):
    left_shoulder = pose[11, :2]
    right_shoulder = pose[12, :2]
    left_hip = pose[23, :2]
    right_hip = pose[24, :2]
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    return int(abs(shoulder_center[1] - hip_center[1]) < 0.08)

def analyze_sway_back(pose):
    left_shoulder = pose[11, :2]
    right_shoulder = pose[12, :2]
    left_hip = pose[23, :2]
    right_hip = pose[24, :2]
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    return int((hip_center[0] - shoulder_center[0]) > 0.08)

def analyze_rounded_shoulders(pose):
    left_shoulder = pose[11, :2]
    right_shoulder = pose[12, :2]
    left_hip = pose[23, :2]
    right_hip = pose[24, :2]
    return int(((left_shoulder[0] - left_hip[0]) > 0.08) and ((right_shoulder[0] - right_hip[0]) > 0.08))

def analyze_weak_abdominals(pose):
    left_hip = pose[23, :2]
    right_hip = pose[24, :2]
    left_knee = pose[25, :2]
    right_knee = pose[26, :2]
    return int(((left_hip[0] - left_knee[0]) > 0.08) and ((right_hip[0] - right_knee[0]) > 0.08))

def analyze_bent_knees(pose):
    left_knee = pose[25, 1]
    right_knee = pose[26, 1]
    left_ankle = pose[27, 1]
    right_ankle = pose[28, 1]
    return int(((left_knee - left_ankle) < -0.08) and ((right_knee - right_ankle) < -0.08))

def analyze_raised_chest(pose):
    chest = pose[12, 1]
    left_shoulder = pose[11, 1]
    right_shoulder = pose[12, 1]
    return int((chest - (left_shoulder + right_shoulder) / 2) < -0.08)

def analyze_bent_neck(pose):
    head = pose[0, :2]
    left_shoulder = pose[11, :2]
    right_shoulder = pose[12, :2]
    neck = (left_shoulder + right_shoulder) / 2
    v1 = head - neck
    v2 = np.array([0, -1])
    angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1))
    return int(angle > np.deg2rad(25))

class PostureMLP(nn.Module):
    def __init__(self, input_dim=66, num_labels=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_labels)
        )
    def forward(self, x):
        return self.net(x)

def test_posture_classifier(model_path, pose_dir, batch_size=8, device='cuda'):
    """
    Loads a trained PostureMLP model, predicts posture issues on a batch of pose keypoints,
    and prints both probabilities and human-readable posture issue names.
    """
    import torch
    import numpy as np
    import os
    from torch.utils.data import DataLoader
    
    dataset = PostureIssueDataset(pose_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = PostureMLP(input_dim=66, num_labels=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        for poses, labels in dataloader:
            poses = poses.to(device)
            logits = model(poses)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).cpu().numpy()
            probs = probs.cpu().numpy()
            for i in range(poses.size(0)):
                print(f"Sample {i+1}:")
                for j, issue in enumerate(posture_issues):
                    print(f"  {issue}: Probability={probs[i][j]:.2f}, Predicted={'Yes' if preds[i][j] else 'No'}")
                print("---")
            break
    print("Testing complete.")