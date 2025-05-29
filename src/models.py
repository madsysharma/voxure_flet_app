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
from moviepy.video.io.VideoFileClip import VideoFileClip
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
    if augment_func is not None:
        y = augment_func(y, sr) if 'sr' in augment_func.__code__.co_varnames else augment_func(y)
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
        if specaug:
            mfcc = spec_augment(mfcc)
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
num_labels = len(posture_issues)

MPII_TO_BLAZEPOSE = {
    0: 28, 1: 26, 2: 24, 3: 23, 4: 25, 5: 27, 6: 0, 7: 12, 8: 0, 9: 0,
    10: 16, 11: 14, 12: 12, 13: 11, 14: 13, 15: 15,
}
MAPPED_INDICES = list(MPII_TO_BLAZEPOSE.values())

def extract_frames(out_path):
    cap = cv2.VideoCapture(out_path)
    trimmed_vid_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_frame = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        trimmed_vid_frames.append(tensor_frame)
    cap.release()

    if not trimmed_vid_frames:
        print(f"No frames extracted for {video_id}")
        return None
    
    return torch.stack(trimmed_vid_frames)

def get_blazepose_keypoints_from_model(image, model, device):
    # image: BGR numpy array
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # (1, 3, H, W)
    img = img.to(device)
    with torch.no_grad():
        keypoints = model(img).cpu().numpy()[0]  # (33, 3)
    return keypoints

class MPIIBlazePoseDataset(TensorDataset):
    def __init__(self, images_dir, keypoints_dir, transform=None):
        self.images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')])
        self.keypoints = sorted([os.path.join(keypoints_dir, f) for f in os.listdir(keypoints_dir) if f.endswith('.npy')])
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        kpts = np.load(self.keypoints[idx])  # (16, 3)
        blaze_kpts = np.zeros((33, 3), dtype=np.float32)
        for mpii_idx, blaze_idx in MPII_TO_BLAZEPOSE.items():
            blaze_kpts[blaze_idx] = kpts[mpii_idx]
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, torch.from_numpy(blaze_kpts)

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

class PostureIssueDataset(TensorDataset):
    def __init__(self, pose_dir):
        self.pose_files = [f for f in os.listdir(pose_dir) if f.endswith('.npy')]
        self.pose_dir = pose_dir
    def __len__(self):
        return len(self.pose_files)
    def __getitem__(self, idx):
        pose = np.load(os.path.join(self.pose_dir, self.pose_files[idx]))
        pose = pose.astype(np.float32)
        labels = np.zeros(num_labels, dtype=np.float32)
        labels[0] = analyze_forward_head(pose)
        labels[1] = analyze_flat_back(pose)
        labels[2] = analyze_sway_back(pose)
        labels[3] = analyze_rounded_shoulders(pose)
        labels[4] = analyze_weak_abdominals(pose)
        labels[5] = analyze_bent_knees(pose)
        labels[6] = analyze_raised_chest(pose)
        labels[7] = analyze_bent_neck(pose)
        pose_flat = pose[:, :2].flatten()
        return torch.tensor(pose_flat, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

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

def parse_mpii_annotations(mat_file, images_dir, out_dir):
    """
    Parse the MPII .mat annotation file and save keypoints for each image as .npy files in out_dir.
    """
    mat = scipy.io.loadmat(mat_file)
    annolist = mat['RELEASE']['annolist'][0,0][0]
    os.makedirs(out_dir, exist_ok=True)
    for rec in annolist:
        img_name = rec['image']['name'][0,0][0]
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            continue
        if rec['annorect'].size == 0:
            continue
        for person in rec['annorect'][0]:
            if 'annopoints' not in person.dtype.fields or person['annopoints'].size == 0:
                continue
            keypoints = np.zeros((16, 3), dtype=np.float32)  # 16 keypoints
            if person['annopoints'][0,0].size > 0:
                for pt in person['annopoints'][0,0]['point'][0]:
                    idx = int(pt['id'][0,0])
                    x = float(pt['x'][0,0])
                    y = float(pt['y'][0,0])
                    v = 2  # visible
                    keypoints[idx] = [x, y, v]
            base = os.path.splitext(os.path.basename(img_path))[0]
            np.save(os.path.join(out_dir, f"{base}_keypoints.npy"), keypoints)
    print(f"Saved keypoints to {out_dir}")

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