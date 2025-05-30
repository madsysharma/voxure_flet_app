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
import json
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import hashlib
import os
import tempfile
import shutil
from pathlib import Path

sys.path.append(os.getcwd()+'/MediaPipePyTorch/')
from blazepose import *
from blazepose_landmark import *

# Lottie animation URL
LOTTIE_LOADING_URL = "https://lottie.host/36fbffbc-b690-4b5c-ac66-942fd30c52bd/ZQTN7tTIMt.lottie"

def download_lottie_animation(url, save_path):
    """
    Download Lottie animation file from URL
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the animation file
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Error downloading Lottie animation: {str(e)}")
        return False

def get_lottie_animation_path():
    """
    Get the path to the Lottie animation file, downloading it if necessary
    """
    # Create a directory for animations in the project
    anim_dir = os.path.join(os.getcwd(), 'storage', 'animations')
    os.makedirs(anim_dir, exist_ok=True)
    
    # Set the path for the animation file
    anim_path = os.path.join(anim_dir, 'loading_animation.lottie')
    
    # Download if not exists
    if not os.path.exists(anim_path):
        print("Downloading Lottie animation...")
        if not download_lottie_animation(LOTTIE_LOADING_URL, anim_path):
            return None
    
    return anim_path

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

# Define audio categories
AUDIO_CATEGORIES = [
    'GRBAS Strain',
    'breath_control',
    'agility',
    'stamina',
    'phonation',
    'resonance'
]

# Define posture categories
POSTURE_CATEGORIES = posture_issues = [
    "forward_head",
    "flat_back",
    "sway_back",
    "rounded_shoulders",
    "weak_abdominals",
    "bent_knees",
    "raised_chest",
    "bent_neck"
]

def validate_predictions(preds, expected_length, pred_type="audio"):
    """
    Validate prediction array format and values.
    
    Args:
        preds: Array of predictions
        expected_length: Expected number of predictions
        pred_type: Type of predictions ("audio" or "posture")
        
    Returns:
        numpy.ndarray: Validated predictions
    """
    if preds is None:
        raise ValueError(f"{pred_type.capitalize()} predictions cannot be None")
    
    # Convert to numpy array if not already
    if not isinstance(preds, np.ndarray):
        try:
            preds = np.array(preds, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Failed to convert {pred_type} predictions to numpy array: {str(e)}")
    
    # Check shape
    if len(preds) != expected_length:
        raise ValueError(f"Expected {expected_length} {pred_type} predictions, got {len(preds)}")
    
    # Check for NaN values
    if np.isnan(preds).any():
        raise ValueError(f"NaN values found in {pred_type} predictions")
    
    # Check for infinite values
    if np.isinf(preds).any():
        raise ValueError(f"Infinite values found in {pred_type} predictions")
    
    # Ensure values are between 0 and 1
    if (preds < 0).any() or (preds > 1).any():
        print(f"Warning: {pred_type} predictions should be between 0 and 1, clipping values")
        preds = np.clip(preds, 0, 1)
    
    return preds

def predict_from_audio_features(features):
    """
    Make predictions from audio features.
    
    Args:
        features (numpy.ndarray): Audio features array
        
    Returns:
        dict: Dictionary mapping categories to their scores and labels
    """
    try:
        # Validate features
        if not isinstance(features, np.ndarray):
            features = np.array(features, dtype=np.float32)
        
        # Ensure features are 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Make predictions (this is where you'd use your audio model)
        # For now, we'll use a simple example
        preds = np.mean(features, axis=1)  # Replace with actual model prediction
        
        # Map predictions to categories
        return map_audio_predictions(preds)
    except Exception as e:
        print(f"Error in predict_from_audio_features: {str(e)}")
        return {category: {'score': 0.0, 'label': 'Error'} for category in AUDIO_CATEGORIES}

def predict_from_posture_features(features):
    """
    Make predictions from posture features.
    
    Args:
        features (numpy.ndarray): Posture features array
        
    Returns:
        dict: Dictionary mapping categories to their scores and labels
    """
    try:
        # Validate features
        if not isinstance(features, np.ndarray):
            features = np.array(features, dtype=np.float32)
        
        # Ensure features are 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Make predictions (this is where you'd use your posture model)
        # For now, we'll use a simple example
        preds = np.mean(features, axis=1)  # Replace with actual model prediction
        
        # Map predictions to categories
        return map_posture_predictions(preds)
    except Exception as e:
        print(f"Error in predict_from_posture_features: {str(e)}")
        return {category: {'score': 0.0, 'label': 'Error'} for category in POSTURE_CATEGORIES}

def map_audio_predictions(preds_audio):
    """
    Map audio predictions to categories with scores.
    
    Args:
        preds_audio (numpy.ndarray): Array of predictions from audio model
        
    Returns:
        dict: Dictionary mapping categories to their scores and labels
    """
    try:
        # Validate predictions
        preds_audio = validate_predictions(preds_audio, len(AUDIO_CATEGORIES), "audio")
        
        results = {}
        for category, pred in zip(AUDIO_CATEGORIES, preds_audio):
            try:
                # Special handling for GRBAS Strain
                if category == 'GRBAS Strain':
                    label = 'Strained' if pred > 0.5 else 'Normal'
                else:
                    label = score_to_label(pred)
                    
                results[category] = {
                    'score': float(pred),
                    'label': label
                }
            except Exception as e:
                print(f"Warning: Error processing {category}: {str(e)}")
                results[category] = {
                    'score': 0.0,
                    'label': 'Error'
                }
        
        return results
    except Exception as e:
        print(f"Error in map_audio_predictions: {str(e)}")
        return {category: {'score': 0.0, 'label': 'Error'} for category in AUDIO_CATEGORIES}

def map_posture_predictions(preds_posture):
    """
    Map posture predictions to categories with scores.
    
    Args:
        preds_posture (numpy.ndarray): Array of predictions from posture model
        
    Returns:
        dict: Dictionary mapping categories to their scores and labels
    """
    try:
        # Validate predictions
        preds_posture = validate_predictions(preds_posture, len(POSTURE_CATEGORIES), "posture")
        
        results = {}
        for category, pred in zip(POSTURE_CATEGORIES, preds_posture):
            try:
                results[category] = {
                    'score': float(pred),
                    'label': 'Issue Detected' if pred > 0.5 else 'Good'
                }
            except Exception as e:
                print(f"Warning: Error processing {category}: {str(e)}")
                results[category] = {
                    'score': 0.0,
                    'label': 'Error'
                }
        
        return results
    except Exception as e:
        print(f"Error in map_posture_predictions: {str(e)}")
        return {category: {'score': 0.0, 'label': 'Error'} for category in POSTURE_CATEGORIES}

def analyze_features(audio_features, posture_features):
    """
    Analyze both audio and posture features.
    
    Args:
        audio_features (numpy.ndarray): Audio features array
        posture_features (numpy.ndarray): Posture features array
        
    Returns:
        dict: Combined analysis with audio and posture results
    """
    try:
        audio_results = predict_from_audio_features(audio_features)
        posture_results = predict_from_posture_features(posture_features)
        
        return {
            'audio_analysis': audio_results,
            'posture_analysis': posture_results,
            'timestamp': time.time(),
            'status': 'success'
        }
    except Exception as e:
        print(f"Error in analyze_features: {str(e)}")
        return {
            'audio_analysis': {category: {'score': 0.0, 'label': 'Error'} for category in AUDIO_CATEGORIES},
            'posture_analysis': {category: {'score': 0.0, 'label': 'Error'} for category in POSTURE_CATEGORIES},
            'timestamp': time.time(),
            'status': 'error',
            'error_message': str(e)
        }

def get_predictions_regression(model, data, device, categories):
    """
    Get predictions from a regression model.
    
    Args:
        model: PyTorch model
        data: Input data
        device: Device to run inference on
        categories: List of category names
        
    Returns:
        numpy.ndarray: Array of predictions
    """
    try:
        model.eval()
        print("\nTest set predictions (predicted | true):")
        with torch.no_grad():
            data = data.to(device)
            outputs = model(data)
            preds = outputs.cpu().numpy()
            
            # Validate predictions
            preds = validate_predictions(preds, len(categories))
            
            # Print predictions
            for i in range(preds.shape[0]):
                pred = preds[i]
                pred_str = ', '.join([f"{cat}: {score_to_label(p)} ({p:.2f})" for cat, p in zip(categories, pred)])
                print(pred_str)
            
            return preds
    except Exception as e:
        print(f"Error in get_predictions_regression: {str(e)}")
        return np.zeros(len(categories), dtype=np.float32)

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

def analyze_frame_realtime(frame, blazepose_model, posture_model, device):
    """
    Analyze a single frame in real-time and return posture issues and timestamps.
    Returns: dict with timestamp and detected issues
    """
    # Get pose keypoints
    keypoints = get_blazepose_keypoints_from_model(frame, blazepose_model, device)
    
    # Analyze posture issues
    issues = {
        "forward_head": analyze_forward_head(keypoints),
        "flat_back": analyze_flat_back(keypoints),
        "sway_back": analyze_sway_back(keypoints),
        "rounded_shoulders": analyze_rounded_shoulders(keypoints),
        "weak_abdominals": analyze_weak_abdominals(keypoints),
        "bent_knees": analyze_bent_knees(keypoints),
        "raised_chest": analyze_raised_chest(keypoints),
        "bent_neck": analyze_bent_neck(keypoints)
    }
    
    # Get ML model predictions
    pose_flat = torch.tensor(keypoints[:, :2].flatten(), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = posture_model(pose_flat)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).cpu().numpy()[0]
    
    # Combine manual analysis with ML predictions
    detected_issues = []
    for i, (issue, detected) in enumerate(issues.items()):
        if detected or preds[i]:
            detected_issues.append(issue)
    
    return {
        "timestamp": time.time(),
        "issues": detected_issues,
        "confidence_scores": probs.cpu().numpy()[0]
    }

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
    # Define the activities/labels we want to process
    activities = ['beatboxing', 'busking', 'playing clarinet', 'playing drums', 'playing flute', 
                 'playing guitar', 'playing piano', 'playing recorder', 'playing saxophone', 
                 'playing trombone', 'playing violin', 'recording music', 'singing', 'whistling']
    
    blazepose_model.eval()
    
    for activity in activities:
        # Handle both space and underscore versions of the label
        activity_space = activity
        activity_underscore = activity.replace(" ", "_")
        
        # Check both possible directory paths
        possible_dirs = [
            os.path.join(video_dir, activity_space),
            os.path.join(video_dir, activity_underscore)
        ]
        
        for dir_path in possible_dirs:
            if not os.path.exists(dir_path):
                continue
                
            # Get all video files in the directory
            video_files = [f for f in os.listdir(dir_path) if f.endswith('.mp4') or f.endswith('.avi')]
            
            for video_file in tqdm(video_files, desc=f'Extracting pose from {activity} videos'):
                video_path = os.path.join(dir_path, video_file)
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

def inspect_mpii_mat_file(mat_file):
    """
    Inspect the structure of an MPII .mat file to understand its contents.
    """
    import scipy.io as sio
    print(f"Loading MPII mat file: {mat_file}")
    mat_data = sio.loadmat(mat_file)
    
    print("\nAvailable fields in the mat file:")
    for key in mat_data.keys():
        if not key.startswith('__'):  # Skip internal fields
            print(f"- {key}")
            if isinstance(mat_data[key], np.ndarray):
                print(f"  Shape: {mat_data[key].shape}")
                print(f"  Type: {mat_data[key].dtype}")
    
    return mat_data

def save_keypoints_cache(keypoints_dir, output_archive):
    """
    Save all processed keypoints to a single archive file.
    
    Args:
        keypoints_dir (str): Directory containing the processed keypoints (.npy files)
        output_archive (str): Path to save the archive file (.zip)
    """
    print(f"\nSaving keypoints cache to {output_archive}...")
    
    # Create a temporary directory for organizing files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a manifest file to store metadata
        manifest = {
            'timestamp': time.time(),
            'files': []
        }
        
        # Copy and organize all .npy files
        for file in tqdm(os.listdir(keypoints_dir), desc="Organizing files"):
            if file.endswith('.npy'):
                src_path = os.path.join(keypoints_dir, file)
                dst_path = os.path.join(temp_dir, file)
                shutil.copy2(src_path, dst_path)
                manifest['files'].append({
                    'name': file,
                    'size': os.path.getsize(src_path),
                    'modified': os.path.getmtime(src_path)
                })
        
        # Save manifest
        manifest_path = os.path.join(temp_dir, 'manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Create zip archive
        shutil.make_archive(output_archive.replace('.zip', ''), 'zip', temp_dir)
    
    print(f"Cache saved successfully to {output_archive}")
    return manifest

def load_keypoints_cache(archive_path, output_dir):
    """
    Load keypoints from a cache archive file.
    
    Args:
        archive_path (str): Path to the cache archive file (.zip)
        output_dir (str): Directory to extract the keypoints to
    
    Returns:
        dict: Manifest information about the loaded cache
    """
    print(f"\nLoading keypoints cache from {archive_path}...")
    
    # Get Lottie animation path
    lottie_path = get_lottie_animation_path()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract archive
        shutil.unpack_archive(archive_path, temp_dir, 'zip')
        
        # Load manifest
        manifest_path = os.path.join(temp_dir, 'manifest.json')
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Copy files to output directory
        for file_info in tqdm(manifest['files'], desc="Extracting files"):
            src_path = os.path.join(temp_dir, file_info['name'])
            dst_path = os.path.join(output_dir, file_info['name'])
            shutil.copy2(src_path, dst_path)
    
    print(f"Cache loaded successfully to {output_dir}")
    print(f"Loaded {len(manifest['files'])} keypoint files")
    print(f"Cache created on: {time.ctime(manifest['timestamp'])}")
    
    return manifest

def parse_keyframes_format(annotations, images_dir, out_dir):
    """
    Parse MPII dataset annotations and process keyframes.
    
    Args:
        annotations: List of annotation dictionaries
        images_dir: Directory containing the images
        out_dir: Directory to save processed keypoints
    """
    def scan_directory(directory):
        """Scan directory for image files and return a dictionary of available images."""
        image_files = {}
        for ext in ['.jpg', '.jpeg', '.png']:
            for file in glob.glob(os.path.join(directory, f'*{ext}')):
                base_name = os.path.splitext(os.path.basename(file))[0]
                image_files[base_name] = file
        return image_files

    def get_cache_path(root_dir, cache_type='scan'):
        """Get path for cache file."""
        cache_dir = os.path.join(root_dir, '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f'{cache_type}_cache.json')

    def save_to_cache(cache_path, data, timestamp):
        """Save data to cache file."""
        cache_data = {
            'timestamp': timestamp,
            'data': data
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)

    def load_from_cache(cache_path, max_age_seconds=3600):
        """Load data from cache file if it exists and is not too old."""
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            if time.time() - cache_data['timestamp'] > max_age_seconds:
                return None
            
            return cache_data['data']
        except Exception as e:
            print(f"Error loading cache: {str(e)}")
            return None

    def scan_subdirectories(root_dir):
        """Scan all subdirectories for images and return a dictionary of available images."""
        cache_path = get_cache_path(root_dir)
        cached_data = load_from_cache(cache_path)
        
        if cached_data is not None:
            return cached_data
        
        image_files = {}
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    base_name = os.path.splitext(file)[0]
                    full_path = os.path.join(root, file)
                    image_files[base_name] = full_path
        
        save_to_cache(cache_path, image_files, time.time())
        return image_files

    def get_keypoint_cache_path(annotation_idx, sequence_info, frame_number, base_name):
        """Get path for keypoint cache file."""
        cache_dir = os.path.join(out_dir, '.keypoint_cache')
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f'{annotation_idx}_{sequence_info}_{frame_number}_{base_name}.npy')

    def check_keypoint_cache(cache_path):
        """Check if keypoint cache exists and is valid."""
        if not os.path.exists(cache_path):
            return False
        try:
            data = np.load(cache_path)
            return data is not None and len(data) > 0
        except Exception:
            return False

    def process_single_annotation(idx):
        """Process a single annotation."""
        try:
            annotation = annotations[idx]
            
            # Debug: Print annotation structure
            print(f"\nDebug - Annotation {idx} structure:")
            print(f"Type: {type(annotation)}")
            print(f"Content: {annotation}")
            
            # Handle different possible annotation formats
            try:
                if isinstance(annotation, str):
                    # If annotation is directly a string (image name)
                    image_name = annotation
                elif isinstance(annotation, dict):
                    # If annotation is a dictionary
                    image_name = annotation.get('image', '')
                elif isinstance(annotation, np.ndarray):
                    # If annotation is a numpy array (MATLAB struct)
                    if annotation.dtype.names is not None and 'image' in annotation.dtype.names:
                        img_info = annotation['image']
                        if isinstance(img_info, np.ndarray) and img_info.size > 0:
                            img_info = img_info[0]
                            if isinstance(img_info, np.ndarray) and img_info.size > 0:
                                img_info = img_info[0]
                                if isinstance(img_info, np.ndarray) and 'name' in img_info.dtype.names:
                                    name_info = img_info['name']
                                    if isinstance(name_info, np.ndarray) and name_info.size > 0:
                                        name_info = name_info[0]
                                        if isinstance(name_info, np.ndarray) and name_info.size > 0:
                                            name_info = name_info[0]
                                            if isinstance(name_info, np.ndarray) and name_info.size > 0:
                                                image_name = name_info[0]
                                            else:
                                                image_name = str(name_info)
                                        else:
                                            image_name = str(name_info)
                                    else:
                                        image_name = str(name_info)
                                else:
                                    image_name = str(img_info)
                            else:
                                image_name = str(img_info)
                        else:
                            image_name = str(img_info)
                    else:
                        image_name = str(annotation)
                else:
                    # Try to convert to string as last resort
                    image_name = str(annotation)
                
            except Exception as e:
                print(f"Warning: Error extracting image name for annotation {idx}: {str(e)}")
                return
            
            if not image_name:
                print(f"Warning: No image name found for annotation {idx}")
                return
            
            # Try to find the image file
            base_name = os.path.splitext(image_name)[0]
            image_path = None
            
            # First try direct path
            direct_path = os.path.join(images_dir, image_name)
            if os.path.exists(direct_path):
                image_path = direct_path
            else:
                # Try to find the image in subdirectories
                available_images = scan_subdirectories(images_dir)
                
                # Try different variations of the image name
                possible_names = [
                    base_name,  # Original name
                    base_name.lstrip('0'),  # Remove leading zeros
                    base_name.zfill(5),  # Pad with zeros to 5 digits
                    base_name.zfill(6),  # Pad with zeros to 6 digits
                    base_name.zfill(7),  # Pad with zeros to 7 digits
                ]
                
                # Try each possible name
                for name in possible_names:
                    if name in available_images:
                        image_path = available_images[name]
                        break
                
                # If still not found, try with different extensions
                if image_path is None:
                    for name in possible_names:
                        for ext in ['.jpg', '.jpeg', '.png']:
                            full_name = name + ext
                            if full_name in available_images:
                                image_path = available_images[full_name]
                                break
                        if image_path is not None:
                            break
            
            if image_path is None:
                print(f"Warning: Could not find image for annotation {idx}. Tried variations of: {base_name}")
                return
            
            # Process the image
            try:
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Warning: Could not read image: {image_path}")
                    return
                
                # Save the processed image as numpy array
                output_path = os.path.join(out_dir, f"{base_name}.npy")
                np.save(output_path, img)
                
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                
        except Exception as e:
            print(f"Error processing annotation {idx}: {str(e)}")

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Process all annotations
    print(f"Processing {len(annotations)} annotations...")
    for idx in tqdm(range(len(annotations))):
        process_single_annotation(idx)

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
    return preds

def load_critiques():
    """
    Load critiques from the JSON file containing blog content
    Returns: dict with categorized critiques
    """
    try:
        with open('storage/data/vocal_critiques.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return default critiques if file doesn't exist
        return {
            "posture": {
                "forward_head": "Try to keep your head aligned with your shoulders. This helps maintain proper vocal alignment.",
                "flat_back": "Maintain natural curve in your lower back. This supports better breath control.",
                "sway_back": "Keep your hips aligned with your shoulders. This helps with overall body balance.",
                "rounded_shoulders": "Roll your shoulders back and down. This opens up your chest for better breathing.",
                "weak_abdominals": "Engage your core muscles. This provides better support for your voice.",
                "bent_knees": "Straighten your knees slightly. This helps maintain proper body alignment.",
                "raised_chest": "Lower your chest to a neutral position. This allows for more natural breathing.",
                "bent_neck": "Keep your neck straight and aligned. This helps prevent vocal strain."
            },
            "vocal_technique": {
                "breath_support": "Focus on deep, diaphragmatic breathing. This provides better breath support.",
                "tension": "Try to release any tension in your jaw and neck. This allows for freer vocal production.",
                "tone": "Work on maintaining a clear, focused tone. This helps with vocal clarity.",
                "pitch": "Pay attention to your pitch accuracy. This helps with overall vocal control."
            },
            "positive": "Excellent form! Your posture and vocal technique are well-aligned. Keep up the good work!"
        }

def generate_realtime_critique(analysis_result):
    """
    Generate human-readable critique from analysis results using blog content
    """
    if not analysis_result["issues"]:
        return load_critiques()["positive"]
    
    critiques = []
    blog_critiques = load_critiques()
    
    for issue in analysis_result["issues"]:
        if issue in blog_critiques["posture"]:
            critiques.append(blog_critiques["posture"][issue])
        elif issue in blog_critiques["vocal_technique"]:
            critiques.append(blog_critiques["vocal_technique"][issue])
    
    if not critiques:
        return "Good posture and vocal technique! Keep it up!"
    
    return "\n".join(critiques)

def process_live_recording(video_capture, blazepose_model, posture_model, device, update_callback=None):
    """
    Process live video feed and provide real-time feedback
    Args:
        video_capture: cv2.VideoCapture object
        blazepose_model: BlazePose model for keypoint detection
        posture_model: PostureMLP model for posture analysis
        device: torch device
        update_callback: Function to call with critique updates (timestamp, critique)
    """
    frame_count = 0
    last_critique_time = 0
    critique_interval = 10.0  # Update critique every 10 seconds
    last_issues = set()  # Keep track of last reported issues
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
            
        frame_count += 1
        current_time = time.time()
        
        # Analyze frame
        analysis = analyze_frame_realtime(frame, blazepose_model, posture_model, device)
        
        # Only update critique if:
        # 1. It's been 10 seconds since the last update
        # 2. There are new issues detected
        # 3. The issues are different from the last reported ones
        if (current_time - last_critique_time >= critique_interval and 
            analysis["issues"] and 
            set(analysis["issues"]) != last_issues):
            
            critique = generate_realtime_critique(analysis)
            if update_callback:
                update_callback(current_time, critique)
            last_critique_time = current_time
            last_issues = set(analysis["issues"])
            
        yield frame, analysis

def process_mpii_video_sequences(mat_file_path, images_dir, output_dir):
    """
    Helper function to process MPII video sequences for vocal coaching.
    
    Args:
        mat_file_path (str): Path to mpii_human_pose_v1_sequences_keyframes.mat
        images_dir (str): Directory containing the MPII video frames
        output_dir (str): Directory to save processed keypoints
    
    Returns:
        dict: Statistics about processed sequences
    """
    print("Step 0: Processing MPII video sequences...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the annotations
    processed_annotations = parse_keyframes_format(mat_file_path, images_dir, output_dir)
    
    # Calculate statistics
    stats = {
        'total_sequences': len(set(ann['sequence_id'] for ann in processed_annotations if ann['sequence_id'] is not None)),
        'total_frames': len(processed_annotations),
        'sequences': {}
    }
    
    # Group by sequence
    for ann in processed_annotations:
        if ann['sequence_id'] is not None:
            seq_id = ann['sequence_id']
            if seq_id not in stats['sequences']:
                stats['sequences'][seq_id] = []
            stats['sequences'][seq_id].append(ann)
    
    # Print summary
    print("\nProcessing Summary:")
    print(f"Total sequences processed: {stats['total_sequences']}")
    print(f"Total frames processed: {stats['total_frames']}")
    print("\nSequence details:")
    for seq_id, frames in stats['sequences'].items():
        print(f"Sequence {seq_id}: {len(frames)} frames")
    
    return stats

# Process the sequences
#stats = process_mpii_video_sequences(mat_file, images_dir, output_dir)

# --------- MAIN PIPELINE ---------
if __name__ == "__main__":
    # Define paths
    base_dir = os.getcwd()
    mpii_images_dir = os.path.join(base_dir, 'mpii_human_pose_v1', 'images')
    mpii_keypoints_dir = os.path.join(base_dir, 'mpii_human_pose_v1', 'keypoints')
    kinetics_video_dir = os.path.join(base_dir, 'kinetics_videos')
    pose_out_dir = os.path.join(base_dir, 'pose_keypoints')
    classifier_pose_dir = os.path.join(base_dir, 'classifier_pose_data')
    blazepose_weights_path = os.path.join(base_dir, 'blazepose_weights.pth')
    
    # Create necessary directories
    for dir_path in [mpii_keypoints_dir, pose_out_dir, classifier_pose_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Step 0: Parse MPII .mat annotation file to extract keypoints as .npy
    print("Step 0: Inspecting MPII .mat file structure and parsing MPII .mat annotation file...")
    mpii_mat_file = os.path.join(base_dir, 'mpii_human_pose_v1_sequences_keyframes.mat')
    
    # Load and inspect the mat file
    mat_data = inspect_mpii_mat_file(mpii_mat_file)
    
    # Print the structure of the mat file for debugging
    print("\nMat file structure:")
    for key in mat_data.keys():
        if not key.startswith('__'):
            print(f"\nKey: {key}")
            if isinstance(mat_data[key], np.ndarray):
                print(f"Shape: {mat_data[key].shape}")
                print(f"Type: {mat_data[key].dtype}")
                if mat_data[key].dtype.names is not None:
                    print("Fields:", mat_data[key].dtype.names)
    
    # Try to find the annotations in the mat file
    annotations = None
    if 'annolist' in mat_data:
        annotations = mat_data['annolist']
    elif 'RELEASE' in mat_data:
        try:
            annotations = mat_data['RELEASE'][0][0]['annolist'][0]
        except (KeyError, IndexError) as e:
            print(f"Error accessing RELEASE structure: {str(e)}")
    else:
        # Try to find any array that might contain annotations
        for key in mat_data.keys():
            if not key.startswith('__'):
                if isinstance(mat_data[key], np.ndarray):
                    print(f"\nTrying key: {key}")
                    try:
                        if mat_data[key].dtype.names is not None and 'image' in mat_data[key].dtype.names:
                            annotations = mat_data[key]
                            print(f"Found annotations in key: {key}")
                            break
                    except Exception as e:
                        print(f"Error checking key {key}: {str(e)}")
    
    if annotations is None:
        raise ValueError("Could not find annotations in the mat file. Please check the file structure.")
    
    print(f"\nFound annotations with shape: {annotations.shape if hasattr(annotations, 'shape') else 'unknown'}")
    
    # Process the annotations
    parse_keyframes_format(annotations, mpii_images_dir, mpii_keypoints_dir)
    
    # 1. Finetune BlazePose on MPII
    print("Step 1: Finetuning BlazePose on MPII...")
    blazepose_model = train_blazepose_on_mpii(
        mpii_images_dir, mpii_keypoints_dir,
        num_epochs=20, batch_size=32, lr=1e-4, weight_decay=1e-4, patience=5, device=device,
        save_path=blazepose_weights_path
    )
    
    # 2. Use trained BlazePose to extract pose keypoints from new videos
    print("Step 2: Extracting pose keypoints from new videos...")
    blazepose_model.eval()
    extract_pose_keypoints_from_videos(kinetics_video_dir, pose_out_dir, blazepose_model, device)
    
    # 3. Train posture classifier on extracted keypoints
    print("Step 3: Training posture classifier on extracted keypoints...")
    train_posture_classifier(classifier_pose_dir, num_epochs=20, batch_size=32, lr=1e-3, weight_decay=1e-4, patience=5, device=device)