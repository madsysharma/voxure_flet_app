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

def parse_mpii_annotations(mat_file, images_dir, out_dir):
    """
    Parse MPII annotations and convert to a more usable format.
    Optimized version with error handling and parallel processing.
    """
    import scipy.io as sio
    import numpy as np
    import os
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing

    print("Loading MPII annotations...")
    mat_data = sio.loadmat(mat_file)
    annotations = mat_data['RELEASE'][0][0]
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Get number of CPU cores for parallel processing
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    
    def process_single_annotation(idx):
        try:
            rec = annotations['annolist'][0][idx]
            
            # Safely get image name
            try:
                img_name = rec['image']['name'][0][0][0]
            except (IndexError, KeyError):
                print(f"Warning: Invalid image name structure for annotation {idx}")
                return None
                
            img_path = os.path.join(images_dir, img_name)
            
            # Skip if image doesn't exist
            if not os.path.exists(img_path):
                return None
            
            # Initialize keypoints array
            keypoints = np.zeros((16, 3), dtype=np.float32)
            
            # Process each person in the image
            if 'annorect' in rec.dtype.fields and rec['annorect'].size > 0:
                person = rec['annorect'][0][0]  # Get first person
                
                # Check if person has valid annotations
                if (person is not None and 
                    'annopoints' in person.dtype.fields and 
                    person['annopoints'].size > 0 and 
                    person['annopoints'][0][0] is not None):
                    
                    points = person['annopoints'][0][0]
                    if 'point' in points.dtype.fields:
                        # Handle both single point and multiple points cases
                        point_data = points['point']
                        if point_data.size == 1:  # Single point case
                            point_data = [point_data[0]]
                        
                        for point in point_data:
                            try:
                                if point.size > 0:
                                    # Extract point data with proper array indexing
                                    x = float(point['x'][0][0][0])
                                    y = float(point['y'][0][0][0])
                                    idx = int(point['id'][0][0][0])
                                    is_visible = float(point['is_visible'][0][0][0]) if 'is_visible' in point.dtype.fields else 1.0
                                    
                                    # Validate coordinates
                                    if np.isnan(x) or np.isnan(y):
                                        continue
                                        
                                    # Ensure index is valid and update keypoints
                                    if 0 <= idx < 16:
                                        keypoints[idx] = [x, y, is_visible]
                                    else:
                                        print(f"Warning: Invalid keypoint index {idx} in annotation {idx}")
                            except (IndexError, KeyError, ValueError) as e:
                                # Only print warning once per annotation
                                if not hasattr(process_single_annotation, 'warned_annotations'):
                                    process_single_annotation.warned_annotations = set()
                                if idx not in process_single_annotation.warned_annotations:
                                    print(f"Warning: Invalid point data in annotation {idx}: {str(e)}")
                                    process_single_annotation.warned_annotations.add(idx)
                                continue
            
            # Validate keypoints before saving
            if np.any(np.isnan(keypoints)):
                print(f"Warning: NaN values found in keypoints for annotation {idx}")
                return None
                
            # Save keypoints
            out_path = os.path.join(out_dir, os.path.splitext(img_name)[0] + '.npy')
            np.save(out_path, keypoints)
            return img_name
            
        except Exception as e:
            print(f"Error processing annotation {idx}: {str(e)}")
            return None
    
    print(f"Processing annotations using {num_cores} cores...")
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        # Process annotations in parallel with progress bar
        results = list(tqdm(
            executor.map(process_single_annotation, range(len(annotations['annolist'][0]))),
            total=len(annotations['annolist'][0]),
            desc="Processing annotations"
        ))
    
    # Filter out None results and print summary
    successful = [r for r in results if r is not None]
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(successful)}/{len(results)} annotations")
    
    return successful

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