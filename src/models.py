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

class MPIIBlazePoseDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, keypoints_dir):
        self.images_dir = os.path.abspath(images_dir)
        self.keypoints_dir = os.path.abspath(keypoints_dir)
        
        print(f"Images directory: {self.images_dir}")
        print(f"Keypoints directory: {self.keypoints_dir}")
        
        # Check if directories exist
        if not os.path.exists(self.images_dir):
            raise ValueError(f"Images directory does not exist: {self.images_dir}")
        if not os.path.exists(self.keypoints_dir):
            raise ValueError(f"Keypoints directory does not exist: {self.keypoints_dir}")
        
        # Get all image files
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
        print(f"Found {len(self.image_files)} image files")
        
        # Get all keypoint files
        keypoint_files = [f for f in os.listdir(self.keypoints_dir) if f.endswith('.npy')]
        print(f"Found {len(keypoint_files)} keypoint files")
        
        # Create a set of keypoint filenames for faster lookup
        keypoint_set = set(keypoint_files)
        
        # Validate that corresponding keypoint files exist
        self.valid_pairs = []
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            keypoint_name = f"{base_name}.npy"
            
            if keypoint_name in keypoint_set:
                self.valid_pairs.append((img_file, keypoint_name))
            else:
                print(f"Warning: No matching keypoint file found for {img_file}")
                print(f"Expected keypoint file: {keypoint_name}")
                # Try to find the file directly
                keypoint_path = os.path.join(self.keypoints_dir, keypoint_name)
                if os.path.exists(keypoint_path):
                    print(f"File exists but wasn't found in directory listing: {keypoint_path}")
                else:
                    print(f"File does not exist: {keypoint_path}")
        
        if not self.valid_pairs:
            raise ValueError(f"No valid image-keypoint pairs found in {self.images_dir} and {self.keypoints_dir}")
        
        print(f"Found {len(self.valid_pairs)} valid image-keypoint pairs")
        
    def __len__(self):
        return len(self.valid_pairs)
        
    def __getitem__(self, idx):
        img_name, keypoint_name = self.valid_pairs[idx]
        img_path = os.getcwd() + '/images/' + img_name
        keypoint_path = os.getcwd() + '/human_pose_keypoints/' + keypoint_name
        
        # Verify files exist before loading
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not os.path.exists(keypoint_path):
            raise FileNotFoundError(f"Keypoint file not found: {keypoint_path}")
        
        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Load keypoints
        try:
            keypoint_data = np.load(keypoint_path)
            print(f"Raw keypoint data shape: {keypoint_data.shape}")  # Debug print
            
            # Extract components from the flattened array
            # Format: [keypoints, visibility, scale, center]
            num_joints = 16  # MPII has 16 joints
            
            # Extract keypoints (first 32 values: 16 joints * 2 coordinates)
            keypoints = keypoint_data[:32].reshape(num_joints, 2)
            
            # Extract visibility (next 16 values)
            visibility = keypoint_data[32:48]
            
            # Extract scale (next 1 value)
            scale = keypoint_data[48]
            
            # Extract center (last 2 values)
            center = keypoint_data[49:51]
            
            # Create (16, 3) array with (x,y,visibility) for each keypoint
            keypoints = np.column_stack([keypoints, visibility])
            
            # Normalize coordinates to [0,1] range using scale and center
            keypoints = keypoints.astype(np.float32)
            
            # Apply scale and center normalization
            keypoints[:, 0] = (keypoints[:, 0] - center[0]) / (scale * img.shape[1])
            keypoints[:, 1] = (keypoints[:, 1] - center[1]) / (scale * img.shape[0])
            
            # Pad to 33 keypoints (BlazePose format)
            padding = np.zeros((33 - num_joints, 3), dtype=np.float32)
            keypoints = np.concatenate([keypoints, padding], axis=0)
            
            # Convert to tensor and add batch dimension
            keypoints_tensor = torch.from_numpy(keypoints).float()
            keypoints_tensor = keypoints_tensor.unsqueeze(0)  # Add batch dimension
            
            print(f"Final keypoints tensor shape: {keypoints_tensor.shape}")  # Debug print
            
        except Exception as e:
            print(f"Error processing keypoints for {keypoint_path}: {str(e)}")
            print(f"Keypoint data: {keypoint_data}")  # Print the actual data for debugging
            raise
        
        return img_tensor, keypoints_tensor

# --------- Loss Function (only on mapped indices) ---------
criterion = nn.MSELoss()
def compute_loss(pred, target):
    # Debug prints for input types
    print(f"Pred type: {type(pred)}, Target type: {type(target)}")
    if hasattr(pred, 'shape'):
        print(f"Pred shape: {pred.shape}")
    if hasattr(target, 'shape'):
        print(f"Target shape: {target.shape}")
    
    # Ensure inputs are tensors
    if not isinstance(pred, torch.Tensor):
        if isinstance(pred, (list, tuple)):
            pred = torch.stack(pred)
        else:
            pred = torch.tensor(pred, device=target.device)
    
    if not isinstance(target, torch.Tensor):
        if isinstance(target, (list, tuple)):
            target = torch.stack(target)
        else:
            target = torch.tensor(target, device=pred.device)
    
    # Ensure same device
    if pred.device != target.device:
        target = target.to(pred.device)
    
    # Print shapes for debugging
    print(f"Final Pred shape: {pred.shape}, Target shape: {target.shape}")
    
    # Reshape model output to match target shape
    # Model outputs [batch, num_keypoints, 12] but we need [batch, num_keypoints, 3]
    if pred.shape[-1] == 12:
        # Take first 3 channels (x, y, confidence)
        pred = pred[..., :3]
    
    # Ensure MAPPED_INDICES are within bounds
    if max(MAPPED_INDICES) >= pred.shape[1] or max(MAPPED_INDICES) >= target.shape[1]:
        raise ValueError(f"MAPPED_INDICES {MAPPED_INDICES} out of bounds for shapes {pred.shape}, {target.shape}")
    
    # Select only the mapped indices
    pred_mapped = pred[:, MAPPED_INDICES, :]
    target_mapped = target[:, MAPPED_INDICES, :]
    
    print(f"Mapped shapes - Pred: {pred_mapped.shape}, Target: {target_mapped.shape}")
    
    return criterion(pred_mapped, target_mapped)

# --------- Training Loop with LR Scheduler, Weight Decay, Early Stopping ---------
def train_blazepose_on_mpii(num_epochs=20, batch_size=8, lr=1e-4, weight_decay=1e-4, patience=5, device='cuda'):
    """
    Train BlazePose model on MPII dataset with memory-efficient processing.
    """
    # Create dataset
    full_dataset = MPIIBlazePoseDataset()
    print(f"Full dataset size: {len(full_dataset)}")
    
    # Split dataset into train, validation, and test sets
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model and optimizer
    model = BlazePose().to(device)
    model.train()
    
    # Use gradient accumulation to simulate larger batch size
    accumulation_steps = 4  # Accumulate gradients for 4 batches
    effective_batch_size = batch_size * accumulation_steps
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Initialize mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    # Training loop variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        # Reset gradients at start of epoch
        optimizer.zero_grad()
        
        for batch_idx, (imgs, kpts) in enumerate(train_pbar):
            # Move data to device
            imgs = imgs.to(device, non_blocking=True)
            kpts = kpts.to(device, non_blocking=True)
            
            # Remove extra dimension from keypoints if present
            if len(kpts.shape) == 4:  # [batch, 1, 33, 3]
                kpts = kpts.squeeze(1)  # [batch, 33, 3]
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                # Handle list output from model
                if isinstance(preds, list):
                    preds = preds[0]  # Take first element if it's a list
                loss = compute_loss(preds, kpts)
                # Normalize loss by accumulation steps
                loss = loss / accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Update weights if we've accumulated enough gradients
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Update statistics
            train_loss += loss.item() * effective_batch_size
            train_pbar.set_postfix({'loss': loss.item() * accumulation_steps})
            
            # Clear memory
            del imgs, kpts, preds, loss
            torch.cuda.empty_cache()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for imgs, kpts in val_pbar:
                # Move data to device
                imgs = imgs.to(device, non_blocking=True)
                kpts = kpts.to(device, non_blocking=True)
                
                # Remove extra dimension from keypoints if present
                if len(kpts.shape) == 4:
                    kpts = kpts.squeeze(1)
                
                # Forward pass
                preds = model(imgs)
                if isinstance(preds, list):
                    preds = preds[0]
                loss = compute_loss(preds, kpts)
                
                # Update statistics
                val_loss += loss.item() * imgs.size(0)
                val_pbar.set_postfix({'loss': loss.item()})
                
                # Clear memory
                del imgs, kpts, preds, loss
                torch.cuda.empty_cache()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        
        # Learning rate scheduling based on validation loss
        scheduler.step(avg_val_loss)
        
        # Model checkpointing based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, save_path)
                print(f"  Saved model checkpoint with validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Test phase
    print("\nEvaluating on test set...")
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Test')
        for imgs, kpts in test_pbar:
            # Move data to device
            imgs = imgs.to(device, non_blocking=True)
            kpts = kpts.to(device, non_blocking=True)
            
            # Remove extra dimension from keypoints if present
            if len(kpts.shape) == 4:
                kpts = kpts.squeeze(1)
            
            # Forward pass
            preds = model(imgs)
            if isinstance(preds, list):
                preds = preds[0]
            loss = compute_loss(preds, kpts)
            
            # Update statistics
            test_loss += loss.item() * imgs.size(0)
            test_pbar.set_postfix({'loss': loss.item()})
            
            # Clear memory
            del imgs, kpts, preds, loss
            torch.cuda.empty_cache()
    
    # Calculate average test loss
    avg_test_loss = test_loss / len(test_dataset)
    print(f"Test Loss: {avg_test_loss:.6f}")
    
    print("BlazePose training complete.")
    
    # Load best model if save path provided
    if os.path.exists(blazepose_weights_path):
        checkpoint = torch.load(blazepose_weights_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def train_posture_classifier(num_epochs=20, batch_size=32, lr=1e-3, weight_decay=1e-4, patience=5, device='cuda'):
    """
    Train PostureMLP model with memory-efficient processing and mixed precision training.
    """
    # Create dataset
    full_dataset = PostureIssueDataset()
    print(f"Full dataset size: {len(full_dataset)}")
    
    # Split dataset into train, validation, and test sets
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model and optimizer
    model = PostureMLP(input_dim=66, num_labels=8).to(device)
    model.train()
    
    # Use gradient accumulation to simulate larger batch size
    accumulation_steps = 4  # Accumulate gradients for 4 batches
    effective_batch_size = batch_size * accumulation_steps
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.BCEWithLogitsLoss()
    
    # Initialize mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    # Training loop variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        # Reset gradients at start of epoch
        optimizer.zero_grad()
        
        for batch_idx, (poses, labels) in enumerate(train_pbar):
            # Move data to device
            poses = poses.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                logits = model(poses)
                loss = criterion(logits, labels)
                # Normalize loss by accumulation steps
                loss = loss / accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Update weights if we've accumulated enough gradients
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Update statistics
            train_loss += loss.item() * effective_batch_size
            train_pbar.set_postfix({'loss': loss.item() * accumulation_steps})
            
            # Clear memory
            del poses, labels, logits, loss
            torch.cuda.empty_cache()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for poses, labels in val_pbar:
                # Move data to device
                poses = poses.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Forward pass
                logits = model(poses)
                loss = criterion(logits, labels)
                
                # Update statistics
                val_loss += loss.item() * poses.size(0)
                val_pbar.set_postfix({'loss': loss.item()})
                
                # Clear memory
                del poses, labels, logits, loss
                torch.cuda.empty_cache()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        
        # Learning rate scheduling based on validation loss
        scheduler.step(avg_val_loss)
        
        # Model checkpointing based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, os.getcwd()+'/best_posture_estimator_model.pth')
            print(f"  Saved model checkpoint with validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Test phase
    print("\nEvaluating on test set...")
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Test')
        for poses, labels in test_pbar:
            # Move data to device
            poses = poses.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            logits = model(poses)
            loss = criterion(logits, labels)
            
            # Update statistics
            test_loss += loss.item() * poses.size(0)
            test_pbar.set_postfix({'loss': loss.item()})
            
            # Clear memory
            del poses, labels, logits, loss
            torch.cuda.empty_cache()
    
    # Calculate average test loss
    avg_test_loss = test_loss / len(test_dataset)
    print(f"Test Loss: {avg_test_loss:.6f}")
    
    print("Posture classifier training complete.")
    
    return model

def get_blazepose_keypoints_from_model(image, model, device):
    # image: BGR numpy array
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # (1, 3, H, W)
    img = img.to(device)
    with torch.no_grad():
        outputs = model(img)
        # Handle list output from model
        if isinstance(outputs, list):
            keypoints = outputs[0]  # Take first element if it's a list
        else:
            keypoints = outputs
        keypoints = keypoints.cpu().numpy()[0]  # (33, 3)
    return keypoints

def normalize_pose_keypoints(keypoints):
    """
    Normalize pose keypoints to make y-coordinates uniform.
    Args:
        keypoints: numpy array of shape (33, 3) containing (x, y, confidence)
    Returns:
        Normalized keypoints with uniform y-coordinates
    """
    # Get hip center (average of left and right hip)
    left_hip = keypoints[23, :2]  # (x, y)
    right_hip = keypoints[24, :2]  # (x, y)
    hip_center = (left_hip + right_hip) / 2
    
    # Get shoulder center (average of left and right shoulder)
    left_shoulder = keypoints[11, :2]  # (x, y)
    right_shoulder = keypoints[12, :2]  # (x, y)
    shoulder_center = (left_shoulder + right_shoulder) / 2
    
    # Calculate scale factor based on shoulder-hip distance
    shoulder_hip_distance = np.linalg.norm(shoulder_center - hip_center)
    scale_factor = 1.0 / (shoulder_hip_distance + 1e-6)  # Add small epsilon to avoid division by zero
    
    # Create normalized keypoints array
    normalized_keypoints = keypoints.copy()
    
    # Center the pose at hip center
    normalized_keypoints[:, 0] = (keypoints[:, 0] - hip_center[0]) * scale_factor
    normalized_keypoints[:, 1] = (keypoints[:, 1] - hip_center[1]) * scale_factor
    
    return normalized_keypoints

def process_single_video(args):
    """
    Process a single video file to extract pose keypoints.
    Args:
        args: Tuple containing (video_path, out_npy_path, blazepose_model, device, frames_per_clip)
    """
    video_path, out_npy_path, blazepose_model, device, frames_per_clip = args
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval to get desired number of frames
        if total_frames > frames_per_clip:
            frame_interval = max(1, total_frames // frames_per_clip)
        else:
            frame_interval = 1
            
        keypoints_list = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                keypoints = get_blazepose_keypoints_from_model(frame, blazepose_model, device)
                # Normalize the keypoints
                normalized_keypoints = normalize_pose_keypoints(keypoints)
                keypoints_list.append(normalized_keypoints)
                
                # Break if we have enough frames
                if len(keypoints_list) >= frames_per_clip:
                    break
                    
            frame_count += 1
            
        cap.release()
        
        if keypoints_list:
            # Pad or truncate to ensure exactly frames_per_clip frames
            if len(keypoints_list) < frames_per_clip:
                # Pad with the last frame
                last_frame = keypoints_list[-1]
                padding = [last_frame] * (frames_per_clip - len(keypoints_list))
                keypoints_list.extend(padding)
            else:
                # Truncate to frames_per_clip
                keypoints_list = keypoints_list[:frames_per_clip]
                
            keypoints_arr = np.stack(keypoints_list)
            np.save(out_npy_path, keypoints_arr)
            print(f"Saved pose keypoints to {out_npy_path} (shape: {keypoints_arr.shape})")
        else:
            print(f"No frames processed for {video_path}")
            
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        if 'cap' in locals():
            cap.release()

def extract_pose_keypoints_from_videos_parallel(video_dir, out_dir, blazepose_model, device, num_workers=4):
    """
    Parallel version of pose keypoint extraction using multiprocessing.
    Args:
        video_dir: Directory containing activity subdirectories with video files
        out_dir: Directory to save keypoint files
        blazepose_model: BlazePose model instance
        device: Device to run inference on
        num_workers: Number of parallel workers
    """
    import multiprocessing as mp
    from functools import partial
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Get list of video files from activity subdirectories
    video_files = []
    for activity in os.listdir(video_dir):
        activity_dir = os.path.join(video_dir, activity)
        if not os.path.isdir(activity_dir):
            continue
            
        # Create activity output directory
        activity_out_dir = os.path.join(out_dir, activity)
        os.makedirs(activity_out_dir, exist_ok=True)
        
        # Get videos in this activity directory
        for file in os.listdir(activity_dir):
            if file.endswith(('.mp4', '.avi')):
                video_files.append((
                    os.path.join(activity_dir, file),  # input path
                    os.path.join(activity_out_dir, os.path.splitext(file)[0] + '_pose.npy')  # output path
                ))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files to process")
    
    # Prepare arguments for parallel processing
    process_args = []
    for video_path, out_npy_path in video_files:
        process_args.append((video_path, out_npy_path, blazepose_model, device, frames_per_clip))
    
    # Create a pool of workers
    with mp.Pool(processes=num_workers) as pool:
        # Process videos in parallel
        list(tqdm(
            pool.imap(process_single_video, process_args),
            total=len(process_args),
            desc="Processing videos in parallel"
        ))
    
    print('All videos processed in parallel.')

def extract_pose_keypoints_from_videos(video_dir, out_dir, blazepose_model, device):
    """
    Original sequential version of pose keypoint extraction.
    Kept for backward compatibility.
    """
    # Get list of video files from activity subdirectories
    video_files = []
    for activity in os.listdir(video_dir):
        activity_dir = os.path.join(video_dir, activity)
        if not os.path.isdir(activity_dir):
            continue
            
        # Create activity output directory
        activity_out_dir = os.path.join(out_dir, activity)
        os.makedirs(activity_out_dir, exist_ok=True)
        
        # Get videos in this activity directory
        for file in os.listdir(activity_dir):
            if file.endswith(('.mp4', '.avi')):
                video_files.append((
                    os.path.join(activity_dir, file),  # input path
                    os.path.join(activity_out_dir, os.path.splitext(file)[0] + '_pose.npy')  # output path
                ))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
        
    print(f"Found {len(video_files)} video files to process")
    blazepose_model.eval()
    
    for video_path, out_npy_path in tqdm(video_files, desc='Extracting pose from videos'):
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

def normalize_poses_from_npy(npy_path, out_path=None):
    """
    Normalize poses from a .npy file.
    Args:
        npy_path: Path to the input .npy file containing poses
        out_path: Path to save normalized poses. If None, overwrites the input file.
    Returns:
        Normalized poses array
    """
    # Load poses
    poses = np.load(npy_path)
    
    # Normalize each frame
    normalized_poses = np.array([normalize_pose_keypoints(pose) for pose in poses])
    
    # Save normalized poses
    if out_path is None:
        out_path = npy_path
    np.save(out_path, normalized_poses)
    
    return normalized_poses

def normalize_poses_directory(input_dir, output_dir=None, num_workers=4):
    """
    Normalize all pose .npy files in a directory and its subdirectories.
    Args:
        input_dir: Directory containing activity subdirectories with .npy files
        output_dir: Directory to save normalized poses. If None, overwrites input files.
        num_workers: Number of parallel workers
    """
    import multiprocessing as mp
    from functools import partial
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get list of .npy files
    npy_files = []
    for activity in os.listdir(input_dir):
        activity_dir = os.path.join(input_dir, activity)
        if not os.path.isdir(activity_dir):
            continue
            
        # Create activity output directory if needed
        if output_dir is not None:
            activity_out_dir = os.path.join(output_dir, activity)
            os.makedirs(activity_out_dir, exist_ok=True)
        
        # Get .npy files in this activity directory
        for file in os.listdir(activity_dir):
            if file.endswith('_pose.npy'):
                input_path = os.path.join(activity_dir, file)
                if output_dir is not None:
                    output_path = os.path.join(activity_out_dir, file)
                else:
                    output_path = None
                npy_files.append((input_path, output_path))
    
    if not npy_files:
        print(f"No pose .npy files found in {input_dir}")
        return
    
    print(f"Found {len(npy_files)} pose files to normalize")
    
    # Process files in parallel
    with mp.Pool(processes=num_workers) as pool:
        list(tqdm(
            pool.starmap(normalize_poses_from_npy, npy_files),
            total=len(npy_files),
            desc="Normalizing poses"
        ))
    
    print('All poses normalized.')