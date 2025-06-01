import os
import json
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import torch
from torch.utils.data import TensorDataset
import shutil
import zipfile
from pathlib import Path
import time

# MPII to BlazePose keypoint mapping
MPII_TO_BLAZEPOSE = {
    0: 0,   # nose
    1: 2,   # right eye
    2: 5,   # left eye
    3: 7,   # right ear
    4: 8,   # left ear
    5: 11,  # right shoulder
    6: 12,  # left shoulder
    7: 13,  # right elbow
    8: 14,  # left elbow
    9: 15,  # right wrist
    10: 16, # left wrist
    11: 23, # right hip
    12: 24, # left hip
    13: 25, # right knee
    14: 26, # left knee
    15: 27,  # right ankle
    16: 28  # left ankle
}

class MPIIBlazePoseDataset(TensorDataset):
    def __init__(self, images_dir, keypoints_dir, transform=None):
        self.images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')])
        self.keypoints = sorted([os.path.join(keypoints_dir, f) for f in os.listdir(keypoints_dir) if f.endswith('.npy')])
        self.transform = transform
        
        # Verify that images and keypoints match
        image_names = {os.path.splitext(os.path.basename(f))[0] for f in self.images}
        keypoint_names = {os.path.splitext(os.path.basename(f))[0] for f in self.keypoints}
        if image_names != keypoint_names:
            raise ValueError("Image and keypoint files do not match!")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load and preprocess image
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load keypoint data
        keypoint_data = np.load(self.keypoints[idx])
        
        # Extract components from the flattened array
        num_joints = 16  # MPII has 16 keypoints
        keypoints = keypoint_data[:num_joints*2].reshape(-1, 2)  # Reshape to (16, 2)
        visibility = keypoint_data[num_joints*2:num_joints*3]  # (16,)
        scale = keypoint_data[num_joints*3]  # (1,)
        center = keypoint_data[num_joints*3+1:num_joints*3+3]  # (2,)
        
        # Normalize keypoints using scale and center
        # First center the keypoints
        keypoints = keypoints - center
        # Then scale them
        keypoints = keypoints / scale
        
        # Create BlazePose format keypoints
        blaze_kpts = np.zeros((33, 3), dtype=np.float32)
        for mpii_idx, blaze_idx in MPII_TO_BLAZEPOSE.items():
            blaze_kpts[blaze_idx, :2] = keypoints[mpii_idx]  # x, y coordinates
            blaze_kpts[blaze_idx, 2] = visibility[mpii_idx]  # visibility
        
        # Apply transforms if any
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img, torch.from_numpy(blaze_kpts)

def load_json(json_path):
    """Load the MPII dataset JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def process_keypoints(item):
    """Process keypoints and additional information from a single item in the JSON data."""
    # Extract keypoints and visibility information
    keypoints = item.get('joints', [])
    visibility = item.get('joints_vis', [])
    scale = item.get('scale', 0.0)
    center = item.get('center', [0, 0])
    
    if not keypoints or not visibility:
        return None
        
    # Convert to numpy arrays
    keypoints = np.array(keypoints, dtype=np.float32)  # Shape: (num_joints, 2)
    visibility = np.array(visibility, dtype=np.float32)  # Shape: (num_joints,)
    scale = np.array([scale], dtype=np.float32)  # Shape: (1,)
    center = np.array(center, dtype=np.float32)  # Shape: (2,)
    
    # Stack all information into a single array
    # Format: [keypoints, visibility, scale, center]
    # keypoints: (num_joints, 2)
    # visibility: (num_joints,)
    # scale: (1,)
    # center: (2,)
    keypoint_data = np.concatenate([
        keypoints.flatten(),  # Flatten keypoints to 1D
        visibility,           # Already 1D
        scale,               # Now 1D
        center               # Already 1D
    ])
    
    return keypoint_data

def process_item(item, existing_npy_files, keypoints_dir):
    """Process a single item and save its keypoint data."""
    image_name = item['image']
    # Remove .jpg extension and add .npy
    base_name = os.path.splitext(image_name)[0]
    output_filename = base_name + '.npy'
    
    # Only process if .npy file exists
    if output_filename in existing_npy_files:
        # Process keypoints
        keypoint_data = process_keypoints(item)
        if keypoint_data is not None:
            # Save keypoints as .npy file
            keypoint_path = os.path.join(keypoints_dir, output_filename)
            np.save(keypoint_path, keypoint_data)
            return True
    return False

def create_archive(source_dir, output_path, chunk_size=100, max_retries=3, retry_delay=5):
    """
    Create a zip archive of the source directory in chunks to avoid memory issues.
    
    Args:
        source_dir (str): Directory to archive
        output_path (str): Path for the output zip file
        chunk_size (int): Number of files to process at once
        max_retries (int): Maximum number of retries for failed files
        retry_delay (int): Delay between retries in seconds
    """
    source_dir = Path(source_dir)
    output_path = Path(output_path)
    
    # Get list of all files
    all_files = list(source_dir.rglob('*'))
    all_files = [f for f in all_files if f.is_file()]
    
    print(f"Found {len(all_files)} files to archive")
    
    # Create zip file with compression level 1 (faster)
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zipf:
        # Process files in chunks
        for i in range(0, len(all_files), chunk_size):
            chunk = all_files[i:i + chunk_size]
            for file_path in tqdm(chunk, desc=f"Archiving chunk {i//chunk_size + 1}"):
                # Try to archive the file with retries
                for attempt in range(max_retries):
                    try:
                        # Get relative path for archive
                        arcname = str(file_path.relative_to(source_dir))
                        
                        # For large files, read and write in chunks
                        if file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB
                            with open(file_path, 'rb') as f:
                                # Add file to archive with a buffer
                                zipf.writestr(arcname, f.read())
                        else:
                            # For smaller files, use normal write
                            zipf.write(str(file_path), arcname)
                        
                        # If successful, break the retry loop
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"\nRetrying {file_path} (attempt {attempt + 1}/{max_retries})")
                            time.sleep(retry_delay)
                        else:
                            print(f"\nError archiving {file_path} after {max_retries} attempts: {str(e)}")
                            # Save failed files to a list
                            with open('failed_files.txt', 'a') as f:
                                f.write(f"{file_path}\n")
                            continue

def main():
    # Define base directory
    base_dir = "C:/Users/lenovo/Downloads/mpii_human_pose_v1"
    
    # Define all paths
    json_path = os.path.join(base_dir, "mpii_human_pose_v1.json")
    images_dir = os.path.join(base_dir, "images")
    output_dir = os.path.join(base_dir, "processed_images")
    keypoints_dir = os.path.join(base_dir, "keypoints")
    
    print(f"Base directory: {base_dir}")
    print(f"Images directory: {images_dir}")
    print(f"Processed images directory: {output_dir}")
    print(f"Keypoints directory: {keypoints_dir}")
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(keypoints_dir, exist_ok=True)
    
    # Load JSON data
    print("\nLoading JSON data...")
    data = load_json(json_path)
    
    # Get list of existing .npy files
    existing_npy_files = set(os.listdir(output_dir))
    print(f"Found {len(existing_npy_files)} existing .npy files in {output_dir}")
    
    # Process keypoints for images that have .npy files
    print("\nProcessing keypoints...")
    
    # Create a partial function with the fixed arguments
    process_func = partial(process_item, 
                         existing_npy_files=existing_npy_files,
                         keypoints_dir=keypoints_dir)
    
    # Determine number of processes to use
    num_processes = max(1, cpu_count() - 1)  # Leave one CPU free
    print(f"Using {num_processes} processes")
    
    # Process items in parallel
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_func, data),
            total=len(data),
            desc="Processing keypoints"
        ))
    
    # Count results
    processed_count = sum(results)
    skipped_count = len(data) - processed_count
    
    print(f"\nProcessing complete:")
    print(f"Successfully processed keypoints for {processed_count} images")
    print(f"Skipped {skipped_count} images (no corresponding .npy file)")
    print(f"\nKeypoints saved in: {keypoints_dir}")
    print(f"Each .npy file contains: keypoints, visibility, scale, and center information")
    
    # Create archive of processed data
    print("\nCreating archive of processed data...")
    archive_path = os.path.join(base_dir, "processed_data.zip")
    create_archive(keypoints_dir, archive_path)
    print(f"Archive created at: {archive_path}")

if __name__ == "__main__":
    main() 