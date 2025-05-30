import os
import json
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

def load_json(json_path):
    """Load the MPII dataset JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def process_image(image_path):
    """Process an image and return its numpy array."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
        
    try:
        # Read image using PIL
        img = Image.open(image_path)
        # Convert to numpy array
        img_array = np.array(img)
        return img_array
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def main():
    # Paths
    json_path = "C:/Users/lenovo/Downloads/mpii_human_pose_v1.json"
    images_dir = "C:/Users/lenovo/Downloads/mpii_human_pose_v1/images"
    output_dir = "C:/Users/lenovo/Downloads/mpii_human_pose_v1/processed_images"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSON data
    print("Loading JSON data...")
    data = load_json(json_path)
    
    # Verify images directory exists
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        return
        
    # Get list of available images
    available_images = set(os.listdir(images_dir))
    print(f"Found {len(available_images)} images in directory")
    
    # Process each image
    print("Processing images...")
    processed_count = 0
    missing_count = 0
    
    for item in tqdm(data):
        image_name = item['image']
        image_path = os.path.join(images_dir, image_name)
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_name}")
            missing_count += 1
            continue
            
        # Process image
        img_array = process_image(image_path)
        if img_array is not None:
            # Create output filename
            output_filename = os.path.splitext(image_name)[0] + '.npy'
            output_path = os.path.join(output_dir, output_filename)
            
            # Save as .npy file
            np.save(output_path, img_array)
            processed_count += 1
    
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {processed_count} images")
    print(f"Missing images: {missing_count}")

if __name__ == "__main__":
    main() 