import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.stats import skew, kurtosis
import os
from tqdm import tqdm
import argparse
import pandas as pd 

# --- Feature Extractor Definitions ---

class SpatialExtractor(nn.Module):
    """XceptionNet based spatial feature extractor."""
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('xception', pretrained=True, num_classes=0)
        self.model.eval()
        # Preprocessing from timm's documentation for Xception
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @torch.no_grad()
    def forward(self, img_path, device):
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(device)
        features = self.model(img_tensor)
        return features.squeeze().cpu().numpy()

class FrequencyExtractor:
    """FFT based frequency feature extractor."""
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def extract(self, img_path):
        img = Image.open(img_path)
        img_tensor = self.transform(img)
        
        # FFT
        f = torch.fft.fft2(img_tensor)
        fshift = torch.fft.fftshift(f)
        magnitude_spectrum = torch.log(torch.abs(fshift) + 1).numpy().flatten()
        
        # Statistical features
        mean = np.mean(magnitude_spectrum)
        var = np.var(magnitude_spectrum)
        skewness = skew(magnitude_spectrum)
        kurt = kurtosis(magnitude_spectrum)
        
        return np.array([mean, var, skewness, kurt])

class SemanticExtractor(nn.Module):
    """DINOv2 based semantic feature extractor."""
    def __init__(self):
        super().__init__()
        # Using dinov2_base with 14x14 patches
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        self.model.eval()
        # DINOv2 preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    @torch.no_grad()
    def forward(self, img_path, device):
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(device)
        features = self.model(img_tensor)
        # Return the CLS token
        return features.squeeze().cpu().numpy()

def process_and_save_features(base_dir, feature_type):
    """
    Iterates through preprocessed faces, extracts features, and saves them.
    """
    preprocessed_dir = os.path.join(base_dir, 'preprocessed_faces')
    feature_dir = os.path.join(base_dir, 'extracted_features', feature_type)
    os.makedirs(feature_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} for {feature_type} extraction")

    # Initialize model
    if feature_type == 'spatial':
        extractor = SpatialExtractor().to(device)
    elif feature_type == 'semantic':
        extractor = SemanticExtractor().to(device)
    elif feature_type == 'frequency':
        extractor = FrequencyExtractor()
    else:
        raise ValueError("Invalid feature type")

    image_paths = []
    for label in ['real', 'fake']:
        label_dir = os.path.join(preprocessed_dir, label)
        if not os.path.exists(label_dir):
            continue
        for video_id in os.listdir(label_dir):
            video_dir = os.path.join(label_dir, video_id)
            if not os.path.isdir(video_dir):
                continue
            for frame_name in os.listdir(video_dir):
                if frame_name.endswith('.png'):
                    image_paths.append(os.path.join(video_dir, frame_name))

    print(f"Extracting {feature_type} features for {len(image_paths)} images...")
    for img_path in tqdm(image_paths, desc=f"Extracting {feature_type}"):
        
        # Create a unique path for the feature file
        relative_path = os.path.relpath(img_path, preprocessed_dir)
        feature_save_path = os.path.join(feature_dir, relative_path).replace('.png', '.npy')
        
        if os.path.exists(feature_save_path):
            continue

        os.makedirs(os.path.dirname(feature_save_path), exist_ok=True)

        try:
            if feature_type == 'frequency':
                features = extractor.extract(img_path)
            else: # Spatial or Semantic
                features = extractor(img_path, device)
            
            np.save(feature_save_path, features)
        except Exception as e:
            print(f"Could not process {img_path}: {e}")


def extract_domain_features(domain, model, transform, device):
    """
    Extracts features for a given domain by reading image paths from the metadata file
    and saves them in a flat directory structure.
    """
    metadata = pd.read_csv('ffpp_metadata.csv')
    
    faces_base_dir = 'preprocessed_faces'
    
    print(f"Extracting {domain} features for {len(metadata)} videos...")

    # Ensure the parent directories exist
    os.makedirs(os.path.join('extracted_features', domain, 'real'), exist_ok=True)
    os.makedirs(os.path.join('extracted_features', domain, 'fake'), exist_ok=True)

    for index, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"Extracting {domain}"):
        label = row['label']
        video_id = row['video_id']
        
        video_faces_dir = os.path.join(faces_base_dir, label, video_id)

        if not os.path.exists(video_faces_dir):
            continue

        for image_file in os.listdir(video_faces_dir):
            image_path = os.path.join(video_faces_dir, image_file)
            
            # --- This is the key change ---
            # The output path is now flat inside the label directory
            feature_filename = image_file.replace('.png', '.npy')
            output_path = os.path.join('extracted_features', domain, label, feature_filename)

            if os.path.exists(output_path):
                continue

            try:
                img = Image.open(image_path).convert('RGB')
                
                if domain == 'frequency':
                    gray_img = img.convert('L')
                    img_array = np.array(gray_img)
                    f_transform = np.fft.fft2(img_array)
                    f_shift = np.fft.fftshift(f_transform)
                    magnitude_spectrum = np.abs(f_shift)
                    
                    mean, var, skewness, kurt = np.mean(magnitude_spectrum), np.var(magnitude_spectrum), skew(magnitude_spectrum.flatten()), kurtosis(magnitude_spectrum.flatten())
                    
                    features = np.array([mean, var, skewness, kurt])
                else:
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        features = model(img_tensor).cpu().numpy().flatten()
                
                np.save(output_path, features)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    print("Feature extraction complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract features from preprocessed faces.")
    parser.add_argument('--base_dir', type=str, default='.', help='Base project directory.')
    parser.add_argument('--domain', type=str, choices=['spatial', 'frequency', 'semantic', 'all'], default='all', help='Feature domain to extract.')
    args = parser.parse_args()

    if args.domain == 'all':
        domains_to_process = ['spatial', 'frequency', 'semantic']
    else:
        domains_to_process = [args.domain]
    
    for domain in domains_to_process:
        process_and_save_features(args.base_dir, domain)
    
    print("Feature extraction complete.")

