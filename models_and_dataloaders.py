import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

# --- MLP Classifier Definition ---

class MLP(nn.Module):
    """A simple MLP for binary classification."""
    def __init__(self, input_dim, hidden_dim1=512, hidden_dim2=256, dropout=0.5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1) # Output is a single logit
        )

    def forward(self, x):
        return self.network(x)

# --- PyTorch Dataset Definition ---
class FeatureDataset(Dataset):
    """
    A PyTorch dataset to load pre-extracted features.
    This version reads from metadata to be more robust.
    """
    def __init__(self, domains, split='train'):
        self.domains = domains
        self.features = []
        self.labels = []

        metadata_path = 'ffpp_metadata.csv'
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Please run download_and_prepare_ffpp.py first.")
        
        metadata = pd.read_csv(metadata_path)
        
        # Simple split for demonstration. For a real project, use a more robust split.
        train_df = metadata.sample(frac=0.8, random_state=42)
        val_df = metadata.drop(train_df.index)
        
        df = train_df if split == 'train' else val_df

        print(f"Loading {split} dataset with {len(df)} videos...")

        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {split} features"):
            label = row['label']
            video_id = row['video_id']
            
            # Directory where preprocessed faces for this video are stored
            faces_dir = os.path.join('preprocessed_faces', label, video_id)
            if not os.path.exists(faces_dir):
                continue
            
            # Find all the frame images that were preprocessed
            for image_file in os.listdir(faces_dir):
                if image_file.endswith('.png'):
                    # Check if the corresponding feature file exists for all requested domains
                    feature_paths_exist = True
                    feature_paths_for_frame = {}

                    for domain in self.domains:
                        feature_filename = image_file.replace('.png', '.npy')
                        # Prefer nested path: extracted_features/<domain>/<label>/<video_id>/<frame>.npy
                        nested_feature_path = os.path.join('extracted_features', domain, label, video_id, feature_filename)
                        # Fallback to flat path if project was extracted that way
                        flat_feature_path = os.path.join('extracted_features', domain, label, feature_filename)
                        feature_path = nested_feature_path if os.path.exists(nested_feature_path) else flat_feature_path
                        
                        if not os.path.exists(feature_path):
                            feature_paths_exist = False
                            break
                        feature_paths_for_frame[domain] = feature_path
                    
                    if feature_paths_exist:
                        self.features.append(feature_paths_for_frame)
                        self.labels.append(1 if label == 'fake' else 0)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature_paths = self.features[idx]
        
        # Load and concatenate features from all specified domains
        loaded_features = [np.load(feature_paths[domain]) for domain in self.domains]
        combined_features = np.concatenate(loaded_features).astype(np.float32)
        
        label = np.float32(self.labels[idx])
        
        return torch.from_numpy(combined_features), torch.tensor(label)

