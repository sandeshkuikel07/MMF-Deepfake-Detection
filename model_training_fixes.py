"""
Model Training Fixes for Deepfake Detection
============================================
This module contains fixed versions of the model components that address 
NaN domain weights and zero learning issues.

To use: Copy-paste the relevant classes/functions into your notebook cells,
replacing the existing implementations.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path

# =============================================================================
# FIX 1: NaN-Safe Dataset Loading
# =============================================================================
# Replace the __getitem__ method in IntegratedMultiDomainDataset with this:

"""
def __getitem__(self, idx):
    sample = self.samples[idx]
    spatial = np.load(sample['spatial']).astype(np.float32)
    frequency = np.load(sample['frequency']).astype(np.float32)
    semantic = np.load(sample['semantic']).astype(np.float32)
    label = np.float32(sample['label'])
    
    # NaN/Inf protection: replace corrupted values with zeros
    for arr in [spatial, frequency, semantic]:
        mask = np.isnan(arr) | np.isinf(arr)
        if np.any(mask):
            arr[mask] = 0.0
    
    return (
        torch.from_numpy(spatial),
        torch.from_numpy(frequency),
        torch.from_numpy(semantic),
        torch.tensor(label)
    )
"""


# =============================================================================
# FIX 2: Numerically Stable DomainAttentionFusion
# =============================================================================
# Replace the DomainAttentionFusion class with this:

class DomainAttentionFusion(nn.Module):
    """
    Learnable attention-based fusion for multi-domain features.
    Dynamically weights contributions from each domain.
    
    FIXED: Added numerical stability with clamping and epsilon.
    """
    def __init__(self, spatial_dim, freq_dim, semantic_dim, fusion_dim=512):
        super().__init__()
        
        # Project each domain to common space
        self.spatial_proj = nn.Linear(spatial_dim, fusion_dim)
        self.freq_proj = nn.Linear(freq_dim, fusion_dim)
        self.semantic_proj = nn.Linear(semantic_dim, fusion_dim)
        
        # Attention network for domain weighting
        self.domain_attention = nn.Sequential(
            nn.Linear(fusion_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
            # NOTE: Removed Softmax here - we'll apply it manually with stability
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, spatial_feat, freq_feat, semantic_feat):
        # Project to common space with clamping for stability
        spatial_proj = torch.clamp(self.spatial_proj(spatial_feat), -100, 100)
        freq_proj = torch.clamp(self.freq_proj(freq_feat), -100, 100)
        semantic_proj = torch.clamp(self.semantic_proj(semantic_feat), -100, 100)
        
        # Stack domains
        domain_stack = torch.stack([spatial_proj, freq_proj, semantic_proj], dim=1)
        
        # Compute attention weights with numerical stability
        concat_features = torch.cat([spatial_proj, freq_proj, semantic_proj], dim=1)
        attention_logits = self.domain_attention(concat_features)
        
        # Stable softmax: subtract max to prevent overflow
        attention_logits = attention_logits - attention_logits.max(dim=1, keepdim=True)[0]
        domain_weights = torch.softmax(attention_logits, dim=1)
        
        # Check for NaN and replace with uniform weights
        if torch.isnan(domain_weights).any():
            domain_weights = torch.ones_like(domain_weights) / 3.0
        
        # Apply attention
        weighted_features = domain_stack * domain_weights.unsqueeze(-1)
        fused = weighted_features.sum(dim=1)
        
        return self.fusion(fused), domain_weights


# =============================================================================
# FIX 3: Training Loop with Gradient Clipping
# =============================================================================
# Add this line after loss.backward() in your training loop:
"""
# Inside the training loop, after loss.backward():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
"""


# =============================================================================
# FIX 4: Feature Validation Cell
# =============================================================================
# Run this cell BEFORE training to check for corrupted features:

def validate_features(features_dir, num_samples=100):
    """
    Validate feature files for NaN/Inf values.
    
    Args:
        features_dir: Path to extracted_features_integrated directory
        num_samples: Number of random samples to check per domain
    
    Returns:
        dict with validation results
    """
    from pathlib import Path
    import random
    
    features_dir = Path(features_dir)
    domains = ['spatial', 'frequency', 'semantic']
    results = {}
    
    print("=" * 70)
    print("FEATURE VALIDATION")
    print("=" * 70)
    
    for domain in domains:
        domain_dir = features_dir / domain
        if not domain_dir.exists():
            print(f"⚠ {domain} directory not found")
            continue
        
        # Collect all .npy files
        all_files = list(domain_dir.rglob('*.npy'))
        if len(all_files) == 0:
            print(f"⚠ No .npy files found in {domain}")
            continue
        
        # Sample files
        sample_files = random.sample(all_files, min(num_samples, len(all_files)))
        
        nan_count = 0
        inf_count = 0
        valid_count = 0
        corrupted_files = []
        
        for f in sample_files:
            try:
                arr = np.load(f)
                has_nan = np.any(np.isnan(arr))
                has_inf = np.any(np.isinf(arr))
                
                if has_nan:
                    nan_count += 1
                    corrupted_files.append(str(f))
                if has_inf:
                    inf_count += 1
                    if str(f) not in corrupted_files:
                        corrupted_files.append(str(f))
                if not has_nan and not has_inf:
                    valid_count += 1
            except Exception as e:
                corrupted_files.append(f"{f} (load error: {e})")
        
        results[domain] = {
            'total_sampled': len(sample_files),
            'valid': valid_count,
            'nan': nan_count,
            'inf': inf_count,
            'corrupted_files': corrupted_files[:5]  # Show first 5
        }
        
        status = "[OK]" if nan_count == 0 and inf_count == 0 else "[FAIL]"
        print(f"{status} {domain.upper()}: {valid_count}/{len(sample_files)} valid, "
              f"{nan_count} NaN, {inf_count} Inf")
        
        if corrupted_files:
            print(f"   Sample corrupted files: {corrupted_files[:3]}")
    
    print("=" * 70)
    return results


# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================
"""
HOW TO APPLY THESE FIXES:

1. BEFORE TRAINING - Validate your features:
   Run a new cell with:
   
   validate_features('E:/MMF Deepfake Detection/extracted_features_integrated')


2. IN DATASET CELL - Replace __getitem__ method:
   Replace the __getitem__ method in IntegratedMultiDomainDataset 
   with the NaN-safe version from FIX 1 above.


3. IN ATTENTION MODULES CELL - Replace DomainAttentionFusion:
   Replace the entire DomainAttentionFusion class with FIX 2 above.


4. IN TRAINING FUNCTION CELL - Add gradient clipping:
   Find the line `optimizer.step()` in the training loop and add:
   
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   
   BEFORE the optimizer.step() call.


5. RE-RUN ALL CELLS and retrain for 2+ epochs.
"""

if __name__ == "__main__":
    # Quick test
    print("Model fixes module loaded successfully!")
    print("\nRun validate_features() to check your feature files.")
