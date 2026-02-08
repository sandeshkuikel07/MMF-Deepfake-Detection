"""
Patch script to apply model training fixes to integrated_training.ipynb
Run this script to automatically update the notebook with the fixes.
"""

import json
from pathlib import Path

def patch_notebook():
    notebook_path = Path('E:/MMF Deepfake Detection/integrated_training.ipynb')
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    changes_made = []
    
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell['source'])
        
        # FIX 1: Patch IntegratedMultiDomainDataset.__getitem__
        if 'def __getitem__(self, idx):' in source and 'IntegratedMultiDomainDataset' in source:
            # Find and replace the __getitem__ method
            old_getitem = '''    def __getitem__(self, idx):
        sample = self.samples[idx]
        spatial = np.load(sample['spatial']).astype(np.float32)
        frequency = np.load(sample['frequency']).astype(np.float32)
        semantic = np.load(sample['semantic']).astype(np.float32)
        label = np.float32(sample['label'])
        
        return (
            torch.from_numpy(spatial),
            torch.from_numpy(frequency),
            torch.from_numpy(semantic),
            torch.tensor(label)
        )'''
            
            new_getitem = '''    def __getitem__(self, idx):
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
        )'''
            
            if old_getitem in source:
                source = source.replace(old_getitem, new_getitem)
                cell['source'] = [line + '\n' if not line.endswith('\n') and i < len(source.split('\n'))-1 else line 
                                  for i, line in enumerate(source.split('\n'))]
                # Fix: properly split into lines
                lines = source.split('\n')
                cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
                changes_made.append("FIX 1: Added NaN/Inf protection to __getitem__")
        
        # FIX 2: Patch DomainAttentionFusion
        if 'class DomainAttentionFusion(nn.Module):' in source:
            old_forward = '''    def forward(self, spatial_feat, freq_feat, semantic_feat):
        # Project to common space
        spatial_proj = self.spatial_proj(spatial_feat)
        freq_proj = self.freq_proj(freq_feat)
        semantic_proj = self.semantic_proj(semantic_feat)
        
        # Stack domains
        domain_stack = torch.stack([spatial_proj, freq_proj, semantic_proj], dim=1)
        
        # Compute attention weights
        concat_features = torch.cat([spatial_proj, freq_proj, semantic_proj], dim=1)
        domain_weights = self.domain_attention(concat_features)
        
        # Apply attention
        weighted_features = domain_stack * domain_weights.unsqueeze(-1)
        fused = weighted_features.sum(dim=1)
        
        return self.fusion(fused), domain_weights'''
            
            new_forward = '''    def forward(self, spatial_feat, freq_feat, semantic_feat):
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
        
        return self.fusion(fused), domain_weights'''
            
            if old_forward in source:
                source = source.replace(old_forward, new_forward)
                lines = source.split('\n')
                cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
                changes_made.append("FIX 2: Added numerical stability to DomainAttentionFusion.forward")
            
            # Also need to remove Softmax from domain_attention Sequential
            old_attention = '''            nn.Linear(256, 3),
            nn.Softmax(dim=1)'''
            new_attention = '''            nn.Linear(256, 3)
            # NOTE: Softmax applied manually in forward() for numerical stability'''
            
            if old_attention in source:
                source = source.replace(old_attention, new_attention)
                lines = source.split('\n')
                cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
                changes_made.append("FIX 2b: Removed Softmax from domain_attention Sequential")
        
        # FIX 3: Patch training function to add gradient clipping
        if 'def train_complete_attention_model' in source and 'optimizer.step()' in source:
            old_step = '''                loss.backward()
                optimizer.step()'''
            new_step = '''                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()'''
            
            if old_step in source and 'clip_grad_norm_' not in source:
                source = source.replace(old_step, new_step)
                lines = source.split('\n')
                cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
                changes_made.append("FIX 3: Added gradient clipping to training loop")
    
    if changes_made:
        # Backup original
        backup_path = notebook_path.with_suffix('.ipynb.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        print(f"Backup saved to: {backup_path}")
        
        # Save patched notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print("\nChanges applied:")
        for change in changes_made:
            print(f"  [OK] {change}")
        print(f"\nNotebook patched successfully: {notebook_path}")
    else:
        print("No changes could be applied - the code patterns may have changed.")
        print("Please apply fixes manually from model_training_fixes.py")
    
    return changes_made

if __name__ == '__main__':
    patch_notebook()
