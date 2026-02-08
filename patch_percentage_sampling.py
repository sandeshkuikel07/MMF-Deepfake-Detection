"""
Patch script to add percentage-based sampling to IntegratedMultiDomainDataset
Run this to allow dynamic data percentage control without re-extracting features.
"""

import json
from pathlib import Path

def patch_dataset_sampling():
    notebook_path = Path('E:/MMF Deepfake Detection/integrated_training.ipynb')
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    changes_made = []
    
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell['source'])
        
        # Find the IntegratedMultiDomainDataset.__init__ method
        if 'class IntegratedMultiDomainDataset' in source and 'all_samples.append(paths)' in source:
            
            # Pattern to find: right after all_samples collection, before length check
            old_pattern = '''                        all_samples.append(paths)
        
        if len(all_samples) == 0:'''
            
            new_pattern = '''                        all_samples.append(paths)
        
        # Apply per-dataset percentage sampling (if configured)
        if any([config.FFPP_PERCENTAGE < 1.0, config.CELEBDF_PERCENTAGE < 1.0, 
                config.DFDC_PERCENTAGE < 1.0, config.HIDF_PERCENTAGE < 1.0]):
            
            dataset_percentages = {
                'FaceForensics++': config.FFPP_PERCENTAGE,
                'Celeb-DF': config.CELEBDF_PERCENTAGE,
                'DFDC': config.DFDC_PERCENTAGE,
                'HiDF': config.HIDF_PERCENTAGE
            }
            
            sampled_samples = []
            for dataset_name, percentage in dataset_percentages.items():
                dataset_samples = [s for s in all_samples if s['dataset'] == dataset_name]
                if len(dataset_samples) > 0:
                    num_to_keep = max(1, int(len(dataset_samples) * percentage))
                    sampled = random.sample(dataset_samples, num_to_keep)
                    sampled_samples.extend(sampled)
                    print(f"  {dataset_name}: {len(dataset_samples)} -> {num_to_keep} samples ({percentage*100:.1f}%)")
            
            all_samples = sampled_samples
            print(f"  Total after percentage sampling: {len(all_samples)} samples")
        
        if len(all_samples) == 0:'''
            
            if old_pattern in source:
                source = source.replace(old_pattern, new_pattern)
                lines = source.split('\n')
                cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
                changes_made.append("Added percentage-based sampling to IntegratedMultiDomainDataset.__init__")
    
    if changes_made:
        # Backup original
        backup_path = notebook_path.with_suffix('.ipynb.backup2')
        
        # Read the current file again to backup
        with open(notebook_path, 'r', encoding='utf-8') as f:
            current_notebook = json.load(f)
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(current_notebook, f, indent=1)
        print(f"Backup saved to: {backup_path}")
        
        # Save patched notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print("\nChanges applied:")
        for change in changes_made:
            print(f"  [OK] {change}")
        print(f"\nNotebook patched successfully: {notebook_path}")
        print("\nNow when you run the notebook:")
        print("  - It will sample data based on FFPP_PERCENTAGE, CELEBDF_PERCENTAGE, etc.")
        print("  - No need to re-extract features!")
        print("  - Just restart kernel and run all cells")
    else:
        print("Could not find the pattern to patch.")
        print("The code may have already been patched or the structure has changed.")
    
    return changes_made

if __name__ == '__main__':
    patch_dataset_sampling()
