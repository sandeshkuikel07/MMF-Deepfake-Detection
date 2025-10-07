import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import pandas as pd
import argparse
import os

# This now correctly imports MLP, which is the class name in the other file
from models_and_dataloaders import MLP, FeatureDataset

def run_pipeline(domains, epochs, batch_size, learning_rate, hidden_dim1=512, hidden_dim2=256, dropout=0.5):
    """
    Trains and evaluates a model for a given set of feature domains.
    This is the corrected and simplified final version.
    """
    print(f"\n--- Running Pipeline for Domain(s): {', '.join(domains)} ---")
    print(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    print(f"Model architecture: hidden_dims=({hidden_dim1}, {hidden_dim2}), dropout={dropout}")

    # This call is now correct and matches the latest FeatureDataset definition
    train_dataset = FeatureDataset(domains=domains, split='train')
    val_dataset = FeatureDataset(domains=domains, split='val')

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("\nERROR: Dataset is empty.")
        print("Please ensure feature extraction was successful and that the 'extracted_features' directory is not empty.")
        return None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Define model, loss, and optimizer
    # We get the input dimension from the first item of the dataset
    input_dim = train_dataset[0][0].shape[0]
    model = MLP(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, dropout=dropout)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
            features, labels = features.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            preds = torch.sigmoid(outputs).cpu().numpy().flatten() >= 0.5
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"Validation Metrics: Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")
    
    return {'Domain': ' + '.join(domains), 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate deepfake detection models.")
    # Note: I've removed the --base_dir argument as it's no longer needed
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    args = parser.parse_args()

    # --- Define model configurations with different hyperparameters ---
    # Dimensions: XceptionNet=2048, DINOv2=768, FFT=4
    configs = {
        'Spatial': {
            'domains': ['spatial'],
            'epochs': 15,
            'batch_size': 64,
            'learning_rate': 2e-4,
            'hidden_dim1': 1024,
            'hidden_dim2': 512,
            'dropout': 0.3
        },
        'Frequency': {
            'domains': ['frequency'],
            'epochs': 20,
            'batch_size': 256,
            'learning_rate': 5e-4,
            'hidden_dim1': 128,
            'hidden_dim2': 64,
            'dropout': 0.1
        },
        'Semantic': {
            'domains': ['semantic'],
            'epochs': 12,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'hidden_dim1': 512,
            'hidden_dim2': 256,
            'dropout': 0.4
        },
        'Fused (All)': {
            'domains': ['spatial', 'frequency', 'semantic'],
            'epochs': 10,
            'batch_size': 128,
            'learning_rate': 1e-4,
            'hidden_dim1': 512,
            'hidden_dim2': 256,
            'dropout': 0.5
        }
    }

    results = []

    for name, config in configs.items():
        # The call to run_pipeline is now simpler and correct
        result = run_pipeline(
            domains=config['domains'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            hidden_dim1=config['hidden_dim1'],
            hidden_dim2=config['hidden_dim2'],
            dropout=config['dropout']
        )
        if result:
            results.append(result)

    # --- Print Final Comparison Table ---
    if results:
        print("\n\n--- Final Performance Comparison ---")
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))

