import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time
import numpy as np
import json
from pathlib import Path
import sys

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.models.model import NCF
from src.utils.metrics import evaluate_1_vs_99

def train_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    
    for user, item, label in train_loader:
        user, item, label = user.to(device), item.to(device), label.to(device)
        
        optimizer.zero_grad()
        prediction = model(user, item)
        loss = loss_fn(prediction, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the NCF model.")
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--embedding_dim', type=int, default=8, help='Embedding dimension')
    parser.add_argument('--hidden_layers', nargs='+', type=int, default=[64, 32, 16], help='MLP hidden layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--top_k', type=int, default=10, help='Top-K for evaluation')
    parser.add_argument('--artifacts_dir', type=str, default='artifacts', help='Directory where pre-processed data is stored and model will be saved.')
    args = parser.parse_args()

    # --- Setup ---
    artifacts_path = project_root / args.artifacts_dir
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data ---
    print("Loading data...")
    try:
        # Note: This script assumes a training dataset with pre-sampled negatives and labels.
        # The 'train_dataset.pt' from prepare_dataset.py contains positive interactions only
        # and is not directly compatible with this training script's DataLoader expectations.
        train_dataset = torch.load(artifacts_path / 'train_dataset.pt')
        test_data = torch.load(artifacts_path / 'test_full.pt', weights_only=True)
        with open(artifacts_path / 'user_map.json', 'r') as f:
            user_map = json.load(f)
        with open(artifacts_path / 'video_map.json', 'r') as f:
            video_map = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find data files in '{artifacts_path}'. ({e})")
        print("Please run the 'src/data/prepare_dataset.py' script first.")
        sys.exit(1)
    print("Data loaded.")

    num_users = len(user_map)
    num_items = len(video_map)

    # --- Create DataLoader ---
    # Note: This simple DataLoader is only suitable for a dataset that already contains
    # user, item, and label columns. It will not work with the 'train_dataset.pt'
    # which only has positive user-item pairs.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # --- Initialize Model ---
    model = NCF(num_users, num_items, args.embedding_dim, args.hidden_layers, args.dropout).to(device)
    
    # --- Loss and Optimizer ---
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Training Loop ---
    model_save_path = artifacts_path / 'ncf_model.pth'
    print("\nStarting training...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        
        # Standardized evaluation call
        hr, ndcg_score = evaluate_1_vs_99(
            model=model,
            test_data=test_data,
            num_items=num_items,
            k=args.top_k,
            device=device
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{args.epochs} [{elapsed_time:.2f}s]: "
              f"Train Loss: {train_loss:.4f}, HR@{args.top_k}: {hr:.4f}, NDCG@{args.top_k}: {ndcg_score:.4f}")

    # --- Save the Model ---
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to '{model_save_path}'.")