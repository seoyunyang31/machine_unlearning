import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time
import numpy as np
import json
import os
import sys

# Add src to path to allow for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import NCF
from utils.metrics import hit_ratio, ndcg

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

def evaluate(model, test_users, test_items, top_k, device):
    model.eval()
    hr_list, ndcg_list = [], []
    
    with torch.no_grad():
        num_test_users = len(torch.unique(test_users))
        for i in range(num_test_users):
            start = i * 100
            end = (i + 1) * 100
            
            user = test_users[start].item()
            items = test_items[start:end]
            
            user_tensor = torch.LongTensor([user] * len(items)).to(device)
            item_tensor = items.to(device)
            
            predictions = model(user_tensor, item_tensor)
            _, indices = torch.topk(predictions, top_k)
            
            recommends = torch.take(item_tensor, indices).cpu().numpy().tolist()
            
            gt_item = items[0].item()
            
            hr_list.append(hit_ratio(recommends, gt_item))
            ndcg_list.append(ndcg(recommends, gt_item))
            
    return np.mean(hr_list), np.mean(ndcg_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the NCF model.")
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--embedding_dim', type=int, default=8, help='Embedding dimension')
    parser.add_argument('--hidden_layers', nargs='+', type=int, default=[64, 32, 16], help='MLP hidden layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--top_k', type=int, default=10, help='Top-K for evaluation')
    parser.add_argument('--artifacts_dir', type=str, default='../../artifacts', help='Directory where pre-processed data is stored and model will be saved.')
    args = parser.parse_args()

    # --- Setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_path = os.path.join(script_dir, args.artifacts_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data ---
    print("Loading data...")
    try:
        train_dataset = torch.load(os.path.join(artifacts_path, 'train_dataset.pt'))
        test_users = torch.load(os.path.join(artifacts_path, 'test_users.pt'), weights_only=True)
        test_items = torch.load(os.path.join(artifacts_path, 'test_items.pt'), weights_only=True)
        with open(os.path.join(artifacts_path, 'user_map.json'), 'r') as f:
            user_map = json.load(f)
        with open(os.path.join(artifacts_path, 'video_map.json'), 'r') as f:
            video_map = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find data files in '{artifacts_path}'.")
        print("Please run the 'src/data/prepare_dataset.py' script first.")
        sys.exit(1)
    print("Data loaded.")

    num_users = len(user_map)
    num_items = len(video_map)

    # --- Create DataLoader ---
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # --- Initialize Model ---
    model = NCF(num_users, num_items, args.embedding_dim, args.hidden_layers, args.dropout).to(device)
    
    # --- Loss and Optimizer ---
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Training Loop ---
    model_save_path = os.path.join(artifacts_path, 'ncf_model.pth')
    print("\nStarting training...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        hr, ndcg_score = evaluate(model, test_users, test_items, args.top_k, device)
        
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{args.epochs} [{elapsed_time:.2f}s]: "
              f"Train Loss: {train_loss:.4f}, HR@{args.top_k}: {hr:.4f}, NDCG@{args.top_k}: {ndcg_score:.4f}")

    # --- Save the Model ---
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to '{model_save_path}'.")