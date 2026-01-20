import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import sys
import json

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
    
    num_users = len(torch.unique(test_users))

    for i in range(num_users):
        # The test data is structured as [pos, neg, neg, ..., neg] for each user (100 items total)
        start_idx = i * 100
        end_idx = (i + 1) * 100
        
        user_batch = test_users[start_idx:end_idx].to(device)
        item_batch = test_items[start_idx:end_idx].to(device)
        
        # The ground truth item is always the first one in the 100-item list
        gt_item = item_batch[0].item()
        
        with torch.no_grad():
            predictions = model(user_batch, item_batch)
            
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(item_batch, indices).cpu().numpy().tolist()
        
        hr_list.append(hit_ratio(recommends, gt_item))
        ndcg_list.append(ndcg(recommends, gt_item))
            
    return np.mean(hr_list), np.mean(ndcg_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the NCF baseline model.")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--embedding_dim', type=int, default=8, help="Should match the model architecture if loading a pre-trained model.")
    parser.add_argument('--hidden_layers', nargs='+', type=int, default=[64, 32, 16], help="Should match the model architecture if loading a pre-trained model.")
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--artifacts_dir', type=str, default='../../artifacts', help='Directory where pre-processed data is stored and model will be saved.')
    args = parser.parse_args()

    # --- Setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_path = os.path.join(script_dir, args.artifacts_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data ---
    print("Loading pre-processed data from artifacts...")
    try:
        train_dataset = torch.load(os.path.join(artifacts_path, 'train_dataset.pt'))
        test_users = torch.load(os.path.join(artifacts_path, 'test_users.pt'), weights_only=True)
        test_items = torch.load(os.path.join(artifacts_path, 'test_items.pt'), weights_only=True)
        with open(os.path.join(artifacts_path, 'user_map.json'), 'r') as f:
            user_map = json.load(f)
        with open(os.path.join(artifacts_path, 'video_map.json'), 'r') as f:
            video_map = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find data files in '{artifacts_path}'.")
        print("Please run the 'src/data/prepare_dataset.py' script first.")
        sys.exit(1)
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    num_users = len(user_map)
    num_items = len(video_map)

    # --- Model ---
    model = NCF(num_users, num_items, args.embedding_dim, args.hidden_layers, args.dropout).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Training Loop ---
    best_hr = 0
    model_save_path = os.path.join(artifacts_path, 'ncf_baseline.pth')

    print("\nStarting training...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        hr, ndcg_score = evaluate(model, test_users, test_items, args.top_k, device)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | HR@{args.top_k}: {hr:.4f} | NDCG@{args.top_k}: {ndcg_score:.4f}")

        if hr > best_hr:
            best_hr = hr
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Best model saved to '{model_save_path}' (HR: {best_hr:.4f})")
    
    print("\nTraining complete.")
    print(f"Best model saved at '{model_save_path}' with HR@{args.top_k} = {best_hr:.4f}")
