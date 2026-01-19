import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time
import numpy as np
import json
import os

from model import NCF
from metrics import hit_ratio, ndcg
from dataset import NCFDataset

def train(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    
    for user, item, label in train_loader:
        user, item, label = user.cuda(), item.cuda(), label.cuda()
        
        optimizer.zero_grad()
        prediction = model(user, item)
        loss = loss_fn(prediction, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate(model, test_users, test_items, top_k):
    model.eval()
    
    hr_list, ndcg_list = [], []
    
    with torch.no_grad():
        # The test data is structured as [pos_item, neg_item_1, neg_item_2, ...] for each user
        # So, for each user, there are 100 items (1 positive + 99 negative)
        num_test_users = len(test_users) // 100
        for i in range(num_test_users):
            start = i * 100
            end = (i + 1) * 100
            
            user = test_users[start] # All users in this block are the same
            items = test_items[start:end]
            
            user_tensor = torch.LongTensor([user] * len(items)).cuda()
            item_tensor = torch.LongTensor(items).cuda()
            
            predictions = model(user_tensor, item_tensor)
            _, indices = torch.topk(predictions, top_k)
            
            recommends = torch.take(item_tensor, indices).cpu().numpy().tolist()
            
            gt_item = items[0]
            
            hr_list.append(hit_ratio(recommends, gt_item))
            ndcg_list.append(ndcg(recommends, gt_item))
            
    return np.mean(hr_list), np.mean(ndcg_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--embedding_dim', type=int, default=8, help='Embedding dimension')
    parser.add_argument('--hidden_layers', nargs='+', type=int, default=[64, 32, 16], help='MLP hidden layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--top_k', type=int, default=10, help='Top-K for evaluation')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load data
    print("Loading data...")
    train_dataset = torch.load(os.path.join(script_dir, 'train_dataset.pt'))
    test_users = torch.load(os.path.join(script_dir, 'test_users.pt'))
    test_items = torch.load(os.path.join(script_dir, 'test_items.pt'))
    with open(os.path.join(script_dir, 'user_map.json'), 'r') as f:
        user_map = json.load(f)
    with open(os.path.join(script_dir, 'video_map.json'), 'r') as f:
        video_map = json.load(f)
    print("Data loaded.")

    num_users = len(user_map)
    num_items = len(video_map)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    model = NCF(num_users, num_items, args.embedding_dim, args.hidden_layers, args.dropout).cuda()
    
    # Loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer, loss_fn)
        hr, ndcg_score = evaluate(model, test_users, test_items, args.top_k)
        
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{args.epochs} [{elapsed_time:.2f}s]: "
              f"Train Loss: {train_loss:.4f}, HR@{args.top_k}: {hr:.4f}, NDCG@{args.top_k}: {ndcg_score:.4f}")

    # Save the model
    torch.save(model.state_dict(), os.path.join(script_dir, 'ncf_model.pt'))
    print(f"Model saved to '{os.path.join(script_dir, 'ncf_model.pt')}'.")