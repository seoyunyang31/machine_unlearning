import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import time
import json
import os
import argparse

# --- Metrics ---
def hit_ratio(recommends, gt_item):
    if gt_item in recommends:
        return 1
    return 0

def ndcg(recommends, gt_item):
    if gt_item in recommends:
        index = recommends.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0

# --- Model ---
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_layers, dropout):
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        self.embed_user_GMF = nn.Embedding(num_users, embedding_dim)
        self.embed_item_GMF = nn.Embedding(num_items, embedding_dim)
        
        self.embed_user_MLP = nn.Embedding(num_users, embedding_dim)
        self.embed_item_MLP = nn.Embedding(num_items, embedding_dim)
        
        mlp_layers = []
        input_size = 2 * embedding_dim
        for output_size in hidden_layers:
            mlp_layers.append(nn.Linear(input_size, output_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=dropout))
            input_size = output_size
        self.mlp_layers = nn.Sequential(*mlp_layers)
        
        self.predict_layer = nn.Linear(embedding_dim + hidden_layers[-1], 1)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        
        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user, item):
        user_emb_gmf = self.embed_user_GMF(user)
        item_emb_gmf = self.embed_item_GMF(item)
        gmf_output = user_emb_gmf * item_emb_gmf
        
        user_emb_mlp = self.embed_user_MLP(user)
        item_emb_mlp = self.embed_item_MLP(item)
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)
        
        concat_output = torch.cat([gmf_output, mlp_output], dim=-1)
        
        prediction = self.predict_layer(concat_output)
        
        return prediction.squeeze()

# --- Dataset ---
class NCFDataset(Dataset):
    def __init__(self, train_data, num_negatives, all_video_ids, user_item_set):
        self.train_data = train_data
        self.num_negatives = num_negatives
        self.all_video_ids = np.array(all_video_ids)
        self.user_item_set = user_item_set
        self.users, self.items, self.labels = self.get_train_instances_batched()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_train_instances_batched(self, batch_size=10000):
        print("  Generating training instances in batches...")
        users, items, labels = [], [], []
        positive_users = self.train_data['user_id'].values
        positive_items = self.train_data['video_id'].values
        
        for i in range(0, len(positive_users), batch_size):
            batch_pos_users = positive_users[i:i+batch_size]
            
            users.extend(batch_pos_users)
            items.extend(positive_items[i:i+batch_size])
            labels.extend([1] * len(batch_pos_users))
            
            num_neg_samples = len(batch_pos_users) * self.num_negatives
            neg_users = np.repeat(batch_pos_users, self.num_negatives)
            neg_items = np.random.choice(self.all_video_ids, size=num_neg_samples)
            
            for j in range(len(neg_users)):
                while (neg_users[j], neg_items[j]) in self.user_item_set:
                    neg_items[j] = np.random.choice(self.all_video_ids)
            
            users.extend(neg_users)
            items.extend(neg_items)
            labels.extend([0] * num_neg_samples)
            
        return torch.LongTensor(users), torch.LongTensor(items), torch.FloatTensor(labels)

# --- Data Processing ---
def process_data(file_path, chunk_size=1000000):
    print("Processing data...")
    user_map, video_map, interactions, user_latest_interaction = {}, {}, [], {}

    reader = pd.read_csv(file_path, usecols=['user_id', 'video_id', 'watch_ratio', 'timestamp'], chunksize=chunk_size)

    for chunk in reader:
        chunk = chunk[chunk['watch_ratio'] > 2.0].copy()
        for user in chunk['user_id'].unique():
            if user not in user_map: user_map[user] = len(user_map)
        for video in chunk['video_id'].unique():
            if video not in video_map: video_map[video] = len(video_map)
        
        chunk['user_id'] = chunk['user_id'].map(user_map)
        chunk['video_id'] = chunk['video_id'].map(video_map)
        interactions.append(chunk[['user_id', 'video_id', 'timestamp']])
        
        chunk = chunk.sort_values('timestamp')
        latest_in_chunk = chunk.groupby('user_id').tail(1)
        for _, row in latest_in_chunk.iterrows():
            user_id = row['user_id']
            if user_id not in user_latest_interaction or row['timestamp'] > user_latest_interaction[user_id]['timestamp']:
                user_latest_interaction[user_id] = {'video_id': row['video_id'], 'timestamp': row['timestamp']}

    all_interactions_df = pd.concat(interactions, ignore_index=True)
    
    test_items_set = set((user_id, data['video_id']) for user_id, data in user_latest_interaction.items())
    
    test_tuples = pd.MultiIndex.from_tuples(test_items_set)
    interaction_tuples = pd.MultiIndex.from_frame(all_interactions_df[['user_id', 'video_id']])
    all_interactions_df['is_test'] = interaction_tuples.isin(test_tuples)
    
    train_df = all_interactions_df[~all_interactions_df['is_test']].drop(columns=['is_test'])
    test_df = all_interactions_df[all_interactions_df['is_test']].drop(columns=['is_test'])
    
    # Ensure IDs are integers
    train_df['user_id'] = train_df['user_id'].astype(int)
    train_df['video_id'] = train_df['video_id'].astype(int)
    test_df['user_id'] = test_df['user_id'].astype(int)
    test_df['video_id'] = test_df['video_id'].astype(int)

    return train_df, test_df, user_map, video_map

# --- Training and Evaluation ---
def train(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for user, item, label in train_loader:
        optimizer.zero_grad()
        prediction = model(user, item)
        loss = loss_fn(prediction, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, test_df, top_k, user_item_set, all_video_ids):
    model.eval()
    hr_list, ndcg_list = [], []
    
    for _, row in test_df.iterrows():
        user_id, gt_item = row['user_id'], row['video_id']
        
        neg_samples = []
        while len(neg_samples) < 99:
            neg_item = np.random.choice(all_video_ids)
            if (user_id, neg_item) not in user_item_set:
                neg_samples.append(neg_item)
        
        items_to_rank = [gt_item] + neg_samples
        
        user_tensor = torch.LongTensor([int(user_id)] * 100)
        item_tensor = torch.LongTensor([int(i) for i in items_to_rank])
        
        with torch.no_grad():
            predictions = model(user_tensor, item_tensor)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(item_tensor, indices).cpu().numpy().tolist()
        
        hr_list.append(hit_ratio(recommends, gt_item))
        ndcg_list.append(ndcg(recommends, gt_item))
            
    return np.mean(hr_list), np.mean(ndcg_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--embedding_dim', type=int, default=8)
    parser.add_argument('--hidden_layers', nargs='+', type=int, default=[64, 32, 16])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--data_path', type=str, default='data/KuaiRec 2.0/data/big_matrix.csv')
    args = parser.parse_args()

    train_df, test_df, user_map, video_map = process_data(args.data_path)
    
    all_video_ids = list(video_map.values())
    user_item_set = set(zip(pd.concat([train_df, test_df])['user_id'], pd.concat([train_df, test_df])['video_id']))
    
    train_dataset = NCFDataset(train_df, 4, all_video_ids, user_item_set)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = NCF(len(user_map), len(video_map), args.embedding_dim, args.hidden_layers, args.dropout)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_hr = 0
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, loss_fn)
        hr, ndcg_score = evaluate(model, test_df, args.top_k, user_item_set, all_video_ids)
        
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, HR@{args.top_k}: {hr:.4f}, NDCG@{args.top_k}: {ndcg_score:.4f}")

        if hr > best_hr:
            best_hr = hr
            torch.save(model.state_dict(), 'ncf_baseline.pth')
            print(f"Best model saved with HR = {best_hr:.4f}")
