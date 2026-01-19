import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import time
import json
import os
from dataset import NCFDataset

def get_test_instances(test_data, num_neg_test, all_video_ids, user_item_set):
    print("Generating test instances...")
    users, items = [], []
    
    for i, (_, row) in enumerate(test_data.iterrows()):
        if (i % 1000 == 0) and (i > 0):
            print(f"  Generated test instances for {i}/{len(test_data)} users...")
        users.append(row.user_id)
        items.append(row.video_id)
        
        neg_samples = []
        while len(neg_samples) < num_neg_test:
            neg_item = np.random.choice(all_video_ids)
            if (row.user_id, neg_item) not in user_item_set:
                neg_samples.append(neg_item)
        
        items.extend(neg_samples)
        users.extend([row.user_id] * num_neg_test)
        
    print("Test instances generated.")
    return torch.LongTensor(users), torch.LongTensor(items)


if __name__ == '__main__':
    print("Starting Step 2: Creating PyTorch datasets...")

    NUM_NEGATIVES = 4
    NUM_NEG_TEST = 99

    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("  Loading intermediate data...")
    train_df = pd.read_pickle(os.path.join(script_dir, 'train_df.pkl'))
    test_df = pd.read_pickle(os.path.join(script_dir, 'test_df.pkl'))
    with open(os.path.join(script_dir, 'user_map.json'), 'r') as f:
        user_map = json.load(f)
    with open(os.path.join(script_dir, 'video_map.json'), 'r') as f:
        video_map = json.load(f)
    print("  Intermediate data loaded.")
    print(f"  Size of training data: {len(train_df)}")

    all_video_ids = list(video_map.values())
    
    print("  Creating user-item set for lookups...")
    user_item_set_train = set(zip(train_df['user_id'], train_df['video_id']))
    user_item_set_test = set(zip(test_df['user_id'], test_df['video_id']))
    user_item_set = user_item_set_train.union(user_item_set_test)
    print("  User-item set created.")
    
    print("  Creating training dataset...")
    train_dataset = NCFDataset(train_df, NUM_NEGATIVES, all_video_ids, user_item_set)
    print("  Training dataset created.")
    
    print(f"\nNumber of users: {len(user_map)}")
    print(f"Number of videos: {len(video_map)}")
    print(f"Training instances (with negatives): {len(train_dataset)}")
    
    print("\n  Creating test instances...")
    test_users, test_items = get_test_instances(test_df, NUM_NEG_TEST, all_video_ids, user_item_set)
    print("  Test instances created.")

    print(f"Test users shape: {test_users.shape}")
    print(f"Test items shape: {test_items.shape}")
    
    torch.save(train_dataset, os.path.join(script_dir, 'train_dataset.pt'))
    torch.save(test_users, os.path.join(script_dir, 'test_users.pt'))
    torch.save(test_items, os.path.join(script_dir, 'test_items.pt'))
    
    print(f"\nPyTorch datasets saved to '{script_dir}'.")
    print("\nStep 2 complete.")