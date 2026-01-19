import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import time
import json

def process_data_chunkwise(file_path='../data/KuaiRec 2.0/data/big_matrix.csv', chunk_size=1000000):
    """
    Loads and preprocesses the KuaiRec dataset in chunks to handle large file sizes,
    using vectorized operations for better performance.
    """
    print("Processing data in chunks...")
    start_time = time.time()

    user_map = {}
    video_map = {}
    interactions = []
    user_latest_interaction = {}

    reader = pd.read_csv(file_path, usecols=['user_id', 'video_id', 'watch_ratio', 'timestamp'], chunksize=chunk_size)

    for i, chunk in enumerate(reader):
        print(f"  Processing chunk {i+1}...")
        
        chunk = chunk[chunk['watch_ratio'] > 2.0].copy()

        new_users = chunk['user_id'].unique()
        new_videos = chunk['video_id'].unique()
        
        for user in new_users:
            if user not in user_map:
                user_map[user] = len(user_map)
        for video in new_videos:
            if video not in video_map:
                video_map[video] = len(video_map)
        
        chunk['user_id'] = chunk['user_id'].map(user_map)
        chunk['video_id'] = chunk['video_id'].map(video_map)

        interactions.append(chunk[['user_id', 'video_id', 'timestamp']])

        chunk = chunk.sort_values('timestamp')
        latest_in_chunk = chunk.groupby('user_id').tail(1)

        for _, row in latest_in_chunk.iterrows():
            user_id = row['user_id']
            if user_id not in user_latest_interaction or row['timestamp'] > user_latest_interaction[user_id]['timestamp']:
                user_latest_interaction[user_id] = {'video_id': row['video_id'], 'timestamp': row['timestamp']}

    print(f"Data processed in {time.time() - start_time:.2f} seconds.")

    all_interactions_df = pd.concat(interactions, ignore_index=True)

    user_map_serializable = {int(k): v for k, v in user_map.items()}
    video_map_serializable = {int(k): v for k, v in video_map.items()}

    print("  Saving user and video maps...")
    with open('user_map.json', 'w') as f:
        json.dump(user_map_serializable, f)
    with open('video_map.json', 'w') as f:
        json.dump(video_map_serializable, f)
    print("  Maps saved.")

    print("Creating train and test sets...")
    
    test_items_set = set()
    for user_id, data in user_latest_interaction.items():
        test_items_set.add((user_id, data['video_id']))

    all_interactions_df['is_test'] = all_interactions_df.apply(lambda row: (row['user_id'], row['video_id']) in test_items_set, axis=1)

    train_df = all_interactions_df[~all_interactions_df['is_test']].drop(columns=['is_test'])
    test_df = all_interactions_df[all_interactions_df['is_test']].drop(columns=['is_test'])
    print("Train and test sets created.")

    return train_df, test_df, user_map, video_map


class NCFDataset(Dataset):
    def __init__(self, train_data, num_negatives, all_video_ids):
        print("Creating NCFDataset...")
        self.train_data = train_data
        self.num_negatives = num_negatives
        self.all_video_ids = np.array(all_video_ids)
        self.user_item_set = set(zip(self.train_data['user_id'], self.train_data['video_id']))
        self.users, self.items, self.labels = self.get_train_instances()
        print("NCFDataset created.")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_train_instances(self):
        print("  Generating training instances with negative samples...")
        start_time = time.time()
        
        positive_users = self.train_data['user_id'].values
        positive_items = self.train_data['video_id'].values
        
        users = list(positive_users)
        items = list(positive_items)
        labels = [1] * len(positive_users)
        
        num_positives = len(positive_users)
        num_neg_samples = num_positives * self.num_negatives
        
        neg_users = np.repeat(positive_users, self.num_negatives)
        neg_items = np.random.choice(self.all_video_ids, size=num_neg_samples)
        
        print(f"    Initial negative sampling done in {time.time() - start_time:.2f}s. Checking for conflicts...")
        
        # Check for conflicts
        for i in range(len(neg_users)):
            if (i % 100000 == 0) and (i > 0):
                print(f"      Checked {i}/{len(neg_users)} conflicts...")
            while (neg_users[i], neg_items[i]) in self.user_item_set:
                neg_items[i] = np.random.choice(self.all_video_ids)
        
        print(f"    Conflict checking done in {time.time() - start_time:.2f}s.")
        
        users.extend(neg_users)
        items.extend(neg_items)
        labels.extend([0] * num_neg_samples)
        
        print("  Training instances generated.")
        return torch.LongTensor(users), torch.LongTensor(items), torch.FloatTensor(labels)


def get_test_instances(test_data, num_neg_test, all_video_ids, user_item_set):
    print("Generating test instances...")
    users, items = [], []
    
    for i, (_, row) in enumerate(test_data.iterrows()):
        if (i % 100 == 0) and (i > 0):
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
    print("Starting data preparation for NCF model...")
    
    NUM_NEGATIVES = 4
    NUM_NEG_TEST = 99
    
    print("\nStep 1: Processing data chunkwise...")
    train_df, test_df, user_map, video_map = process_data_chunkwise()
    all_video_ids = list(video_map.values())
    print("Step 1 complete.")
    
    print("\nStep 2: Creating user-item set for lookups...")
    user_item_set = set(zip(pd.concat([train_df, test_df])['user_id'], pd.concat([train_df, test_df])['video_id']))
    print("Step 2 complete.")
    
    print("\nStep 3: Creating training dataset...")
    train_dataset = NCFDataset(train_df, NUM_NEGATIVES, all_video_ids)
    print("Step 3 complete.")
    
    print(f"\nNumber of users: {len(user_map)}")
    print(f"Number of videos: {len(video_map)}")
    print(f"Training instances (with negatives): {len(train_dataset)}")
    
    print("\nStep 4: Creating test instances...")
    test_users, test_items = get_test_instances(test_df, NUM_NEG_TEST, all_video_ids, user_item_set)
    print("Step 4 complete.")

    print(f"Test users shape: {test_users.shape}")
    print(f"Test items shape: {test_items.shape}")
    
    print("\nData preparation complete.")
    print("You can now use the 'train_dataset' for training and 'test_users', 'test_items' for evaluation.")
