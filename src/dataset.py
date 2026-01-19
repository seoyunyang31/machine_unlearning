import torch
from torch.utils.data import Dataset
import numpy as np
import time

class NCFDataset(Dataset):
    def __init__(self, train_data, num_negatives, all_video_ids, user_item_set):
        print("Creating NCFDataset...")
        self.train_data = train_data
        self.num_negatives = num_negatives
        self.all_video_ids = np.array(all_video_ids)
        self.user_item_set = user_item_set
        self.users, self.items, self.labels = self.get_train_instances_batched()
        print("NCFDataset created.")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_train_instances_batched(self, batch_size=10000):
        print("  Generating training instances in batches...")
        start_time = time.time()
        
        users, items, labels = [], [], []
        
        positive_users = self.train_data['user_id'].values
        positive_items = self.train_data['video_id'].values
        
        for i in range(0, len(positive_users), batch_size):
            batch_start_time = time.time()
            print(f"    Processing batch {i//batch_size + 1}...")
            
            batch_pos_users = positive_users[i:i+batch_size]
            batch_pos_items = positive_items[i:i+batch_size]
            
            users.extend(batch_pos_users)
            items.extend(batch_pos_items)
            labels.extend([1] * len(batch_pos_users))
            
            num_positives_in_batch = len(batch_pos_users)
            num_neg_samples = num_positives_in_batch * self.num_negatives
            
            neg_users = np.repeat(batch_pos_users, self.num_negatives)
            neg_items = np.random.choice(self.all_video_ids, size=num_neg_samples)
            
            # Conflict checking
            for j in range(len(neg_users)):
                if (j % 10000 == 0) and (j > 0):
                    print(f"      Checked {j}/{len(neg_users)} conflicts in batch...")
                while (neg_users[j], neg_items[j]) in self.user_item_set:
                    neg_items[j] = np.random.choice(self.all_video_ids)
            
            users.extend(neg_users)
            items.extend(neg_items)
            labels.extend([0] * num_neg_samples)
            
            print(f"      Batch processed in {time.time() - batch_start_time:.2f}s.")
            
        print(f"  Training instances generated in {time.time() - start_time:.2f}s.")
        return torch.LongTensor(users), torch.LongTensor(items), torch.FloatTensor(labels)