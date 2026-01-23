import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NCFDataset(Dataset):
    """
    PyTorch Dataset for NCF training.
    Handles Just-In-Time (JIT) negative sampling to conserve memory.
    This version creates the user-item lookup map internally for efficiency.
    """

    def __init__(
        self,
        train_interactions: torch.Tensor,
        num_items: int,
        num_neg_samples: int = 4,
        seed: int = 42
    ):
        """
        Args:
            train_interactions (torch.Tensor): Tensor of user-item interaction pairs.
            num_items (int): Total number of unique items in the dataset.
            num_neg_samples (int): Number of negative samples per positive interaction.
            seed (int): Random seed for reproducibility.
        """
        super().__init__()
        self.train_interactions = train_interactions
        self.num_items = num_items
        self.num_neg_samples = num_neg_samples
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        # Efficiently build the user-item map internally
        logging.info("Building user-item map internally within NCFDataset...")
        df = pd.DataFrame(train_interactions.numpy(), columns=['user_id', 'item_id'])
        self.user_item_map = df.groupby('user_id')['item_id'].apply(set)
        logging.info("User-item map built successfully.")


    def resample_negatives(self):
        """
        Re-initializes the random number generator at the start of each epoch.
        This ensures that the on-the-fly negative samples are different per epoch.
        """
        logging.info("Resetting RNG for new epoch of on-the-fly negative sampling.")
        # Increment seed to get different samples next epoch
        self.seed += 1
        self.rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return len(self.train_interactions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of (user, positive_item, negative_items).
        Negative items are sampled on-the-fly here.
        """
        user_tensor, pos_item_tensor = self.train_interactions[idx]
        user = user_tensor.item()
        
        pos_items = self.user_item_map.get(user, set())
        
        neg_samples = []
        while len(neg_samples) < self.num_neg_samples:
            # Generate a batch of random samples
            samples = self.rng.integers(0, self.num_items, size=(self.num_neg_samples * 2))
            # Filter out samples that are positive items for the user
            valid_samples = [s for s in samples if s not in pos_items]
            neg_samples.extend(valid_samples)
            
        # Trim to the required number of samples
        neg_items = torch.tensor(neg_samples[:self.num_neg_samples], dtype=torch.long)
        
        return user_tensor, pos_item_tensor, neg_items