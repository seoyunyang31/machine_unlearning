import torch
import numpy as np
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def hit_ratio_at_k(predictions: torch.Tensor, k: int) -> float:
    """
    Calculates the Hit Ratio @ K.
    The ground truth item is assumed to be at index 0.
    """
    _, top_k_indices = torch.topk(predictions, k)
    # The positive item is always at index 0
    if 0 in top_k_indices:
        return 1.0
    return 0.0

def ndcg_at_k(predictions: torch.Tensor, k: int) -> float:
    """
    Calculates the Normalized Discounted Cumulative Gain @ K.
    The ground truth item is assumed to be at index 0.
    """
    _, top_k_indices = torch.topk(predictions, k)
    # Find the rank of the positive item (at index 0)
    # The 'nonzero()' function returns a tensor of indices where the condition is true.
    # We get the first element because there will be only one match.
    res = (top_k_indices == 0).nonzero()
    
    if res.shape[0] > 0:
        rank = res[0].item()
        # IDCG is always 1 in this setting (one positive item)
        # DCG = 1 / log2(rank + 2)
        return 1.0 / np.log2(rank + 2)
    return 0.0

def evaluate_1_vs_99(
    model,
    test_data: torch.Tensor,
    num_items: int,
    k: int = 10,
    device: str = 'cpu'
):
    """
    Performs evaluation using the 1 vs. 99 protocol.
    For each user-item pair in the test set, the item is ranked against
    99 randomly sampled negative items.

    Args:
        model: The trained NCF model.
        test_data (torch.Tensor): The test set (user, item pairs).
        num_items (int): Total number of items in the dataset.
        k (int): The K for HR@K and NDCG@K.
        device (str): The device to run evaluation on ('cpu' or 'cuda').

    Returns:
        A tuple of (mean HR, mean NDCG).
    """
    model.eval()
    model.to(device)

    total_hr = []
    total_ndcg = []
    
    # Generate a fixed set of negative samples for the entire test set for consistency
    # For each test user, we sample 99 negatives
    test_negatives = {}
    rng = np.random.default_rng(42)
    for user, pos_item in test_data:
        user = user.item()
        pos_item = pos_item.item()
        if user not in test_negatives:
            neg_samples = set()
            while len(neg_samples) < 99:
                sample = rng.integers(0, num_items)
                if sample != pos_item:
                    neg_samples.add(sample)
            test_negatives[user] = list(neg_samples)

    with torch.no_grad():
        for user, pos_item in tqdm(test_data, desc="Evaluating"):
            user_id = user.item()
            pos_item_id = pos_item.item()

            # Get the 99 negative items for this user
            neg_item_ids = test_negatives[user_id]

            # Create the 100 items to rank: 1 positive + 99 negative
            items_to_rank = torch.tensor([pos_item_id] + neg_item_ids, dtype=torch.long).to(device)
            # Create user tensor, repeated 100 times
            user_tensor = torch.full((100,), user_id, dtype=torch.long).to(device)

            # Get model predictions
            predictions = model(user_tensor, items_to_rank)

            # Calculate metrics
            hr = hit_ratio_at_k(predictions, k)
            ndcg = ndcg_at_k(predictions, k)

            total_hr.append(hr)
            total_ndcg.append(ndcg)

    mean_hr = np.mean(total_hr)
    mean_ndcg = np.mean(total_ndcg)
    
    return mean_hr, mean_ndcg

if __name__ == '__main__':
    # --- Example Usage ---
    # This is a dummy example to show how the functions work.
    
    # A set of predictions for 1 positive item (at index 0) and 4 negative items
    # Example 1: Positive item is ranked 2nd
    predictions1 = torch.tensor([3.5, 4.0, 2.0, 1.0, 3.8])
    hr_ex1 = hit_ratio_at_k(predictions1, k=3)
    ndcg_ex1 = ndcg_at_k(predictions1, k=3)
    print(f"Example 1 (Hit @ 3): HR = {hr_ex1:.4f}, NDCG = {ndcg_ex1:.4f}")

    # Example 2: Positive item is ranked 4th (miss)
    predictions2 = torch.tensor([2.0, 4.0, 3.5, 3.8, 1.0])
    hr_ex2 = hit_ratio_at_k(predictions2, k=3)
    ndcg_ex2 = ndcg_at_k(predictions2, k=3)
    print(f"Example 2 (Miss @ 3): HR = {hr_ex2:.4f}, NDCG = {ndcg_ex2:.4f}")