import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_dataset(
    raw_data_path: Path,
    output_dir: Path,
    min_watch_ratio: float = 2.0,
    chunksize: int = 1_000_000,
    seed: int = 42
):
    """
    Prepares the dataset from raw KuaiRec 2.0 data, processing the file in chunks
    to handle large file sizes and avoid out-of-memory errors.

    Args:
        raw_data_path (Path): Path to the raw 'big_matrix.csv' file.
        output_dir (Path): Directory to save the processed files.
        min_watch_ratio (float): Minimum watch ratio to consider as a positive interaction.
        chunksize (int): The number of rows to read into memory at a time.
        seed (int): Random seed for reproducibility.
    """
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- First Pass: Discover all unique user and item IDs ---
    logging.info("First pass: Discovering all unique user and item IDs from positive interactions.")
    user_ids = set()
    item_ids = set()
    
    reader = pd.read_csv(raw_data_path, chunksize=chunksize, usecols=['user_id', 'video_id', 'watch_ratio'])
    for chunk in tqdm(reader, desc="Scanning for positive interactions"):
        positive_chunk = chunk[chunk['watch_ratio'] >= min_watch_ratio]
        user_ids.update(positive_chunk['user_id'].unique())
        item_ids.update(positive_chunk['video_id'].unique())

    # --- Create deterministic mappings for the discovered IDs ---
    logging.info("Creating user and item ID mappings...")
    unique_users = sorted(list(user_ids))
    unique_items = sorted(list(item_ids))

    user_map = {user_id: i for i, user_id in enumerate(unique_users)}
    item_map = {video_id: i for i, video_id in enumerate(unique_items)}

    # Convert numpy types to native Python types for JSON serialization
    user_map_py = {int(k): v for k, v in user_map.items()}
    item_map_py = {int(k): v for k, v in item_map.items()}

    # Save mappings
    with open(output_dir / 'user_map.json', 'w') as f:
        json.dump(user_map_py, f)
    with open(output_dir / 'video_map.json', 'w') as f:
        json.dump(item_map_py, f)
    logging.info(f"Saved user_map.json and video_map.json to {output_dir}")
    
    # --- Second Pass: Process data in chunks ---
    logging.info("Second pass: Processing interactions in chunks...")
    processed_chunks = []
    reader = pd.read_csv(raw_data_path, chunksize=chunksize)
    for chunk in tqdm(reader, desc="Processing chunks"):
        # Filter for positive interactions
        chunk_pos = chunk[chunk['watch_ratio'] >= min_watch_ratio].copy()
        
        # Apply mappings. Only keep users/items that are in our final maps.
        chunk_pos['user_id'] = chunk_pos['user_id'].map(user_map)
        chunk_pos['video_id'] = chunk_pos['video_id'].map(item_map)
        
        # Drop rows where mapping resulted in NaN
        chunk_pos.dropna(subset=['user_id', 'video_id'], inplace=True)
        chunk_pos['user_id'] = chunk_pos['user_id'].astype(int)
        chunk_pos['video_id'] = chunk_pos['video_id'].astype(int)

        processed_chunks.append(chunk_pos[['user_id', 'video_id', 'timestamp']])

    # --- Combine, Split, and Save ---
    logging.info("Combining processed chunks...")
    df_pos = pd.concat(processed_chunks, ignore_index=True)

    # Ensure all interactions are unique
    logging.info(f"Dropping duplicate user-item interactions. Original count: {len(df_pos)}")
    df_pos.drop_duplicates(subset=['user_id', 'video_id'], inplace=True)
    logging.info(f"Count after dropping duplicates: {len(df_pos)}")

    # Sort by timestamp to prepare for leave-one-out split
    df_pos = df_pos.sort_values(by='timestamp').reset_index(drop=True)

    # --- 2. Guaranteed Zero-Leakage (Leave-One-Out) ---
    logging.info("Performing leave-one-out split with guaranteed zero leakage...")
    test_data = df_pos.groupby('user_id').tail(1).copy()

    # Explicitly remove test interactions from the training set using an anti-join
    # This is more robust than relying on index dropping.
    train_data = df_pos.merge(
        test_data[['user_id', 'video_id']],
        on=['user_id', 'video_id'],
        how='left',
        indicator=True
    )
    train_data = train_data[train_data['_merge'] == 'left_only'].drop(columns=['_merge'])

    # --- 3. Integrity Report ---
    logging.info("Verifying train/test split integrity...")
    train_tuples = set(map(tuple, train_data[['user_id', 'video_id']].values))
    test_tuples = set(map(tuple, test_data[['user_id', 'video_id']].values))
    
    overlap = train_tuples.intersection(test_tuples)
    
    logging.info(f"Integrity Check: Overlap between train and test sets: {len(overlap)}")
    if len(overlap) > 0:
        logging.error("FATAL: Data leakage detected! Train and test sets have overlapping user-item pairs.")
        raise ValueError("Data leakage detected between train and test sets.")
    else:
        logging.info("SUCCESS: Zero data leakage confirmed.")
        
    # Final processed datasets
    train_dataset = torch.tensor(train_data[['user_id', 'video_id']].values, dtype=torch.long)
    test_dataset = torch.tensor(test_data[['user_id', 'video_id']].values, dtype=torch.long)
    
    # Save datasets
    torch.save(train_dataset, output_dir / 'train_dataset.pt')
    torch.save(test_dataset, output_dir / 'test_full.pt')

    logging.info(f"Saved train_dataset.pt ({len(train_dataset)} interactions) and test_full.pt ({len(test_dataset)} interactions).")
    logging.info("Dataset preparation complete.")


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[2]
    # Assuming the user has placed the data in the nested directory
    raw_data_path = project_root / 'data' / 'KuaiRec 2.0' / 'data' / 'big_matrix.csv'
    output_dir = project_root / 'artifacts'

    if not raw_data_path.exists():
        logging.error(f"FATAL: Raw data not found at {raw_data_path}")
        logging.error("Please ensure 'big_matrix.csv' is in the 'data/KuaiRec 2.0/data/' directory.")
    else:
        prepare_dataset(raw_data_path=raw_data_path, output_dir=output_dir)
