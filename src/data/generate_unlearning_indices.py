import pandas as pd
import numpy as np
import os
from collections import Counter

def get_value_counts_from_chunks(data_path, column_name, dtype, chunk_size=1_000_000):
    """Reads a column from a CSV in chunks and returns value counts."""
    counts = Counter()
    for chunk in pd.read_csv(data_path, usecols=[column_name], dtype=dtype, chunksize=chunk_size):
        counts.update(chunk[column_name])
    return pd.Series(counts)

def generate_unlearning_indices(data_path, output_path='.'):
    """
    Generates deterministic train/forget splits for three unlearning scenarios
    by processing the data in chunks to avoid memory errors.

    Args:
        data_path (str): Path to the big_matrix.csv file.
        output_path (str): Directory to save the output files.
    """
    print("Starting unlearning index generation...")
    # Set seed for reproducibility at the very beginning
    np.random.seed(42)
    os.makedirs(output_path, exist_ok=True)

    dtype = {
        'user_id': 'int32',
        'video_id': 'int32',
        'watch_ratio': 'float32',
        'date': 'int32'
    }
    usecols = ['user_id', 'video_id', 'watch_ratio', 'date']
    
    try:
        # Test if the file is accessible before starting the whole process
        with open(data_path, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"Error: The file '{data_path}' was not found.")
        return

    # --- Scenario 1: User Unlearning ---
    print("\nGenerating User Unlearning Scenario...")
    print("  Loading user IDs to determine top 10 users...")
    user_interaction_counts = get_value_counts_from_chunks(data_path, 'user_id', dtype)
    top_10_users = user_interaction_counts.nlargest(10).index
    print(f"  Top 10 users identified: {top_10_users.tolist()}")

    # --- Scenario 2: Item Unlearning ---
    print("\nGenerating Item Unlearning Scenario...")
    print("  Loading video IDs to determine top 5 items...")
    item_interaction_counts = get_value_counts_from_chunks(data_path, 'video_id', dtype)
    top_5_items = item_interaction_counts.nlargest(5).index
    print(f"  Top 5 items identified: {top_5_items.tolist()}")

    # --- Scenario 3: Context Unlearning ---
    print("\nGenerating Context Unlearning Scenario...")
    print("  Loading dates to determine busiest date...")
    date_interaction_counts = get_value_counts_from_chunks(data_path, 'date', dtype)
    busiest_date = date_interaction_counts.nlargest(1).index[0]
    print(f"  Busiest date identified: {busiest_date}")


    print("\nProcessing full dataset in chunks to create splits...")
    # Re-initialize files
    # User scenario
    user_forget_file = os.path.join(output_path, 'user_forget.csv')
    user_retain_file = os.path.join(output_path, 'user_retain.csv')
    user_forget_idx = []
    # Item scenario
    item_forget_file = os.path.join(output_path, 'item_forget.csv')
    item_retain_file = os.path.join(output_path, 'item_retain.csv')
    item_forget_idx = []
    # Context scenario
    context_forget_file = os.path.join(output_path, 'context_forget.csv')
    context_retain_file = os.path.join(output_path, 'context_retain.csv')
    context_forget_idx = []

    header_written = {
        'user': False, 'item': False, 'context': False
    }

    chunk_iterator = pd.read_csv(data_path, dtype=dtype, usecols=usecols, chunksize=1_000_000, index_col=False)

    for i, chunk in enumerate(chunk_iterator):
        print(f"  Processing chunk {i+1}...")

        # User Unlearning
        user_forget_mask = chunk['user_id'].isin(top_10_users)
        user_forget_idx.extend(chunk.index[user_forget_mask] + i * 1_000_000)
        
        chunk[user_forget_mask].to_csv(user_forget_file, mode='a', header=not header_written['user'], index=False)
        chunk[~user_forget_mask].to_csv(user_retain_file, mode='a', header=not header_written['user'], index=False)
        if not header_written['user']:
            header_written['user'] = True

        # Item Unlearning
        item_forget_mask = chunk['video_id'].isin(top_5_items)
        item_forget_idx.extend(chunk.index[item_forget_mask] + i * 1_000_000)

        chunk[item_forget_mask].to_csv(item_forget_file, mode='a', header=not header_written['item'], index=False)
        chunk[~item_forget_mask].to_csv(item_retain_file, mode='a', header=not header_written['item'], index=False)
        if not header_written['item']:
            header_written['item'] = True
            
        # Context Unlearning
        context_forget_mask = chunk['date'] == busiest_date
        context_forget_idx.extend(chunk.index[context_forget_mask] + i * 1_000_000)

        chunk[context_forget_mask].to_csv(context_forget_file, mode='a', header=not header_written['context'], index=False)
        chunk[~context_forget_mask].to_csv(context_retain_file, mode='a', header=not header_written['context'], index=False)
        if not header_written['context']:
            header_written['context'] = True

    print("  Finished processing chunks.")

    # Save the indices
    np.save(os.path.join(output_path, 'user_forget_idx.npy'), np.array(user_forget_idx))
    np.save(os.path.join(output_path, 'item_forget_idx.npy'), np.array(item_forget_idx))
    np.save(os.path.join(output_path, 'context_forget_idx.npy'), np.array(context_forget_idx))

    print(f"\nSaved user splits and indices to '{output_path}'")
    print(f"Saved item splits and indices to '{output_path}'")
    print(f"Saved context splits and indices to '{output_path}'")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate unlearning indices from KuaiRec dataset.")
    parser.add_argument('--data_path', type=str, default='data/"KuaiRec 2.0"/data/big_matrix.csv', help='Path to the big_matrix.csv file.')
    parser.add_argument('--output_path', type=str, default='artifacts/unlearning_indices', help='Directory to save the output files.')
    args = parser.parse_args()
    
    # Sanitize data_path by removing quotes if present
    sanitized_data_path = args.data_path.replace('"', '')
    
    generate_unlearning_indices(sanitized_data_path, args.output_path)
    print("\nAll unlearning scenarios generated successfully.")
