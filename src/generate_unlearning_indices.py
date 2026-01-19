import pandas as pd
import numpy as np
import os

def generate_unlearning_indices(data_path, output_path='.'):
    """
    Generates deterministic train/forget splits for three unlearning scenarios.

    Args:
        data_path (str): Path to the big_matrix.csv file.
        output_path (str): Directory to save the output files.
    """
    print("Loading data for unlearning index generation...")
    # Use dtype optimization and usecols to save memory
    dtype = {
        'user_id': 'int32',
        'video_id': 'int32',
        'watch_ratio': 'float32',
        'date': 'int32'
    }
    usecols = ['user_id', 'video_id', 'watch_ratio', 'date']
    try:
        df = pd.read_csv(data_path, dtype=dtype, usecols=usecols)
    except FileNotFoundError:
        print(f"Error: The file '{data_path}' was not found.")
        return

    # Set seed for reproducibility
    np.random.seed(42)
    
    os.makedirs(output_path, exist_ok=True)

    # --- Scenario 1: User Unlearning ---
    print("\nGenerating User Unlearning Scenario...")
    user_interaction_counts = df['user_id'].value_counts()
    top_10_users = user_interaction_counts.nlargest(10).index
    print(f"  Top 10 users identified: {top_10_users.tolist()}")

    user_forget_mask = df['user_id'].isin(top_10_users)
    user_forget_idx = df.index[user_forget_mask].to_numpy()
    
    user_forget_df = df[user_forget_mask]
    user_retain_df = df[~user_forget_mask]

    user_forget_df.to_csv(os.path.join(output_path, 'user_forget.csv'), index=False)
    user_retain_df.to_csv(os.path.join(output_path, 'user_retain.csv'), index=False)
    np.save(os.path.join(output_path, 'user_forget_idx.npy'), user_forget_idx)
    print(f"  Saved user_forget.csv, user_retain.csv, and user_forget_idx.npy to '{output_path}'")


    # --- Scenario 2: Item Unlearning ---
    print("\nGenerating Item Unlearning Scenario...")
    item_interaction_counts = df['video_id'].value_counts()
    top_5_items = item_interaction_counts.nlargest(5).index
    print(f"  Top 5 items identified: {top_5_items.tolist()}")

    item_forget_mask = df['video_id'].isin(top_5_items)
    item_forget_idx = df.index[item_forget_mask].to_numpy()

    item_forget_df = df[item_forget_mask]
    item_retain_df = df[~item_forget_mask]
    
    item_forget_df.to_csv(os.path.join(output_path, 'item_forget.csv'), index=False)
    item_retain_df.to_csv(os.path.join(output_path, 'item_retain.csv'), index=False)
    np.save(os.path.join(output_path, 'item_forget_idx.npy'), item_forget_idx)
    print(f"  Saved item_forget.csv, item_retain.csv, and item_forget_idx.npy to '{output_path}'")
    

    # --- Scenario 3: Context Unlearning ---
    print("\nGenerating Context Unlearning Scenario...")
    date_interaction_counts = df['date'].value_counts()
    busiest_date = date_interaction_counts.nlargest(1).index[0]
    print(f"  Busiest date identified: {busiest_date}")

    context_forget_mask = df['date'] == busiest_date
    context_forget_idx = df.index[context_forget_mask].to_numpy()

    context_forget_df = df[context_forget_mask]
    context_retain_df = df[~context_forget_mask]

    context_forget_df.to_csv(os.path.join(output_path, 'context_forget.csv'), index=False)
    context_retain_df.to_csv(os.path.join(output_path, 'context_retain.csv'), index=False)
    np.save(os.path.join(output_path, 'context_forget_idx.npy'), context_forget_idx)
    print(f"  Saved context_forget.csv, context_retain.csv, and context_forget_idx.npy to '{output_path}'")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate unlearning indices from KuaiRec dataset.")
    parser.add_argument('--data_path', type=str, default='data/KuaiRec 2.0/data/big_matrix.csv', help='Path to the big_matrix.csv file.')
    parser.add_argument('--output_path', type=str, default='unlearning_indices', help='Directory to save the output files.')
    args = parser.parse_args()
    
    generate_unlearning_indices(args.data_path, args.output_path)
    print("\nAll unlearning scenarios generated successfully.")
