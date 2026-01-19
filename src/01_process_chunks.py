import pandas as pd
import time
import json
import os

def process_data_chunkwise(file_path, chunk_size=1000000):
    """
    Loads and preprocesses the KuaiRec dataset in chunks to handle large file sizes,
    using vectorized operations for better performance.
    Saves the train and test dataframes and mappings to disk.
    """
    print("Processing data in chunks...")
    start_time = time.time()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
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
    with open(os.path.join(script_dir, 'user_map.json'), 'w') as f:
        json.dump(user_map_serializable, f)
    with open(os.path.join(script_dir, 'video_map.json'), 'w') as f:
        json.dump(video_map_serializable, f)
    print("  Maps saved.")

    print("Creating and saving train and test sets...")
    
    test_items_set = set()
    for user_id, data in user_latest_interaction.items():
        test_items_set.add((user_id, data['video_id']))

    all_interactions_df['is_test'] = all_interactions_df.apply(lambda row: (row['user_id'], row['video_id']) in test_items_set, axis=1)

    train_df = all_interactions_df[~all_interactions_df['is_test']].drop(columns=['is_test'])
    test_df = all_interactions_df[all_interactions_df['is_test']].drop(columns=['is_test'])
    
    train_df.to_pickle(os.path.join(script_dir, 'train_df.pkl'))
    test_df.to_pickle(os.path.join(script_dir, 'test_df.pkl'))
    
    print(f"Train and test sets saved to '{os.path.join(script_dir, 'train_df.pkl')}' and '{os.path.join(script_dir, 'test_df.pkl')}'.")


if __name__ == '__main__':
    print("Starting Step 1: Processing data and creating train/test splits...")
    # The script is in /app/src, so the data is at ../data/...
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'KuaiRec 2.0', 'data', 'big_matrix.csv')
    process_data_chunkwise(data_path)
    print("Step 1 complete.")
