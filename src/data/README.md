# Data Processing and Preparation

This directory contains the scripts responsible for processing the raw KuaiRec dataset and preparing it for model training and unlearning experiments.

## Scripts

### `prepare_dataset.py`

This is the primary script for creating the training and evaluation datasets. It should be run before any model training.

-   **Input:** The raw `big_matrix.csv` from the KuaiRec dataset.
-   **Process:**
    1.  Reads the raw data in chunks for memory efficiency.
    2.  Filters for strong user preference signals (`watch_ratio > 2.0`).
    3.  Builds integer mappings for user and video IDs.
    4.  Performs a "leave-one-out" split, using each user's latest interaction for the test set and all other interactions for the training set.
    5.  Generates training instances with 4 negative samples for each positive interaction.
    6.  Generates test instances with 99 negative samples for each positive interaction to create a ranked list for evaluation.
-   **Output (in `artifacts/` directory):**
    -   `train_dataset.pt`: A PyTorch `Dataset` object containing all training instances.
    -   `test_users.pt`: A tensor of user IDs for the evaluation set.
    -   `test_items.pt`: A tensor of item IDs (1 positive + 99 negative per user) for the evaluation set.
    -   `user_map.json` & `video_map.json`: Dictionaries mapping original user/video IDs to their new integer indices.

### `generate_unlearning_indices.py`

This script generates the data files required for the machine unlearning scenarios.

-   **Input:** The raw `big_matrix.csv`.
-   **Process:** Identifies target users, items, and dates based on interaction counts and splits the entire raw dataset into `forget` and `retain` files for each scenario.
-   **Output (in `artifacts/unlearning_indices/` directory):**
    -   `user_forget.csv` / `user_retain.csv`
    -   `item_forget.csv` / `item_retain.csv`
    -   `context_forget.csv` / `context_retain.csv`
    -   `..._forget_idx.npy` files containing the row indices for each forget set.

### `dataset.py`

This module defines the `NCFDataset` class, which is used by `prepare_dataset.py` to create the training dataset with on-the-fly negative sampling.
