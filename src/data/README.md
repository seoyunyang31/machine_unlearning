# Data Processing and Preparation

This directory contains the scripts responsible for processing the raw KuaiRec dataset. There are two primary, separate workflows.

## Workflows

### 1. Baseline Model Training

This is the standard workflow for training the initial "teacher" model.

1.  **Run `prepare_dataset.py`:** This script processes `big_matrix.csv` to create the main training and test sets (`train_dataset.pt` and `test_full.pt`). This is a required first step before running `train_baseline.py`.
    ```bash
    python src/data/prepare_dataset.py
    ```

### 2. Unlearning Scenario Preparation

This workflow is only required when conducting unlearning experiments.

1.  **Run `generate_unlearning_indices.py`:** This script also processes `big_matrix.csv` to identify different subsets of data to be "forgotten" and "retained," saving them as separate `.csv` and `.npy` files. These artifacts are used by specialized unlearning scripts.
    ```bash
    python src/data/generate_unlearning_indices.py
    ```

---

## Script Details

### `prepare_dataset.py`

This is the primary script for creating the datasets for baseline training.

-   **Input:** The raw `big_matrix.csv` from the KuaiRec dataset.
-   **Process:**
    1.  Reads the raw data in chunks for memory efficiency.
    2.  Filters for positive user interactions (`watch_ratio >= 2.0`).
    3.  Builds and saves integer mappings for user and video IDs.
    4.  Deduplicates interactions to ensure only unique user-item pairs are kept.
    5.  Performs a "leave-one-out" split, using each user's most recent interaction for the test set.
    6.  Uses a robust anti-join to guarantee zero data leakage between the train and test sets.
-   **Output (in `artifacts/` directory):**
    -   `train_dataset.pt`: A tensor of positive user-item interaction pairs for training.
    -   `test_full.pt`: A tensor containing the single "left-out" positive interaction for each user, to be used for evaluation.
    -   `user_map.json` & `video_map.json`: Dictionaries mapping original user/video IDs to their new integer indices.

### `generate_unlearning_indices.py`

This script generates the data files required for machine unlearning scenarios. It identifies target users/items and splits the raw dataset into `forget` and `retain` files.

-   **Output (in `artifacts/unlearning_indices/` directory):**
    -   `..._forget.csv` / `..._retain.csv` files for different unlearning scenarios.
    -   `..._forget_idx.npy` files containing the row indices for each forget set.

### `dataset.py`

This module defines the `NCFDataset` class. It is **used by the `train_baseline.py` script**, not by the data preparation scripts. It is not a script and should not be run directly. Its primary role is to enable efficient, on-the-fly negative sampling during the training loop.
