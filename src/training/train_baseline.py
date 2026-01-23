import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import json
from pathlib import Path
import logging
from tqdm import tqdm
import yaml
import time

# Add project root to the Python path
import sys
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.models.model import NCF
from src.data.dataset import NCFDataset
from src.utils.metrics import evaluate_1_vs_99

# --- 1. Global Reproducibility ---
def set_seed(seed):
    """Sets the seed for reproducibility across all relevant libraries."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- Configuration ---
config = {
    "artifacts_dir": "artifacts",
    "models_dir": "models",
    "output_model_name": "ncf_best.pth",
    "embedding_dim": 32,
    "hidden_layers": [64, 32, 16, 8],
    "dropout": 0.1,
    "learning_rate": 0.001,
    "epochs": 50,  # Set a higher max for early stopping
    "batch_size": 1024,
    "num_neg_samples": 4,
    "num_workers": 0,
    "seed": 42,
    "patience": 5, # For early stopping
    "k_eval": 10 # For HR@K and NDCG@K
}

# --- Logging Setup ---
# Remove any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout) # Ensure logs go to console
    ]
)

def train_production():
    """Main production-grade training function."""
    
    set_seed(config['seed'])
    
    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    project_root = Path(__file__).resolve().parents[2]
    artifacts_dir = project_root / config['artifacts_dir']
    models_dir = project_root / config['models_dir']
    models_dir.mkdir(exist_ok=True)
    best_model_path = models_dir / config['output_model_name']
    
    logging.info("--- Configuration ---")
    logging.info(f"Configuration:\n{yaml.dump(config, indent=2)}")

    # --- Load Data ---
    logging.info("STEP 1: Loading data artifacts...")
    train_data = torch.load(artifacts_dir / 'train_dataset.pt', weights_only=True)
    test_data = torch.load(artifacts_dir / 'test_full.pt', weights_only=True)
    with open(artifacts_dir / 'user_map.json', 'r') as f:
        user_map = json.load(f)
    with open(artifacts_dir / 'video_map.json', 'r') as f:
        item_map = json.load(f)
    num_users = len(user_map)
    num_items = len(item_map)
    logging.info("Data loading complete.")
    logging.info(f"Number of users: {num_users}, Number of items: {num_items}")

    
    # --- Create Dataset and DataLoader ---
    logging.info("STEP 2: Creating Dataset and DataLoader...")
    
    train_dataset = NCFDataset(
        train_interactions=train_data, 
        num_items=num_items, 
        num_neg_samples=config['num_neg_samples'], 
        seed=config['seed']
    )
    logging.info("NCFDataset created. Now creating DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=False
    )
    logging.info("Dataset and DataLoader created.")

    # --- Initialize Model, Loss, and Optimizer ---
    logging.info("STEP 3: Initializing NCF model...")
    model = NCF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=config['embedding_dim'],
        hidden_layers=config['hidden_layers'],
        dropout=config['dropout']
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    logging.info("Model, Loss Function, and Optimizer initialized.")
    logging.info(model)
    
    # --- Checkpointing and Early Stopping Setup ---
    best_ndcg = -1
    patience_counter = 0

    logging.info("--- STEP 4: Starting Training ---")
    for epoch in range(config['epochs']):
        logging.info(f"--- Epoch {epoch+1}/{config['epochs']} ---")
        epoch_start_time = time.time()
        
        # --- 2. Dynamic Training Loop (Training Phase) ---
        model.train()
        logging.info("Resampling negatives for the new epoch...")
        train_loader.dataset.resample_negatives()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]", file=sys.stdout)
        for batch_idx, (users, pos_items, neg_items) in enumerate(progress_bar):
            
            # --- 3. Tensor Flattening (Batch Compatibility) ---
            users_rep = users.repeat_interleave(1 + config['num_neg_samples'])
            items = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1).view(-1)
            
            pos_labels = torch.ones(users.size(0), 1)
            neg_labels = torch.zeros(users.size(0), config['num_neg_samples'])
            labels = torch.cat([pos_labels, neg_labels], dim=1).view(-1)
            
            users_rep, items, labels = users_rep.to(device), items.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(users_rep, items)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            if (batch_idx + 1) % 200 == 0:
                logging.info(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # --- 4. Metric-Driven Evaluation (Validation Phase) ---
        model.eval()
        logging.info(f"Epoch {epoch+1} training complete. Starting evaluation...")
        hr, ndcg = evaluate_1_vs_99(
            model=model,
            test_data=test_data,
            num_items=num_items,
            k=config['k_eval'],
            device=device
        )
        
        # --- 7. Detailed Logging ---
        log_message = (
            f"EVALUATION | Epoch {epoch+1:02d} | Duration: {epoch_duration:.2f}s | Train Loss: {avg_train_loss:.4f} | "
            f"HR@{config['k_eval']}: {hr:.4f} | NDCG@{config['k_eval']}: {ndcg:.4f}"
        )
        logging.info(log_message)

        # --- 5. Best-Model Checkpointing ---
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"--> NEW BEST MODEL! Saved to {best_model_path} with NDCG@{config['k_eval']}: {best_ndcg:.4f}")
        else:
            # --- 6. Early Stopping ---
            patience_counter += 1
            logging.info(f"No improvement in NDCG@{config['k_eval']} for {patience_counter} epoch(s). Patience: {patience_counter}/{config['patience']}")
            if patience_counter >= config['patience']:
                logging.warning(f"EARLY STOPPING! Stopping training as NDCG did not improve for {config['patience']} epochs.")
                break # Exit training loop
    
    logging.info("--- Training Finished ---")
    logging.info(f"Best model was saved at '{best_model_path}' with final NDCG@{config['k_eval']}: {best_ndcg:.4f}")

if __name__ == '__main__':
    train_production()
