#load baseline model
#load forget/retain data
#compute Fisher on retain set
#save original params theta_star
#run unlearning optimization
#evaluate and save result




#Load train_dataset, test_full, user_map, video_map
#build the same NCFDataset/Dataloader
#initialize NCF
#load baseline weights from models/ncf_best

#load forget indices and pairs
#make retain data
#implement fisher computation ___

#save copy of original weights
#do unlearning training loop




import torch
import numpy as np
import random
import json
from pathlib import Path
import logging
import yaml
import sys

#need project root
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.models.model import NCF
from src.data.dataset import NCFDataset
from src.utils.metrics import evaluate_1_vs_99


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


config = {
    "artifacts_dir": "artifacts",
    "models_dir": "models",
    "baseline_model_name": "ncf_best.pth",

    "embedding_dim": 32,
    "hidden_layers": [64, 32, 16, 8],
    "dropout": 0.1,

    "batch_size": 1024,
    "num_neg_samples": 4,
    "num_workers": 0,
    "seed": 42,
    "k_eval": 10,
}


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def main():
    set_seed(config["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    project_root = Path(__file__).resolve().parents[2]
    artifacts_dir = project_root / config["artifacts_dir"]
    models_dir = project_root / config["models_dir"]
    baseline_model_path = models_dir / config["baseline_model_name"]

    logging.info("--- Configuration ---")
    logging.info(f"\n{yaml.dump(config, indent=2)}")

    # 1. Load artifacts
    logging.info("Loading data artifacts...")
    train_data = torch.load(artifacts_dir / "train_dataset.pt", weights_only=True)
    test_data = torch.load(artifacts_dir / "test_full.pt", weights_only=True)

    with open(artifacts_dir / "user_map.json", "r") as f:
        user_map = json.load(f)

    with open(artifacts_dir / "video_map.json", "r") as f:
        item_map = json.load(f)

    num_users = len(user_map)
    num_items = len(item_map)

    logging.info(f"Loaded train_data shape: {train_data.shape}")
    logging.info(f"Loaded test_data shape: {test_data.shape}")
    logging.info(f"num_users: {num_users}")
    logging.info(f"num_items: {num_items}")

    # 2. Build dataset just to make sure it works
    logging.info("Building NCFDataset...")
    train_dataset = NCFDataset(
        train_interactions=train_data,
        num_items=num_items,
        num_neg_samples=config["num_neg_samples"],
        seed=config["seed"]
    )

    logging.info(f"Dataset built successfully. Length: {len(train_dataset)}")

    # 3. Build model
    logging.info("Building NCF model...")
    model = NCF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=config["embedding_dim"],
        hidden_layers=config["hidden_layers"],
        dropout=config["dropout"]
    ).to(device)

    logging.info("Model created successfully.")

    # 4. Load baseline weights
    logging.info(f"Loading baseline weights from: {baseline_model_path}")
    state_dict = torch.load(baseline_model_path, map_location=device)
    model.load_state_dict(state_dict)

    logging.info("Baseline model loaded successfully.")

    # 5. Optional sanity evaluation
    logging.info("Running sanity evaluation...")
    model.eval()
    hr, ndcg = evaluate_1_vs_99(
        model=model,
        test_data=test_data,
        num_items=num_items,
        k=config["k_eval"],
        device=device
    )

    logging.info(f"Sanity check complete | HR@{config['k_eval']}: {hr:.4f} | NDCG@{config['k_eval']}: {ndcg:.4f}")
    logging.info("Everything loaded correctly.")


if __name__ == "__main__":
    main()