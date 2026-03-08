import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import json
import pandas as pd
from pathlib import Path
import logging
import yaml
import sys
import time

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
    "unlearning_scenario": "user",          # "user" | "item" | "context"
    "output_model_name": "ncf_unlearned.pth",
    "metrics_output_name": "unlearning_metrics.json",

    "embedding_dim": 32,
    "hidden_layers": [64, 32, 16, 8],
    "dropout": 0.1,

    "batch_size": 1024,
    "num_neg_samples": 4,
    "num_workers": 0,
    "seed": 42,
    "k_eval": 10,

    "unlearn_epochs": 5,
    "unlearn_lr": 5e-4,
    "ewc_lambda": 1000.0,
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


def calculate_fisher(model, model_path, retain_loader, device):

    # Load the saved baseline model weights into the model.
    # Fisher should be computed from the trained model, not a random one.
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Put the model in evaluation mode.

    criterion = nn.BCEWithLogitsLoss() # Define loss here so this function is self-contained

    fisher = {} # Fisher information for each parameter
    for name, param in model.named_parameters():
        if param.requires_grad: # Only compute Fisher for trainable parameters
            fisher[name] = torch.zeros_like(param) # Initialize the Fisher value for this parameter as all zeros and keep same shape

    num_batches = 0 # Start a counter for how many retain batches we process

    for users, items, labels in retain_loader: # Loop through the retain dataset batch by batch

        users = users.to(device)
        items = items.to(device)
        labels = labels.float().to(device)

        model.zero_grad() # Clear old gradients from the previous batch

        outputs = model(users, items) # Run a forward pass through the model to get predictions

        loss = criterion(outputs.squeeze(), labels) # Compute the batch loss

        loss.backward() # Backpropagate the loss. This computes gradients of the batch loss with respect to every parameter

        for name, param in model.named_parameters(): # Loop again through every parameter so we can collect its gradient
            if param.requires_grad and param.grad is not None: # Only use trainable parameters that actually got a gradient
                fisher[name] += param.grad.detach() ** 2 # F_i <- F_i + (dL_b / dtheta_i)^2
        num_batches += 1 # Later we divide by total number of batches to get the average

    for name in fisher: # Divide the accumulated squared gradients by the number of batches
        fisher[name] /= max(num_batches, 1) # F_i = (1 / B) * sum_b (dL_b / dtheta_i)^2

    return fisher


def save_original_weights(model): # model's original trainable weights
    original_weight = {} # Create an empty dictionary to store the baseline parameter values
    for name, param in model.named_parameters(): # Only save parameters that are trainable. These are the weights that unlearning may later change
        if param.requires_grad:
            original_weight[name] = param.detach().clone()
            # Save a separate copy of this parameter.
            # detach() removes it from gradient tracking.
            # clone() makes a real copy so it will not change when the model updates.
    return original_weight


# computes the EWC regularization penalty
def ewc_penalty(model, fisher, theta_star):
    # let the model change enough to learn/forget something new
    # but slow down changes on important weights so it does not destroy old useful knowledge
    penalty = 0.0 # Start the total penalty at 0.

    for name, param in model.named_parameters(): # Loop through every parameter in the current model
        if param.requires_grad: # Only use trainable parameters
            f   = fisher[name].to(param.device)     # move to same device as param
            ref = theta_star[name].to(param.device) # move to same device as param
            penalty += (f * (param - ref) ** 2).sum()
            # Compute the weighted squared distance between current weights
            # and original baseline weights, then add it to the total penalty
            # (param - theta_star[name])
            # = how far this weight moved from the original model
            #
            # ** 2
            # = square that movement
            #
            # fisher[name] * ...
            # = weight the movement by Fisher importance
            #
            # .sum()
            # = turn the whole tensor into one scalar contribution
    return penalty


# takes original train_data and removes all forget pairs
# result: only the data the model is still allowed to remember
def make_retain_data(train_data, forget_pairs):

    keep_mask = torch.ones(len(train_data), dtype=torch.bool) # Start by marking every row as keep

    for idx, (user_id, item_id) in enumerate(train_data.tolist()): # Loop through every interaction
        if (int(user_id), int(item_id)) in forget_pairs: # Check if this pair is in the forget set
            keep_mask[idx] = False # If so, mark it for removal

    retain_data = train_data[keep_mask] # Apply the mask to get only the rows we are keeping

    logging.info(f"make_retain_data | Removed {len(train_data) - len(retain_data)} forget interactions. {len(retain_data)} retain interactions remaining.")
    return retain_data


# runs the actual forgetting process
# updates the model to perform worse on forget samples while using the EWC penalty to protect retained knowledge
# result: the model becomes unlearned
def unlearning_train_loop(model, forget_loader, retain_data, fisher, theta_star,
                          num_items, device, epochs, lr, ewc_lambda):

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Adam optimiser same as baseline training
    criterion = nn.BCEWithLogitsLoss() # Same loss function as baseline training

    for epoch in range(epochs):
        epoch_start        = time.time()
        total_forget_loss  = 0.0
        total_penalty      = 0.0
        total_combined     = 0.0
        num_batches        = 0

        for users, items, labels in forget_loader: # Loop through forget batches
            users  = users.to(device)
            items  = items.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad() # Clear old gradients

            outputs     = model(users, items) # Forward pass on forget pairs
            forget_loss = criterion(outputs.squeeze(), labels) # Compute forget loss

            # Gradient ascent — negate the loss so we go uphill instead of downhill.
            # Normal training minimises loss to get better. Here we maximise it to get worse.
            # That is what makes the model forget.

            penalty = ewc_penalty(model, fisher, theta_star) # Compute EWC penalty to protect retain weights

            combined_loss = -forget_loss + ewc_lambda * penalty # Negate forget loss to ascend, add EWC to protect

            combined_loss.backward() # Backpropagate combined loss
            optimizer.step() # Update weights

            total_forget_loss += forget_loss.item()
            total_penalty     += penalty.item()
            total_combined    += combined_loss.item()
            num_batches       += 1

        epoch_time   = time.time() - epoch_start
        avg_forget   = total_forget_loss / max(num_batches, 1)
        avg_penalty  = total_penalty     / max(num_batches, 1)
        avg_combined = total_combined    / max(num_batches, 1)

        logging.info(
            f"Epoch {epoch+1}/{epochs} [{epoch_time:.1f}s] | "
            f"forget_loss: {avg_forget:.4f} | "
            f"ewc_penalty: {avg_penalty:.4f} | "
            f"combined_loss: {avg_combined:.4f}"
        )

    logging.info("unlearning_train_loop | Finished.")
    return model


# tests how well the unlearned model performs then saves the final model and metrics
# result: we can check whether forgetting worked and keep the output
def evaluate_and_save(model, test_data, forget_data, num_items, device,
                      models_dir, artifacts_dir, output_model_name,
                      metrics_output_name, k):

    model.eval() # Put model in evaluation mode
    models_dir.mkdir(parents=True, exist_ok=True)

    metrics = {} # Empty dict to collect all results

    # Run the standard 1-vs-99 evaluation on the full test set.
    # This checks overall recommendation quality did not collapse after unlearning.
    logging.info("  Evaluating on full test set...")
    hr, ndcg = evaluate_1_vs_99(
        model=model,
        test_data=test_data,
        num_items=num_items,
        k=k,
        device=device,
    )
    metrics[f"test_HR@{k}"]   = float(hr)
    metrics[f"test_NDCG@{k}"] = float(ndcg)
    logging.info(f"  Full test | HR@{k}: {hr:.4f} | NDCG@{k}: {ndcg:.4f}")

    # Compute average raw logit score on forget pairs.
    # After unlearning the model should score these LOW — it no longer confidently recommends them.
    logging.info("  Computing average score on forget pairs...")
    all_scores = [] # Collect scores batch by batch
    with torch.no_grad(): # No gradients needed for evaluation
        for start in range(0, len(forget_data), config["batch_size"]):
            batch  = forget_data[start : start + config["batch_size"]]
            users  = batch[:, 0].to(device)
            items  = batch[:, 1].to(device)
            scores = model(users, items) # Get raw logit scores
            all_scores.append(scores.cpu())
    forget_avg_score = float(torch.cat(all_scores).mean().item()) # Average across all forget pairs
    metrics["forget_avg_score"] = forget_avg_score
    logging.info(f"  Forget avg score: {forget_avg_score:.4f}  (lower = better forgetting)")

    # Save the unlearned model weights so we can use or inspect them later
    model_save_path = models_dir / output_model_name
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"  Model saved to {model_save_path}")

    # Save all metrics to a JSON file so we have a record of how well unlearning worked
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_save_path = artifacts_dir / metrics_output_name
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"  Metrics saved to {metrics_save_path}")

    logging.info("=" * 55)
    logging.info("UNLEARNING EVALUATION SUMMARY")
    logging.info("=" * 55)
    for key, val in metrics.items():
        logging.info(f"  {key:<25} : {val:.4f}")
    logging.info("=" * 55)

    return metrics


def main():
    set_seed(config["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    project_root = Path(__file__).resolve().parents[2]
    artifacts_dir       = project_root / config["artifacts_dir"]
    models_dir          = project_root / config["models_dir"]
    unlearn_dir         = artifacts_dir / "unlearning_indices"
    baseline_model_path = models_dir / config["baseline_model_name"]

    logging.info("--- Configuration ---")
    logging.info(f"\n{yaml.dump(config, indent=2)}")

    # 1. Load artifacts
    logging.info("Loading data artifacts...")
    train_data = torch.load(artifacts_dir / "train_dataset.pt", weights_only=True)
    test_data  = torch.load(artifacts_dir / "test_full.pt",     weights_only=True)

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

    # 2. Build model
    logging.info("Building NCF model...")
    model = NCF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=config["embedding_dim"],
        hidden_layers=config["hidden_layers"],
        dropout=config["dropout"]
    ).to(device)

    logging.info("Model created successfully.")

    # 3. Load baseline weights
    logging.info(f"Loading baseline weights from: {baseline_model_path}")
    state_dict = torch.load(baseline_model_path, map_location=device)
    model.load_state_dict(state_dict)

    logging.info("Baseline model loaded successfully.")

    # 4. Sanity check — make sure the baseline model loaded correctly before we touch it
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

    # 5. Load forget CSV and find all train_data rows belonging to forget users.
    # The forget scenario is user-based so we forget ALL interactions for those users.
    # We get the unique original user IDs from the CSV, map them to model indices,
    # then find every row in train_data whose user index is in that set.
    logging.info("Loading forget data...")
    scenario        = config["unlearning_scenario"]
    forget_csv_path = unlearn_dir / f"{scenario}_forget.csv"
    forget_df       = pd.read_csv(forget_csv_path, usecols=["user_id", "video_id"])

    # Convert original user IDs to model indices using user_map
    forget_user_ids    = set(str(int(u)) for u in forget_df["user_id"].unique())
    forget_user_mapped = set(user_map[u] for u in forget_user_ids if u in user_map)
    logging.info(f"Forget users mapped indices: {forget_user_mapped}")

    # Find all rows in train_data where the user is one of the forget users
    train_list  = train_data.tolist() # Convert once for fast iteration
    forget_mask = torch.tensor([int(u) in forget_user_mapped for u, i in train_list], dtype=torch.bool)
    forget_data = train_data[forget_mask] # All interactions belonging to forget users
    forget_pairs = {(int(u), int(i)) for u, i in forget_data.tolist()} # For make_retain_data lookup
    logging.info(f"Forget interactions found in train_data: {len(forget_data)}")

    # 6. make_retain_data — remove forget pairs from training data
    logging.info("Step 6 | make_retain_data...")
    retain_data = make_retain_data(train_data, forget_pairs)

    # 7. Build retain DataLoader for calculate_fisher.
    # Fisher expects (users, items, labels) so we wrap the retain tensor in a TensorDataset.
    # All retain interactions are positive so we label them all 1.
    logging.info("Building retain DataLoader for Fisher computation...")
    retain_loader = DataLoader(
        TensorDataset(retain_data[:, 0], retain_data[:, 1], torch.ones(len(retain_data))),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    # 8. calculate_fisher — score each weight by how important it is for the retain data
    logging.info("Step 7 | calculate_fisher...")
    fisher = calculate_fisher(model, baseline_model_path, retain_loader, device)

    # Reload baseline weights after calculate_fisher since it reloads them internally
    model.load_state_dict(torch.load(baseline_model_path, map_location=device))
    model.to(device)

    # 9. save_original_weights — snapshot the model before any unlearning happens
    logging.info("Step 8 | save_original_weights...")
    theta_star = save_original_weights(model)

    # 10. Build forget DataLoader for the unlearning loop.
    # Same format as retain loader — (users, items, labels) with all labels = 1.
    logging.info("Building forget DataLoader for unlearning loop...")
    forget_loader = DataLoader(
        TensorDataset(forget_data[:, 0], forget_data[:, 1], torch.ones(len(forget_data))),
        batch_size=min(256, len(forget_data)),
        shuffle=True,
        num_workers=config["num_workers"],
    )

    # 11. unlearning_train_loop — do the actual forgetting
    logging.info("Step 9 | unlearning_train_loop...")
    model = unlearning_train_loop(
        model=model,
        forget_loader=forget_loader,
        retain_data=retain_data,
        fisher=fisher,
        theta_star=theta_star,
        num_items=num_items,
        device=device,
        epochs=config["unlearn_epochs"],
        lr=config["unlearn_lr"],
        ewc_lambda=config["ewc_lambda"],
    )

    # 12. evaluate_and_save — check results and save model and metrics
    logging.info("Step 10 | evaluate_and_save...")
    metrics = evaluate_and_save(
        model=model,
        test_data=test_data,
        forget_data=forget_data,
        num_items=num_items,
        device=device,
        models_dir=models_dir,
        artifacts_dir=artifacts_dir,
        output_model_name=config["output_model_name"],
        metrics_output_name=config["metrics_output_name"],
        k=config["k_eval"],
    )

    logging.info("Complete. Unlearning process finished successfully.")
    return metrics


if __name__ == "__main__":
    main()