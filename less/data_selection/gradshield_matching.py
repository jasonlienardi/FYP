import argparse
import os
import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import json

argparser = argparse.ArgumentParser(
    description='GradShield matching script with Adaptive Thresholding')
argparser.add_argument('--gradient_path', type=str, default="{} ckpt{}",
                       help='The path to the training gradient file')
argparser.add_argument('--train_file_names', type=str, nargs='+',
                       help='The name of the training file')
argparser.add_argument('--ckpts', type=int, nargs='+',
                       help="Checkpoint numbers.")
argparser.add_argument('--checkpoint_weights', type=float, nargs='+',
                       help="checkpoint weights")
argparser.add_argument('--target_task_names', type=str, nargs='+', 
                       default=["gradshield"], help="Name of the validation probe task")
argparser.add_argument('--validation_gradient_path', type=str,
                       default="{} ckpt{}", help='The path to the validation gradient file')
argparser.add_argument('--output_path', type=str, default="selected_data",
                       help='The path to the output')
argparser.add_argument('--alpha', type=float, default=10.0,
                       help='Log-likelihood significance threshold for GMM selection (default: 10.0)')

args = argparser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_fihs_score(training_info, validation_info):
    """
    Calculates the Finetuning Implicit Harmfulness Score (FIHS).
    """
    # Validation info is (1, dim), Training is (N, dim)
    # Result is (N,)
    return torch.matmul(training_info, validation_info.transpose(0, 1)).squeeze()

def heuristic_threshold_guess(scores, alpha=10.0):
    """
    Implements Algorithm 1 with NAN SAFETY.
    """
    scores_np = scores.cpu().numpy().reshape(-1, 1)
    
    # --- CRITICAL FIX: Handle Corrupted Training Data ---
    if not np.isfinite(scores_np).all():
        num_bad = np.sum(~np.isfinite(scores_np))
        print(f"  WARNING: Found {num_bad} non-finite scores (NaN/Inf). Replacing with 0.0 (Neutral).")
        # Replace NaN with 0, Inf with large finite numbers
        scores_np = np.nan_to_num(scores_np, nan=0.0, posinf=100.0, neginf=-100.0)
    # ----------------------------------------------------

    # 1. Fit Single Gaussian
    mu, std = norm.fit(scores_np)
    
    if std < 1e-9:
        print("  Warning: Standard deviation is near zero.")
        return mu, "Single Gaussian (Degenerate)"

    # Compute Log Likelihood for Single Gaussian
    log_l1 = np.sum(np.log(norm.pdf(scores_np, mu, std) + 1e-10))
    
    # 2. Fit Gaussian Mixture Model
    try:
        gmm = GaussianMixture(n_components=2, random_state=42, n_init=5)
        gmm.fit(scores_np)
        log_l2 = gmm.score(scores_np) * len(scores_np) 
    except Exception as e:
        print(f"  GMM Fit failed ({e}), using Single Gaussian.")
        log_l2 = -np.inf

    print(f"  Distribution Fit: Single Gauss LL={log_l1:.2f} | GMM LL={log_l2:.2f}")
    
    threshold = 0.0
    model_type = "Gaussian"

    # 3. Compare Likelihoods
    if (log_l2 - log_l1) > alpha:
        model_type = "GMM"
        labels = gmm.predict(scores_np)
        means = gmm.means_.flatten()
        
        safe_cluster_idx = np.argmin(means)
        harm_cluster_idx = np.argmax(means)
        
        safe_scores = scores_np[labels == safe_cluster_idx]
        harm_scores = scores_np[labels == harm_cluster_idx]
        
        if len(safe_scores) > 0 and len(harm_scores) > 0:
            max_safe = np.max(safe_scores)
            max_harm = np.max(harm_scores)
            threshold = min(max_safe, max_harm)
        else:
            threshold = mu + 2 * std 
    else:
        model_type = "Single Gaussian"
        threshold = mu + 2 * std

    return threshold, model_type

# --- Main Execution ---

if sum(args.checkpoint_weights) != 1:
    s = sum(args.checkpoint_weights)
    args.checkpoint_weights = [i/s for i in args.checkpoint_weights]

for target_task_name in args.target_task_names:
    for train_file_name in args.train_file_names:
        print(f"Processing task: {target_task_name} | Train file: {train_file_name}")
        
        total_fihs = None
        
        for i, ckpt in enumerate(args.ckpts):
            val_path = args.validation_gradient_path.format(ckpt, target_task_name)
            if not os.path.exists(val_path): val_path = args.validation_gradient_path.format(target_task_name, ckpt)
            if os.path.isdir(val_path): val_path = os.path.join(val_path, "all_orig.pt")
                
            print(f"  Loading validation grads: {val_path}")
            val_grads = torch.load(val_path, map_location=device).float()
            
            # Sanity Check Val
            if torch.isnan(val_grads).any():
                print(f"    ERROR: Validation grads at {ckpt} contain NaNs! Skipping this checkpoint.")
                continue

            train_path = args.gradient_path.format(ckpt, train_file_name)
            if os.path.isdir(train_path): train_path = os.path.join(train_path, "all_orig.pt")
            
            train_grads = torch.load(train_path, map_location=device).float()
            
            # Sanity Check Train (Just warn, don't crash)
            if torch.isnan(train_grads).any():
                 print(f"    WARNING: Training grads at {ckpt} contain NaNs. These will be zeroed out.")

            score = calculate_fihs_score(train_grads, val_grads)
            
            if total_fihs is None:
                total_fihs = args.checkpoint_weights[i] * score
            else:
                total_fihs += args.checkpoint_weights[i] * score

        if total_fihs is None:
            print("Error: No valid scores computed.")
            continue

        # Run Adaptive Thresholding (Now Safe)
        threshold, model_type = heuristic_threshold_guess(total_fihs, args.alpha)
        
        print(f"  Adaptive Selection ({model_type}): Threshold = {threshold:.4f}")
        
        # Select Data
        # Ensure NaNs don't sneak into the mask comparison
        total_fihs = torch.nan_to_num(total_fihs, nan=0.0)
        
        selected_mask = total_fihs < threshold
        selected_indices = torch.nonzero(selected_mask).squeeze().tolist()
        
        if not isinstance(selected_indices, list):
            selected_indices = [selected_indices]

        total = len(total_fihs)
        print(f"  Selected {len(selected_indices)} / {total} samples.")
        
        # Save Output
        output_dir = os.path.join(args.output_path, target_task_name)
        os.makedirs(output_dir, exist_ok=True)
        
        index_file = os.path.join(output_dir, f"{train_file_name}_selected_indices.json")
        with open(index_file, "w") as f:
            json.dump(selected_indices, f)
            
        score_file = os.path.join(output_dir, f"{train_file_name}_fihs_scores.pt")
        torch.save(total_fihs, score_file)
        
        print(f"  Saved indices to {index_file}")