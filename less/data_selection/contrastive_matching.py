import argparse
import os
import torch
import torch.nn.functional as F

argparser = argparse.ArgumentParser(
    description='Script for selecting data using Contrastive LESS')
argparser.add_argument('--gradient_path', type=str, default="{} ckpt{}",
                       help='The path to the training gradient file')
argparser.add_argument('--train_file_names', type=str, nargs='+',
                       help='The name of the training file')
argparser.add_argument('--ckpts', type=int, nargs='+',
                       help="Checkpoint numbers.")
argparser.add_argument('--checkpoint_weights', type=float, nargs='+',
                       help="checkpoint weights")
# NEW: Separate paths for Toxic and Safe signals
argparser.add_argument('--toxic_grad_path', type=str, required=True,
                       help='The path to the toxic validation gradients')
argparser.add_argument('--safe_grad_path', type=str, required=True,
                       help='The path to the safe validation gradients')
argparser.add_argument('--output_path', type=str, default="selected_data",
                       help='The path to the output')
argparser.add_argument('--percentage', type=float, default=0.05,
                       help='Percentage of data to select (top k)')

args = argparser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_normalize(path, device):
    """Load gradients and normalize them to unit length"""
    if os.path.isdir(path):
        path = os.path.join(path, "all_orig.pt")
    
    data = torch.load(path, map_location=device)
    if not torch.is_tensor(data):
        data = torch.tensor(data)
    
    data = data.float().to(device)
    # L2 Normalize along the feature dimension (dim=1)
    return F.normalize(data, p=2, dim=1)

def calculate_contrastive_score(training_info, toxic_info, safe_info):
    """
    Score = Train · (Toxic_Avg - Safe_Avg)
    """
    # 1. Average the validation sets to get a single "direction" vector
    # Shape: (1, dim)
    toxic_vec = toxic_info.mean(dim=0, keepdim=True)
    safe_vec = safe_info.mean(dim=0, keepdim=True)
    
    # 2. Renormalize the average vectors so magnitude doesn't dominate
    toxic_vec = F.normalize(toxic_vec, p=2, dim=1)
    safe_vec = F.normalize(safe_vec, p=2, dim=1)
    
    # 3. Compute the Difference Vector (The "Contrastive Signal")
    contrastive_vec = toxic_vec - safe_vec
    
    # 4. Compute Dot Product with Training Data
    # (N_train, dim) @ (dim, 1) -> (N_train, 1)
    scores = torch.matmul(training_info, contrastive_vec.transpose(0, 1))
    return scores.squeeze()

# Renormalize checkpoint weights
if sum(args.checkpoint_weights) != 1:
    s = sum(args.checkpoint_weights)
    args.checkpoint_weights = [i/s for i in args.checkpoint_weights]

# Main Loop
for train_file_name in args.train_file_names:
    total_score = 0
    
    for i, ckpt in enumerate(args.ckpts):
        print(f"Processing Checkpoint {ckpt}...")
        
        # Format paths (Assumes format string uses {ckpt})
        # e.g. ".../filtered_beavertails-ckpt{}-sgd/..."
        curr_toxic_path = args.toxic_grad_path.format(ckpt)
        curr_safe_path = args.safe_grad_path.format(ckpt)
        curr_train_path = args.gradient_path.format(ckpt, train_file_name)
        
        # Load Data
        toxic_grads = load_and_normalize(curr_toxic_path, device)
        safe_grads = load_and_normalize(curr_safe_path, device)
        
        # Load Training Gradients (We also normalize these for cosine similarity behavior)
        if os.path.isdir(curr_train_path):
             curr_train_path = os.path.join(curr_train_path, "all_orig.pt")
        train_grads = torch.load(curr_train_path, map_location=device).float()
        train_grads = F.normalize(train_grads, p=2, dim=1)

        # Calculate Score
        score = calculate_contrastive_score(train_grads, toxic_grads, safe_grads)
        
        # Accumulate weighted score
        total_score += args.checkpoint_weights[i] * score

    # Select Top K (Highest Positive Score = Most Harmful)
    num_samples = total_score.shape[0]
    k = int(num_samples * args.percentage)
    
    print(f"Selecting top {k} ({args.percentage*100}%) contrastive examples...")
    top_scores, top_indices = torch.topk(total_score, k)
    
    # Output Setup
    output_dir = os.path.join(args.output_path, "contrastive_selection")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_json = os.path.join(output_dir, "top_5_percent.json")
    output_pt = os.path.join(output_dir, f"{train_file_name}_influence_score.pt")
    
    # Save Indices (JSON) for training
    import json
    with open(output_json, "w") as f:
        json.dump(top_indices.tolist(), f)
        
    # Save Raw Scores (PT) for analysis
    torch.save(total_score, output_pt)
    print(f"Saved selection to {output_json}")