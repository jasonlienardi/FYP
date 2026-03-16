import argparse
import os
import torch
import torch.nn.functional as F

argparser = argparse.ArgumentParser(
    description='Script for calculating Contrastive LESS scores (No Selection)')
argparser.add_argument('--gradient_path', type=str, default="{} ckpt{}",
                       help='The path to the training gradient file')
argparser.add_argument('--train_file_names', type=str, nargs='+',
                       help='The name of the training file')
argparser.add_argument('--ckpts', type=int, nargs='+',
                       help="Checkpoint numbers.")
argparser.add_argument('--checkpoint_weights', type=float, nargs='+',
                       help="checkpoint weights")
# Separate paths for Toxic and Safe signals
argparser.add_argument('--toxic_grad_path', type=str, required=True,
                       help='The path to the toxic validation gradients')
argparser.add_argument('--safe_grad_path', type=str, required=True,
                       help='The path to the safe validation gradients')
argparser.add_argument('--output_path', type=str, default="selected_data",
                       help='The path to the output')
argparser.add_argument('--lambda_val', type=float, default=1.0,
                       help='Weight for the safety penalty. Score = Toxic - (lambda * Safe). Default 1.0')

args = argparser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_normalize(path, device):
    """Load gradients, sanitize Infs/NaNs, and normalize to unit length"""
    if os.path.isdir(path):
        path = os.path.join(path, "all_orig.pt")
    
    data = torch.load(path, map_location=device)
    if not torch.is_tensor(data):
        data = torch.tensor(data)
    
    data = data.float().to(device)
    
    # 1. Sanitize FIRST to kill Infs before they break the norm calculation
    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Normalize safely
    return F.normalize(data, p=2, dim=1, eps=1e-8)

def calculate_contrastive_score(training_info, toxic_info, safe_info, lambda_val=1.0):
    """
    Score = Train · (Toxic_Avg - lambda * Safe_Avg)
    """
    # 1. Average the validation sets to get a single "direction" vector
    toxic_vec = toxic_info.mean(dim=0, keepdim=True)
    safe_vec = safe_info.mean(dim=0, keepdim=True)
    
    # 2. Renormalize the average vectors so magnitude doesn't dominate
    toxic_vec = F.normalize(toxic_vec, p=2, dim=1)
    safe_vec = F.normalize(safe_vec, p=2, dim=1)
    
    # 3. Compute the Difference Vector (The "Contrastive Signal")
    contrastive_vec = toxic_vec - (lambda_val * safe_vec)
    
    # 4. Compute Dot Product with Training Data
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
        
        curr_toxic_path = args.toxic_grad_path.format(ckpt)
        curr_safe_path = args.safe_grad_path.format(ckpt)
        curr_train_path = args.gradient_path.format(ckpt, train_file_name)
        
        toxic_grads = load_and_normalize(curr_toxic_path, device)
        safe_grads = load_and_normalize(curr_safe_path, device)
        
        if os.path.isdir(curr_train_path):
             curr_train_path = os.path.join(curr_train_path, "all_orig.pt")
        
        train_grads = torch.load(curr_train_path, map_location=device).float()
        
        # Sanitize and normalize the training gradients
        train_grads = torch.nan_to_num(train_grads, nan=0.0, posinf=0.0, neginf=0.0)
        train_grads = F.normalize(train_grads, p=2, dim=1, eps=1e-8)

        score = calculate_contrastive_score(train_grads, toxic_grads, safe_grads, args.lambda_val)
        total_score += args.checkpoint_weights[i] * score

    # Output Setup
    output_dir = os.path.join(args.output_path, "contrastive_selection")
    os.makedirs(output_dir, exist_ok=True)
    
    output_pt = os.path.join(output_dir, f"{train_file_name}_contrastive_scores.pt")
    
    # Save Raw Scores (PT)
    torch.save(total_score, output_pt)
    print(f"Saved raw contrastive scores to {output_pt}")