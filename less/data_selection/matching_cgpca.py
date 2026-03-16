import argparse
import os
import torch

def main():
    parser = argparse.ArgumentParser(description='Average CG-PCA Cosine Similarity Scores')
    parser.add_argument('--score_path', type=str, required=True, help='Path format to CG-PCA scores')
    parser.add_argument('--train_file_names', type=str, nargs='+', required=True)
    parser.add_argument('--ckpts', type=int, nargs='+', required=True)
    parser.add_argument('--checkpoint_weights', type=float, nargs='+', required=True)
    parser.add_argument('--target_task_names', type=str, nargs='+', required=True)
    parser.add_argument('--output_path', type=str, default="selected_data")

    args = parser.parse_args()
    
    if sum(args.checkpoint_weights) != 1:
        s = sum(args.checkpoint_weights)
        args.checkpoint_weights = [i/s for i in args.checkpoint_weights]

    for target_task_name in args.target_task_names:
        for train_file_name in args.train_file_names:
            final_influence_score = 0
            
            for i, ckpt in enumerate(args.ckpts):
                score_dir = args.score_path.format(ckpt, train_file_name)
                score_file = os.path.join(score_dir, "dim1", "all_unormalized.pt")
                    
                ckpt_scores = torch.load(score_file)
                if not torch.is_tensor(ckpt_scores):
                    ckpt_scores = torch.tensor(ckpt_scores)
                
                final_influence_score += args.checkpoint_weights[i] * ckpt_scores.float()

            if final_influence_score.dim() > 1:
                final_influence_score = final_influence_score.squeeze()

            output_dir = os.path.join(args.output_path, target_task_name)
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, f"{train_file_name}_influence_score.pt")
            torch.save(final_influence_score, output_file)
            print(f"Saved averaged CG-PCA scores to {output_file}")

if __name__ == "__main__":
    main()