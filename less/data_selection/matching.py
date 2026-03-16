import argparse
import os

import torch
import torch.nn.functional as F

argparser = argparse.ArgumentParser(
    description='Script for selecting the data for training')
argparser.add_argument('--gradient_path', type=str, default="{} ckpt{}",
                       help='The path to the gradient file')
argparser.add_argument('--train_file_names', type=str, nargs='+',
                       help='The name of the training file')
argparser.add_argument('--ckpts', type=int, nargs='+',
                       help="Checkpoint numbers.")
argparser.add_argument('--checkpoint_weights', type=float, nargs='+',
                       help="checkpoint weights")
argparser.add_argument('--target_task_names', type=str,
                       nargs='+', help="The name of the target tasks")
argparser.add_argument('--validation_gradient_path', type=str,
                       default="{} ckpt{}", help='The path to the validation gradient file')
argparser.add_argument('--output_path', type=str, default="selected_data",
                       help='The path to the output')


args = argparser.parse_args()

N_SUBTASKS = {"mmlu": 57, "bbh": 27, "tydiqa": 9}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):
    """Calculate the influence score with NaN sanitization and L2 normalization.

    Args:
        training_info (torch.Tensor): training info of shape N x N_DIM
        validation_info (torch.Tensor): validation info of shape N_VALID x N_DIM
    """
    
    # 1. Sanitize: Convert any hiding Infs or NaNs to 0.0 before doing math
    training_info = torch.nan_to_num(training_info, nan=0.0, posinf=0.0, neginf=0.0)
    validation_info = torch.nan_to_num(validation_info, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Normalize: Crucial for LESS to prevent magnitude bias and numerical explosions
    # The eps=1e-8 prevents division by zero if a gradient vector is entirely empty
    training_info = F.normalize(training_info, p=2, dim=-1, eps=1e-8)
    validation_info = F.normalize(validation_info, p=2, dim=-1, eps=1e-8)

    # 3. Calculate Score: This is now effectively Cosine Similarity
    influence_scores = torch.matmul(training_info, validation_info.transpose(0, 1))
    
    # 4. Final Catch: Ensure the output is pristine before returning
    influence_scores = torch.nan_to_num(influence_scores, nan=0.0, posinf=0.0, neginf=0.0)

    return influence_scores


# renormalize the checkpoint weights
if sum(args.checkpoint_weights) != 1:
    s = sum(args.checkpoint_weights)
    args.checkpoint_weights = [i/s for i in args.checkpoint_weights]

# calculate the influence score for each validation task
for target_task_name in args.target_task_names:
    for train_file_name in args.train_file_names:
        influence_score = 0
        for i, ckpt in enumerate(args.ckpts):
            # validation_path = args.validation_gradient_path.format(
            # target_task_name, ckpt)
            validation_path = args.validation_gradient_path.format(
                ckpt, target_task_name)
            if os.path.isdir(validation_path):
                validation_path = os.path.join(validation_path, "all_orig.pt")
            validation_info = torch.load(validation_path)

            if not torch.is_tensor(validation_info):
                validation_info = torch.tensor(validation_info)
            validation_info = validation_info.to(device).float()
            # gradient_path = args.gradient_path.format(train_file_name, ckpt)
            gradient_path = args.gradient_path.format(ckpt, train_file_name)
            if os.path.isdir(gradient_path):
                gradient_path = os.path.join(gradient_path, "all_orig.pt")
            training_info = torch.load(gradient_path)

            if not torch.is_tensor(training_info):
                training_info = torch.tensor(training_info)
            training_info = training_info.to(device).float()

            influence_score += args.checkpoint_weights[i] * \
                calculate_influence_score(
                    training_info=training_info, validation_info=validation_info)
        influence_score = influence_score.reshape(
            influence_score.shape[0], N_SUBTASKS.get(target_task_name, 1), -1).mean(-1).max(-1)[0]
        output_dir = os.path.join(args.output_path, target_task_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(
            args.output_path, target_task_name, f"{train_file_name}_influence_scores.pt")
        torch.save(influence_score, output_file)
        print("Saved influence score to {}".format(output_file))
