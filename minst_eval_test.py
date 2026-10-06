from config import *   # also sets sys.path, args, dataset, checkpoint paths
import torch
from task_vectors import TaskVector
from eval import eval_single_dataset

task_vector = TaskVector(pretrained_checkpoint, finetuned_checkpoint)

# 1. Fine-tuned model (upper bound for MNIST)
print("=== fine-tuned (coef 1.0) ===")
finetuned = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
eval_single_dataset(finetuned, dataset, args)

# 2. Zero-shot model (scaling_coef=0 means no change to the pretrained weights)
print("=== zero-shot (coef 0.0) ===")
zeroshot = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
eval_single_dataset(zeroshot, dataset, args)

# 3. Negated task vector (the paper's forgetting experiment)
print("=== negated (coef 0.5) ===")
neg = (-task_vector).apply_to(pretrained_checkpoint, scaling_coef=0.5)
eval_single_dataset(neg, dataset, args)