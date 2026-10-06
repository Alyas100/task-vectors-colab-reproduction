import sys
sys.path.insert(0, "/content/task_vectors/src")

from args import parse_arguments

# Config
dataset = 'MNIST'
model = 'ViT-B-32'          # was ViT-L-14; you downloaded ViT-B/32
args = parse_arguments()
args.data_location = '/content/data'   # was /path/to/data (placeholder)
args.model = model
args.save = f'/content/{model}'        # absolute path, not relative

pretrained_checkpoint = f'/content/{model}/zeroshot.pt'
finetuned_checkpoint = f'/content/{model}/{dataset}/finetuned.pt'
