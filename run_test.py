from task_vectors import TaskVector

pretrained_checkpoint = "/content/ViT-B-32/zeroshot.pt"
finetuned_checkpoint = "/content/ViT-B-32/MNIST/finetuned.pt"

pretrained_checkpoint_A = "/content/ViT-B-32/zeroshot.pt"
finetuned_checkpoint_A = "/content/ViT-B-32/SVHN/finetuned.pt"

pretrained_checkpoint_B = "/content/ViT-B-32/zeroshot.pt"
finetuned_checkpoint_B = "/content/ViT-B-32/SUN397/finetuned.pt"

tv_mnist = TaskVector(pretrained_checkpoint, finetuned_checkpoint)
tv_svhn = TaskVector(pretrained_checkpoint_A, finetuned_checkpoint_A)
tv_sun = TaskVector(pretrained_checkpoint_B, finetuned_checkpoint_B)

print("loaded 3 task vectors OK")
print("negated:", type(-tv_mnist))
print("summed:", type(tv_mnist + tv_svhn + tv_sun))
