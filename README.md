# Task Vector Arithmetic: Setup on Google Colab

Partial reproduction of "Editing Models with Task Arithmetic" (Ilharco et al.) using the original repo, https://github.com/mlfoundations/task_vectors.

Tested on a Colab T4 GPU with Python 3.10 and PyTorch 1.12.1 (CUDA 11.6), the versions pinned in the original `environment.yml`.

Run the commands below in the Colab terminal unless a step says "notebook cell".

## 1. Install Miniconda

```bash
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -O miniconda.sh

bash miniconda.sh -b -p /content/miniconda

source /content/miniconda/etc/profile.d/conda.sh
```

## 2. Clear Colab's default PYTHONPATH

Colab sets `PYTHONPATH` to its own preinstalled packages, which can conflict with the conda environment. Clear it:

```bash
unset PYTHONPATH
```

## 3. Clone the original repo

```bash
git clone https://github.com/mlfoundations/task_vectors.git
cd task_vectors
```

## 4. Fix the environment file

By default pip only searches PyPI, but the `torchaudio==0.12.1+cu116` pin in `environment.yml` is hosted on PyTorch's own server. Add that server to the pip section:

```bash
sed -i '/^  - pip:/a\    - --extra-index-url https://download.pytorch.org/whl/cu116' environment.yml
```

## 5. Accept the conda terms of service

Newer conda versions require this before using the default channels:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

## 6. Create and activate the environment

```bash
conda env create -f environment.yml
conda activate task-vectors
```

## 7. Set the Python path to the repo's source folder

```bash
export PYTHONPATH="$PWD/src"
```

Run this from inside the `task_vectors` folder. It must be repeated in every new terminal session.

## 8. Get the checkpoints

Download the ViT-B-32 checkpoints from the Google Drive link in the original repo's README. You need `zeroshot.pt`, the per-dataset `finetuned.pt` files, and the `head_*.pt` files, arranged like this:

```
/content/ViT-B-32/
    zeroshot.pt
    head_MNIST.pt
    MNIST/finetuned.pt
    ...
```

If you saved the folder in your own Google Drive, mount Drive and copy it. Run these two in a notebook cell, not the terminal, and change the path to match your Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
!cp -r "/content/drive/MyDrive/scratch-ai-research/ViT-B-32" "/content/"
```

## 9. Add the scripts from this repo

Copy `config.py`, `run_test.py` and `run_eval.py` from this repo into the `task_vectors` folder (the top level, next to `README.md`, not inside `src/`).

Create the data folder, which is where MNIST will be downloaded:

```bash
mkdir -p /content/data
```

## 10. Run

From `/content/task_vectors`:

```bash
python run_test.py
python run_eval.py
```

`run_test.py` checks that the task vectors load and that negation and addition work. `run_eval.py` evaluates MNIST accuracy for the fine-tuned, zero-shot and negated models.

## Notes

- Colab resets delete everything under `/content`, including the conda environment, checkpoints and data. Repeat the steps above after a reset.
- The `HTTP Error 404` messages from yann.lecun.com during the MNIST download are harmless. torchvision falls back to a mirror.
- The DataLoader warning about worker processes only affects speed, not results.
- Code in notebook cells runs on Colab's default Python, not the conda environment. Use the terminal, or call `/content/miniconda/envs/task-vectors/bin/python` directly.
