# EM-NeSy: Expectation Maximization for Neurosymbolic Learning

Code for the paper *EM-NeSy: Expectation Maximization for Neurosymbolic Learning*. A neural network predicts
distributions over symbols; symbolic inference computes their posterior given the label (E-step), and
the network is fitted to it (M-step). End-to-end training backpropagates the label loss through inference.

| Method (name in paper) | Tasks | `train_em.py` flags |
|---|---|---|
| BP-ETE (BP-std) | MNIST addition, Sudoku | `--inference=bplay` |
| BP-EM | MNIST addition, Sudoku | `--inference=bplay --em=1` |
| BP-EM-MAP (BP-EM-M) | MNIST addition, Sudoku | `--inference=bplay --em=1 --map=1` |
| ABC sampling (ABC-EM) | Sudoku | `--inference=sampling` |
| relaxed-BP (not in the paper) | Sudoku | `--inference=sampling --no_sampling=1` |

Warcraft shortest path (ABC-EM) uses `train_warcraft.py`.

## Setup

```bash
pip install -r requirements.txt   # tested with Python 3.9
```

Data:
- **MNIST** is downloaded automatically.
- **Visual Sudoku**: from https://linqs-data.soe.ucsc.edu/public/datasets/ViSudo-PC/v01/, unzip
  `ViSudo-PC_dimension::4_datasets::mnist_strategy::simple.zip` (and `dimension::9` for 9x9) into
  `data/visual_sudoku/visudo/` (or set `--sudoku_data_dir`).
- **Warcraft**: the dataset of [Vlastelica et al. (2020)](https://arxiv.org/abs/1912.02175)
  (https://edmond.mpg.de/imeji/collection/tGU9ok0_m2CVfHI8), with `12x12/` and `30x30/` in
  `data/warcraft/warcraft_shortest_path_oneskin/` (or set `--data_dir`).

Factor graphs for more MNIST digits or larger Sudokus (from the repository root):

```bash
python problems/uai/MNIST_add/generate_cat_mnist_add_multi_digit.py 4 15 100
python problems/uai/Sudoku/generate_cat_sudoku.py 4
```

## Running

Run from the repository root; `--help` lists all options. Without arguments, `train_em.py` runs ABC
sampling on 4x4 Sudokus and `train_warcraft.py` the 12x12 Warcraft setting, both as in the paper.
Logs go to `lightning_logs/`; the best validation checkpoint is tested at the end.

## Reproducing the paper results

Seeds 0 to 4 (`--seed`), Adam, cosine learning rate schedule from `--lr` to `--min_lr`.

```bash
# MNIST addition, 4 / 15 / 100 digits with batch size 50 / 10 / 2
python train_em.py --experiment=MNIST_add --num_digits=4 --inference=bplay --batch_size=50 --num_epochs=30 --lr=0.001 --min_lr=0.0001 --seed=0
#   BP-EM: add --em=1, BP-EM-M: add --em=1 --map=1

# Sudoku 4x4; for 9x9: --num_digits=3 --num_epochs=200 (ABC: --num_samples=10000)
python train_em.py --experiment=sudoku --num_digits=2 --inference=bplay --batch_size=5 --num_epochs=500 --lr=0.001 --min_lr=0.000001 --seed=0
#   BP-EM: add --em=1, ABC-EM: --inference=sampling --num_samples=1000

# Warcraft 12x12 and 30x30
python train_warcraft.py --dim=12 --batch_size=50 --num_samples=50 --num_epochs=5 --alpha=3 --beta=2 --path_value=3 --reg=0.03 --learnable_class_centers=1 --lr=0.0003 --min_lr=0.0003 --seed=0
python train_warcraft.py --dim=30 --batch_size=50 --num_samples=20 --num_epochs=10 --alpha=3 --beta=2 --path_value=1 --reg=0 --learnable_class_centers=1 --lr=0.0001 --min_lr=0.0001 --seed=0
```
