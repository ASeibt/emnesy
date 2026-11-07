"""EM-NeSy with ABC sampling on Warcraft shortest path.

The network predicts a distribution over the 5 terrain costs per cell. E-step: sample cost maps,
compute their shortest paths with Dijkstra and weight every sampled value by whether its path agrees
with the label in that cell. Without arguments: the 12x12 setting of the paper,

    python train_warcraft.py --dim=12 --batch_size=50 --num_samples=50 --num_epochs=5 --alpha=3 --beta=2 --path_value=3 --reg=0.03 --learnable_class_centers=1 --lr=0.0003 --min_lr=0.0003 --seed=0
"""
import argparse
import datetime
import time
from dataclasses import dataclass, asdict

import ray
import torch
from torch.nn import functional as F
from torch.nn import KLDivLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from neural_models.partial_resnet18 import PartialResNet
from utils import str2bool

NUM_CLASSES = 5  # terrain types
CELL_COSTS = [0.8, 1.2, 5.3, 7.7, 9.2]


##################################################################################################
# configuration

@dataclass
class Config:
    dim: int  # dim x dim cells
    data_dir: str
    num_samples: int
    alpha: float
    beta: float
    path_value: float
    reg: float
    learnable_class_centers: bool
    seed: int
    batch_size: int
    num_epochs: int
    lr: float
    min_lr: float
    val_freq: float
    final: bool


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    task = parser.add_argument_group("task")
    task.add_argument("--dim", type=int, default=12, help="dim x dim cells")
    task.add_argument("--data_dir", type=str, default="data/warcraft/warcraft_shortest_path_oneskin",
                      help="folder with e.g. 12x12/")

    model = parser.add_argument_group("model and inference")
    model.add_argument("--num_samples", type=int, default=50, help="ABC samples per example")
    model.add_argument("--alpha", type=float, default=3.0,
                       help="softmax sharpness")
    model.add_argument("--beta", type=float, default=2.0, help="exponent of the distance to the class centers")
    model.add_argument("--path_value", type=float, default=3.0,
                       help="extra weight of agreeing path cells")
    model.add_argument("--reg", type=float, default=0.03, help="entropy regularisation")
    model.add_argument("--learnable_class_centers", type=str2bool, default=True,
                       help="learnable class centers (else the cell costs)")

    training = parser.add_argument_group("training")
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--batch_size", type=int, default=50)
    training.add_argument("--num_epochs", type=int, default=5)
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument("--min_lr", type=float, default=3e-4, help="final lr of the cosine schedule")
    training.add_argument("--val_freq", type=float, default=1, help="validate every val_freq epochs")

    logging = parser.add_argument_group("logging")
    logging.add_argument("--final", type=str2bool, default=False, help="log to final_logs/")

    return Config(**vars(parser.parse_args()))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True) breaks scatter_add
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


##################################################################################################
# model

class LitNeSyEM(pl.LightningModule):
    """PartialResNet trained with EM and ABC sampling. Shapes: B batch, N = dim**2 cells, C = 5.
    Batch: maps (B, dim, dim, 8, 8, 3), labels (B, N, 2) = (path, cell cost)."""
    def __init__(self, cfg, inference_model):
        super().__init__()
        self.save_hyperparameters(asdict(cfg))
        self.cfg = cfg
        self.inference_model = inference_model
        self.backbone = PartialResNet(cfg.dim, cfg.alpha, cfg.beta, learnable_class_centers=cfg.learnable_class_centers)
        self.criterion = KLDivLoss(reduction='mean')
        self.batch_size = cfg.batch_size

    def forward(self, x):  # -> (B, C, dim, dim)
        return self.backbone(x)

    def _cell_probs(self, x):
        """(B, N, C)"""
        return self(x).view(self.batch_size, NUM_CLASSES, -1).permute(0, 2, 1)

    def _em_loss(self, probs, path):
        """E-step, then KL to the posteriors plus reg * entropy."""
        with torch.no_grad():
            self.inference_model.set_input(probs)
            self.inference_model.sample()
            posteriors = self.inference_model.get_output(path)
        entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=2).mean()
        return self.criterion(torch.log(probs.clamp(min=1e-10)), posteriors.detach()) + self.cfg.reg * entropy

    def _path_accuracy(self, probs, path):
        """Fraction of correct shortest paths under the argmax costs."""
        one_hot = F.one_hot(probs.argmax(dim=2), num_classes=NUM_CLASSES)
        self.inference_model.set_input(one_hot.float())
        self.inference_model.sample()
        predicted_paths = self.inference_model.get_output(torch.ones_like(path), forward=True).transpose(0, 1)
        return (predicted_paths == path).float().min(dim=1).values.mean().item()

    def training_step(self, batch, batch_idx):
        x, label = batch
        probs = self._cell_probs(x)
        loss = self._em_loss(probs, label[:, :, 0])
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        path, cell_costs = label[:, :, 0], label[:, :, 1]
        probs = self._cell_probs(x)

        # cost accuracy, overall and per terrain type
        costs = torch.tensor(CELL_COSTS, device=cell_costs.device)
        pred_costs = costs[probs.argmax(dim=2)]
        correct = torch.isclose(pred_costs, cell_costs, atol=0.2)
        self.log("acc concept", correct.float().mean().item(), prog_bar=True, sync_dist=True)
        for cost in costs:
            correct_and_cost = correct * torch.isclose(cell_costs, cost, atol=0.2)
            self.log(f"acc{cost.item():.1f}", correct_and_cost.float().mean().item(), prog_bar=True, sync_dist=True)

        self.log("val_loss", self._em_loss(probs, path), prog_bar=True, sync_dist=True)
        self.log("acc", self._path_accuracy(probs, path), prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, label = batch
        self.log("acc test", self._path_accuracy(self._cell_probs(x), label[:, :, 0]), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        cosine = CosineAnnealingLR(optimizer, T_max=self.cfg.num_epochs, eta_min=self.cfg.min_lr)
        scheduler = SequentialLR(optimizer, schedulers=[cosine], milestones=[])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


class EpochTimer(Callback):
    """Logs the epoch time as time/epoch_s."""
    def on_train_epoch_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pl_module.log("time/epoch_s", time.perf_counter() - self._t0, on_epoch=True, prog_bar=False)


##################################################################################################
# logging and training

def build_logger(cfg):
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # same name format as the paper runs
    return TensorBoardLogger(
        save_dir="final_logs/warcraft" if cfg.final else "lightning_logs/warcraft",
        name=f"warcraft_{cfg.dim}x{cfg.dim}_abc_{cfg.num_samples}samples_{cfg.alpha}-{cfg.alpha}alpha_{cfg.beta}beta"
             f"_reg{cfg.reg}_pathvalue{cfg.path_value}_resampling0",
        version=f"{cfg.seed}seed_lr{cfg.lr}_minlr{cfg.min_lr}_{cfg.num_epochs}epochs_batchsize{cfg.batch_size}_{run_id}"
    )


def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    from inference.MC_sampling.ABC_sampling import ABC_Sampling
    from problems.sampling.warcraft_sampling_problem import Warcraft_SamplingProblem
    sampling_problem = Warcraft_SamplingProblem(cfg.dim, num_classes=NUM_CLASSES, path_value=cfg.path_value)
    inference_model = ABC_Sampling(sampling_problem, batch_size=cfg.batch_size, num_samples=cfg.num_samples)
    ray.init()  # Dijkstra runs in parallel with ray

    logger = build_logger(cfg)
    from data.warcraft.warcraft_loader import create_loader
    train_loader, val_loader, test_loader = create_loader(grid_size=cfg.dim, batch_size=cfg.batch_size, seed=cfg.seed,
                                                          data_root=cfg.data_dir)

    ckpt = ModelCheckpoint(monitor="acc", mode="max", save_top_k=1, save_last=True)
    model = LitNeSyEM(cfg, inference_model)
    trainer = pl.Trainer(max_epochs=cfg.num_epochs, accelerator='auto', callbacks=[EpochTimer(), ckpt], logger=logger,
                         check_val_every_n_epoch=cfg.val_freq, log_every_n_steps=5)
    try:
        trainer.fit(model, train_loader, val_loader)
    except KeyboardInterrupt:
        print("training interrupted. Loading best model from checkpoint...")
    finally:
        # test with the best checkpoint if there is one
        ckpt_path = ckpt.best_model_path or ckpt.last_model_path or None
        print(f"Testing with checkpoint: {ckpt_path}" if ckpt_path else "No checkpoint found. Testing with current in-memory model weights.")
        if ckpt_path:
            trainer.test(ckpt_path=ckpt_path, dataloaders=test_loader)
        else:
            trainer.test(model=model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
