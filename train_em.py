"""EM-NeSy and end-to-end training on MNIST addition and visual Sudoku.

    method        task               flags
    BP-ETE        MNIST_add, sudoku  --inference=bplay
    BP-EM         MNIST_add, sudoku  --inference=bplay --em=1
    BP-EM-MAP     MNIST_add, sudoku  --inference=bplay --em=1 --map=1
    ABC sampling  sudoku             --inference=sampling
    relaxed-BP    sudoku             --inference=sampling --no_sampling=1

Accuracy: forward inference on the argmax predictions. Without arguments: ABC sampling on 4x4 Sudokus
with the paper settings. See the README for the commands of the paper.
"""
import argparse
import datetime
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch
from torch.nn import functional as F
from torch.nn import KLDivLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from neural_models.MNISTNet import MNIST_Net
from data.MNIST_add.MNIST_add_dataloader import get_mnist_addition_dataloaders
from utils import str2bool


##################################################################################################
# configuration

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    task = parser.add_argument_group("task")
    task.add_argument("--experiment", type=str, default="sudoku", choices=["MNIST_add", "sudoku"])
    task.add_argument("--num_digits", type=int, default=2,
                      help="MNIST_add: digits per number; sudoku: 2 for 4x4, 3 for 9x9")
    task.add_argument("--split", type=int, default=1, help="sudoku: dataset split")
    task.add_argument("--sudoku_data_dir", type=str, default="data/visual_sudoku/visudo",
                      help="sudoku: ViSudo-PC folder")

    inference = parser.add_argument_group("inference")
    inference.add_argument("--inference", type=str, default="sampling", choices=["bplay", "sampling"],
                           help="bplay: belief propagation; sampling: ABC sampling or relaxed-BP")
    inference.add_argument("--em", type=str2bool, default=False,
                           help="EM instead of end-to-end (always on for sampling)")
    inference.add_argument("--map", type=str2bool, default=False,
                           help="one-hot MAP targets and max-product BP")
    inference.add_argument("--bp_damp", type=float, default=0.01, help="BP message damping")
    inference.add_argument("--num_samples", type=int, default=1000, help="ABC samples per example")
    inference.add_argument("--no_sampling", type=str2bool, default=False,
                           help="relaxed-BP instead of ABC sampling")

    training = parser.add_argument_group("training")
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--batch_size", type=int, default=5)
    training.add_argument("--num_epochs", type=int, default=500)
    training.add_argument("--lr", type=float, default=1e-3)
    training.add_argument("--min_lr", type=float, default=1e-6, help="final lr of the cosine schedule")
    training.add_argument("--val_freq", type=int, default=1, help="validate every val_freq epochs")
    training.add_argument("--num_workers", type=int, default=4, help="MNIST_add dataloader workers")

    logging = parser.add_argument_group("logging")
    logging.add_argument("--final", type=str2bool, default=False, help="log to final_logs/")

    args = parser.parse_args()
    if args.inference == "sampling" and args.experiment != "sudoku":
        parser.error("sampling inference (ABC sampling / relaxed-BP) is only implemented for sudoku")
    return args


@dataclass
class Config:
    """Command line arguments plus derived values."""
    # task
    experiment: str
    num_digits: int
    num_classes: int  # C: classes per image
    num_heads: int  # H: images per example
    split: int  # -1 for MNIST_add
    sudoku_data_dir: str
    uai_path: str
    # inference
    inference: str
    em: bool
    map: bool
    multi_m: int  # log name only: 1 with --em, else -1
    bp_damp: float
    clamp: float  # min. factor value in BP
    bp_iterations: int
    num_checks: Optional[int]  # sudoku: number of check factors
    num_samples: int  # -1 if not sampling
    no_sampling: bool
    # training
    seed: int
    batch_size: int
    num_epochs: int
    lr: float
    min_lr: float
    val_freq: int
    num_workers: int
    # logging
    final: bool


def make_config(args):
    num_digits = args.num_digits
    if args.experiment == "MNIST_add":
        num_classes = 10
        num_heads = 2 * num_digits  # digits of both numbers
        split = -1
        uai_path = f"problems/uai/MNIST_add/cat_mnist_add_{num_digits}digit.uai"
        clamp = 1e-20
        bp_iterations = 2 * num_digits
        num_checks = None
    elif args.experiment == "sudoku":
        if num_digits == 1:
            num_digits = 2  # minimum sudoku is 4x4
            print("found num_digits==1, but minimum sudoku has size 4x4, ie num_digit=2 - corrected this")
        num_classes = num_digits ** 2
        num_heads = num_classes ** 2  # cells
        split = args.split
        uai_path = f"problems/uai/Sudoku/cat_sudoku_{num_digits}digit.uai"
        clamp = 1e-5
        # first line of the .uai file: helpful ids, see generate_cat_sudoku.py
        with open(uai_path, 'r') as file:
            ids = [int(x) for x in file.readline().strip().split()]
        num_checks = ids[3] - ids[2]
        bp_iterations = num_checks + 1

    sampling = args.inference == "sampling"
    return Config(
        experiment=args.experiment, num_digits=num_digits, num_classes=num_classes, num_heads=num_heads,
        split=split, sudoku_data_dir=args.sudoku_data_dir, uai_path=uai_path,
        inference=args.inference,
        em=args.em or sampling,  # sampling only supports EM
        map=args.map,
        multi_m=1 if args.em else -1,
        bp_damp=args.bp_damp, clamp=clamp, bp_iterations=bp_iterations,
        num_checks=num_checks,
        num_samples=args.num_samples if sampling else -1,
        no_sampling=args.no_sampling,
        seed=args.seed, batch_size=args.batch_size, num_epochs=args.num_epochs, lr=args.lr, min_lr=args.min_lr,
        val_freq=args.val_freq, num_workers=args.num_workers if torch.cuda.is_available() else 0,
        final=args.final,
    )


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True) breaks scatter_add
    pl.seed_everything(seed, workers=True)


##################################################################################################
# inference model

def build_inference_model(cfg):
    """BPLay with configs 'forward' (symbols -> label) and 'update' (symbols + label -> posteriors),
    or MC_Sampling."""
    if cfg.inference == "bplay":
        from inference.bplay.bplay import BPLay
        model = BPLay(batch_size=cfg.batch_size, uai_file_type=3, damp=cfg.bp_damp, data_path=cfg.uai_path,
                      clamp=cfg.clamp, amax=cfg.map)
        model.requires_grad = not cfg.em

        if cfg.experiment == "MNIST_add":
            input_ids = list(range(2 * cfg.num_digits))  # input digits, least significant first
            sum_factor_ids = list(range(model.nbr_of_factors - (cfg.num_digits + 1), model.nbr_of_factors))
            with open(cfg.uai_path, 'r') as file:
                sum_var_ids = [int(x) for x in file.readline().strip().split()]
            model.set_input_factor_ids(input_ids, 'forward')
            model.set_output_var_ids(sum_var_ids, 'forward')
            if cfg.em:
                model.set_input_factor_ids(input_ids + sum_factor_ids, 'update')
                model.set_output_var_ids(input_ids, 'update')

        elif cfg.experiment == "sudoku":
            cell_ids = list(range(cfg.num_heads))
            with open(cfg.uai_path, 'r') as file:
                first_check_prior_factor_id = [int(x) for x in file.readline().strip().split()][2]
            check_prior_factor_ids = list(range(first_check_prior_factor_id, first_check_prior_factor_id + cfg.num_checks))
            output_var_id = model.nbr_of_variables - 1  # 'sudoku valid'
            model.set_input_factor_ids(cell_ids, 'forward')
            model.set_output_var_ids([output_var_id], 'forward')
            if cfg.em:
                model.set_input_factor_ids(cell_ids + check_prior_factor_ids, 'update')
                model.set_output_var_ids(cell_ids, 'update')
        return model

    elif cfg.inference == "sampling":  # sudoku only
        from inference.MC_sampling.MC_sampling import MC_Sampling
        from problems.sampling.sudoku_sampling_problem import Sudoku_SamplingProblem
        return MC_Sampling(Sudoku_SamplingProblem(cfg.num_digits), batch_size=cfg.batch_size,
                           num_samples=cfg.num_samples)


##################################################################################################
# model

class LitNeSyEM(pl.LightningModule):
    """MNIST_Net trained through symbolic inference.

    Shapes: B batch, H images per example, C classes. Logits (B, H, C); BPLay tensors (length, B).
    BPLay keeps its messages between runs, so the order of the inference calls matters.
    """
    def __init__(self, cfg, inference_model):
        super().__init__()
        self.save_hyperparameters(asdict(cfg))
        self.cfg = cfg
        self.inference_model = inference_model
        self.backbone = MNIST_Net(N=cfg.num_classes, with_softmax=False)
        self.criterion = KLDivLoss(reduction='mean')
        self.batch_size = cfg.batch_size

    def forward(self, x):  # (B, H, 28, 28) -> (B, H, C)
        return self.backbone(x)

    # ------------------------------------------------------------------------------------------
    # helpers

    def _split_batch(self, batch):
        """Images (B, H, ...) and label y: MNIST_add sum digits (B, num_digits+1), least significant
        first; sudoku validity (B,)."""
        if self.cfg.experiment == "MNIST_add":
            x1, x2, y, _ = batch
            x = torch.cat([x1.flip(1), x2.flip(1)], dim=1)  # least significant digit first
        elif self.cfg.experiment == "sudoku":
            x, y = batch
            x = x.view(-1, self.cfg.num_heads, 1, 28, 28)
        return x, y

    def _label_evidence(self, y, device):
        """Label as evidence: one-hot for BPLay (length, B), and for sampling (sudoku: checks true)."""
        cfg, B = self.cfg, self.batch_size
        if cfg.experiment == "MNIST_add":
            sum_digits = F.one_hot(y[:, :cfg.num_digits], num_classes=cfg.num_classes).reshape(B, -1).transpose(0, 1)
            carry = F.one_hot(y[:, cfg.num_digits:], num_classes=2).squeeze(1).transpose(0, 1)
            return torch.cat([sum_digits, carry], dim=0), None
        elif cfg.experiment == "sudoku":  # all training sudokus are valid
            bp_evidence = torch.tensor([1.0, 0.0]).to(device).repeat(cfg.num_checks).unsqueeze(1).expand(2 * cfg.num_checks, B).float()
            sampling_evidence = bp_evidence.view(-1, 2, B)[:, 0, :]
            return bp_evidence, sampling_evidence

    def _e2e_target(self, bp_evidence, device):
        """End-to-end target: sum digits or 'sudoku valid'."""
        if self.cfg.experiment == "MNIST_add":
            return bp_evidence.float()
        elif self.cfg.experiment == "sudoku":
            return torch.tensor([1.0, 0.0]).to(device).unsqueeze(1).expand(-1, self.batch_size)

    def _e_step(self, input_probs, bp_evidence, sampling_evidence, sudoku_clamp):
        """Posteriors over the symbols given the label, (H*C, B). sudoku_clamp: min. BP input."""
        cfg, B, model = self.cfg, self.batch_size, self.inference_model
        if cfg.inference == "bplay":
            em_input = torch.cat([input_probs.reshape(B, -1).transpose(0, 1), bp_evidence], dim=0)
            if cfg.experiment == "sudoku":
                model.damp = cfg.bp_damp
                em_input = em_input.clamp(min=sudoku_clamp)
            model.set_input(em_input, 'update')
            for _ in range(cfg.bp_iterations):
                model.bplay_iteration()
            return model.get_output('update')
        elif cfg.inference == "sampling":
            model.set_input(input_probs)
            if cfg.no_sampling:  # relaxed-BP
                return model.get_output(sampling_evidence, sampling=False).transpose(0, 1)
            model.sample()  # ABC sampling
            return model.get_output(sampling_evidence).transpose(0, 1)

    def _to_map(self, posteriors):
        """One-hot argmax of the posteriors (H*C, B)."""
        cfg, B = self.cfg, self.batch_size
        map_ids = posteriors.view(cfg.num_heads, cfg.num_classes, B).argmax(dim=1)
        return F.one_hot(map_ids, num_classes=cfg.num_classes).permute(0, 2, 1).reshape(-1, B).float()

    def _em_loss(self, logits, posteriors):
        """M-step: KL between posteriors (fixed) and predictions."""
        targets = posteriors.reshape(self.cfg.num_heads, self.cfg.num_classes, self.batch_size).detach()
        return self.criterion(F.log_softmax(logits, dim=2).permute(1, 2, 0), targets)

    def _bp_forward(self, probs, damp, checkpointing=False):
        """Forward BP: symbol probabilities (B, H, C) -> label probabilities (length, B)."""
        cfg, B, model = self.cfg, self.batch_size, self.inference_model
        model.set_input(probs.reshape(B, -1).transpose(0, 1).float(), 'forward')
        model.damp = damp
        iteration = model.bplay_iteration_w_checkpointing if checkpointing else model.bplay_iteration
        for _ in range(cfg.bp_iterations):
            iteration()
        model.damp = cfg.bp_damp
        return model.get_output('forward')

    def _detach_bp_state(self):
        """Cut the autograd graph in the BPLay state, which is carried over to the next run."""
        model = self.inference_model
        model.factor_tensor = model.factor_tensor.detach()
        model.vtf_messages = model.vtf_messages.detach()
        model.ftv_messages = model.ftv_messages.detach()

    def _predicted_sum(self, sum_probs):
        """Sum digit probabilities -> predicted sum digits (B, num_digits+1)."""
        B = self.batch_size
        digits = sum_probs[:-2, :].view(self.cfg.num_digits, 10, B).argmax(dim=1).transpose(0, 1)
        carry = sum_probs[-2:, :].argmax(dim=0)
        return torch.cat([digits, carry.unsqueeze(1)], dim=1)

    def _one_hot_accuracy(self, logits, y, sampling_evidence):
        """Label accuracy of forward inference on the argmax digits; also per-digit accuracy (MNIST_add)."""
        cfg, B, model = self.cfg, self.batch_size, self.inference_model
        one_hot = F.one_hot(logits.argmax(dim=2), num_classes=cfg.num_classes)
        if cfg.inference == "bplay":
            label_probs = self._bp_forward(one_hot, damp=0.0 if cfg.experiment == "sudoku" else cfg.bp_damp)
        elif cfg.inference == "sampling":
            model.set_input(one_hot.float())
            num_samples_before = model.num_samples
            model.num_samples = 1  # one sample of a one-hot distribution
            model.sample()
            model.num_samples = num_samples_before
            label_probs = model.get_output(sampling_evidence, forward=True).view(-1, B)

        if cfg.experiment == "MNIST_add":
            correct = self._predicted_sum(label_probs) == y
            return correct.min(dim=1).values.float().mean().item(), correct.float().mean().item()
        elif cfg.experiment == "sudoku":
            true_dim = 0 if cfg.inference == "bplay" else 1  # index of 'true'
            pred_validity = label_probs.view(-1, 2, B)[:, true_dim, :].round().min(dim=0).values
            return (pred_validity == y).float().mean().item(), None

    def _log_memory(self):
        cuda = torch.cuda.is_available()
        self.log("memory_alloc_gb", torch.cuda.max_memory_allocated() / 1e9 if cuda else 0, sync_dist=True)
        self.log("memory_reserved_gb", torch.cuda.max_memory_reserved() / 1e9 if cuda else 0, sync_dist=True)

    # ------------------------------------------------------------------------------------------
    # lightning steps

    def training_step(self, batch, batch_idx):
        cfg = self.cfg
        x, y = self._split_batch(batch)
        bp_evidence, sampling_evidence = self._label_evidence(y, x.device)
        logits = self(x)
        input_probs = F.softmax(logits, dim=2)

        if cfg.em:
            with torch.no_grad():
                posteriors = self._e_step(input_probs, bp_evidence, sampling_evidence, sudoku_clamp=1e-3)
            if cfg.map:
                posteriors = self._to_map(posteriors)
            self._log_memory()
            loss = self._em_loss(logits, posteriors)

        else:  # end-to-end through BP
            label_probs = self._bp_forward(input_probs, damp=0.0 if cfg.experiment == "sudoku" else cfg.bp_damp, checkpointing=True)
            self._log_memory()
            loss = self.criterion(label_probs.log(), self._e2e_target(bp_evidence, label_probs.device))
            self._detach_bp_state()

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        cfg = self.cfg
        x, y = self._split_batch(batch)
        bp_evidence, sampling_evidence = self._label_evidence(y, x.device)
        logits = self(x)
        input_probs = F.softmax(logits, dim=2)

        if cfg.em:
            posteriors = self._e_step(input_probs, bp_evidence, sampling_evidence, sudoku_clamp=1e-1)
            if cfg.map:
                posteriors = self._to_map(posteriors)
            self.log("val_loss", self._em_loss(logits, posteriors), prog_bar=True, sync_dist=True)

        else:  # end-to-end
            label_probs = self._bp_forward(input_probs, damp=cfg.bp_damp)
            loss = self.criterion(label_probs.log(), self._e2e_target(bp_evidence, label_probs.device))
            self.log("val_loss", loss, prog_bar=True, sync_dist=True)
            self._detach_bp_state()

        acc, per_digit_acc = self._one_hot_accuracy(logits, y, sampling_evidence)
        if per_digit_acc is not None:
            self.log("one_hot_val_acc_sum_digits", per_digit_acc, prog_bar=True, sync_dist=True)
        self.log("one_hot_val_acc_entire_sum", acc, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = self._split_batch(batch)
        _, sampling_evidence = self._label_evidence(y, x.device)
        logits = self(x)
        acc, _ = self._one_hot_accuracy(logits, y, sampling_evidence)
        self.log("test_one_hot_val_acc_entire_sum", acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.num_epochs, eta_min=self.cfg.min_lr)
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
# data, logging, training

def build_dataloaders(cfg):
    if cfg.experiment == "sudoku":
        from data.visual_sudoku.load_visudo_dataset import create_loader
        return create_loader(
            grid_size=cfg.num_classes,
            batch_size=cfg.batch_size,
            num_train="00100",
            overlap="0.00",
            split=cfg.split,
            use_negative=False,  # only valid sudokus for training
            data_root=cfg.sudoku_data_dir,
            # loader seed stays 0 for every --seed, as in the paper
        )
    elif cfg.experiment == "MNIST_add":
        return get_mnist_addition_dataloaders(n_digits=cfg.num_digits, batch_size=cfg.batch_size,
                                              num_workers=cfg.num_workers, output_digits=True, seed=cfg.seed)


def build_logger(cfg):
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    method = "relaxed_bp" if cfg.inference == "sampling" and cfg.no_sampling else cfg.inference
    # same name format as the paper runs
    return TensorBoardLogger(
        save_dir=f"final_logs/{cfg.experiment}" if cfg.final else f"lightning_logs/{cfg.experiment}",
        name=f"{cfg.experiment}_{cfg.num_digits}digits_{method}_map_{True if cfg.map else 0}_em{1 if cfg.em else 0}"
             f"_{cfg.multi_m}Msteps_{cfg.num_samples}samples",
        version=f"{cfg.seed}seed_{cfg.split}split_lr{cfg.lr}_minlr{cfg.min_lr}_{cfg.num_epochs}epochs_batchsize{cfg.batch_size}_{run_id}"
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = make_config(args)

    inference_model = build_inference_model(cfg)
    logger = build_logger(cfg)
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    ckpt = ModelCheckpoint(monitor="one_hot_val_acc_entire_sum", mode="max", save_top_k=1, save_last=True)
    model = LitNeSyEM(cfg, inference_model)
    trainer = pl.Trainer(max_epochs=cfg.num_epochs, accelerator='auto', callbacks=[EpochTimer(), ckpt], logger=logger,
                         check_val_every_n_epoch=cfg.val_freq, log_every_n_steps=5)

    try:
        trainer.fit(model, train_loader, val_loader)
    except KeyboardInterrupt:
        print("training interrupted. Loading best model from checkpoint...")
    finally:
        # test the best checkpoint
        ckpt_path = None
        if getattr(ckpt, "best_model_path", ""):
            ckpt_path = ckpt.best_model_path
            print(f"Testing with best checkpoint: {ckpt_path}")
        elif getattr(ckpt, "last_model_path", ""):
            ckpt_path = ckpt.last_model_path
            print(f"Testing with last checkpoint: {ckpt_path}")
        else:
            print("No checkpoint found. Testing with current in-memory model weights.")

        if ckpt_path:
            trainer.test(ckpt_path=ckpt_path, dataloaders=test_loader)
        else:
            trainer.test(model=model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
