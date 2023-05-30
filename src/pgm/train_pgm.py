import sys

sys.path.append("..")
from train_setup import *
from datasets import ukbb, get_attr_max_min, morphomnist, cmnist
from utils import seed_all, seed_worker, EMA

import os
import copy
import argparse
import pyro
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from utils_pgm import update_stats, plot_joint
from layers import TraceStorage_ELBO


def preprocess(batch, dataset="ukbb", split="l"):
    if "x" in batch.keys():
        batch["x"] = (batch["x"].float().cuda() - 127.5) / 127.5  # [-1,1]
    # for all other variables except x
    not_x = [k for k in batch.keys() if k != "x"]
    for k in not_x:
        if split == "u":  # unlabelled
            batch[k] = None
        elif split == "l":  # labelled
            batch[k] = batch[k].float().cuda()
            if len(batch[k].shape) < 2:
                batch[k] = batch[k].unsqueeze(-1)
        else:
            NotImplementedError
    if "ukbb" in dataset:
        for k in not_x:
            if k in ["age", "brain_volume", "ventricle_volume"]:
                k_max, k_min = get_attr_max_min(k)
                batch[k] = (batch[k] - k_min) / (k_max - k_min)  # [0,1]
                batch[k] = 2 * batch[k] - 1  # [-1,1]
            # else:
            # print(k, batch[k].max(), batch[k].min())
    return batch


def ss_train_epoch(args, model, ema, dataloaders, elbo_fn, aux_elbo_fn, optimizer):
    "semi-supervised training epoch"
    stats = {"loss": 0, "aux_loss": 0, "n": 0}  # sample counter
    alpha = args.alpha * len(dataloaders["l"].dataset)

    # outer loop over largest set, (u) unlabelled or (l) labelled
    if len(dataloaders["u"]) > len(dataloaders["l"]):
        outer, inner = "u", "l"
    else:
        outer, inner = "l", "u"
    iter_outer = iter(dataloaders[outer])
    iter_inner = iter(dataloaders[inner])
    loader = tqdm(range(len(iter_outer)))

    model.train()
    for i in loader:
        batch = {}
        batch[outer] = next(iter_outer)
        batch[outer] = preprocess(batch[outer], args.dataset, split=outer)

        try:
            batch[inner] = next(iter_inner)
        except StopIteration:
            iter_inner = iter(dataloaders[inner])  # restart inner iterator
            batch[inner] = next(iter_inner)
        batch[inner] = preprocess(batch[inner], args.dataset, split=inner)

        # supervised update
        loss = elbo_fn(model.svi_model, model.guide, **batch["l"])
        # unsupervised update
        loss = loss + elbo_fn(model.svi_model, model.guide, **batch["u"])
        # aux supervised update
        aux_loss = aux_elbo_fn(model.model_anticausal, model.guide_pass, **batch["l"])
        loss = loss + alpha * aux_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update()

        stats["loss"] += loss.item()
        stats["aux_loss"] += aux_loss.item()
        stats["n"] += batch[outer]["x"].shape[0]

        loader.set_description(
            f' => train | -elbo: {stats["loss"] / stats["n"]:.4f}'
            + f' - aux_loss: {stats["aux_loss"] / stats["n"] * alpha:.4f}'
        )

    stats = {k: v / stats["n"] for k, v in stats.items() if k != "n"}
    stats["aux_loss"] *= alpha
    return stats


def sup_epoch(args, model, ema, dataloader, elbo_fn, optimizer=None, is_train=True):
    "supervised epoch"
    stats = {"loss": 0, "n": 0}  # sample counter
    loader = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        miniters=len(dataloader) // 100,
        mininterval=5,
    )

    model.train(is_train)
    for i, batch in loader:
        bs = batch["x"].shape[0]
        batch = preprocess(batch, args.dataset, split="l")

        with torch.set_grad_enabled(is_train):
            if args.setup == "sup_aux":
                loss = (
                    elbo_fn.differentiable_loss(
                        model.model_anticausal, model.guide_pass, **batch
                    )
                    / bs
                )
            elif args.setup == "sup_pgm":
                loss = (
                    elbo_fn.differentiable_loss(
                        model.svi_model, model.guide_pass, **batch
                    )
                    / bs
                )
            else:
                NotImplementedError

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 200)
            optimizer.step()
            ema.update()

        stats["loss"] += loss.item() * bs
        stats["n"] += bs
        stats = update_stats(stats, elbo_fn)
        loader.set_description(
            f' => {("train" if is_train else "eval")} | '
            + f", ".join(
                f'{k}: {v / stats["n"]:.4f}' for k, v in stats.items() if k != "n"
            )
            +
            # ', probs: ' + f', '.join(f'{v:.4f}' for v in F.softmax(model.digit_probs.data, dim=-1).squeeze().tolist()) +
            (f", grad_norm: {grad_norm:.3f}" if is_train else ""),  # refresh=False
        )
    return {k: v / stats["n"] for k, v in stats.items() if k != "n"}


@torch.no_grad()
def eval_epoch(args, model, dataloader):
    "this can consume lots of memory if dataset is large"
    model.eval()
    preds = {k: [] for k in model.variables.keys()}
    targets = {k: [] for k in model.variables.keys()}

    for batch in tqdm(dataloader):
        for k in targets.keys():
            targets[k].extend(copy.deepcopy(batch[k]))
        # predict
        batch = preprocess(batch, args.dataset, split="l")
        out = model.predict(**batch)

        for k, v in out.items():
            preds[k].extend(v)

    for k, v in preds.items():
        preds[k] = torch.stack(v).squeeze().cpu()
        targets[k] = torch.stack(targets[k])

    stats = {}
    for k in model.variables.keys():
        if "ukbb" in args.dataset:
            if k == "mri_seq" or k == "sex":
                stats[k + "_rocauc"] = roc_auc_score(
                    targets[k].numpy(), preds[k].numpy(), average="macro"
                )
                stats[k + "_acc"] = (
                    targets[k] == torch.round(preds[k])
                ).sum().item() / targets[k].shape[0]
            else:  # continuous variables
                preds_k = (preds[k] + 1) / 2  # [-1,1] -> [0,1]
                _max, _min = get_attr_max_min(k)
                preds_k = preds_k * (_max - _min) + _min
                norm = 1000 if "volume" in k else 1  # for volume in ml
                stats[k + "_mae"] = (targets[k] - preds_k).abs().mean().item() / norm
        elif args.dataset == "morphomnist":
            if k == "digit":
                num_corrects = (targets[k].argmax(-1) == preds[k].argmax(-1)).sum()
                stats[k + "_acc"] = num_corrects.item() / targets[k].shape[0]
            else:  # continuous variables
                # unormalize from [-1,1] back to original range
                min_max = dataloader.dataset.min_max[k]
                _min, _max = min_max[0], min_max[1]
                preds_k = ((preds[k] + 1) / 2) * (_max - _min) + _min
                targets_k = ((targets[k] + 1) / 2) * (_max - _min) + _min
                stats[k + "_mae"] = (targets_k - preds_k).abs().mean().item()
        elif args.dataset == "cmnist":
            num_corrects = (targets[k].argmax(-1) == preds[k].argmax(-1)).sum()
            stats[k + "_acc"] = num_corrects.item() / targets[k].shape[0]
        else:
            NotImplementedError
    return stats


def setup_dataloaders(args):
    if "ukbb" in args.dataset:
        datasets = ukbb(args)
    elif args.dataset == "morphomnist":
        assert args.input_channels == 1
        assert args.input_res == 32
        assert args.pad == 4
        args.parents_x = ["thickness", "intensity", "digit"]
        args.context_norm = "[-1,1]"
        args.concat_pa = False
        datasets = morphomnist(args)
    elif args.dataset == "cmnist":
        assert args.input_channels == 3
        assert args.input_res == 32
        assert args.pad == 4
        args.parents_x = ["digit", "colour"]
        args.concat_pa = False
        datasets = cmnist(args)
    else:
        NotImplementedError

    kwargs = {
        "batch_size": args.bs,
        "num_workers": 4,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
    }
    dataloaders = {}
    if args.setup == "sup_pgm":
        dataloaders["train"] = DataLoader(
            datasets["train"], shuffle=True, drop_last=True, **kwargs
        )
    else:
        args.n_total = len(datasets["train"])
        args.n_labelled = int(args.sup_frac * args.n_total)
        args.n_unlabelled = args.n_total - args.n_labelled
        idx = np.arange(args.n_total)
        rng = np.random.RandomState(1)
        rng.shuffle(idx)
        train_l = torch.utils.data.Subset(datasets["train"], idx[: args.n_labelled])

        if args.setup == "semi_sup":
            train_u = torch.utils.data.Subset(datasets["train"], idx[args.n_labelled :])
            dataloaders["train_l"] = DataLoader(  # labelled
                train_l, shuffle=True, drop_last=True, **kwargs
            )
            dataloaders["train_u"] = DataLoader(  # unlabelled
                train_u, shuffle=True, drop_last=True, **kwargs
            )
        elif args.setup == "sup_aux":
            dataloaders["train"] = DataLoader(  # labelled
                train_l, shuffle=True, drop_last=True, **kwargs
            )

    dataloaders["valid"] = DataLoader(datasets["valid"], shuffle=False, **kwargs)
    dataloaders["test"] = DataLoader(datasets["test"], shuffle=False, **kwargs)
    return dataloaders


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", help="Experiment name.", type=str, default="")
    parser.add_argument("--dataset", help="Dataset name.", type=str, default="ukbb")
    parser.add_argument(
        "--data_dir", help="Data directory to load form.", type=str, default=""
    )
    parser.add_argument(
        "--load_path", help="Path to load checkpoint.", type=str, default=""
    )
    parser.add_argument(
        "--setup",  # semi_sup/sup_pgm/sup_aux
        help="training setup.",
        type=str,
        default="sup_pgm",
    )
    parser.add_argument("--seed", help="Set random seed.", type=int, default=7)
    parser.add_argument(
        "--deterministic",
        help="Toggle cudNN determinism.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--testing", help="Test model.", action="store_true", default=False
    )
    # training
    parser.add_argument(
        "--epochs", help="Number of training epochs.", type=int, default=1000
    )
    parser.add_argument("--bs", help="Batch size.", type=int, default=32)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=1e-4)
    parser.add_argument(
        "--lr_warmup_steps", help="lr warmup steps.", type=int, default=1
    )
    parser.add_argument("--wd", help="Weight decay penalty.", type=float, default=0.1)
    parser.add_argument(
        "--input_res", help="Input image crop resolution.", type=int, default=192
    )
    parser.add_argument(
        "--input_channels", help="Input image num channels.", type=int, default=1
    )
    parser.add_argument("--pad", help="Input padding.", type=int, default=9)
    parser.add_argument(
        "--hflip", help="Horizontal flip prob.", type=float, default=0.5
    )
    parser.add_argument(
        "--sup_frac", help="Labelled data fraction.", type=float, default=1
    )
    parser.add_argument("--eval_freq", help="Num epochs per eval.", type=int, default=1)
    # model
    parser.add_argument(
        "--widths",
        help="Cond flow fc network width per layer.",
        nargs="+",
        type=int,
        default=[32, 32],
    )
    parser.add_argument(
        "--parents_x", help="Parents of x to load.", nargs="+", default=[]
    )
    parser.add_argument(
        "--alpha",  # for semi_sup learning only
        help="aux loss multiplier.",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--std_fixed", help="Fix aux dist std value (0 is off).", type=float, default=0
    )
    args = parser.parse_known_args()[0]

    seed_all(args.seed, args.deterministic)

    # update hparams if loading checkpoint
    if args.load_path:
        if os.path.isfile(args.load_path):
            print(f"\nLoading checkpoint: {args.load_path}")
            ckpt = torch.load(args.load_path)
            ckpt_args = {k: v for k, v in ckpt["hparams"].items() if k != "load_path"}
            if args.data_dir is not None:
                ckpt_args["data_dir"] = args.data_dir
            if args.testing:
                ckpt_args["testing"] = args.testing
            vars(args).update(ckpt_args)
        else:
            print(f"Checkpoint not found at: {args.load_path}")

    # Load data
    dataloaders = setup_dataloaders(args)

    # Init model
    pyro.clear_param_store()
    if "ukbb" in args.dataset:
        from flow_pgm import FlowPGM

        model = FlowPGM(args)
    elif args.dataset == "morphomnist":
        from flow_pgm import MorphoMNISTPGM

        model = MorphoMNISTPGM(args)
    elif args.dataset == "cmnist":
        from flow_pgm import ColourMNISTPGM

        model = ColourMNISTPGM(args)
    else:
        NotImplementedError
    ema = EMA(model, beta=0.999)
    model.cuda()
    ema.cuda()

    # Init loss & optimizer
    elbo_fn = TraceStorage_ELBO(num_particles=2)
    aux_elbo_fn = TraceStorage_ELBO(num_particles=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if not args.testing:
        # Train model
        args.save_dir = setup_directories(args, ckpt_dir="../../checkpoints")
        writer = setup_tensorboard(args, model)
        logger = setup_logging(args)

        for k in sorted(vars(args)):
            logger.info(f"--{k}={vars(args)[k]}")
        if args.setup != "sup_pgm":
            logger.info(
                f"Data splits: #labelled: {args.n_labelled}"
                + f" - #unlabelled: {args.n_unlabelled}"
            )
        args.best_loss = float("inf")

        for epoch in range(args.epochs):
            logger.info(f"Epoch {epoch+1}:")

            # semi supervised training
            if args.setup == "semi_sup":
                stats = ss_train_epoch(
                    args,
                    model,
                    ema,
                    {"l": dataloaders["train_l"], "u": dataloaders["train_u"]},
                    elbo_fn,
                    aux_elbo_fn,
                    optimizer,
                )
                # valid aux loss on labelled data only
                if epoch % args.eval_freq == 0:
                    valid_stats = sup_epoch(
                        args,
                        ema.ema_model,
                        None,
                        dataloaders["valid"],
                        elbo_fn,
                        is_train=False,
                    )
                    steps = (epoch + 1) * max(
                        len(dataloaders["train_l"]), len(dataloaders["train_u"])
                    )
                    stats["aux_loss"] *= args.alpha * len(
                        dataloaders["train_l"].dataset
                    )

                    logger.info(
                        f'loss: {stats["loss"]:.4f}'
                        + f' - aux_loss: {stats["aux_loss"]:.4f}'
                        + f' - valid_aux_loss: {valid_stats["aux_loss"]:.4f} - steps: {steps}'
                    )
            # supervised training of PGM or aux models
            elif args.setup == "sup_pgm" or args.setup == "sup_aux":
                stats = sup_epoch(
                    args,
                    model,
                    ema,
                    dataloaders["train"],
                    elbo_fn,
                    optimizer,
                    is_train=True,
                )
                if epoch % args.eval_freq == 0:
                    valid_stats = sup_epoch(
                        args,
                        ema.ema_model,
                        None,
                        dataloaders["valid"],
                        elbo_fn,
                        is_train=False,
                    )
                    steps = (epoch + 1) * len(dataloaders["train"])
                    if args.setup == "sup_pgm":
                        plot_joint(
                            args, ema.ema_model, dataloaders["train"].dataset, steps
                        )

                    logger.info(
                        f'loss | train: {stats["loss"]:.4f}'
                        + f' - valid: {valid_stats["loss"]:.4f} - steps: {steps}'
                    )

                    for k, v in stats.items():
                        writer.add_scalar("train/" + k, v, steps)
                        writer.add_scalar("valid/" + k, valid_stats[k], steps)

                    writer.add_custom_scalars(
                        {"elbo": {"elbo": ["Multiline", ["elbo/train", "elbo/valid"]]}}
                    )
                    writer.add_scalar("elbo/train", stats["loss"], steps)
                    writer.add_scalar("elbo/valid", valid_stats["loss"], steps)
            else:
                NotImplementedError

            if epoch % args.eval_freq == 0:
                if not args.setup == "sup_pgm":  # eval aux classifiers
                    metrics = eval_epoch(args, ema.ema_model, dataloaders["valid"])
                    logger.info(
                        "valid | "
                        + " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                    )

            if valid_stats["loss"] < args.best_loss:
                args.best_loss = valid_stats["loss"]
                ckpt_path = os.path.join(args.save_dir, "checkpoint.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "step": steps,
                        "best_loss": args.best_loss,
                        "model_state_dict": model.state_dict(),
                        "ema_model_state_dict": ema.ema_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "hparams": vars(args),
                    },
                    ckpt_path,
                )
                logger.info(f"Model saved: {ckpt_path}")

    else:
        # test model
        model.load_state_dict(ckpt["model_state_dict"])
        ema.ema_model.load_state_dict(ckpt["ema_model_state_dict"])
        print("Evaluating test set:\n")
        stats = sup_epoch(
            args,
            ema.ema_model,
            None,
            dataloaders["test"],
            elbo_fn,
            optimizer=None,
            is_train=False,
        )
        if not args.setup == "sup_pgm":  # eval aux classifiers
            stats = eval_epoch(args, ema.ema_model, dataloaders["test"])
            print("test | " + " - ".join(f"{k}: {v:.4f}" for k, v in stats.items()))
        else:
            plot_joint(args, ema.ema_model, dataloaders["test"].dataset, 0)
            plt.show()
