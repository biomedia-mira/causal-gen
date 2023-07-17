import argparse
import copy
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from dscm import DSCM
from flow_pgm import FlowPGM
from layers import TraceStorage_ELBO
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_pgm import eval_epoch, preprocess, sup_epoch
from utils_pgm import plot_cf, update_stats

sys.path.append("..")
from datasets import get_attr_max_min
from hps import Hparams
from train_setup import setup_directories, setup_logging, setup_tensorboard
from utils import EMA, seed_all
from vae import HVAE


def loginfo(title: str, logger: Any, stats: Dict[str, Any]):
    logger.info(f"{title} | " + " - ".join(f"{k}: {v:.4f}" for k, v in stats.items()))


def inv_preprocess(pa: Dict[str, Tensor]) -> Dict[str, Tensor]:
    # undo [-1,1] parent preprocessing back to original range
    for k, v in pa.items():
        if k != "mri_seq" and k != "sex":
            pa[k] = (v + 1) / 2  # [-1,1] -> [0,1]
            _max, _min = get_attr_max_min(k)
            pa[k] = pa[k] * (_max - _min) + _min
    return pa


def save_plot(
    save_path: str,
    obs: Dict[str, Tensor],
    cfs: Dict[str, Tensor],
    do: Dict[str, Tensor],
    var_cf_x: Optional[Tensor] = None,
    num_images: int = 10,
) -> None:
    _ = plot_cf(
        obs["x"],
        cfs["x"],
        inv_preprocess({k: v for k, v in obs.items() if k != "x"}),  # pa
        inv_preprocess({k: v for k, v in cfs.items() if k != "x"}),  # cf_pa
        inv_preprocess(do),
        var_cf_x,  # counterfactual variance per pixel
        num_images=num_images,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_metrics(
    dataset: str, preds: Dict[str, List[Tensor]], targets: Dict[str, List[Tensor]]
) -> Dict[str, Tensor]:
    for k, v in preds.items():
        preds[k] = torch.stack(v).squeeze().cpu()
        targets[k] = torch.stack(targets[k]).squeeze().cpu()
    stats = {}
    for k in preds.keys():
        if "ukbb" in dataset:
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
        elif "mimic" in dataset:
            if k in ["sex", "finding"]:
                stats[k + "_rocauc"] = roc_auc_score(
                    targets[k].numpy(), preds[k].numpy(), average="macro"
                )
                stats[k + "_acc"] = (
                    targets[k] == torch.round(preds[k])
                ).sum().item() / targets[k].shape[0]
            elif k == "age":
                preds_k = (preds[k] + 1) * 50  # unormalize
                targets_k = (targets[k] + 1) * 50  # unormalize
                stats[k + "_mae"] = (targets_k - preds_k).abs().mean().item()
            elif k == "race":
                num_corrects = (targets[k].argmax(-1) == preds[k].argmax(-1)).sum()
                stats[k + "_acc"] = num_corrects.item() / targets[k].shape[0]
                stats[k + "_rocauc"] = roc_auc_score(
                    targets[k].numpy(),
                    preds[k].numpy(),
                    multi_class="ovr",
                    average="macro",
                )
        else:
            NotImplementedError
    return stats


def cf_epoch(
    args: Hparams,
    model: nn.Module,
    ema: nn.Module,
    dataloaders: Dict[str, DataLoader],
    elbo_fn: TraceStorage_ELBO,
    optimizers: Optional[Tuple] = None,
    split: str = "train",
):
    "counterfactual auxiliary training/eval epoch"
    is_train = split == "train"
    model.vae.train(is_train)
    model.pgm.eval()
    model.predictor.eval()
    stats = {k: 0 for k in ["loss", "aux_loss", "elbo", "nll", "kl", "n"]}
    steps_skipped = 0

    dag_vars = list(model.pgm.variables.keys())
    if is_train and isinstance(optimizers, tuple):
        optimizer, lagrange_opt = optimizers
    else:
        preds = {k: [] for k in dag_vars}
        targets = {k: [] for k in dag_vars}
        train_set = copy.deepcopy(dataloaders["train"].dataset.samples)

    loader = tqdm(
        enumerate(dataloaders[split]), total=len(dataloaders[split]), mininterval=0.1
    )

    for i, batch in loader:
        bs = batch["x"].shape[0]
        batch = preprocess(batch)

        with torch.no_grad():
            # randomly intervene on a single parent do(pa_k ~ p(pa_k))
            do = {}
            do_k = copy.deepcopy(args.do_pa) if args.do_pa else random.choice(dag_vars)
            if is_train:
                do[do_k] = batch[do_k].clone()[torch.randperm(bs)]
            else:
                idx = torch.randperm(train_set[do_k].shape[0])
                do[do_k] = train_set[do_k].clone()[idx][:bs]
                do = preprocess(do)

        with torch.set_grad_enabled(is_train):
            # if not is_train:
            #     args.cf_particles = 5 if i == 0 else 1

            out = model.forward(batch, do, elbo_fn, cf_particles=args.cf_particles)

            if torch.isnan(out["loss"]):
                model.zero_grad(set_to_none=True)
                steps_skipped += 1
                continue

        if is_train:
            args.step = i + (args.epoch - 1) * len(dataloaders[split])
            optimizer.zero_grad(set_to_none=True)
            lagrange_opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if grad_norm < args.grad_skip:
                optimizer.step()
                lagrange_opt.step()  # gradient ascent on lmbda
                model.lmbda.data.clamp_(min=0)
                ema.update()
            else:
                steps_skipped += 1
                print(f"Steps skipped: {steps_skipped} - grad_norm: {grad_norm:.3f}")
        else:  # evaluation
            with torch.no_grad():
                preds_cf = ema.ema_model.predictor.predict(**out["cfs"])
                for k, v in preds_cf.items():
                    preds[k].extend(v)
                # interventions are the targets for prediction
                for k in targets.keys():
                    t_k = do[k].clone() if k in do.keys() else out["cfs"][k].clone()
                    targets[k].extend(inv_preprocess({k: t_k})[k])

        if i % args.plot_freq == 0:
            if is_train:
                copy_do_pa = copy.deepcopy(args.do_pa)
                for pa_k in dag_vars + [None]:
                    args.do_pa = pa_k
                    valid_stats, valid_metrics = cf_epoch(  # recursion
                        args, model, ema, dataloaders, elbo_fn, None, split="valid"
                    )
                    loginfo(f"valid do({pa_k})", logger, valid_stats)
                    loginfo(f"valid do({pa_k})", logger, valid_metrics)
                args.do_pa = copy_do_pa
            # save_path = os.path.join(args.save_dir, f'{args.step}_{split}_{do_k}_cfs.pdf')
            # save_plot(save_path, batch, out['cfs'], do, out['var_cf_x'], num_images=args.imgs_plot)

        stats["n"] += bs
        stats["loss"] += out["loss"].item() * bs
        stats["aux_loss"] += out["aux_loss"].item() * args.alpha * bs
        stats["elbo"] += out["elbo"] * bs
        stats["nll"] += out["nll"] * bs
        stats["kl"] += out["kl"] * bs
        stats = update_stats(stats, elbo_fn)  # aux_model stats
        loader.set_description(
            f"[{split}] lmbda: {model.lmbda.data.item():.3f}, "
            + f", ".join(
                f'{k}: {v / stats["n"]:.3f}' for k, v in stats.items() if k != "n"
            )
            + (f", grad_norm: {grad_norm:.3f}" if is_train else "")
        )
    stats = {k: v / stats["n"] for k, v in stats.items() if k != "n"}
    return stats if is_train else (stats, get_metrics(args.dataset, preds, targets))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", help="experiment name.", type=str, default="")
    parser.add_argument(
        "--data_dir", help="data directory to load form.", type=str, default=""
    )
    parser.add_argument(
        "--load_path", help="Path to load checkpoint.", type=str, default=""
    )
    parser.add_argument(
        "--pgm_path",
        help="path to load pgm checkpoint.",
        type=str,
        default="../../checkpoints/sup_pgm/checkpoint.pt",
    )
    parser.add_argument(
        "--predictor_path",
        help="path to load predictor checkpoint.",
        type=str,
        default="../../checkpoints/sup_aux_prob/checkpoint.pt",
    )
    parser.add_argument(
        "--vae_path",
        help="path to load vae checkpoint.",
        type=str,
        default="../../checkpoints/from_server/m_b_v_s/ukbb192_beta5_dgauss_b33/checkpoint.pt",
    )
    parser.add_argument("--seed", help="random seed.", type=int, default=7)
    parser.add_argument(
        "--deterministic",
        help="toggle cudNN determinism.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--testing", help="test model.", action="store_true", default=False
    )
    # training
    parser.add_argument("--epochs", help="num training epochs.", type=int, default=5000)
    parser.add_argument("--bs", help="batch size.", type=int, default=32)
    parser.add_argument("--lr", help="learning rate.", type=float, default=1e-4)
    parser.add_argument(
        "--lr_lagrange", help="learning rate for multipler.", type=float, default=1e-2
    )
    parser.add_argument(
        "--ema_rate", help="Exp. moving avg. model rate.", type=float, default=0.999
    )
    parser.add_argument("--alpha", help="aux loss multiplier.", type=float, default=1)
    parser.add_argument(
        "--lmbda_init", help="lagrange multiplier init.", type=float, default=0
    )
    parser.add_argument(
        "--damping", help="lagrange damping scalar.", type=float, default=100
    )
    parser.add_argument("--do_pa", help="intervened parent.", type=str, default=None)
    parser.add_argument("--eval_freq", help="epochs per eval.", type=int, default=1)
    parser.add_argument("--plot_freq", help="steps per plot.", type=int, default=500)
    parser.add_argument("--imgs_plot", help="num images to plot.", type=int, default=10)
    parser.add_argument(
        "--cf_particles", help="num counterfactual samples.", type=int, default=1
    )
    args = parser.parse_known_args()[0]

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

    seed_all(args.seed, args.deterministic)

    # Load predictors
    print(f"\nLoading predictor checkpoint: {args.predictor_path}")
    predictor_checkpoint = torch.load(args.predictor_path)
    predictor_args = Hparams()
    predictor_args.update(predictor_checkpoint["hparams"])
    predictor = FlowPGM(predictor_args).cuda()
    predictor.load_state_dict(predictor_checkpoint["ema_model_state_dict"])

    # for backwards compatibility
    if not hasattr(predictor_args, "dataset"):
        predictor_args.dataset = "ukbb"
    if hasattr(predictor_args, "loss_norm"):
        args.loss_norm

    from train_pgm import setup_dataloaders

    if args.data_dir != "":
        predictor_args.data_dir = args.data_dir
    dataloaders = setup_dataloaders(predictor_args)
    elbo_fn = TraceStorage_ELBO(num_particles=1)

    test_stats = sup_epoch(
        predictor_args,
        predictor,
        None,
        dataloaders["test"],
        elbo_fn,
        optimizer=None,
        is_train=False,
    )
    stats = eval_epoch(predictor_args, predictor, dataloaders["test"])
    print("test | " + " - ".join(f"{k}: {v:.4f}" for k, v in stats.items()))

    # Load PGM
    print(f"\nLoading PGM checkpoint: {args.pgm_path}")
    pgm_checkpoint = torch.load(args.pgm_path)
    pgm_args = Hparams()
    pgm_args.update(pgm_checkpoint["hparams"])
    pgm = FlowPGM(pgm_args).cuda()
    pgm.load_state_dict(pgm_checkpoint["ema_model_state_dict"])

    # for backwards compatibility
    if not hasattr(pgm_args, "dataset"):
        pgm_args.dataset = "ukbb"
    if args.data_dir != "":
        pgm_args.data_dir = args.data_dir
    dataloaders = setup_dataloaders(pgm_args)
    elbo_fn = TraceStorage_ELBO(num_particles=1)

    test_stats = sup_epoch(
        pgm_args, pgm, None, dataloaders["test"], elbo_fn, is_train=False
    )

    # Load deep VAE
    print(f"\nLoading VAE checkpoint: {args.vae_path}")
    vae_checkpoint = torch.load(args.vae_path)
    vae_args = Hparams()
    vae_args.update(vae_checkpoint["hparams"])
    if not hasattr(vae_args, "cond_prior"):  # for backwards compatibility
        vae_args.cond_prior = False
    vae_args.kl_free_bits = vae_args.free_bits
    vae = HVAE(vae_args).cuda()
    vae.load_state_dict(vae_checkpoint["ema_model_state_dict"])

    # vae_args.data_dir = None  # adjust data_dir as needed
    if args.data_dir != "":
        vae_args.data_dir = args.data_dir
    dataloaders = setup_dataloaders(vae_args)

    @torch.no_grad()
    def vae_epoch(args, vae, dataloader):
        vae.eval()
        stats = {k: 0 for k in ["elbo", "nll", "kl", "n"]}
        loader = tqdm(enumerate(dataloader), total=len(dataloader))

        for i, batch in loader:
            # preprocessing
            batch["x"] = (batch["x"].cuda().float() - 127.5) / 127.5  # [-1, 1]
            batch["pa"] = (
                batch["pa"][..., None, None]
                .repeat(1, 1, args.input_res, args.input_res)
                .cuda()
                .float()
            )
            # forward pass
            out = vae(batch["x"], batch["pa"], beta=args.beta)
            # update stats
            bs = batch["x"].shape[0]
            stats["n"] += bs  # samples seen counter
            stats["elbo"] += out["elbo"] * bs
            stats["nll"] += out["nll"] * bs
            stats["kl"] += out["kl"] * bs
            loader.set_description(
                f' => eval | nelbo: {stats["elbo"] / stats["n"]:.4f}'
                + f' - nll: {stats["nll"] / stats["n"]:.4f}'
                + f' - kl: {stats["kl"] / stats["n"]:.4f}'
            )
        return {k: v / stats["n"] for k, v in stats.items() if k != "n"}

    stats = vae_epoch(vae_args, vae, dataloaders["test"])

    # setup current experiment args
    args.beta = vae_args.beta
    args.parents_x = vae_args.parents_x
    args.input_res = vae_args.input_res
    args.grad_clip = vae_args.grad_clip
    args.grad_skip = vae_args.grad_skip
    args.elbo_constraint = 1.841216802597046  # train set elbo constraint
    args.wd = vae_args.wd
    args.betas = vae_args.betas

    # init model
    if not hasattr(vae_args, "dataset"):
        args.dataset = "ukbb"
    model = DSCM(args, pgm, predictor, vae)
    ema = EMA(model, beta=args.ema_rate)
    model.cuda()
    ema.cuda()

    # setup data
    pgm_args.concat_pa = False
    pgm_args.bs = args.bs
    from train_pgm import setup_dataloaders

    dataloaders = setup_dataloaders(pgm_args)

    # Train model
    if not args.testing:
        args.save_dir = setup_directories(args, ckpt_dir="../../checkpoints")
        writer = setup_tensorboard(args, model)
        logger = setup_logging(args)
        writer.add_custom_scalars(
            {
                "loss": {"loss": ["Multiline", ["loss/train", "loss/valid"]]},
                "aux_loss": {
                    "aux_loss": ["Multiline", ["aux_loss/train", "aux_loss/valid"]]
                },
            }
        )

        # setup loss & optimizer
        elbo_fn = TraceStorage_ELBO(num_particles=1)
        optimizer = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if n != "lmbda"],
            lr=args.lr,
            weight_decay=args.wd,
            betas=args.betas,
        )
        lagrange_opt = torch.optim.AdamW(
            [model.lmbda],
            lr=args.lr_lagrange,
            betas=args.betas,
            weight_decay=0,
            maximize=True,
        )
        optimizers = (optimizer, lagrange_opt)

        # load checkpoint
        if args.load_path:
            if os.path.isfile(args.load_path):
                args.start_epoch = ckpt["epoch"]
                args.step = ckpt["step"]
                args.best_loss = ckpt["best_loss"]
                model.load_state_dict(ckpt["model_state_dict"])
                ema.ema_model.load_state_dict(ckpt["ema_model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                lagrange_opt.load_state_dict(ckpt["lagrange_opt_state_dict"])
            else:
                print("Checkpoint not found: {}".format(args.load_path))
        else:
            args.start_epoch, args.step = 0, 0
            args.best_loss = float("inf")

        for k in sorted(vars(args)):
            logger.info(f"--{k}={vars(args)[k]}")

        # training loop
        for epoch in range(args.start_epoch, args.epochs):
            args.epoch = epoch + 1
            logger.info(f"Epoch: {args.epoch}")
            stats = cf_epoch(
                args, model, ema, dataloaders, elbo_fn, optimizers, split="train"
            )
            loginfo("train", logger, stats)

            if epoch % args.eval_freq == 0:
                # evaluate single parent interventions
                copy_do_pa = copy.deepcopy(args.do_pa)
                for pa_k in list(model.pgm.variables.keys()) + [None]:
                    args.do_pa = pa_k
                    valid_stats, metrics = cf_epoch(
                        args, model, ema, dataloaders, elbo_fn, None, split="valid"
                    )
                    loginfo(f"valid do({pa_k})", logger, valid_stats)
                    loginfo(f"valid do({pa_k})", logger, metrics)
                args.do_pa = copy_do_pa

                for k, v in stats.items():
                    writer.add_scalar("train/" + k, v, args.step)
                    writer.add_scalar("valid/" + k, valid_stats[k], args.step)

                for k, v in metrics.items():
                    writer.add_scalar("valid/" + k, v, args.step)

                writer.add_scalar("loss/train", stats["loss"], args.step)
                writer.add_scalar("loss/valid", valid_stats["loss"], args.step)
                writer.add_scalar("aux_loss/train", stats["aux_loss"], args.step)
                writer.add_scalar("aux_loss/valid", valid_stats["aux_loss"], args.step)

                if valid_stats["loss"] < args.best_loss:
                    args.best_loss = valid_stats["loss"]
                    ckpt_path = os.path.join(
                        args.save_dir, f"{args.step}_checkpoint.pt"
                    )
                    torch.save(
                        {
                            "epoch": args.epoch,
                            "step": args.step,
                            "best_loss": args.best_loss,
                            "model_state_dict": model.state_dict(),
                            "ema_model_state_dict": ema.ema_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "lagrange_opt_state_dict": lagrange_opt.state_dict(),
                            "hparams": vars(args),
                        },
                        ckpt_path,
                    )
                    logger.info(f"Model saved: {ckpt_path}")
    else:
        # test model
        model.load_state_dict(ckpt["model_state_dict"])
        ema.ema_model.load_state_dict(ckpt["ema_model_state_dict"])
        stats, metrics = cf_epoch(
            args, model, ema, dataloaders, elbo_fn, None, split="test"
        )
        print(f"\n[test] " + " - ".join(f"{k}: {v:.4f}" for k, v in stats.items()))
        print(f"[test] " + " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
