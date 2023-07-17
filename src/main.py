import argparse
import gc
import os
import traceback

import send2trash
import torch

from hps import Hparams
from simple_vae import VAE
from train_setup import (
    setup_dataloaders,
    setup_directories,
    setup_logging,
    setup_optimizer,
    setup_tensorboard,
)
from trainer import trainer
from utils import EMA, seed_all
from vae import HVAE


def main(args: Hparams):
    seed_all(args.seed, args.deterministic)
    # update hyperparams if resuming from a checkpoint
    ckpt = None
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\nLoading checkpoint: {args.resume}")
            ckpt = torch.load(args.resume)
            ckpt_args = {k: v for k, v in ckpt["hparams"].items() if k != "resume"}
            if args.data_dir is not None:
                ckpt_args["data_dir"] = args.data_dir
            if args.lr < ckpt_args["lr"]:
                ckpt_args["lr"] = args.lr
            vars(args).update(ckpt_args)
        else:
            print(f"Checkpoint not found at: {args.resume}")

    # load data
    dataloaders = setup_dataloaders(args)

    # init model
    if args.vae == "hierarchical":
        model = HVAE(args)
    elif args.vae == "simple":
        model = VAE(args)
    else:
        NotImplementedError

    def init_bias(m):
        if type(m) == torch.nn.Conv2d:
            torch.nn.init.zeros_(m.bias)

    model.apply(init_bias)
    ema = EMA(model, beta=args.ema_rate)
    ema.ema_model.eval()

    # setup model save directory, logging and tensorboard summaries
    assert args.exp_name != "", "No experiment name given."
    args.save_dir = setup_directories(args)
    writer = setup_tensorboard(args, model)
    logger = setup_logging(args)

    # setup optimizer
    optimizer, scheduler = setup_optimizer(args, model)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.device)
    model.to(args.device)
    ema.to(args.device)

    # load checkpoint state dicts
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        ema.ema_model.load_state_dict(ckpt["ema_model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        # update lr of the loaded optimizer
        for p_group in optimizer.param_groups:
            p_group["lr"] = args.lr
            p_group["initial_lr"] = args.lr  # needed to init the scheduler lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda x: x * 0 + 1
        )
        args.start_epoch, args.iter = ckpt["epoch"], ckpt["step"]
        args.best_loss = ckpt["best_loss"]
        del ckpt  # remove reference to checkpoint
    else:
        args.start_epoch, args.iter, args.best_loss = 0, 0, float("inf")

    # train
    try:
        gc.collect()
        torch.cuda.empty_cache()
        trainer(args, model, ema, dataloaders, optimizer, scheduler, writer, logger)
    except KeyboardInterrupt:
        print(traceback.format_exc())
        if input("Training interrupted, keep logs? [Y/n]: ") == "n":
            if input(f"Send '{args.save_dir}' to Trash? [y/N]: ") == "y":
                send2trash.send2trash(args.save_dir)
                print("Done.")


if __name__ == "__main__":
    from hps import add_arguments, setup_hparams

    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = setup_hparams(parser)
    main(args)
