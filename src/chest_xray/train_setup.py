import os
import logging
import argparse
import torch
import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import linear_warmup
from chexpert import CheXpertDataset
from mimic import MimicDataset


def setup_dataloaders(args):
    if not args.data_dir:
        args.data_dir = "../chest_xray/CheXpert-v1.0"
    if not args.csv_dir:
        args.csv_dir = "../cheXpert"

    transf = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((args.input_res, args.input_res)),
        ]
    )

    if args.use_dataset == "chexpert":
        train_set = CheXpertDataset(
            root=args.data_dir,
            # csv_file=os.path.join(csv_dir, 'chexpert.sample.val.csv'),
            csv_file=os.path.join(args.csv_dir, "chexpert.sample.train.csv"),
            columns=args.parents_x,
            transform=transf,
        )
        valid_set = CheXpertDataset(
            root=args.data_dir,
            csv_file=os.path.join(args.csv_dir, "chexpert.sample.val.csv"),
            columns=args.parents_x,
            transform=transf,
        )
        test_set = CheXpertDataset(
            root=args.data_dir,
            csv_file=os.path.join(args.csv_dir, "chexpert.sample.test.csv"),
            columns=args.parents_x,
            transform=transf,
        )
    elif args.use_dataset == "mimic":
        train_set = MimicDataset(
            root=args.data_dir,
            # csv_file=os.path.join(csv_dir, 'chexpert.sample.val.csv'),
            csv_file=os.path.join(args.csv_dir, "mimic.sample.train.csv"),
            columns=args.parents_x,
            transform=transf,
            # create_bias=True,
        )
        valid_set = MimicDataset(
            root=args.data_dir,
            csv_file=os.path.join(args.csv_dir, "mimic.sample.val.csv"),
            columns=args.parents_x,
            transform=transf,
        )
        test_set = MimicDataset(
            root=args.data_dir,
            csv_file=os.path.join(args.csv_dir, "mimic.sample.test.csv"),
            columns=args.parents_x,
            transform=transf,
        )

    kwargs = {
        "batch_size": args.bs,
        "num_workers": os.cpu_count() // 2,
        "pin_memory": True,
    }
    dataloaders = {
        "train": DataLoader(train_set, shuffle=True, drop_last=True, **kwargs),
        "valid": DataLoader(valid_set, shuffle=False, drop_last=True, **kwargs),
        "test": DataLoader(test_set, shuffle=False, **kwargs),
    }

    return dataloaders


def setup_optimizer(args, model):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd, betas=args.betas
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=linear_warmup(args.lr_warmup_steps)
    )

    return optimizer, scheduler


def setup_directories(args, ckpt_dir="../checkpoints"):
    parents_folder = "_".join([k[0] for k in args.parents_x])
    save_dir = os.path.join(ckpt_dir, parents_folder, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def setup_tensorboard(args, model):
    """Setup metric summary writer."""
    writer = SummaryWriter(args.save_dir)

    hparams = {
        k: str(v) if isinstance(v, list) or isinstance(v, torch.device) else v
        for k, v in vars(args).items()
    }
    writer.add_hparams(hparams, {"hparams": 0}, run_name=os.path.abspath(args.save_dir))

    if "vae" in type(model).__name__.lower():
        z_str = []
        if hasattr(model.decoder, "blocks"):
            for i, block in enumerate(model.decoder.blocks):
                if block.stochastic:
                    z_str.append(f"z{i}_{block.res}x{block.res}")
        else:
            z_str = ["z0_" + str(args.z_dim)]

        writer.add_custom_scalars(
            {
                "nelbo": {"nelbo": ["Multiline", ["nelbo/train", "nelbo/valid"]]},
                "nll": {"kl": ["Multiline", ["nll/train", "nll/valid"]]},
                "kl": {"kl": ["Multiline", ["kl/train", "kl/valid"]]},
            }
        )
    return writer


def setup_logging(args):
    # reset root logger
    [logging.root.removeHandler(h) for h in logging.root.handlers[:]]
    # info logger for saving command line outputs during training
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(args.save_dir, "trainlog.txt")),
            logging.StreamHandler(),
        ],
        # filemode='a',  # append to file, 'w' for overwrite
        format="%(asctime)s, %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(args.exp_name)  # name the logger
    return logger


if __name__ == "__main__":
    from hps import add_arguments, setup_hparams

    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = setup_hparams(parser)
    args.data_dir = "../mimic-cxr-jpg-224/data/"
    args.csv_dir = "../mimic_meta"
    args.use_dataset = "mimic"
    args.free_bits = 0.0
    dataloaders = setup_dataloaders(args)
    train_loader = dataloaders["train"]
    sample_bacth = next(iter(dataloaders["train"]))
