import gzip
import os
import random
import struct
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as TF
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from hps import Hparams
from utils import log_standardize, normalize


class UKBBDataset(Dataset):
    def __init__(
        self,
        root: str,
        csv_file: str,
        transform: Optional[torchvision.transforms.Compose],
        columns: Optional[List[str]],
        norm: Optional[str],
        concat_pa=True,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.concat_pa = concat_pa  # return concatenated parents

        print(f"\nLoading csv data: {csv_file}")
        self.df = pd.read_csv(csv_file)
        self.columns = columns
        if self.columns is None:
            # ['eid', 'sex', 'age', 'brain_volume', 'ventricle_volume', 'mri_seq']
            self.columns = list(self.df.columns)  # return all
            self.columns.pop(0)  # remove redundant 'index' column
        print(f"columns: {self.columns}")
        self.samples = {i: torch.as_tensor(self.df[i]).float() for i in self.columns}

        for k in ["age", "brain_volume", "ventricle_volume"]:
            print(f"{k} normalization: {norm}")
            if k in self.columns:
                if norm == "[-1,1]":
                    self.samples[k] = normalize(self.samples[k])
                elif norm == "[0,1]":
                    self.samples[k] = normalize(self.samples[k], zero_one=True)
                elif norm == "log_standard":
                    self.samples[k] = log_standardize(self.samples[k])
                elif norm == None:
                    pass
                else:
                    NotImplementedError(f"{norm} not implemented.")
        print(f"#samples: {len(self.df)}")
        self.return_x = True if "eid" in self.columns else False

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sample = {k: v[idx] for k, v in self.samples.items()}

        if self.return_x:
            mri_seq = "T1" if sample["mri_seq"] == 0.0 else "T2_FLAIR"
            # Load scan
            filename = (
                f'{int(sample["eid"])}_' + mri_seq + "_unbiased_brain_rigid_to_mni.png"
            )
            x = Image.open(os.path.join(self.root, "thumbs_192x192", filename))

            if self.transform is not None:
                sample["x"] = self.transform(x)
            sample.pop("eid", None)

        if self.concat_pa:
            sample["pa"] = torch.cat(
                [torch.tensor([sample[k]]) for k in self.columns if k != "eid"], dim=0
            )

        return sample


def get_attr_max_min(attr: str):
    # some ukbb dataset (max, min) stats
    if attr == "age":
        return 73, 44
    elif attr == "brain_volume":
        return 1629520, 841919
    elif attr == "ventricle_volume":
        return 157075, 7613.27001953125
    else:
        NotImplementedError


def ukbb(args: Hparams) -> Dict[str, UKBBDataset]:
    # Load data
    if not args.data_dir:
        args.data_dir = "../ukbb/"
    csv_dir = os.path.join(args.data_dir, "brain_csv")

    augmentation = {
        "train": TF.Compose(
            [
                TF.Resize((args.input_res, args.input_res)),
                TF.RandomCrop(
                    size=(args.input_res, args.input_res),
                    padding=[2 * args.pad, args.pad],
                ),
                TF.RandomHorizontalFlip(p=args.hflip),
                TF.PILToTensor(),
            ]
        ),
        "eval": TF.Compose(
            [TF.Resize((args.input_res, args.input_res)), TF.PILToTensor()]
        ),
    }

    datasets = {}
    for split in ["train", "valid", "test"]:
        datasets[split] = UKBBDataset(
            root=args.data_dir,
            csv_file=os.path.join(csv_dir, split + ".csv"),
            transform=augmentation[("eval" if split != "train" else split)],
            columns=(None if not args.parents_x else ["eid"] + args.parents_x),
            norm=(None if not hasattr(args, "context_norm") else args.context_norm),
            concat_pa=(True if not hasattr(args, "concat_pa") else args.concat_pa),
        )

    return datasets


def _load_uint8(f):
    idx_dtype, ndim = struct.unpack("BBBB", f.read(4))[2:]
    shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data


def load_idx(path: str) -> np.ndarray:
    """Reads an array in IDX format from disk.
    Parameters
    ----------
    path : str
        Path of the input file. Will uncompress with `gzip` if path ends in '.gz'.
    Returns
    -------
    np.ndarray
        Output array of dtype ``uint8``.
    References
    ----------
    http://yann.lecun.com/exdb/mnist/
    """
    open_fcn = gzip.open if path.endswith(".gz") else open
    with open_fcn(path, "rb") as f:
        return _load_uint8(f)


def _get_paths(root_dir, train):
    prefix = "train" if train else "t10k"
    images_filename = prefix + "-images-idx3-ubyte.gz"
    labels_filename = prefix + "-labels-idx1-ubyte.gz"
    metrics_filename = prefix + "-morpho.csv"
    images_path = os.path.join(root_dir, images_filename)
    labels_path = os.path.join(root_dir, labels_filename)
    metrics_path = os.path.join(root_dir, metrics_filename)
    return images_path, labels_path, metrics_path


def load_morphomnist_like(
    root_dir, train: bool = True, columns=None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Args:
        root_dir: path to data directory
        train: whether to load the training subset (``True``, ``'train-*'`` files) or the test
            subset (``False``, ``'t10k-*'`` files)
        columns: list of morphometrics to load; by default (``None``) loads the image index and
            all available metrics: area, length, thickness, slant, width, and height
    Returns:
        images, labels, metrics
    """
    images_path, labels_path, metrics_path = _get_paths(root_dir, train)
    images = load_idx(images_path)
    labels = load_idx(labels_path)

    if columns is not None and "index" not in columns:
        usecols = ["index"] + list(columns)
    else:
        usecols = columns
    metrics = pd.read_csv(metrics_path, usecols=usecols, index_col="index")
    return images, labels, metrics


class MorphoMNIST(Dataset):
    def __init__(
        self,
        root_dir: str,
        train: bool = True,
        transform: Optional[torchvision.transforms.Compose] = None,
        columns: Optional[List[str]] = None,
        norm: Optional[str] = None,
        concat_pa: bool = True,
    ):
        self.train = train
        self.transform = transform
        self.columns = columns
        self.concat_pa = concat_pa
        self.norm = norm

        cols_not_digit = [c for c in self.columns if c != "digit"]
        images, labels, metrics_df = load_morphomnist_like(
            root_dir, train, cols_not_digit
        )
        self.images = torch.from_numpy(np.array(images)).unsqueeze(1)
        self.labels = F.one_hot(
            torch.from_numpy(np.array(labels)).long(), num_classes=10
        )

        if self.columns is None:
            self.columns = metrics_df.columns
        self.samples = {k: torch.tensor(metrics_df[k]) for k in cols_not_digit}

        self.min_max = {
            "thickness": [0.87598526, 6.255515],
            "intensity": [66.601204, 254.90317],
        }

        for k, v in self.samples.items():  # optional preprocessing
            print(f"{k} normalization: {norm}")
            if norm == "[-1,1]":
                self.samples[k] = normalize(
                    v, x_min=self.min_max[k][0], x_max=self.min_max[k][1]
                )
            elif norm == "[0,1]":
                self.samples[k] = normalize(
                    v, x_min=self.min_max[k][0], x_max=self.min_max[k][1], zero_one=True
                )
            elif norm == None:
                pass
            else:
                NotImplementedError(f"{norm} not implemented.")
        print(f"#samples: {len(metrics_df)}\n")

        self.samples.update({"digit": self.labels})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sample = {}
        sample["x"] = self.images[idx]

        if self.transform is not None:
            sample["x"] = self.transform(sample["x"])

        if self.concat_pa:
            sample["pa"] = torch.cat(
                [
                    v[idx] if k == "digit" else torch.tensor([v[idx]])
                    for k, v in self.samples.items()
                ],
                dim=0,
            )
        else:
            sample.update({k: v[idx] for k, v in self.samples.items()})
        return sample


def morphomnist(args: Hparams) -> Dict[str, MorphoMNIST]:
    # Load data
    if not args.data_dir:
        args.data_dir = "../morphomnist/"

    augmentation = {
        "train": TF.Compose(
            [
                TF.RandomCrop((args.input_res, args.input_res), padding=args.pad),
            ]
        ),
        "eval": TF.Compose(
            [
                TF.Pad(padding=2),  # (32, 32)
            ]
        ),
    }

    datasets = {}
    for split in ["train", "valid", "test"]:
        datasets[split] = MorphoMNIST(
            root_dir=args.data_dir,
            train=(split == "train"),  # test set is valid set
            transform=augmentation[("eval" if split != "train" else split)],
            columns=args.parents_x,
            norm=args.context_norm,
            concat_pa=args.concat_pa,
        )
    return datasets


class ColourMNIST(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[torchvision.transforms.Compose] = None,
        concat_pa: bool = True,
        corrupt_p: float = 0,
    ):
        self.transform = transform
        self.concat_pa = concat_pa
        root = os.path.join(root, "train") if train else os.path.join(root, "test")
        self.images = np.load(os.path.join(root, "images.npy"))
        pickle_load = lambda *a, **k: np.load(*a, allow_pickle=True, **k)
        parents = pickle_load(os.path.join(root, "parents.npy")).item()

        if train and corrupt_p > 0:
            # corrupt first 'corrupt_p' percentage of labels
            corrupt_indices = torch.randperm(len(self.images))[
                : int(corrupt_p * len(self.images))
            ]

            def sample_set(n, exclude=None):
                s = set(np.arange(n))
                s.remove(exclude)
                return random.choice(tuple(s))

            for idx in corrupt_indices:
                new_y = sample_set(10, exclude=parents["digit"][idx])
                parents["digit"][idx] = new_y

                new_y = sample_set(10, exclude=parents["colour"][idx])
                parents["colour"][idx] = new_y

        self.samples = {
            "digit": F.one_hot(torch.tensor(parents["digit"]), num_classes=10),
            "colour": F.one_hot(torch.tensor(parents["colour"]), num_classes=10),
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sample = {}
        sample["x"] = self.images[idx]

        if self.transform:
            sample["x"] = self.transform(sample["x"])

        if self.concat_pa:
            sample["pa"] = torch.cat([v[idx] for _, v in self.samples.items()], dim=0)
        else:
            sample.update({k: v[idx] for k, v in self.samples.items()})
        return sample


def cmnist(args: Hparams) -> Dict[str, ColourMNIST]:
    if not args.data_dir:
        args.data_dir = "../mnist_digit_colour"

    augmentation = {
        "train": TF.Compose(
            [
                TF.ToPILImage(),
                TF.RandomCrop((args.input_res, args.input_res), padding=args.pad),
                TF.PILToTensor(),
            ]
        ),
        "eval": TF.Compose(
            [TF.ToPILImage(), TF.Pad(padding=2), TF.PILToTensor()]  # (32, 32)
        ),
    }

    datasets = {}
    for split in ["train", "valid", "test"]:
        datasets[split] = ColourMNIST(
            root=args.data_dir,
            train=(split == "train"),  # test set is valid set
            transform=augmentation[("eval" if split != "train" else split)],
            concat_pa=args.concat_pa,
        )

    return datasets


class MIMICMetadata(TypedDict):
    age: float  # age in years
    sex: int  # 0 -> male , 1 -> female
    race: int  # 0 -> white , 1 -> asian , 2 -> black


def read_mimic_from_df(
    idx: int, df: pd.DataFrame, data_dir: str
) -> Tuple[Image.Image, torch.Tensor, MIMICMetadata]:
    """Get a single data point from the MIMIC-CXR dataframe.

    References:
    Written by Charles Jones.
    https://github.com/biomedia-mira/chexploration/blob/main/notebooks/mimic.sample.ipynb

    Args:
        idx (int): Index of the data point to retrieve.
        df (pd.DataFrame): Dataframe containing the MIMIC-CXR data.
        data_dir (str): Path to the directory containing the preprocessed data.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing the image and binary label.
            label 0 represents no finding, 1 represents pleural effusion.
    """
    img_path = os.path.join(data_dir, df.iloc[idx]["path_preproc"])
    img = Image.open(img_path)  # .convert("RGB")
    if df.iloc[idx]["disease"] == "Pleural Effusion":
        label = torch.tensor(1)
    elif df.iloc[idx]["disease"] == "No Finding":
        label = torch.tensor(0)
    else:
        raise ValueError(
            f"Invalid label {df.iloc[idx]['disease']}.",
            "We expect either 'pleural_effusion' or 'no_finding'.",
        )

    age = df.iloc[idx]["age"]
    sex = df.iloc[idx]["sex_label"]
    race = df.iloc[idx]["race_label"]

    meta = MIMICMetadata(age=age, sex=sex, race=race)
    return img, label, meta


class MIMIC(Dataset):
    def __init__(
        self,
        split_path,
        data_dir,
        cache=False,
        transform=None,
        parents_x=None,
        concat_pa=False,
    ):
        super().__init__()
        self.concat_pa = concat_pa
        self.parents_x = parents_x
        split_df = pd.read_csv(split_path)
        # remove rows whose disease label is neither No Finding nor Pleural Effusion
        self.split_df = split_df[
            (split_df["disease"] == "No Finding")
            | (split_df["disease"] == "Pleural Effusion")
        ].reset_index(drop=True)

        self.data_dir = data_dir
        self.cache = cache
        self.transform = transform

        if self.cache:
            self.imgs = []
            self.labels = []
            self.meta = []
            for idx, _ in tqdm(
                self.split_df.iterrows(), total=len(self.split_df), desc="Caching MIMIC"
            ):
                assert isinstance(idx, int)
                img, label, meta = read_mimic_from_df(idx, self.split_df, self.data_dir)
                self.imgs.append(img)
                self.labels.append(label)
                self.meta.append(meta)

    def __len__(self):
        return len(self.split_df)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        if self.cache:
            img = self.imgs[idx]
            label = self.labels[idx]
            meta = self.meta[idx]
        else:
            img, label, meta = read_mimic_from_df(idx, self.split_df, self.data_dir)
        sample = {}
        sample["x"] = self.transform(img)
        sample["finding"] = label
        sample.update(meta)
        sample = preprocess_mimic(sample)
        if self.concat_pa:
            sample["pa"] = torch.cat(
                [sample[k] for k in self.parents_x],
                dim=0,
            )
        return sample


def preprocess_mimic(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
    for k, v in sample.items():
        if k != "x":
            sample[k] = torch.tensor([v])
            if k == "race":
                sample[k] = F.one_hot(sample[k], num_classes=3).squeeze()
            elif k == "age":
                sample[k] = sample[k] / 100 * 2 - 1  # [-1,1]
    return sample


def mimic(
    args: Hparams,
    augmentation: Optional[Dict[str, torchvision.transforms.Compose]] = None,
) -> Dict[str, MIMIC]:
    if augmentation is None:
        augmentation = {}
        augmentation["train"] = TF.Compose(
            [
                TF.Resize((args.input_res, args.input_res), antialias=None),
                TF.PILToTensor(),
            ]
        )
        augmentation["eval"] = augmentation["train"]

    datasets = {}
    for split in ["train", "valid", "test"]:
        datasets[split] = MIMIC(
            data_dir=os.path.join(args.data_dir, "data"),
            split_path=os.path.join(args.data_dir, "meta", f"{split}.csv"),
            cache=False,
            parents_x=args.parents_x,  # ["age", "race", "sex", "finding"],
            concat_pa=(True if not hasattr(args, "concat_pa") else args.concat_pa),
            transform=augmentation[("eval" if split != "train" else split)],
        )
    return datasets
