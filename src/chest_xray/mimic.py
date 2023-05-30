import os
import copy
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage.io import imread
from torch.utils.data import Dataset
import glob
from matplotlib import pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F


def get_attr_max_min(attr):
    if attr == "age":
        return 90, 18
    else:
        NotImplementedError


def preprocess_mimic(batch):
    for k, v in batch.items():
        if k == "x":
            batch[k] = (v.float() - 127.5) / 127.5  # [-1,1]
        elif k in ["age"]:
            batch[k] = v.float().unsqueeze(-1) / 100 * 2 - 1  # [-1,1]
        elif k in ["race"]:
            batch[k] = F.one_hot(v, num_classes=3).squeeze().float()
        else:
            batch[k] = v.float().unsqueeze(-1)
    return batch


class MimicDataset(Dataset):
    def __init__(
        self,
        root,
        csv_file,
        transform=None,
        columns=None,
        concat_pa=True,
        use_only_pleural_effusion=True,
        create_bias=False,
    ):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.disease_labels = [
            "No Finding",
            "Other",
            "Pleural Effusion",
            # "Lung Opacity",
        ]

        self.samples = {
            "age": [],
            "sex": [],
            "finding": [],
            "x": [],
            "race": [],
            # "lung_opacity": [],
            "pleural_effusion": [],
        }

        for idx, _ in enumerate(tqdm(range(len(self.data)), desc="Loading Data")):
            if use_only_pleural_effusion and self.data.loc[idx, "disease"] == "Other":
                continue
            img_path = os.path.join(root, self.data.loc[idx, "path_preproc"])

            # lung_opacity = self.data.loc[idx, "Lung Opacity"]
            # self.samples["lung_opacity"].append(lung_opacity)

            pleural_effusion = self.data.loc[idx, "Pleural Effusion"]
            self.samples["pleural_effusion"].append(pleural_effusion)
            disease = self.data.loc[idx, "disease"]
            finding = 0 if disease == "No Finding" else 1

            # Create a biased dataset
            if create_bias:
                if self.data.loc[idx, "sex"] == "Male" and finding == 0:
                    continue
                if self.data.loc[idx, "sex"] == "Female" and finding == 1:
                    continue

            self.samples["x"].append(img_path)
            self.samples["finding"].append(finding)
            self.samples["age"].append(self.data.loc[idx, "age"])
            self.samples["race"].append(self.data.loc[idx, "race_label"])
            self.samples["sex"].append(self.data.loc[idx, "sex_label"])

        self.columns = columns
        if self.columns is None:
            # ['age', 'race', 'sex']
            self.columns = list(self.data.columns)  # return all
            self.columns.pop(0)  # remove redundant 'index' column
        self.concat_pa = concat_pa

    def __len__(self):
        return len(self.samples["x"])

    def __getitem__(self, idx):
        sample = {k: v[idx] for k, v in self.samples.items()}
        sample["x"] = imread(sample["x"]).astype(np.float32)[None, ...]

        for k, v in sample.items():
            sample[k] = torch.tensor(v)

        if self.transform:
            sample["x"] = self.transform(sample["x"])

        sample = preprocess_mimic(sample)
        if self.concat_pa:
            sample["pa"] = torch.cat([sample[k] for k in self.columns], dim=0)
        return sample


if __name__ == "__main__":
    data_dir = "mimic-cxr-jpg-224/data/"
    csv_dir = "mimic_meta"

    csv_pd = pd.read_csv(csv_dir + "/mimic.sample.train.csv")
    d = glob.glob(csv_dir + "/*")

    train_set = MimicDataset(
        root=data_dir,
        csv_file=os.path.join(csv_dir, "mimic.sample.train.csv"),
        transform=None,
        columns=["age", "race", "sex", "finding"],
        concat_pa=True,
    )
