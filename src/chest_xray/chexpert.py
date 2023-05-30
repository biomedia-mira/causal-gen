import os
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage.io import imread
from torch.utils.data import Dataset
import torch.nn.functional as F


def get_attr_max_min(attr):
    if attr == "age":
        return 90, 18
    else:
        NotImplementedError


def preprocess_chexpert(batch):
    for k, v in batch.items():
        if k == "x":
            batch["x"] = (v.float() - 127.5) / 127.5  # [-1,1]
        elif k in ["age"]:
            batch[k] = v.float().unsqueeze(-1) / 100 * 2 - 1
        elif k in ["race"]:
            batch[k] = F.one_hot(v, num_classes=3).squeeze().float()
        else:
            batch[k] = v.float().unsqueeze(-1)
    return batch


class CheXpertDataset(Dataset):
    def __init__(self, root, csv_file, transform=None, columns=None, concat_pa=True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
        ]

        self.samples = {
            "age": [],
            "sex": [],
            "finding": [],
            "x": [],
            "race": [],
        }
        for idx, _ in enumerate(tqdm(range(len(self.data)), desc="Loading Data")):
            img_path = os.path.join(root, self.data.loc[idx, "path_preproc"])

            disease = np.zeros(len(self.labels) - 1, dtype=int)
            for i in range(1, len(self.labels)):
                disease[i - 1] = np.array(self.data.loc[idx, self.labels[i]] == 1)

            finding = 0 if disease.sum() == 0 else 1

            self.samples["x"].append(img_path)
            self.samples["finding"].append(finding)
            self.samples["age"].append(self.data.loc[idx, "age"])
            self.samples["race"].append(self.data.loc[idx, "race_label"])
            self.samples["sex"].append(self.data.loc[idx, "sex_label"])

        # self.samples = np.array(self.samples)
        # self.samples_copy = copy.deepcopy(self.samples)

        self.columns = columns
        if self.columns is None:
            # ['age', 'race', 'sex']
            self.columns = list(self.data.columns)  # return all
            self.columns.pop(0)  # remove redundant 'index' column
        self.concat_pa = concat_pa

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {k: v[idx] for k, v in self.samples.items()}
        sample["x"] = imread(sample["x"]).astype(np.float32)[None, ...]

        for k, v in sample.items():
            sample[k] = torch.tensor(v)

        if self.transform:
            sample["x"] = self.transform(sample["x"])

        sample = preprocess_chexpert(sample)
        if self.concat_pa:
            sample["pa"] = torch.cat([sample[k] for k in self.columns], dim=0)
        return sample
