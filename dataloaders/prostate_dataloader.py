import torch.utils.data as data
import os
import json
import torch
import numpy as np
from .transformations import get_transform


class ProstateDataset(data.Dataset):
    def __init__(self, data_root, domain_name, phase="train", split_train=True,
                 img_size=(384, 384)):
        self.data_root = data_root
        self.domain_name = domain_name
        self.phase = phase
        self.img_size = img_size
        self.augmenter = get_transform(self.phase, New_size=img_size)

        metadata_path = os.path.join(data_root, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None

        self.slice_dir = os.path.join(data_root, domain_name, "slices")
        split_key = "train" if split_train else "test"
        self.all_data_path = []
        self.name_list = []

        if self.metadata and "splits" in self.metadata:
            case_ids = set(self.metadata["splits"][domain_name][split_key])
            for f_name in sorted(os.listdir(self.slice_dir)):
                if not f_name.endswith(".npz"):
                    continue
                basename = f_name.replace(".npz", "")
                parts = basename.split("_slice_")
                case_name = parts[0].replace("vol_", "", 1)
                if case_name in case_ids:
                    self.all_data_path.append(os.path.join(self.slice_dir, f_name))
                    self.name_list.append(basename)
        else:
            for f_name in sorted(os.listdir(self.slice_dir)):
                if not f_name.endswith(".npz"):
                    continue
                self.all_data_path.append(os.path.join(self.slice_dir, f_name))
                self.name_list.append(f_name.replace(".npz", ""))

    def __len__(self):
        return len(self.all_data_path)

    def __getitem__(self, index):
        raw = np.load(self.all_data_path[index])
        name = self.name_list[index]

        img = raw["img"].astype(np.float32)
        img -= img.min()
        mx = img.max()
        if mx > 0:
            img /= mx
        img = np.stack([img, img, img], axis=-1)  # (H, W, 3)

        seg = raw["label"].astype(np.int64)

        transformed = self.augmenter(image=img, mask=seg)
        img = transformed["image"].to(torch.float32)
        seg = transformed["mask"].to(torch.long)
        return img, seg, name
