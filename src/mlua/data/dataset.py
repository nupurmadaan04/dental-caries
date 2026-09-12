import random
from pathlib import Path
from typing import List, Union, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

from evaluate.utils import get_data_test_overlap, rgb2gray


class TrainDataset(Dataset):
    """
    Official MLUA TrainDataset implementation for dental caries patch segmentation.
    Loads 384x384 single-channel grayscale patches and binary masks.
    Applies synchronized spatial augmentation (RandomHorizontalFlip, RandomRotation(45))
    and photometric jitter (ColorJitter(brightness=0.5, contrast=0.5)).
    """
    def __init__(
        self,
        image_list: List[Union[str, Path]],
        label_list: List[Union[str, Path]],
        ul_image_list: Optional[List[Union[str, Path]]] = None,
        transize: int = 384,
    ):
        self.transize = transize
        self.data_list = []

        for image_path, label_path in zip(image_list, label_list):
            self.data_list.append([Path(image_path), Path(label_path) if label_path is not None else None])

        if ul_image_list is not None:
            for ul_image_path in ul_image_list:
                self.data_list.append([Path(ul_image_path), None])

        self.img_transform = T.Compose([
            T.ColorJitter(brightness=0.5, contrast=0.5),
        ])
        self.both_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(45),
        ])
        self.resize_transform = T.Resize((self.transize, self.transize))
        self.nomalize_transform = T.ToTensor()

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.data_list[index]
        image = Image.open(str(image_path)).convert("L")

        if label_path is not None and label_path.exists():
            label = Image.open(str(label_path)).convert("L")
        else:
            label = Image.fromarray(np.zeros((self.transize, self.transize), dtype=np.uint8))

        seed = random.randint(0, 10000)
        torch.random.manual_seed(seed)
        image = self.both_transform(image)
        torch.random.manual_seed(seed)
        label = self.both_transform(label)

        image = self.img_transform(image)
        image = self.resize_transform(image)
        label = self.resize_transform(label)

        image = self.nomalize_transform(image)
        label = self.nomalize_transform(label)

        image_tensor = torch.tensor(np.array(image), dtype=torch.float32)
        label_tensor = torch.tensor(np.array(label), dtype=torch.float32)

        return image_tensor, label_tensor


class ValDataset(Dataset):
    """
    Official MLUA ValDataset implementation for full panoramic dental arch validation.
    Extracts 21 overlapping 384x384 patches (stride 192x192) from 768x1536 panoramic crops.
    """
    def __init__(
        self,
        img_path_list: List[Union[str, Path]],
        gt_path_list: List[Union[str, Path]],
    ):
        self.img_path_list = [Path(p) for p in img_path_list]
        self.gt_path_list = [Path(p) for p in gt_path_list]
        self.resize_transform = T.Resize((384, 384))
        self.nomalize_transform = T.ToTensor()

    def __len__(self) -> int:
        return len(self.img_path_list)

    def normalize(self, inputs: np.ndarray) -> np.ndarray:
        return (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-8)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = str(self.img_path_list[index])
        gt_path = str(self.gt_path_list[index])

        imgs_patch, _, _, gt = get_data_test_overlap(img_path, gt_path, 384, 384, 192, 192)
        assert imgs_patch.shape[0] == 21, f"Expected 21 patches, got {imgs_patch.shape[0]}"
        imgs_patch = rgb2gray(imgs_patch)

        final_img = torch.zeros(21, 384, 384, dtype=torch.float32)
        for i in range(imgs_patch.shape[0]):
            image = Image.fromarray(np.uint8(imgs_patch[i].squeeze()))
            image = self.resize_transform(image)
            image = self.nomalize_transform(image)
            final_img[i] = torch.tensor(np.array(image).squeeze(), dtype=torch.float32)

        gt = self.normalize(gt)
        final_gt = torch.tensor(gt.squeeze(), dtype=torch.float32)

        return final_img, final_gt
