import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from utils import Paths


class ToBinaryMask(object):
    def __call__(self, img):
        img = img.convert("L")  # Convert to grayscale
        img = torch.tensor(np.array(img), dtype = torch.float32)
        img = (img > 1).float()
        return img

class BiomedImages(Dataset):
    def __init__(self, image_path, mask_path, augment = True, crop = True):
        self.image_path = image_path
        self.mask_path = mask_path
        self.augment = augment
        self.crop = crop
        self.image_transform = data_augmentation() if augment else no_augmentation()
        self.mask_transform = data_augmentation_mask() if augment else no_augmentation()
        self.image_files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
        self.mask_files = [f for f in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path, f))]

        self.image_ids = [f.split('_')[0] for f in self.image_files]
        self.mask_ids = [f.split('_')[1].split('.')[0] for f in self.mask_files]
        self.matched_files = self._match_images_and_masks()

    def _match_images_and_masks(self):
        matched = []
        for img_file, img_id in zip(self.image_files, self.image_ids):
            for mask_file, mask_id in zip(self.mask_files, self.mask_ids):
                if img_id == mask_id:
                    matched.append((img_file, mask_file))
        return matched

    def __len__(self):
        return len(self.matched_files)

    def __getitem__(self, idx):
        img_file, mask_file = self.matched_files[idx]

        image = Image.open(os.path.join(self.image_path, img_file))
        mask = Image.open(os.path.join(self.mask_path, mask_file))

        if self.crop:
            image = crop_image(image)
            mask = crop_image(mask)

        if self.augment:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.image_transform(image)
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)
        else:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)

        return image, mask


def data_augmentation():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(degrees = (-15, 15)),
        transforms.RandomAffine(degrees = 0, translate = (0.1, 0.1), scale = (0.9, 1.1), shear = (-5, 5)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    return transform


def data_augmentation_mask():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(degrees = (-15, 15)),
        transforms.RandomAffine(degrees = 0, translate = (0.1, 0.1), scale = (0.9, 1.1), shear = (-5, 5)),
        transforms.RandomHorizontalFlip(),
        ToBinaryMask()
    ])
    return transform


def no_augmentation():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform


def crop_image(image, top_left = (166, 54), top_right = (892, 54), bottom_left = (166, 780), bottom_right = (892, 780)):
    left = top_left[0]
    upper = top_left[1]
    right = top_right[0]
    lower = bottom_left[1]
    return image.crop((left, upper, right, lower))


def dataloaders(batch_size):
    train_dataset = BiomedImages(image_path = Paths.train_image_path, mask_path = Paths.mask_path, augment = True, crop = True)
    val_dataset = BiomedImages(image_path = Paths.val_image_path, mask_path = Paths.mask_path, augment = False, crop = True)
    test_dataset = BiomedImages(image_path = Paths.test_image_path, mask_path = Paths.mask_path, augment = False, crop = True)

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return train_dataloader, val_dataloader, test_dataloader
