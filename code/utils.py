import os
import re
import random
import matplotlib.pyplot as plt
import torch


class Paths:
    images_dir = "../datasets/segmentation_final/SQR_YOLO/Roberta/"

    train_image_path = "../datasets/segmentation_final/SQR_YOLO/Roberta/fold_1/train/images/"
    val_image_path = "../datasets/segmentation_final/SQR_YOLO/Roberta/fold_1/val/images/"
    test_image_path = "../datasets/segmentation_final/SQR_YOLO/Roberta/fold_1/test/images/"
    mask_path = "../datasets/segmentation_final/masks/Roberta/"

    predicted_masks_path_unet = "../sangiovanni_project/PredictedMasks/UNet/"
    predicted_masks_path_unetpp = "../sangiovanni_project/PredictedMasks/UNet++/"
    predicted_masks_path_attunet = "../sangiovanni_project/PredictedMasks/AttUNet/"
    predicted_masks_path_r2att = "../sangiovanni_project/PredictedMasks/R2AttUNet/"

    unet_bce_sgd = "Models/unet_BCE_SGD.pth"
    unet_bce_adamw = "Models/unet_BCE_ADAMW.pth"
    unet_dice_sgd = "Models/unet_Dice_SGD.pth"
    unet_dice_adamw = "Models/unet_Dice_ADAMW.pth"
    unet_tversky_adamw = "Models/unet_Tversky_ADAMW.pth"
    unet_tversky_sgd = "Models/unet_Tversky_SGD.pth"
    unetpp_bce_sgd = "Models/unetpp_BCE_SGD.pth"
    unetpp_bce_adamw = "Models/unetpp_BCE_ADAMW.pth"
    unetpp_dice_sgd = "Models/unetpp_Dice_SGD.pth"
    unetpp_dice_adamw = "Models/unetpp_Dice_ADAMW.pth"
    unetpp_tversky_sgd = "Models/unetpp_Tversky_SGD.pth"
    unetpp_tversky_adamw = "Models/unetpp_Tversky_ADAMW.pth"
    attunet_bce_sgd = "Models/attunet_BCE_SGD.pth"
    attunet_bce_adamw = "Models/attunet_BCE_ADAMW.pth"
    attunet_dice_sgd = "Models/attunet_Dice_SGD.pth"
    attunet_dice_adamw = "Models/attunet_Dice_ADAMW.pth"
    attunet_tversky_sgd = "Models/attunet_Tversky_SGD.pth"
    attunet_tversky_adamw = "Models/attunet_Tversky_ADAMW.pth"

    def __init__(self):
        self.train_ids = self.images_list(self.train_image_path)  # List of training image IDs
        self.val_ids = self.images_list(self.val_image_path)  # List of validation image IDs
        self.test_ids = self.images_list(self.test_image_path)  # List of testing image IDs
        self.mask_ids = self.masks_list(self.mask_path)  # List of all mask IDs

        self.train_ids = [image_id for image_id in self.train_ids if image_id in self.mask_ids]
        self.val_ids = [image_id for image_id in self.val_ids if image_id in self.mask_ids]
        self.test_ids = [image_id for image_id in self.test_ids if image_id in self.mask_ids]

    @staticmethod
    def images_list(filetype_path):  # filetype_path = train_image_path, val_image_path, test_image_path, mask_path
        image_files = os.listdir(filetype_path)
        image_files = [f for f in image_files if f.endswith(('.png', '.jpg', '.jpeg'))]
        image_ids = [re.match(r'^\d+', f).group() for f in image_files if re.match(r'^\d+', f)]
        return image_ids

    @staticmethod
    def masks_list(mask_path):
        mask_files = os.listdir(mask_path)
        mask_files = [f for f in mask_files if f.endswith(('.png', '.jpg', '.jpeg'))]
        mask_ids = [re.search(r'_(\d+)', f).group(1) for f in mask_files if re.search(r'_(\d+)', f)]
        return mask_ids


def show_random_picture(filetype, specific_id=None, random_choice=False):
    from preprocessing import BiomedImages, crop_image  # imported here in order to avoid circular import error

    paths = Paths()
    filetype = filetype.lower()
    if filetype == "validation":
        chosen_path = paths.val_image_path
        chosen_ids = paths.val_ids
    elif filetype == "training":
        chosen_path = paths.train_image_path
        chosen_ids = paths.train_ids
    elif filetype == "testing":
        chosen_path = paths.test_image_path
        chosen_ids = paths.test_ids
    else:
        return ("You must choose between training, validation or testing")

    if random_choice:
        specific_id = random.choice(chosen_ids)

    mask_id = f"roberta.gualtierotti@unimi.it_{specific_id}.jpg"

    chosen_dataset = BiomedImages(image_path=chosen_path, mask_path=paths.mask_path, augment=False, crop=True)  ####

    chosen_id_index = chosen_ids.index(specific_id)
    image, mask = chosen_dataset[chosen_id_index]

    if isinstance(image, torch.Tensor) and image.ndimension() == 3 and image.shape[0] == 3:
        image = image.numpy().transpose(1, 2, 0)
    if isinstance(mask, torch.Tensor) and mask.ndimension() == 3:
        mask = mask.squeeze(0).numpy()  # remove first dimension of mask, so that it can be displayed

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Image - ID: {specific_id}")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title(f"Mask - ID: {specific_id}")
    plt.axis('off')
    plt.show()


# Loss functions

def dice_loss(y_pred, target, smooth=1e-6):
    intersection = (y_pred * target).sum(dim=(1, 2, 3))
    union = y_pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-6):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    TP = torch.sum(y_true * y_pred)
    FP = torch.sum((1 - y_true) * y_pred)
    FN = torch.sum(y_true * (1 - y_pred))
    tversky_index = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return 1 - tversky_index


def dice_score_by_epoch(y_pred, target, smooth=1e-6):
    try:
        y_pred = (y_pred > 0.5).float()
        intersection = (y_pred * target).sum(dim=(1, 2, 3))
        union = y_pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.mean().item()
    except ZeroDivisionError:
        return 1.0


def iou_score_by_epoch(y_pred, target, smooth=1e-6):
    y_pred = (y_pred > 0.5).float()
    intersection = (y_pred * target).sum(dim=(1, 2, 3))
    union = y_pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


# Metrics

def iou_metric(y_pred, target):
    y_pred = (y_pred > 0).float()
    target = (target > 0).float()
    intersection = (y_pred * target).sum()
    union = y_pred.sum() + target.sum() - intersection
    try:
        iou = intersection / union
    except ZeroDivisionError:
        iou = torch.tensor(1.0)
    return iou


def dice_metric(y_pred, target):
    try:
        y_pred = (y_pred > 0).bool()
        target = (target > 0).bool()
        intersection = (y_pred & target).sum().item()
        return 2 * intersection / (y_pred.sum().item() + target.sum().item())
    except ZeroDivisionError:
        return 1.0


def precision_metric(y_pred, target):
    y_pred = y_pred > 0
    target = target > 0
    tp = torch.count_nonzero(y_pred & target).item()
    fp = torch.count_nonzero(y_pred & ~target).item()
    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0
    return precision


def pixel_accuracy(y_pred, target):
    y_pred = y_pred.view(-1)
    target = target.view(-1)
    correct_pixels = (y_pred == target).sum().item()
    total_pixels = y_pred.numel()
    accuracy = correct_pixels / total_pixels
    return accuracy
