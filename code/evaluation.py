import torch
import matplotlib.pyplot as plt
from utils import Paths, iou_metric, dice_metric, pixel_accuracy, precision_metric
from models import UNet, UNetPP, AttUNet
from preprocessing import dataloaders


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(chosen_model_path):
    if "unetpp_" in chosen_model_path:
        model = UNetPP(in_channels=3, out_channels=1).to(DEVICE)
    elif "attunet_" in chosen_model_path:
        model = AttUNet(in_channels=3, out_channels=1).to(DEVICE)
    elif "unet_" in chosen_model_path:
        model = UNet(in_channels=3, out_channels=1).to(DEVICE)

    model.load_state_dict(torch.load(chosen_model_path))
    model.eval()
    return model, chosen_model_path


def evaluate_model(model, chosen_model_path, chosen_dataloader, predicted_masks_path):
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    precision_scores = []

    for idx, (images, ground_truth_masks) in enumerate(chosen_dataloader):
        images = images.float().to(DEVICE)
        ground_truth_masks = ground_truth_masks.float().to(DEVICE).unsqueeze(
            1)  # add 1 dimension (channel of image, black&white is 1)

        with torch.no_grad():
            predictions = model(images)
            predictions = torch.sigmoid(predictions)

            # visualize_predictions(predictions)

            # print(f"Predictions shape: {predictions.shape}")
            # print(f"Predictions (first batch): {predictions[0].cpu().numpy()}")

            predicted_masks = (predictions > 0.5).float()

            # print(f"Predicted Masks (first batch): {predicted_masks[0].cpu().numpy()}")

            iou = iou_metric(predicted_masks, ground_truth_masks)
            dice = dice_metric(predicted_masks, ground_truth_masks)
            precision = precision_metric(predicted_masks, ground_truth_masks)
            accuracy = pixel_accuracy(predicted_masks, ground_truth_masks)

            image_id = chosen_dataloader.dataset.matched_files[idx][0].split('_')[0]
            print(
                f"Image ID: {image_id} ---> IoU: {iou:.4f} - Dice: {dice:.4f} - Precision: {precision:.4f} - Pixel Acc: {accuracy:.4f}\n")

            iou_scores.append((image_id, iou))
            dice_scores.append((image_id, dice))
            pixel_accuracies.append((image_id, accuracy))
            precision_scores.append((image_id, precision))

        images_np = images.cpu().numpy().squeeze().transpose(1, 2,
                                                             0)  # removing all values of 1, and putting it in (height, width, channels) for visualization
        ground_truth_np = ground_truth_masks.cpu().numpy().squeeze()  # removing all 1 so that it can be visualized
        predicted_masks_np = predicted_masks.cpu().numpy().squeeze()  # removing all 1 so that it can be visualized

        # print(f"Ground Truth Mask shape: {ground_truth_np.shape}")
        # print(f"Predicted Mask shape: {predicted_masks_np.shape}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(images_np)
        axes[0].set_title("Input Image")
        axes[0].axis("off")
        axes[1].imshow(ground_truth_np, cmap="gray")
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")
        axes[2].imshow(predicted_masks_np, cmap="gray")
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")

        plt.savefig(f"{predicted_masks_path}/predicted_mask_{image_id}.png")
        plt.close()

    average_iou = sum(iou for _, iou in iou_scores) / len(iou_scores)
    average_dice = sum(dice for _, dice in dice_scores) / len(dice_scores)
    average_precision = sum(precision for _, precision in precision_scores) / len(precision_scores)
    average_pixel_acc = sum(accuracy for _, accuracy in pixel_accuracies) / len(pixel_accuracies)

    print("-" * 30)
    print(f"Average IoU: {average_iou:.4f}")
    print(f"Average Dice Score: {average_dice:.4f}")
    print(f"Average Precision: {average_precision:.4f}")
    print(f"Average Pixel Accuracy: {average_pixel_acc:.4f}")
    print("-" * 30)

    if chosen_model_path == Paths.unet_dice_sgd:
        metrics_filename = "unet_dice_sgd.txt"
    elif chosen_model_path == Paths.unet_dice_adamw:
        metrics_filename = "unet_dice_adamw.txt"
    elif chosen_model_path == Paths.unet_bce_sgd:
        metrics_filename = "unet_bce_sgd.txt"
    elif chosen_model_path == Paths.unet_bce_adamw:
        metrics_filename = "unet_bce_adamw.txt"
    elif chosen_model_path == Paths.unet_tversky_adamw:
        metrics_filename = "unet_tversky_adamw.txt"
    elif chosen_model_path == Paths.unet_tversky_sgd:
        metrics_filename = "unet_tversky_sgd.txt"
    elif chosen_model_path == Paths.unetpp_dice_sgd:
        metrics_filename = "unetpp_dice_sgd.txt"
    elif chosen_model_path == Paths.unetpp_dice_adamw:
        metrics_filename = "unetpp_dice_adamw.txt"
    elif chosen_model_path == Paths.unetpp_bce_sgd:
        metrics_filename = "unetpp_bce_sgd.txt"
    elif chosen_model_path == Paths.unetpp_bce_adamw:
        metrics_filename = "unetpp_bce_adamw.txt"
    elif chosen_model_path == Paths.unetpp_tversky_sgd:
        metrics_filename = "unetpp_tversky_sgd.txt"
    elif chosen_model_path == Paths.unetpp_tversky_adamw:
        metrics_filename = "unetpp_tversky_adamw.txt"
    elif chosen_model_path == Paths.attunet_bce_sgd:
        metrics_filename = "attunet_bce_sgd.txt"
    elif chosen_model_path == Paths.attunet_bce_adamw:
        metrics_filename = "attunet_bce_adamw.txt"
    elif chosen_model_path == Paths.attunet_dice_sgd:
        metrics_filename = "attunet_dice_sgd.txt"
    elif chosen_model_path == Paths.attunet_dice_adamw:
        metrics_filename = "attunet_dice_adamw.txt.txt"
    elif chosen_model_path == Paths.attunet_tversky_sgd:
        metrics_filename = "attunet_tversky_sgd.txt"
    elif chosen_model_path == Paths.attunet_tversky_adamw:
        metrics_filename = "attunet_tversky_adamw.txt"

    with open(f"../image-segmentation-unet/Metrics/{metrics_filename}", "w") as f:
        f.write(f"Average IoU: {average_iou:.4f}\n")
        f.write(f"Average Dice Score: {average_dice:.4f}\n")
        f.write(f"Average Precision: {average_precision:.4f}\n")
        f.write(f"Average Pixel Acc: {average_pixel_acc:.4f}\n")
        f.write("\nMetrics by image:\n")
        for (image_id, iou), (_, dice), (_, precision), (_, accuracy) in zip(iou_scores, dice_scores, precision_scores,
                                                                             pixel_accuracies):
            f.write(
                f"Image ID: {image_id} ---> IoU: {iou:.4f} - Dice: {dice:.4f} - Precision: {precision:.4f} - Pixel Acc: {accuracy:.4f}\n")


def run_evaluation(chosen_set, chosen_model_path):
    if chosen_set == "validation":
        _, val_dataloader, _ = dataloaders(batch_size=1)
        chosen_dataloader = val_dataloader
    elif chosen_set == "test":
        _, _, test_dataloader = dataloaders(batch_size=1)
        chosen_dataloader = test_dataloader
    if "unetpp_" in chosen_model_path:
        predicted_masks_path = Paths.predicted_masks_path_unetpp
    elif "attunet_" in chosen_model_path:
        predicted_masks_path = Paths.predicted_masks_path_attunet
    elif "unet_" in chosen_model_path:
        predicted_masks_path = Paths.predicted_masks_path_unet
    model, _ = load_model(chosen_model_path)
    evaluate_model(model, chosen_model_path, chosen_dataloader, predicted_masks_path)


def visualize_predictions(predictions):
    raw_predictions = predictions.squeeze().cpu().numpy()  # removing all 1 so that it can be visualized

    plt.imshow(raw_predictions, cmap="hot")
    plt.colorbar()
    plt.title("Raw Predictions Heatmap")
    plt.axis('off')
    plt.show()
