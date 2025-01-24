import pandas as pd
import os
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import copy
import time
from models import UNet, UNetPP, AttUNet
from utils import dice_loss, tversky_loss, dice_score_by_epoch, iou_score_by_epoch
from preprocessing import dataloaders


def train_model(chosen_type, chosen_criterion, chosen_optimizer, num_epochs, batch_size=4, learning_rate=0.01):
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # using the GPU (or CPU if not available)

    TRAIN_DATALOADER, VAL_DATALOADER, TEST_DATALOADER = dataloaders(batch_size=batch_size)

    metrics_df = pd.DataFrame(
        columns=["epoch", "delta_time_seconds", "cum_time_seconds", "train_loss", "val_loss", "train_dice", "val_dice"])

    chosen_criterion = chosen_criterion.lower()
    chosen_optimizer = chosen_optimizer.lower()

    if chosen_type == "unet":
        model = UNet(in_channels=3, out_channels=1).to(DEVICE)
        if chosen_criterion == "bce":
            criterion = nn.BCELoss()
            if chosen_optimizer == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                model_save_path = "../sangiovanni_project/Models/unet_BCE_SGD.pth"
                dataframe_path = "../sangiovanni_project/Dataframes/unet_BCE_SGD.csv"
            if chosen_optimizer == "adamw":
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
                model_save_path = "../sangiovanni_project/Models/unet_BCE_ADAMW.pth"
                dataframe_path = "../sangiovanni_project/Dataframes/unet_BCE_ADAMW.csv"
        if chosen_criterion == "dice":
            criterion = dice_loss
            if chosen_optimizer == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                model_save_path = "../sangiovanni_project/Models/unet_Dice_SGD.pth"
                dataframe_path = "../sangiovanni_project/Dataframes/unet_Dice_SGD.csv"
            if chosen_optimizer == "adamw":
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
                model_save_path = "../sangiovanni_project/Models/unet_Dice_ADAMW.pth"
                dataframe_path = "../sangiovanni_project/Dataframes/unet_Dice_ADAMW.csv"
        if chosen_criterion == "tversky":
            criterion = tversky_loss
            if chosen_optimizer == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                model_save_path = "../sangiovanni_project/Models/unet_Tversky_SGD.pth"
                dataframe_path = "../sangiovanni_project/Dataframes/unet_Tversky_SGD.csv"
            if chosen_optimizer == "adamw":
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
                model_save_path = "../sangiovanni_project/Models/unet_Tversky_ADAMW.pth"
                dataframe_path = "../sangiovanni_project/Dataframes/unet_Tversky_ADAMW.csv"

    elif chosen_type == "unetpp":
        model = UNetPP(in_channels=3, out_channels=1).to(DEVICE)
        if chosen_criterion == "bce":
            criterion = nn.BCELoss()
            if chosen_optimizer == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                model_save_path = "../sangiovanni_project/Models/unetpp_BCE_SGD.pth"
                dataframe_path = "../sangiovanni_project/Dataframes/unetpp_BCE_SGD.csv"
            if chosen_optimizer == "adamw":
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
                model_save_path = "../sangiovanni_project/Models/unetpp_BCE_ADAMW.pth"
                dataframe_path = "../sangiovanni_project/Dataframes/unetpp_BCE_ADAMW.csv"
        elif chosen_criterion == "dice":
            criterion = dice_loss
            if chosen_optimizer == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                model_save_path = "../sangiovanni_project/Models/unetpp_Dice_SGD.pth"
                dataframe_path = "../sangiovanni_project/Dataframes/unetpp_Dice_SGD.csv"
            if chosen_optimizer == "adamw":
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
                model_save_path = "../sangiovanni_project/Models/unetpp_Dice_ADAMW.pth"
                dataframe_path = "../sangiovanni_project/Dataframes/unetpp_Dice_ADAMW.csv"
        elif chosen_criterion == "tversky":
            criterion = tversky_loss
            if chosen_optimizer == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                model_save_path = "../sangiovanni_project/Models/unetpp_Tversky_SGD.pth"
                dataframe_path = "../sangiovanni_project/Dataframes/unetpp_Tversky_SGD.csv"
            if chosen_optimizer == "adamw":
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
                model_save_path = "../sangiovanni_project/Models/unetpp_Tversky_ADAMW.pth"
                dataframe_path = "../sangiovanni_project/Dataframes/unetpp_Tversky_ADAMW.csv"

    elif chosen_type == "attunet":
        model = AttUNet(in_channels=3, out_channels=1).to(DEVICE)
        if chosen_criterion == "bce":
            criterion = nn.BCELoss()
            if chosen_optimizer == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                model_save_path = "../sangiovanni_project/Models/attunet_BCE_SGD.pth"
                dataframe_path = "../sangiovanni_project/Dataframes/attunet_BCE_SGD.csv"
            if chosen_optimizer == "adamw":
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
                model_save_path = "../sangiovanni_project/Models/attunet_BCE_ADAMW.pth"
                dataframe_path = "../sangiovanni_project/Dataframes/attunet_BCE_ADAMW.csv"
        elif chosen_criterion == "dice":
            criterion = dice_loss
            if chosen_optimizer == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                model_save_path = "../sangiovanni_project/Models/attunet_Dice_SGD.pth"
                dataframe_path = "../sangiovanni_project/Dataframes/attunet_Dice_SGD.csv"
            if chosen_optimizer == "adamw":
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
                model_save_path = "../sangiovanni_project/Models/attunet_Dice_ADAMW.pth"
                dataframe_path = "../sangiovanni_project/Dataframes/attunet_Dice_ADAMW.csv"
        elif chosen_criterion == "tversky":
            criterion = tversky_loss
            if chosen_optimizer == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                model_save_path = "../sangiovanni_project/Models/attunet_Tversky_SGD.pth"
                dataframe_path = "../sangiovanni_project/Dataframes/attunet_Tversky_SGD.csv"
            if chosen_optimizer == "adamw":
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
                model_save_path = "../sangiovanni_project/Models/attunet_Tversky_ADAMW.pth"
                dataframe_path = "../sangiovanni_project/Dataframes/attunet_Tversky_ADAMW.csv"
    else:
        return ("Choose an adequate model type, loss function and optimizer")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                                    steps_per_epoch=len(TRAIN_DATALOADER), epochs=num_epochs)

    best_loss = float("inf")
    best_dice = 0.0
    best_iou = 0.0
    best_model_weights = None
    patience = 15

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):

        start_time = time.time()

        model.train()  # beginning of training loop

        train_running_loss = 0
        train_dice = 0.0
        train_iou = 0.0
        for img_mask in TRAIN_DATALOADER:
            image = img_mask[0].float().to(DEVICE)
            mask = img_mask[1].float().to(DEVICE)
            mask = mask.unsqueeze(1)  # Add channel dimension (at the second position)

            y_pred = model(image)
            y_pred = torch.sigmoid(y_pred)
            loss = criterion(y_pred, mask)

            train_running_loss += loss.item()
            train_dice += dice_score_by_epoch(y_pred, mask)
            train_iou += iou_score_by_epoch(y_pred, mask)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss = train_running_loss / (len(TRAIN_DATALOADER))
        train_dice /= len(TRAIN_DATALOADER)
        train_iou /= len(TRAIN_DATALOADER)

        model.eval()  # beginning of validation loop

        val_running_loss = 0
        val_dice = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for img_mask in VAL_DATALOADER:
                image = img_mask[0].float().to(DEVICE)
                mask = img_mask[1].float().to(DEVICE)

                y_pred = model(image)
                y_pred = torch.sigmoid(y_pred)
                loss = criterion(y_pred, mask)
                val_running_loss += loss.item()
                val_dice += dice_score_by_epoch(y_pred, mask)
                val_iou += iou_score_by_epoch(y_pred, mask)

        val_loss = val_running_loss / (len(VAL_DATALOADER))
        val_dice /= len(VAL_DATALOADER)
        val_iou /= len(VAL_DATALOADER)

        if val_dice > best_dice:
            best_dice = val_dice
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())  # Deep copy here
            patience = 15  # Reset patience counter
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping due to no improvement.")
                break

        delta_time_seconds = time.time() - start_time
        cum_time_seconds = metrics_df["delta_time_seconds"].sum() + delta_time_seconds

        updated_metrics_df = pd.DataFrame({
            "epoch": [epoch + 1],
            "delta_time_seconds": [delta_time_seconds],
            "cum_time_seconds": [cum_time_seconds],
            "train_loss": [train_loss],
            "val_loss": [val_loss],
            "train_dice": [train_dice],
            "val_dice": [val_dice]
        })

        metrics_df = pd.concat([metrics_df, updated_metrics_df], ignore_index=True)
        metrics_df["epoch"] = metrics_df["epoch"].astype(int)
        metrics_df["delta_time_seconds"] = metrics_df["delta_time_seconds"].round(2)
        metrics_df["cum_time_seconds"] = metrics_df["cum_time_seconds"].round(2)
        metrics_df["train_loss"] = metrics_df["train_loss"].round(4)
        metrics_df["val_loss"] = metrics_df["val_loss"].round(4)
        metrics_df["train_dice"] = metrics_df["train_dice"].round(4)
        metrics_df["val_dice"] = metrics_df["val_dice"].round(4)

        print("-" * 30)
        print(
            f"Train Loss EPOCH {epoch + 1}: {train_loss:.4f} | Train Dice Score: {train_dice:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Valid Loss EPOCH {epoch + 1}: {val_loss:.4f} | Val Dice Score: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        print("-" * 30)

        torch.cuda.empty_cache()

    model.load_state_dict(best_model_weights)  # load the best model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # if the path doesn't exist, we create it
    torch.save(model.state_dict(), model_save_path)
    print("Best model saved with:\n")
    print(f"Validation loss: {best_loss:.4f}")
    print(f"Dice score: {best_dice:.4f}")

    os.makedirs(os.path.dirname(dataframe_path), exist_ok=True)
    metrics_df.to_csv(dataframe_path, index=False)  # create dataframe for the plots
    print(f"\nBy-epoch metrics saved to {dataframe_path}")
