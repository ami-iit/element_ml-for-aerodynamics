"""
Author: Antonello Paolino
Date: 2025-01-31
Description:    Module for training the learning algorithm.
"""

import time
import wandb
import torch.optim as optim

from modules.constants import Const


def train_MLP(train_dataloader, val_dataloader, model, loss, optimizer, device):
    outputs = []
    train_loss_avg = []
    min_loss_avg = []
    val_loss_avg = []
    lr_history = []

    stop_train = False
    epoch = 0

    start_time = time.time()
    min_val_loss = 1e6

    if Const.lr_scheduler == "linear":
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=1e-3, total_iters=Const.lr_iters
        )
    elif Const.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=Const.lr_patience
        )

    print("\nStarting the training \n")
    while not stop_train:
        time_start = time.time()
        model.train()

        train_loss_avg.append(0)
        min_loss_avg.append(0)
        num_batches_train = 0
        for features_batch, target_batch in train_dataloader:

            features_batch = features_batch.to(device)
            target_batch = target_batch.to(device)

            # Compute forward pass
            pred = model(features_batch)

            # Compute loss
            train_loss = loss(pred, target_batch)

            # Backward step
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Update loss
            train_loss_avg[-1] += train_loss.item()
            num_batches_train += 1

        model.eval()
        val_loss_avg.append(0)
        num_batches_val = 0
        for features_batch, target_batch in val_dataloader:
            features_batch = features_batch.to(device)
            target_batch = target_batch.to(device)
            pred = model(features_batch)
            val_loss = loss(pred, target_batch)
            val_loss_avg[-1] += val_loss.item()
            num_batches_val += 1

        train_loss_avg[-1] /= num_batches_train
        val_loss_avg[-1] /= num_batches_val

        if val_loss_avg[-1] < min_val_loss:
            best_model = model
            best_epoch = epoch
            min_val_loss = val_loss_avg[-1]

        # Update learning rate
        lr_history.append(scheduler.get_last_lr()[0])
        if Const.lr_scheduler == "linear":
            if epoch < Const.lr_iters:
                scheduler.step()
        elif Const.lr_scheduler == "plateau":
            scheduler.step(val_loss_avg[-1])

        time_end = time.time()
        outputs.append((epoch, train_loss_avg[-1], val_loss_avg[-1], lr_history[-1]))
        print(
            f"Epoch {epoch+1}/{Const.epochs}: Train loss: {train_loss_avg[-1]:5f}, Val loss: {val_loss_avg[-1]:.5f}, lr: {lr_history[-1]:.3e}, iter time: {time_end - time_start:.2f} s"
        )

        epoch = epoch + 1
        if epoch >= Const.epochs:
            stop_train = True

        if Const.wandb_logging:
            opt_str = f"_{Const.optuna_trial}" if Const.optuna_trial >= 0 else ""
            wandb.log(
                {
                    ("iteration" + opt_str): epoch,
                    ("train_loss" + opt_str): train_loss_avg[-1],
                    ("val_loss" + opt_str): val_loss_avg[-1],
                    ("best_epoch" + opt_str): best_epoch,
                    ("learning_rate" + opt_str): lr_history[-1],
                }
            )

    elapsed_time_seconds = time.time() - start_time
    print("=================================================")
    print("\nTime for training:", "{:.{}f}".format(elapsed_time_seconds / 60, 3), "min")

    return outputs, model, best_model
