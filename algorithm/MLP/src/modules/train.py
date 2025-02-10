"""
Author: Antonello Paolino
Date: 2025-01-31
Description:    Module for training the learning algorithm.
"""

import time
import wandb

from modules import globals as glvar


def train_MLP(train_dataloader, val_dataloader, model, loss, optimizer, device):
    outputs = []
    train_loss_avg = []
    min_loss_avg = []
    val_loss_avg = []

    stop_train = False
    epoch = 0

    start_time = time.time()
    min_val_loss = 1e6
    while not stop_train:
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

        outputs.append((epoch, train_loss_avg[-1], val_loss_avg[-1]))
        print(
            f"Epoch {epoch+1}/{glvar.epochs}: Train loss: {train_loss_avg[-1]:5f}, Val loss: {val_loss_avg[-1]:.5f}"
        )

        epoch = epoch + 1
        if epoch >= glvar.epochs:
            stop_train = True

        if glvar.wandb_logging:
            wandb.log(
                {
                    "train_loss": train_loss_avg[-1],
                    "val_loss": val_loss_avg[-1],
                    "best_epoch": best_epoch,
                }
            )

    elapsed_time_seconds = time.time() - start_time
    print("=================================================")
    print("\nTime for training:", "{:.{}f}".format(elapsed_time_seconds / 60, 3), "min")

    return outputs, model, best_model
