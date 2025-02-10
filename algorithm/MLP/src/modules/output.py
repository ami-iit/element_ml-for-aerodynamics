"""
Author: Antonello Paolino
Date: 2025-01-31
Description:    Module for generating the output files.
"""

from pathlib import Path
import numpy as np
import wandb
import torch
import shutil
import pandas

from modules import globals as glvar


def gen_folder(fname):
    folder_path = str(fname)
    if not Path(folder_path).is_dir():
        Path(folder_path).mkdir()
        suff = ""
    else:
        print(folder_path + " dir already existing")
        i = 1
        stop = False
        while not stop:
            suff = "_" + str(i)
            try:
                Path(folder_path + suff).mkdir()
                stop = True
            except:
                i += 1
                stop = False
    print("Generated output folder: ./" + folder_path + suff)
    return folder_path + suff


def save_scaling(scaling_params):
    print("Saving scaling parameters as scaling.npy")
    np.save(glvar.out_dir + "/scaling", scaling_params)
    if glvar.wandb_logging:
        scale_artifact = wandb.Artifact("scaling-parameters", type="parameters")
        scale_artifact.add_file(glvar.out_dir + "/scaling.npy")
        wandb.log_artifact(scale_artifact, aliases=["latest", glvar.run_name])


def save_model(model, optimizer, example, best_model=None):
    print("Saving model as ckp_model.pt")
    checkpoint = {
        "epoch": glvar.epochs,
        "model": model,
        "model_state": model.state_dict(),
        "optimizer": optimizer,
        "optimizer_state": optimizer.state_dict(),
    }
    best_model_path = glvar.out_dir + "/model.pt"
    checkpoint_path = glvar.out_dir + "/ckp_model.pt"
    save_ckp(checkpoint, False, checkpoint_path, best_model_path)

    # save model for inference
    scripted_model = torch.jit.trace(model.eval().to("cpu"), example)
    scripted_model.save(glvar.out_dir + "/scripted_model.pt")
    if best_model is not None:
        best_scripted_model = torch.jit.trace(best_model.eval().to("cpu"), example)
        best_scripted_model.save(glvar.out_dir + "/scripted_model_best.pt")

    # Log the model to wandb
    if glvar.wandb_logging:
        model_artifact = wandb.Artifact("model", type="model")
        model_artifact.add_file(checkpoint_path)
        model_artifact.add_file(glvar.out_dir + "/scripted_model.pt")
        if best_model is not None:
            model_artifact.add_file(glvar.out_dir + "/scripted_model_best.pt")
        wandb.log_artifact(model_artifact, aliases=["latest", glvar.run_name])


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)


def write_hystory(history):
    print("Saving loss history as loss.npy")
    np.save(glvar.out_dir + "/loss", history)


def write_datasets(indices, train_num, val_num):
    print("\nGenerating dataset indices file as indices.xlsx")
    # Create a DataFrame with different lengths
    df = pandas.concat(
        [
            pandas.Series(indices[:train_num], name="Training set"),
            pandas.Series(
                indices[train_num : train_num + val_num],
                name="Validation set",
            ),
            pandas.Series(indices[train_num + val_num :], name="Testing set"),
        ],
        axis=1,
    )
    df.to_excel(glvar.out_dir + "/indices.xlsx", index=False)
    # Log the dataset indices to wandb
    if glvar.wandb_logging:
        table_artifact = wandb.Artifact("dataset-indices", type="run_table")
        table_artifact.add(obj=wandb.Table(dataframe=df), name="df")
        wandb.log_artifact(table_artifact, aliases=["latest", glvar.run_name])
