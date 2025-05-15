"""
Author: Antonello Paolino
Date: 2025-03-25
"""

import numpy as np
import torch
from pathlib import Path

from src.wing_flow import FlowGenerator, FlowVisualizer

IM_RES = (64, 64)
DENSITY_EXP = 4


def main():
    root = Path(__file__).parents[0]
    out_dir = root / "training" / "rbf"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Initialize flow object
    flow = FlowGenerator()
    # Load the dataset
    dataset_file = root / "dataset" / f"wing-images-{IM_RES[0]}-{DENSITY_EXP}.npz"
    dataset = np.load(dataset_file, allow_pickle=True)
    sweeps = dataset["sweep_angles"]
    aoas = dataset["angles_of_attack"]
    images = dataset["database"]
    # Import the trained models
    model_path = root / "training" / f"{DENSITY_EXP}"
    flow.load_models(
        encoder_path=str(model_path / "scripted_enc.pt"),
        decoder_path=str(model_path / "scripted_dec.pt"),
    )

    # Generate the latent space from the encoder
    in_image = torch.tensor(images).to("cpu")
    latent_space = flow.encoder(in_image).cpu().detach().numpy()

    # Plots
    viz = FlowVisualizer(flow)
    # Plot the latent space 2D projections
    input = {
        "sweep_angles": sweeps,
        "angles_of_attack": aoas,
    }
    viz.plot_2D_latent_space_projections(input, latent_space)

    # Plot 3D latent space
    if latent_space.shape[1] == 3:
        viz.plot_3D_latent_space(input, latent_space)

    # Plot latent space projected on the input space
    # Train RBF using TPS
    X = np.array([sweeps, aoas]).T
    flow.train_rbf_tps(X, latent_space)
    # Predict using TPS
    x_pred = np.linspace(sweeps.min(), sweeps.max(), 300)
    y_pred = np.linspace(aoas.min(), aoas.max(), 300)
    X_new = np.array(np.meshgrid(x_pred, y_pred)).T.reshape(-1, 2)
    latent_space_pred = flow.predict_rbf_tps(X_new)
    X_image = np.meshgrid(x_pred, y_pred)
    viz.plot_rbf_result(input, X_image, latent_space_pred)

    rbf_data = {
        "centers": flow.rbf_centers,
        "weights": flow.rbf_weights,
        "epsilon": flow.rbf_epsilon,
    }
    np.save(out_dir / f"rbf_data_{DENSITY_EXP}.npy", rbf_data)
    print("RBF mapping saved")


if __name__ == "__main__":
    main()
