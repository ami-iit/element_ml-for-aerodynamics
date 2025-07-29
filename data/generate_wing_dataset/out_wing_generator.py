"""
Author: Antonello Paolino
Date: 2025-03-25
"""

import numpy as np
import torch
from pathlib import Path

from src.wing_flow import FlowGenerator, FlowVisualizer

IM_RES = (64, 64)
DENSITY_EXP = 3
DEVICE = "cpu"

AOA = 0.0
SWEEP = 20.0


def main():
    root = Path(__file__).parents[0]
    # Initialize flow object
    flow = FlowGenerator()
    # Load the dataset
    dataset_file = root / "dataset" / f"wing-images-{IM_RES[0]}-{DENSITY_EXP}.npz"
    dataset = np.load(dataset_file, allow_pickle=True)
    sweeps = dataset["sweep_angles"]
    aoas = dataset["angles_of_attack"]
    images = dataset["database"]
    # Import and set mesh mapping
    map_file = root / "maps" / f"S_0-map-eem-{DENSITY_EXP}.npy"
    map_data = np.load(map_file, allow_pickle=True).item()
    flow.import_mapping(map_data)
    # Initialize interpolator
    flow.compute_interpolator(images.shape[2:])
    # Import the trained models
    model_path = root / "training" / f"{DENSITY_EXP}_v2"
    flow.load_models(
        encoder_path=str(model_path / "scripted_enc.pt"),
        decoder_path=str(model_path / "scripted_dec.pt"),
        device=DEVICE,
    )
    # Load the RBF mapping
    rbf_file = root / "training" / "rbf" / f"rbf_data_{DENSITY_EXP}.npy"
    rbf_data = np.load(rbf_file, allow_pickle=True).item()
    flow.rbf_centers = rbf_data["centers"]
    flow.rbf_weights = rbf_data["weights"]
    flow.rbf_epsilon = rbf_data["epsilon"]

    # Generate a solution from the dataset using the image generator
    index = np.where((aoas == AOA) & (sweeps == SWEEP))[0]
    in_training = True if index.size > 0 else False
    if in_training:
        index = index[0]
        image = images[index]
        flow.interpolate_flow_data(image)
        # Plot the solution
        viz = FlowVisualizer(flow)
        viz.plot_wing_pressure(window_name="training image to 3D geometry")

    # Generate a solution from the ML models using the image generator
    if in_training:
        in_image = torch.tensor(image[np.newaxis, ...]).to(DEVICE)
        latent_space_val = flow.encoder(in_image)
        out_image = flow.decoder(latent_space_val).cpu().detach().numpy()
        flow.interpolate_flow_data(out_image[0])
        # Plot the solution
        viz = FlowVisualizer(flow)
        viz.plot_wing_pressure(window_name="AE predicted image to 3D geometry")

    # Generate a solution from the input AoA and sweep
    latent_space_pred = flow.predict_rbf_tps(np.array([[SWEEP, AOA]]))
    latent_space_pred = torch.tensor(latent_space_pred).to(DEVICE).type(torch.float32)
    out_image = flow.decoder(latent_space_pred).cpu().detach().numpy()
    flow.interpolate_flow_data(out_image[0])
    # Plot the solution
    viz = FlowVisualizer(flow)
    viz.plot_wing_pressure(window_name="RBF-GAE predicted image to 3D geometry")
    viz.plot_wing_geometry(window_name="RBF-GAE predicted geometry")


if __name__ == "__main__":
    main()
