"""
Author: Antonello Paolino
Date: 2025-03-25
"""

import numpy as np
import torch
from pathlib import Path

from src.wing_flow import FlowGenerator, FlowVisualizer

AOA = 12.5
SWEEP = 15.0


def main():
    root = Path(__file__).parents[0]
    # Initialize flow object
    flow = FlowGenerator()
    # Load the dataset
    dataset_file = root / "dataset" / "wing-images-256.npz"
    dataset = np.load(dataset_file, allow_pickle=True)
    sweeps = dataset["sweep_angles"]
    aoas = dataset["angles_of_attack"]
    images = dataset["database"]
    # Import and set mesh mapping
    map_file = root / "maps" / "S_10-map.npy"
    map_data = np.load(map_file, allow_pickle=True).item()
    flow.import_mapping(map_data)
    # Initialize interpolator
    flow.compute_interpolator(images.shape[2:])
    # Import the trained models
    model_path = root / "training" / "Out"
    flow.load_models(
        encoder_path=str(model_path / "scripted_enc.pt"),
        decoder_path=str(model_path / "scripted_dec.pt"),
    )
    # Load the RBF mapping
    rbf_file = root / "training" / "rbf" / "rbf_data.npy"
    rbf_data = np.load(rbf_file, allow_pickle=True).item()
    flow.rbf_centers = rbf_data["centers"]
    flow.rbf_weights = rbf_data["weights"]
    flow.rbf_epsilon = rbf_data["epsilon"]

    # Generate a solution from the dataset using the image generator
    index = np.where((aoas == AOA) & (sweeps == SWEEP))[0][0]
    image = images[index]
    flow.interpolate_flow_data(image)
    # Plot the solution
    viz = FlowVisualizer(flow)
    viz.plot_wing_pressure()

    # Generate a solution from the ML models using the image generator
    in_image = torch.tensor(image[np.newaxis, ...]).to("cpu")
    latent_space_val = flow.encoder(in_image)
    out_image = flow.decoder(latent_space_val).cpu().detach().numpy()
    flow.interpolate_flow_data(out_image[0])
    # Plot the solution
    viz = FlowVisualizer(flow)
    viz.plot_wing_pressure()

    # Generate a solution from the input AoA and sweep
    latent_space_pred = flow.predict_rbf_tps(np.array([[SWEEP, AOA]]))
    latent_space_pred = torch.tensor(latent_space_pred).to("cpu").type(torch.float32)
    out_image = flow.decoder(latent_space_pred).cpu().detach().numpy()
    flow.interpolate_flow_data(out_image[0])
    # Plot the solution
    viz = FlowVisualizer(flow)
    viz.plot_wing_pressure()

    print("stop")


if __name__ == "__main__":
    main()
