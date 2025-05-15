"""
Author: Antonello Paolino
Date: 2025-05-06
"""

import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt

from src.wing_flow import FlowImporter, FlowGenerator

MODE = "rbf-dec"  # "rbf-dec" or "ae"
IM_RES = (256, 256)
DENSITY_EXP = 1
WING_AREA = 1.83


def main():
    root = Path(__file__).parents[0]
    # Initialize flow object
    flow_in = FlowImporter()
    flow_out = FlowGenerator()
    # Import the image dataset
    dataset_file = root / "dataset" / f"wing-images-{IM_RES[0]}-eem-{DENSITY_EXP}.npz"
    dataset = np.load(dataset_file, allow_pickle=True)
    sweeps = dataset["sweep_angles"]
    aoas = dataset["angles_of_attack"]
    images = dataset["database"]
    # Flatten images array to 1D
    image_data_vec = images.reshape(-1)

    # Create raw data single vector
    data_dir = root / "simulations"
    files = [file for file in data_dir.rglob("*.vtu") if file.is_file()]
    raw_data_vec = np.empty(shape=(0,))
    for idx, file in enumerate(files):
        flow_in.import_solution_data(file)
        # Create a vector with all the data
        data_vec = np.concatenate(
            (
                flow_in.points.reshape(-1),
                flow_in.cp.reshape(-1),
                flow_in.cf.reshape(-1),
            )
        )
        raw_data_vec = np.concatenate((raw_data_vec, data_vec))
        print(f"Raw data progress: {idx+1}/{len(files)}", end="\r", flush=True)

    # Create reconstructed image data
    map_dir = root / "maps"
    map_file = list(map_dir.rglob(f"S_0-map-eem-{DENSITY_EXP}.npy"))[0]
    map_data = np.load(map_file, allow_pickle=True).item()
    flow_out.import_mapping(map_data)
    flow_out.compute_interpolator(IM_RES)
    recon_data_vec = np.empty(shape=(0,))
    for image in images:
        flow_out.interpolate_flow_data(image)
        # Create a vector with all the data
        data_vec = np.concatenate(
            (
                flow_out.points.reshape(-1),
                flow_out.cp.reshape(-1),
                flow_out.cf.reshape(-1),
            )
        )
        recon_data_vec = np.concatenate((recon_data_vec, data_vec))
        print(f"Recon data progress: {idx+1}/{len(files)}", end="\r", flush=True)

    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.hist(raw_data_vec, bins=100, alpha=0.5, density=True, color="g", label="CFD")
    # plt.hist(
    #     image_data_vec, bins=100, alpha=0.6, density=True, color="r", label="images"
    # )
    plt.hist(
        recon_data_vec, bins=100, alpha=0.5, density=True, color="b", label="recon"
    )
    plt.xlabel("Pixel value")
    plt.ylabel("Density")
    plt.title("Histogram of dataset values")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
