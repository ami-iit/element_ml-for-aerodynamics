"""
Author: Antonello Paolino
Date: 2025-03-25
"""

import numpy as np
from pathlib import Path

from src.wing_flow import FlowGenerator, FlowVisualizer

AOA = 10.0
SWEEP = 10.0


def main():
    root = Path(__file__).parents[0]
    # Initialize flow object
    flow = FlowGenerator()
    # Load the dataset
    dataset_file = root / "dataset" / "wing-images-new.npz"
    dataset = np.load(dataset_file, allow_pickle=True)
    sweeps = dataset["sweep_angles"]
    aoas = dataset["angles_of_attack"]
    images = dataset["database"]
    # Import and set mesh mapping
    map_file = root / "maps" / "S_10-map-new.npy"
    map_data = np.load(map_file, allow_pickle=True).item()
    flow.import_mapping(map_data)
    # Initialize interpolator
    flow.compute_interpolator(images.shape[2:])

    # Generate a solution from the dataset using the image generator
    index = np.where((aoas == AOA) & (sweeps == SWEEP))[0][0]
    image = images[index]

    # Generate the solution
    flow.interpolate_flow_data(image)

    # Plot the solution
    viz = FlowVisualizer(flow)
    viz.plot_wing_pressure()


if __name__ == "__main__":
    main()
