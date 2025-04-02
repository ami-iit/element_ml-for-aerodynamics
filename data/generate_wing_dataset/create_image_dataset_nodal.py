"""
Author: Antonello Paolino
Date: 2025-03-25
"""

import numpy as np
import pickle
from pathlib import Path
from matplotlib import pyplot as plt

from src.wing_flow import FlowImporter

IM_RES = (1024, 1024)


def main():
    # Initialize flow object
    flow = FlowImporter()
    # Create a dataset output directory if not existing
    dataset_dir = Path(__file__).parents[0] / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # Import and set mesh mapping
    map_dir = Path(__file__).parents[0] / "maps"
    map_file = list(map_dir.rglob(f"S_10-map-new.npy"))[0]
    map_data = np.load(map_file, allow_pickle=True).item()
    flow.import_mapping(map_data)
    # Get the path to the raw data
    data_dir = Path(__file__).parents[0] / "simulations"
    files = [file for file in data_dir.rglob("*.vtu") if file.is_file()]
    # Initialize dataset variables
    database = np.empty(shape=(len(files), 7, IM_RES[0], IM_RES[1]), dtype=np.float32)
    sweep_angles = []
    angles_of_attack = []
    for idx, file in enumerate(files):
        sweep = float(file.parent.parent.stem.split("_")[-1])
        aoa = float(file.parent.stem.split("_")[-1])
        # Import solution data
        flow.import_solution_data(file)
        # flow.reorder_data()
        # Data Interpolation and Image Generation
        flow.interp_3d_to_image(IM_RES)
        database[idx, :, :, :] = flow.image
        sweep_angles.append(sweep)
        angles_of_attack.append(aoa)
        print(f"Generation progress: {idx+1}/{len(files)}", end="\r", flush=True)
    # Assign dataset variables
    dataset = {
        "sweep_angles": np.array(sweep_angles).astype(np.float32),
        "angles_of_attack": np.array(angles_of_attack).astype(np.float32),
        "database": np.array(database).astype(np.float32),
    }
    # Save compressed dataset using compressed numpy
    with open(str(dataset_dir / f"wing-images-new.npz"), "wb") as f:
        pickle.dump(dataset, f, protocol=4)
    print(f"Nodal image dataset for wings saved.")

    ## Testing
    n_sim = len(files)
    rand_ids = np.random.randint(0, n_sim, 5)
    for i in rand_ids:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(
            f"Simulation {i}: Sweep {sweep_angles[i]} - AoA {angles_of_attack[i]}"
        )
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        titles = ["x", "y", "z", "Cp", "Cfx", "Cfy", "Cfz"]
        for idx, title in enumerate(titles):
            ax = axes[idx // 4, idx % 4]
            image = database[i, idx, :, :]
            last_im = ax.imshow(image, origin="lower", cmap="jet")
            ax.set_title(title)
            ax.set_xlim([-10, IM_RES[1] + 10])
            ax.set_ylim([-10, IM_RES[0] + 10])
            # fig.colorbar(last_im, ax=ax)
    plt.show()


if __name__ == "__main__":
    main()
