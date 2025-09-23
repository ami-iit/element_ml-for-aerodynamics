"""
Author: Antonello Paolino
Date: 2025-03-25
"""

import numpy as np
import pickle
from pathlib import Path
from matplotlib import pyplot as plt

from src.wing_flow import FlowImporter

GEOM = "struct"  # "struct" | "unstruct"
DENSITY_EXP = 0
IM_RES = (64, 64)
MIRROR_DATA = True


def main():
    root = Path(__file__).parents[0]
    # Initialize flow object
    flow = FlowImporter()
    # Get the path to the raw data
    data_dir = root / "simulations" / GEOM
    files = [file for file in data_dir.rglob("*.vtu") if file.is_file()]
    # Get the pah of mesh files
    map_dir = root / "maps" / GEOM
    # Create a dataset output directory if not existing
    dataset_dir = root / "dataset" / GEOM
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dataset variables
    database = np.empty(shape=(len(files), 7, IM_RES[0], IM_RES[1]), dtype=np.float32)
    sweep_angles = []
    angles_of_attack = []
    for idx, file in enumerate(files):
        if GEOM == "struct":
            parent = file.parent.parent
        elif GEOM == "unstruct":
            parent = file.parent
        wing_name = parent.stem
        sweep = float(wing_name.split("_")[-1])
        aoa = float(file.stem[16:])
        # Import and set mesh mapping
        map_file = map_dir / f"{wing_name}-map-eem-{DENSITY_EXP}.npy"
        map_data = np.load(map_file, allow_pickle=True).item()
        flow.import_mapping(map_data)
        # Data Import, Interpolation, and Image Generation
        flow.import_solution_data(file)
        flow.interp_3d_to_image(IM_RES)
        database[idx, :, :, :] = flow.image
        sweep_angles.append(sweep)
        angles_of_attack.append(aoa)
        print(f"Generation progress: {idx+1}/{len(files)}", end="\r", flush=True)
    # Mirror data if needed
    if MIRROR_DATA:
        database_mirrored = np.empty_like(database)
        sweep_angles_mirrored = []
        angles_of_attack_mirrored = []
        for idx, sweep in enumerate(sweep_angles):
            aoa = -angles_of_attack[idx]
            sweep_angles_mirrored.append(sweep)
            angles_of_attack_mirrored.append(aoa)
        database_mirrored = database.copy()
        database_mirrored[:, 3:, :, :] = np.transpose(
            database_mirrored[:, 3:, :, :], (0, 1, 3, 2)
        )
        # Concatenate original and mirrored data
        database = np.concatenate((database, database_mirrored), axis=0)
        sweep_angles += sweep_angles_mirrored
        angles_of_attack += angles_of_attack_mirrored
        print(f"Mirroring completed.")
    # Assign dataset variables
    dataset = {
        "sweep_angles": np.array(sweep_angles).astype(np.float32),
        "angles_of_attack": np.array(angles_of_attack).astype(np.float32),
        "database": np.array(database).astype(np.float32),
    }
    # Save compressed dataset using compressed numpy
    if MIRROR_DATA:
        dataset_name = f"wing-images-{IM_RES[0]}-{DENSITY_EXP}-mirrored.npz"
    else:
        dataset_name = f"wing-images-{IM_RES[0]}-{DENSITY_EXP}.npz"
    with open(str(dataset_dir / dataset_name), "wb") as f:
        pickle.dump(dataset, f, protocol=4)
    print(f"Image dataset for wings saved.")

    # Testing
    n_sim = len(database)
    rand_ids = np.random.randint(0, n_sim, 10)
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
            ax.imshow(image, origin="lower", cmap="jet")
            ax.set_title(title)
            ax.set_xlim([-10, IM_RES[1] + 10])
            ax.set_ylim([-10, IM_RES[0] + 10])
    plt.show()


if __name__ == "__main__":
    main()
