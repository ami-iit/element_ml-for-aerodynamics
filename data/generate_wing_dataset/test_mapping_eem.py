"""
Author: Antonello Paolino
Date: 2025-05-01
"""

import numpy as np
import pickle
from pathlib import Path
from matplotlib import pyplot as plt

from src.wing_flow import FlowImporter, FlowGenerator

IM_RES = (256, 256)
DENSITY_EXP = 1
WING_AREA = 1.83


def main():
    # Initialize flow object
    flow_in = FlowImporter()
    flow_out = FlowGenerator()
    # Create a dataset output directory if not existing
    dataset_dir = Path(__file__).parents[0] / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # Import and set mesh mapping
    map_dir = Path(__file__).parents[0] / "maps"
    map_file = list(map_dir.rglob(f"S_0-map-eem-{DENSITY_EXP}.npy"))[0]
    map_data = np.load(map_file, allow_pickle=True).item()
    flow_in.import_mapping(map_data)
    flow_out.import_mapping(map_data)
    # Initialize interpolator
    flow_out.compute_interpolator(IM_RES)
    # Get the path to the raw data
    data_dir = Path(__file__).parents[0] / "simulations"
    files = [file for file in data_dir.rglob("*.vtu") if file.is_file()]
    # Initialize data variables
    sweep_angles = []
    angles_of_attack = []
    in_drag_coef, in_lift_coef = [], []
    out_drag_coef, out_lift_coef = [], []
    for idx, file in enumerate(files):
        wing_name = file.parents[1].stem
        sweep = float(file.parent.parent.stem.split("_")[-1])
        aoa = float(file.parent.stem.split("_")[-1])
        # Import solution data
        flow_in.import_solution_data(file)
        # Import mesh
        mesh_file = list(map_dir.rglob(f"{wing_name}-map-eem-0.npy"))[0]
        mesh_data = np.load(mesh_file, allow_pickle=True).item()
        flow_in.import_mesh(mesh_data["nodes"], mesh_data["faces"])
        flow_in.compute_aerodynamic_coefficients(aoa, WING_AREA)
        # flow_in.reorder_data()
        # Data Interpolation and Image Generation
        flow_in.interp_3d_to_image(IM_RES)
        image = flow_in.image
        # Generate a solution from the dataset using the image generator
        flow_out.interpolate_flow_data(image)
        flow_out.import_mesh(mesh_data["nodes"], mesh_data["faces"])
        flow_out.compute_aerodynamic_coefficients(aoa, WING_AREA)

        # Save data
        sweep_angles.append(sweep)
        angles_of_attack.append(aoa)
        in_drag_coef.append(flow_in.drag_coefficient)
        in_lift_coef.append(flow_in.lift_coefficient)
        out_drag_coef.append(flow_out.drag_coefficient)
        out_lift_coef.append(flow_out.lift_coefficient)

        print(f"Testing progress: {idx+1}/{len(files)}", end="\r", flush=True)

    # Plots

    # Enable LaTeX text rendering
    plt.rcParams["text.usetex"] = True

    unique_sweep_angles = sorted(set(sweep_angles))
    for sweep in unique_sweep_angles:
        # Get the indices of the images with the current sweep angle
        indices = [i for i, x in enumerate(sweep_angles) if x == sweep]
        aoas = np.array(angles_of_attack)[indices]
        in_cl = np.array(in_lift_coef)[indices]
        out_cl = np.array(out_lift_coef)[indices]
        in_cd = np.array(in_drag_coef)[indices]
        out_cd = np.array(out_drag_coef)[indices]
        # Sort the data by angle of attack
        indices = np.argsort(aoas)
        aoas = np.array(aoas)[indices]
        in_cl = np.array(in_cl)[indices]
        out_cl = np.array(out_cl)[indices]
        in_cd = np.array(in_cd)[indices]
        out_cd = np.array(out_cd)[indices]
        # Create a figure with subplots for each coefficient
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        plt.suptitle(f"Sweep angle: {sweep}°")
        # Plot the lift coefficient
        axs[0].plot(aoas, in_cl, label="input", color="blue")
        axs[0].plot(aoas, out_cl, label="output", color="orange")
        axs[0].set_xlabel(r"$\alpha$ (deg)")
        axs[0].set_ylabel(r"$C_L$")
        axs[0].legend()
        axs[0].grid()
        # Plot the drag coefficient
        axs[1].plot(aoas, in_cd, label="input", color="blue")
        axs[1].plot(aoas, out_cd, label="output", color="orange")
        axs[1].set_xlabel(r"$\alpha$ (deg)")
        axs[1].set_ylabel(r"$C_D$")
        axs[1].legend()
        axs[1].grid()
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
    plt.show()


if __name__ == "__main__":
    main()
