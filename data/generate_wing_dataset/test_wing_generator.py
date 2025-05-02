"""
Author: Antonello Paolino
Date: 2025-05-01
"""

import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt

from src.wing_flow import FlowImporter, FlowGenerator

IM_RES = (256, 256)
DENSITY_EXP = 1
WING_AREA = 1.83


def main():
    root = Path(__file__).parents[0]
    # Initialize flow object
    flow_in = FlowImporter()
    flow_out = FlowGenerator()
    # Create a dataset output directory if not existing
    dataset_dir = root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # Import and set mesh mapping
    map_dir = root / "maps"
    map_file = list(map_dir.rglob(f"S_0-map-eem-{DENSITY_EXP}.npy"))[0]
    map_data = np.load(map_file, allow_pickle=True).item()
    flow_in.import_mapping(map_data)
    flow_out.import_mapping(map_data)
    # Initialize interpolator
    flow_out.compute_interpolator(IM_RES)
    # Import the trained models
    model_path = root / "training" / "Out_1"
    flow_out.load_models(
        encoder_path=str(model_path / "scripted_enc.pt"),
        decoder_path=str(model_path / "scripted_dec.pt"),
    )
    # Load the RBF mapping
    rbf_file = root / "training" / "rbf" / "rbf_data_1.npy"
    rbf_data = np.load(rbf_file, allow_pickle=True).item()
    flow_out.rbf_centers = rbf_data["centers"]
    flow_out.rbf_weights = rbf_data["weights"]
    flow_out.rbf_epsilon = rbf_data["epsilon"]
    # Get the path to the raw data
    data_dir = Path(__file__).parents[0] / "simulations"
    files = [file for file in data_dir.rglob("*.vtu") if file.is_file()]
    # Initialize data variables
    sweep_angles = []
    angles_of_attack = []
    in_drag_coef, in_lift_coef = [], []
    out_drag_coef, out_lift_coef = [], []
    for idx, file in enumerate(files):
        # Set variables
        wing_name = file.parents[1].stem
        sweep = float(file.parent.parent.stem.split("_")[-1])
        aoa = float(file.parent.stem.split("_")[-1])
        # Define the mesh file name based on the wing name
        mesh_file = list(map_dir.rglob(f"{wing_name}-map-eem-0.npy"))[0]
        mesh_data = np.load(mesh_file, allow_pickle=True).item()

        # Import solution data and compute input coefficients
        flow_in.import_solution_data(file)
        flow_in.import_mesh(mesh_data["nodes"], mesh_data["faces"])
        flow_in.compute_aerodynamic_coefficients(aoa, WING_AREA)

        # Generate a solution with RBF+Decoder and compute output coefficients
        ls_val = flow_out.predict_rbf_tps(np.array([[sweep, aoa]]))
        ls_val = torch.tensor(ls_val).to("cpu").type(torch.float32)
        out_image = flow_out.decoder(ls_val).cpu().detach().numpy()
        flow_out.interpolate_flow_data(out_image[0])
        flow_out.import_mesh(mesh_data["nodes"], mesh_data["faces"])
        flow_out.compute_aerodynamic_coefficients(aoa, WING_AREA)

        # Collect data
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
        axs[0].plot(aoas, in_cl, label="dataset", color="blue")
        axs[0].plot(aoas, out_cl, label="prediction", color="orange")
        axs[0].set_xlabel(r"$\alpha$ (deg)")
        axs[0].set_ylabel(r"$C_L$")
        axs[0].legend()
        axs[0].grid()
        # Plot the drag coefficient
        axs[1].plot(aoas, in_cd, label="dataset", color="blue")
        axs[1].plot(aoas, out_cd, label="prediction", color="orange")
        axs[1].set_xlabel(r"$\alpha$ (deg)")
        axs[1].set_ylabel(r"$C_D$")
        axs[1].legend()
        axs[1].grid()
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
    plt.show()


if __name__ == "__main__":
    main()
