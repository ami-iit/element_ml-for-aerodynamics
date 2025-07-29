"""
Author: Antonello Paolino
Date: 2025-05-01
"""

import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt

from src.wing_flow import FlowImporter, FlowGenerator, FlowVisualizer

MODE = "ae"  # "rbf-dec" or "ae" or "map"
IM_RES = (64, 64)
DENSITY_EXP = 0
DEVICE = "cpu"
WING_AREA = 1.83


def main():
    root = Path(__file__).parents[0]
    # Initialize flow object
    flow_in = FlowImporter()
    flow_out = FlowGenerator()
    # Import the image dataset
    dataset_file = root / "dataset" / f"wing-images-{IM_RES[0]}-{DENSITY_EXP}.npz"
    dataset = np.load(dataset_file, allow_pickle=True)
    sweeps = dataset["sweep_angles"]
    aoas = dataset["angles_of_attack"]
    images = dataset["database"]
    # Import and set mesh mapping
    map_dir = root / "maps"
    map_file = list(map_dir.rglob(f"S_0-map-eem-{DENSITY_EXP}.npy"))[0]
    map_data = np.load(map_file, allow_pickle=True).item()
    flow_in.import_mapping(map_data)
    flow_out.import_mapping(map_data)
    # Initialize interpolator
    flow_out.compute_interpolator(IM_RES)
    # Import the trained models
    model_path = root / "training" / f"{DENSITY_EXP}"
    flow_out.load_models(
        encoder_path=str(model_path / "scripted_enc.pt"),
        decoder_path=str(model_path / "scripted_dec.pt"),
        device=DEVICE,
    )
    # Load the RBF mapping
    rbf_file = root / "training" / "rbf" / f"rbf_data_{DENSITY_EXP}.npy"
    rbf_data = np.load(rbf_file, allow_pickle=True).item()
    flow_out.rbf_centers = rbf_data["centers"]
    flow_out.rbf_weights = rbf_data["weights"]
    flow_out.rbf_epsilon = rbf_data["epsilon"]
    # Get the path to the raw data
    data_dir = root / "simulations"
    files = [file for file in data_dir.rglob("*.vtu") if file.is_file()]
    # Initialize data variables
    sweep_angles = []
    angles_of_attack = []
    in_drag_coef, in_lift_coef = [], []
    out_drag_coef, out_lift_coef = [], []
    output_data = np.zeros(shape=(len(files), len(flow_in.map_nodes), 7))
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

        if sweep == 0.0 and aoa == 6.0:
            viz = FlowVisualizer(flow_in)
            viz.plot_wing_pressure()

        if MODE == "rbf-dec":
            # Generate a solution with RBF+Decoder and compute output coefficients
            ls_val = flow_out.predict_rbf_tps(np.array([[sweep, aoa]]))
            ls_val = torch.tensor(ls_val).to(DEVICE).type(torch.float32)
            out_image = flow_out.decoder(ls_val).cpu().detach().numpy()[0]
        elif MODE == "ae":
            # Generate a solution from the dataset using the autoencoder
            index = np.where((aoas == aoa) & (sweeps == sweep))[0][0]
            image = images[index]
            in_image = torch.tensor(image[np.newaxis, ...]).to(DEVICE)
            latent_space_val = flow_out.encoder(in_image)
            out_image = flow_out.decoder(latent_space_val).cpu().detach().numpy()[0]
        elif MODE == "map":
            # Generate a solution identical to the input
            index = np.where((aoas == aoa) & (sweeps == sweep))[0][0]
            out_image = images[index]

        flow_out.interpolate_flow_data(out_image)
        flow_out.import_mesh(mesh_data["nodes"], mesh_data["faces"])
        flow_out.compute_aerodynamic_coefficients(aoa, WING_AREA)

        output_data[idx, :, :3] = flow_out.points
        output_data[idx, :, 3] = flow_out.cp
        output_data[idx, :, 4:] = flow_out.cf

        # Collect data
        sweep_angles.append(sweep)
        angles_of_attack.append(aoa)
        in_drag_coef.append(flow_in.drag_coefficient)
        in_lift_coef.append(flow_in.lift_coefficient)
        out_drag_coef.append(flow_out.drag_coefficient)
        out_lift_coef.append(flow_out.lift_coefficient)

        print(f"{MODE} testing progress: {idx+1}/{len(files)}", end="\r", flush=True)

    # Save data
    # Save the mapping
    data = {
        "sweeps": np.array(sweep_angles),
        "aoas": np.array(angles_of_attack),
        "output_data": output_data,
    }
    np.save(root / "dataset" / f"output-{IM_RES[0]}-{DENSITY_EXP}.npy", data)

    # Plots
    plt.rcParams["text.usetex"] = True
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    colors = ["blue", "yellow", "green", "red", "purple"]
    unique_sweep_angles = sorted(set(sweep_angles))
    for idx, sweep in enumerate(unique_sweep_angles):
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
        # Plot the coefficients
        label = f"$\Lambda = {sweep}^\circ$"
        axs[0].plot(aoas, out_cl, label=None, color=colors[idx])
        axs[0].scatter(aoas, in_cl, label=label, color=colors[idx], s=12)
        axs[1].plot(aoas, out_cd, label=None, color=colors[idx])
        axs[1].scatter(aoas, in_cd, label=label, color=colors[idx], s=12)
    axs[0].set_xlabel(r"$\alpha [^\circ]$")
    axs[0].set_ylabel(r"$C_L$")
    axs[0].grid()
    axs[0].legend()
    axs[1].set_xlabel(r"$\alpha [^\circ]$")
    axs[1].set_ylabel(r"$C_D$")
    axs[1].grid()
    axs[1].legend()
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.suptitle(f"{MODE} testing")
    plt.show()


if __name__ == "__main__":
    main()
