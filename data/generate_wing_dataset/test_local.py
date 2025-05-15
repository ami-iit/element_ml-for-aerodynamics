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
DENSITY_EXP = 4
WING_AREA = 1.83
OUT_STATIONS = [0.1, 0.5, 0.9]
SWEEP = 30.0
AOA = 0.0


def main():
    root = Path(__file__).parents[0]
    # Initialize flow object
    flow_in = FlowImporter()
    flow_out = FlowGenerator()
    # Load image dataset
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
    )
    # Load the RBF mapping
    # rbf_file = root / "training" / "rbf" / f"rbf_data_{DENSITY_EXP}.npy"
    # rbf_data = np.load(rbf_file, allow_pickle=True).item()
    # flow_out.rbf_centers = rbf_data["centers"]
    # flow_out.rbf_weights = rbf_data["weights"]
    # flow_out.rbf_epsilon = rbf_data["epsilon"]
    # Get the path to the raw data
    data_dir = root / "simulations"
    files = [file for file in data_dir.rglob("*.vtu") if file.is_file()]

    #
    for file in files:
        wing_name = file.parents[1].stem
        sweep = float(file.parent.parent.stem.split("_")[-1])
        aoa = float(file.parent.stem.split("_")[-1])
        if sweep == SWEEP and aoa == AOA:
            break

    # Define the mesh file name based on the wing name
    mesh_file = list(map_dir.rglob(f"{wing_name}-map-eem-0.npy"))[0]
    mesh_data = np.load(mesh_file, allow_pickle=True).item()

    # Import solution data and compute input coefficients
    flow_in.import_solution_data(file)
    flow_in.import_mesh(mesh_data["nodes"], mesh_data["faces"])
    flow_in.compute_aerodynamic_coefficients(aoa, WING_AREA)

    if MODE == "rbf-dec":
        # Generate a solution with RBF+Decoder and compute output coefficients
        ls_val = flow_out.predict_rbf_tps(np.array([[sweep, aoa]]))
        ls_val = torch.tensor(ls_val).to("cpu").type(torch.float32)
        out_image = flow_out.decoder(ls_val).cpu().detach().numpy()[0]
    elif MODE == "ae":
        # Generate a solution from the dataset using the autoencoder
        index = np.where((aoas == aoa) & (sweeps == sweep))[0][0]
        image = images[index]
        in_image = torch.tensor(image[np.newaxis, ...]).to("cpu")
        latent_space_val = flow_out.encoder(in_image)
        out_image = flow_out.decoder(latent_space_val).cpu().detach().numpy()[0]
    elif MODE == "map":
        # Generate a solution identical to the input
        index = np.where((aoas == aoa) & (sweeps == sweep))[0][0]
        out_image = images[index]

    flow_out.interpolate_flow_data(out_image)
    flow_out.import_mesh(mesh_data["nodes"], mesh_data["faces"])
    flow_out.compute_aerodynamic_coefficients(aoa, WING_AREA)

    # Compute stations for Cp evaluation
    stations, clusters = clusterize_input_points(flow_in.points)

    # Cp local plots
    plt.rcParams["text.usetex"] = True
    fig, ax = plt.subplots(figsize=(10, 5))
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    fig.suptitle(f"Pressure Coefficient, Sweep {sweep} - AoA {aoa}")
    colors = ["blue", "green", "red", "yellow", "purple"]
    for c, out_station in zip(colors, OUT_STATIONS):
        # Get the index of the closest cluster to the station
        st_idx = np.argmin(np.abs(np.array(stations) - out_station))
        station = stations[st_idx]
        cluster = clusters[st_idx]
        # Get the input values in the cluster
        points = flow_in.points[cluster]
        x_in = points[:, 0]
        x_in = (x_in - x_in.min()) / (x_in.max() - x_in.min())
        cp_in = flow_in.cp[cluster]
        # Get the output values in the cluster
        points = flow_in.points[cluster]
        x_out = points[:, 0]
        x_out = (x_out - x_out.min()) / (x_out.max() - x_out.min())
        z_out = points[:, 2]
        cp_out = flow_out.cp[cluster]
        pos = np.where(z_out >= 0)[0]
        neg = np.where(z_out < 0)[0]
        pos_ord = np.argsort(x_out[pos])
        neg_ord = np.argsort(x_out[neg])
        # Plotting
        l = f" at $y={station:.2f}$"
        ax.scatter(x_in, cp_in, s=12, color=c, label="Input" + l)
        ax.plot(x_out[pos][pos_ord], cp_out[pos][pos_ord], color=c, label="Output" + l)
        ax.plot(x_out[neg][neg_ord], cp_out[neg][neg_ord], color=c, label=None)
    ax.yaxis.set_inverted(True)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$C_p$")
    ax.grid()
    ax.legend()

    # Plotting errors
    # cp_error = np.abs(flow_in.cp - flow_out.cp) / np.abs(flow_in.cp)
    # viz = FlowVisualizer(flow_out)
    # viz.plot_wing_pressure_error(
    #     cp_error, window_name="Pressure Coefficient Error"
    # )

    # Cp images
    image_idx = np.where((aoas == aoa) & (sweeps == sweep))[0][0]
    input_image = images[image_idx][3]
    output_image = out_image[3]
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    fig.suptitle(f"Comparison, Sweep {sweep} - AoA {aoa}")
    axs[0].imshow(input_image, origin="lower", cmap="jet")
    axs[0].set_title("Input Image")
    axs[0].set_xlim([-10, IM_RES[1] + 10])
    axs[0].set_ylim([-10, IM_RES[0] + 10])
    axs[1].imshow(output_image, origin="lower", cmap="jet")
    axs[1].set_title("Output Image")
    axs[1].set_xlim([-10, IM_RES[1] + 10])
    axs[1].set_ylim([-10, IM_RES[0] + 10])

    # Cp image error
    error_image = np.abs(input_image - output_image)
    fig, ax = plt.subplots(figsize=(10, 5))
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    fig.suptitle(f"Comparison, Sweep {sweep} - AoA {aoa}")
    im = ax.imshow(error_image, origin="lower", cmap="jet", vmin=0, vmax=0.12)
    ax.set_title("Absolute Error Image")
    ax.set_xlim([-10, IM_RES[1] + 10])
    ax.set_ylim([-10, IM_RES[0] + 10])
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Error", rotation=270, labelpad=15)
    plt.show()


def clusterize_input_points(input_points, tolerance=0.001):
    """
    Clusterize input geometry points based on the tolerance value along y axis.
    """
    # Sort the input points and indices based on the y coordinate
    wing_span = np.abs(input_points[:, 1].max() - input_points[:, 1].min())
    sorted_indices = np.argsort(input_points[:, 1])
    sorted_points = input_points[sorted_indices]
    clusters = []
    stations = []
    current_cluster = [sorted_indices[0]]

    for point, idx in zip(sorted_points[1:], sorted_indices[1:]):
        if abs(point[1] - sorted_points[current_cluster[0]][1]) <= tolerance:
            current_cluster.append(idx)
        else:
            y_span = (
                np.abs(np.mean([sorted_points[i][1] for i in current_cluster]))
                / wing_span
            )
            stations.append(y_span)
            clusters.append(current_cluster)
            current_cluster = [idx]

    clusters.append(current_cluster)
    return stations, clusters


if __name__ == "__main__":
    main()
