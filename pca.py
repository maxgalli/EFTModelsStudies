import argparse
import yaml
import json
import numpy as np
from scipy.sparse import block_diag
from numpy import linalg
import mplhep as hep
import matplotlib.pyplot as plt
import logging
from pathlib import Path

hep.style.use("CMS")

from differential_combination_postprocess.utils import setup_logging
from differential_combination_postprocess.matrix import MatricesExtractor
from differential_combination_postprocess.cosmetics import SMEFT_parameters_labels
from utils import (
    refactor_predictions_multichannel,
    robusthesse_paths,
    max_to_matt,
    ggH_production_files,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--prediction-dir", type=str, required=True, help="")

    parser.add_argument("--model-yaml", type=str, required=True, help="")

    parser.add_argument(
        "--channels", nargs="+", type=str, required=True, default=[], help=""
    )

    parser.add_argument("--output-dir", type=str, required=True, help="")

    parser.add_argument("--debug", action="store_true", help="Print debug messages")

    return parser.parse_args()


def get_p_ij(mu, wilson_coeff, production_dct, decays_dct, channel):
    if channel == "hgg":
        channel = "gamgam"
    res = 0
    try:
        res += production_dct[mu][f"A_{wilson_coeff}"]
    except KeyError:
        pass
    try:
        res += production_dct[mu][f"B_{wilson_coeff}_2"]
    except KeyError:
        pass
    try:
        res += decays_dct[channel][f"A_{wilson_coeff}"]
    except KeyError:
        pass
    try:
        res += decays_dct[channel][f"B_{wilson_coeff}_2"]
    except KeyError:
        pass
    try:
        res -= decays_dct["tot"][f"A_{wilson_coeff}"]
    except KeyError:
        pass
    try:
        res -= decays_dct["tot"][f"B_{wilson_coeff}_2"]
    except KeyError:
        pass

    return res


def plot_rotation_matrix(
    rot, eigenvalues, wilson_coeffs, channels, output_dir, full=False
):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("bwr")
    # plot only for eigenvalus > 0.001
    n_eigen_great = (
        len(eigenvalues) if full else int(np.sum(np.real(eigenvalues) > 0.001))
    )
    logging.debug(f"n_eigen_great {n_eigen_great}")
    if not full:
        rot = rot[:n_eigen_great, :]
    logging.debug(f"rot shape {rot.shape}")
    logging.debug(f"rot {rot}")
    cax = ax.matshow(rot, cmap=cmap, vmin=-1, vmax=1)
    # cbar = fig.colorbar(cax)

    for i in range(rot.shape[0]):
        for j in range(rot.shape[1]):
            ax.text(j, i, f"{rot[i, j]:.2f}", va="center", ha="center", fontsize=12)

    try:
        xlabels = [SMEFT_parameters_labels[c] for c in wilson_coeffs]
    except KeyError:
        xlabels = wilson_coeffs
    ylabels = [f"EV{i}" for i in list(range(n_eigen_great))]

    ax.set_xticks(np.arange(rot.shape[1]), minor=False)
    ax.set_yticks(np.arange(rot.shape[0]), minor=False)
    ax.set_xticklabels(xlabels, rotation=45, fontsize=12)
    ax.set_yticklabels(ylabels, fontsize=12)
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.tick_params(axis="y", which="both", left=False, right=False)

    # write eigenvalues on the right
    for i, ev in enumerate(eigenvalues[:n_eigen_great]):
        ax.text(
            rot.shape[1] - 0.45,
            i,
            f"$\lambda_{i}$ = {ev:.3f}",
            va="center",
            ha="left",
            fontsize=12,
            color="black",
        )

    fig.tight_layout()

    # save
    fig.savefig(f"{output_dir}/PCA-{''.join(channels)}{'-Full' if full else ''}.png")
    fig.savefig(f"{output_dir}/PCA-{''.join(channels)}{'-Full' if full else ''}.pdf")


def plot_diag_fisher(C_inv_smeft_dct, wilson_coeffs, output_dir):
    # plot at page 34 here https://arxiv.org/pdf/2105.00006.pdf
    # input is a dictionary where every key is a channel and the value is a matrix
    channels = list(C_inv_smeft_dct.keys())
    diagonals = []
    for channel, C_inv in C_inv_smeft_dct.items():
        diagonals.append(np.diag(C_inv))
    logging.debug(f"diagonals {diagonals}")
    matrix = np.array(diagonals).T
    # normalize to 100 along rows
    matrix = matrix / np.sum(matrix, axis=1)[:, np.newaxis] * 100
    logging.debug(f"Produced fake Fisher matrix of shape {matrix.shape}")
    # plot
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("bwr")
    cax = ax.matshow(matrix, cmap=cmap, vmin=0, vmax=100)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", va="center", ha="center", fontsize=12)

    try:
        ylabels = [SMEFT_parameters_labels[c] for c in wilson_coeffs]
    except KeyError:
        ylabels = wilson_coeffs
    xlabels = [f"{c}" for c in channels]

    ax.set_xticks(np.arange(len(xlabels)), minor=False)
    ax.set_yticks(np.arange(len(ylabels)), minor=False)
    ax.set_xticklabels(xlabels, rotation=45, fontsize=12)
    ax.set_yticklabels(ylabels, fontsize=12)
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.tick_params(axis="y", which="both", left=False, right=False)

    fig.tight_layout()

    # save
    fig.savefig(f"{output_dir}/diag_fisher-{''.join(channels)}.png")
    fig.savefig(f"{output_dir}/diag_fisher-{''.join(channels)}.pdf")


def rotate_back(dct, rot, eigenvalues, wilson_coeffs):
    """
    dct is a dictionary in the form {
        'A_c1': val,
        'B_c1_2': val,
        'B_c1_c2': val
    }
    """
    A = []
    for wc in wilson_coeffs:
        try:
            A.append(dct[f"A_{wc}"])
        except KeyError:
            A.append(0.0)
    A = np.array(A)

    B = []
    for iwc in wilson_coeffs:
        B_row = []
        for jwc in wilson_coeffs:
            if iwc == jwc:
                try:
                    B_row.append(dct[f"B_{iwc}_2"])
                except KeyError:
                    B_row.append(0.0)
            else:
                try:
                    B_row.append(0.5 * dct[f"B_{iwc}_{jwc}"])
                except KeyError:
                    try:
                        B_row.append(0.5 * dct[f"B_{jwc}_{iwc}"])
                    except KeyError:
                        B_row.append(0.0)
        B.append(B_row)
    B = np.array(B)

    A_rot = A.dot(rot.T)
    B_rot = rot.dot(B).dot(rot.T)

    v_rot = {}
    for i, iwc in enumerate(wilson_coeffs):
        iwcrot = f"EV{i}"
        if A_rot[i] != 0:
            v_rot[f"A_{iwcrot}"] = np.real(A_rot[i])
        for j, jwc in enumerate(wilson_coeffs):
            if i == j:
                if B_rot[i, j] != 0:
                    v_rot[f"B_{iwcrot}_2"] = np.real(B_rot[i, j])
            elif j > i:
                jwcrot = f"EV{j}"
                if B_rot[i, j] != 0:
                    v_rot[f"B_{iwcrot}_{jwcrot}"] = 2 * np.real(B_rot[i, j])

    return v_rot


def main(args):
    if args.debug:
        logger = setup_logging(level="DEBUG")
    else:
        logger = setup_logging(level="INFO")

    # Extract inputs
    with open(args.model_yaml, "r") as f:
        model = yaml.safe_load(f)
    wilson_coeffs = list(model.keys())
    decays_dct, production_dct_of_dcts, edges_dct = refactor_predictions_multichannel(
        args.prediction_dir, args.channels
    )
    logger.debug(f"decays_dct {decays_dct}")
    logger.debug(f"production_dct {production_dct_of_dcts}")
    logger.debug(f"edges {edges_dct}")
    mus_dct = {}
    for channel in args.channels:
        mus_dct[channel] = list(production_dct_of_dcts[channel].keys())
    logger.debug(f"Found mus {mus_dct}")

    # Build C_diff_inv, P, C_smeft_inv for each channel
    C_diff_inv_dct = {}
    P_dct = {}
    C_smeft_inv_dct = {}

    for channel in args.channels:
        logger.info(f"Processing channel {channel} for C_diff_inv, P, C_smeft_inv")
        # Build C_diff
        # c_diff = np.random.rand(len(production_dct), len(production_dct)) # debug
        me = MatricesExtractor(mus_dct[channel])
        me.extract_from_robusthesse(robusthesse_paths[channel])
        C_diff = me.matrices["hessian_covariance"]
        logger.debug(f"C_diff shape {C_diff.shape}")
        C_diff_inv_dct[channel] = linalg.inv(C_diff)
        logger.debug(f"C_diff_inv shape {C_diff_inv_dct[channel].shape}")

        # Build P
        lol = []
        for mu in mus_dct[channel]:
            row = []
            for wc in wilson_coeffs:
                row.append(
                    get_p_ij(
                        mu, wc, production_dct_of_dcts[channel], decays_dct, channel
                    )
                )
            lol.append(row)
        P_dct[channel] = np.array(lol)
        logger.debug(f"P shape {P_dct[channel].shape}")

        # Build C_smeft
        C_smeft_inv_dct[channel] = np.dot(
            P_dct[channel].T, np.dot(C_diff_inv_dct[channel], P_dct[channel])
        )

    # Before putting them togheter, plot diag_fisher
    plot_diag_fisher(C_smeft_inv_dct, wilson_coeffs, args.output_dir)

    # Now attach them and proceed with PCA
    C_diff_inv = block_diag(
        [C_diff_inv_dct[channel] for channel in args.channels]
    ).toarray()
    # concatenate Ps along y
    P = np.concatenate([P_dct[channel] for channel in args.channels], axis=0)
    logger.debug(f"Concatenated P shape {P.shape}")

    # eigenvalue decomposition
    C_smeft_inv = np.dot(P.T, np.dot(C_diff_inv, P))
    logger.debug(f"C_smeft_inv shape {C_smeft_inv.shape}")
    l, v = linalg.eig(C_smeft_inv)
    logger.debug(f"Eigenvalues: {l}")
    logger.debug(f"Eigenvectors: {v}")

    # order by decreasing eigenvalues
    idx = l.argsort()[::-1]
    l = l[idx]
    v = v[:, idx]
    # v = v.T[idx] # jonno
    logger.debug(f"Ordered eigenvalues: {l}")
    logger.debug(f"Ordered eigenvectors: {v}")

    # define rotation matrix
    rot = linalg.inv(v)
    # rot = linalg.inv(v).T # jonno
    logger.debug(f"Rotation matrix: {rot}")

    # plot rotation matrix
    # debug
    # rot = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # l = np.array([3, 2, 0.00001])
    # wilson_coeffs = ["C7", "C8", "C9"]
    plot_rotation_matrix(rot, l, wilson_coeffs, args.channels, args.output_dir)
    plot_rotation_matrix(
        rot, l, wilson_coeffs, args.channels, args.output_dir, full=True
    )

    model_name = args.model_yaml.split("/")[-1].split(".")[0]
    base_output_dir = (
        f"{args.prediction_dir}_rotated_{model_name}_{''.join(args.channels)}"
    )
    for channel in args.channels:
        mus = mus_dct[channel]
        production_dct = production_dct_of_dcts[channel]
        edges = edges_dct[channel]

        # create and dump equations (both prod and decay) with eigenvectors as coefficients
        production_dct_rot = {}
        for mu, edge in zip(mus, edges[:-1]):
            production_dct_rot[str(edge)] = rotate_back(
                production_dct[mu], rot, l, wilson_coeffs
            )
        logger.info(f"Rotated production dct: {production_dct_rot}")

        # create output directory with usual structure
        channel_output_dir = f"{base_output_dir}/differentials/{channel}"
        p = Path(channel_output_dir)
        p.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory {channel_output_dir}")

        # dump production
        file_name = ggH_production_files[channel].split("/")[-1]
        with open(f"{channel_output_dir}/{file_name}", "w") as f:
            json.dump(production_dct_rot, f, indent=4)
        logger.info(f"Dumped rotated production to {channel_output_dir}/{file_name}")

    # dump decay
    decays_dct_rot = {}
    for dec in [*[max_to_matt[c] for c in args.channels], "tot"]:
        decays_dct_rot[dec] = rotate_back(decays_dct[dec], rot, l, wilson_coeffs)
    logger.info(f"Rotated decays dct: {decays_dct_rot}")

    with open(f"{base_output_dir}/decay.json", "w") as f:
        json.dump(decays_dct_rot, f, indent=4)
    logger.info(f"Dumped rotated decays to {base_output_dir}/decay.json")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
