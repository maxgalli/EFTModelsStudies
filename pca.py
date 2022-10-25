import argparse
import yaml
import json
import numpy as np
from numpy import linalg
import mplhep as hep
import matplotlib.pyplot as plt
import logging
from pathlib import Path

hep.style.use("CMS")

from differential_combination_postprocess.utils import setup_logging
from differential_combination_postprocess.matrix import MatricesExtractor
from differential_combination_postprocess.cosmetics import SMEFT_parameters_labels
from chi_square_fitter import refactor_predictions


def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--robusthesse-file",
        type=str,
        required=True,
        help="Path to the robusthesse file",
    )

    parser.add_argument("--prediction-dir", type=str, required=True, help="")

    parser.add_argument("--model-yaml", type=str, required=True, help="")

    parser.add_argument("--channel", type=str, required=True, help="")

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


def plot_rotation_matrix(rot, eigenvalues, wilson_coeffs, output_dir):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("bwr")
    # plot only for eigenvalus > 0.001
    n_eigen_great = int(np.sum(np.real(eigenvalues) > 0.001))
    logging.debug(f"n_eigen_great {n_eigen_great}")
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
    fig.savefig(f"{output_dir}/PCA.png")
    fig.savefig(f"{output_dir}/PCA.pdf")


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
    decays_dct, production_dct, edges = refactor_predictions(
        args.prediction_dir, args.channel
    )
    logger.debug(f"decays_dct {decays_dct}")
    logger.debug(f"production_dct {production_dct}")
    logger.debug(f"edges {edges}")
    mus = list(production_dct.keys())
    logger.debug(f"Found mus {mus}")

    # Build C_diff
    # c_diff = np.random.rand(len(production_dct), len(production_dct))
    me = MatricesExtractor(mus)
    me.extract_from_robusthesse(args.robusthesse_file)
    c_diff = me.matrices["hessian_covariance"]
    logger.debug(f"C_diff shape {c_diff.shape}")
    c_diff_inv = linalg.inv(c_diff)
    logger.debug(f"C_diff_inv shape {c_diff_inv.shape}")

    # Build P
    lol = []
    for mu in mus:
        row = []
        for wc in wilson_coeffs:
            row.append(get_p_ij(mu, wc, production_dct, decays_dct, args.channel))
        lol.append(row)
    p = np.array(lol)
    logger.debug(f"P shape {p.shape}")

    # eigenvalue decomposition
    c_smeft_inv = np.dot(p.T, np.dot(c_diff_inv, p))
    l, v = linalg.eig(c_smeft_inv)
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
    plot_rotation_matrix(rot, l, wilson_coeffs, args.output_dir)

    # create and dump equations (both prod and decay) with eigenvectors as coefficients
    production_dct_rot = {}
    for mu, edge in zip(mus, edges[:-1]):
        production_dct_rot[str(edge)] = rotate_back(
            production_dct[mu], rot, l, wilson_coeffs
        )
    logger.info(f"Rotated production dct: {production_dct_rot}")

    decays_dct_rot = {}
    max_to_matt = {"hgg": "gamgam"}
    for dec in [max_to_matt[args.channel], "tot"]:
        decays_dct_rot[dec] = rotate_back(decays_dct[dec], rot, l, wilson_coeffs)
    logger.info(f"Rotated decays dct: {decays_dct_rot}")

    # create output directory with usual structure
    model_name = args.model_yaml.split("/")[-1].split(".")[0]
    base_output_dir = f"{args.prediction_dir}_rotated_{model_name}"
    channel_output_dir = f"{base_output_dir}/differentials/{args.channel}"
    p = Path(channel_output_dir)
    p.mkdir(parents=True, exist_ok=True)

    # dump production
    with open(f"{channel_output_dir}/ggH_SMEFTatNLO_pt_gg.json", "w") as f:
        json.dump(production_dct_rot, f, indent=4)
    with open(f"{base_output_dir}/decay.json", "w") as f:
        json.dump(decays_dct_rot, f, indent=4)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
