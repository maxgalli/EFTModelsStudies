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
import os

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

from pca import get_p_ij, plot_rotation_matrix, rotate_back


def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--prediction-dir", type=str, required=True, help="")

    parser.add_argument(
        "--submodel-yamls", nargs="+", type=str, required=True, default=True, help=""
    )

    parser.add_argument(
        "--channels", nargs="+", type=str, required=True, default=[], help=""
    )

    parser.add_argument("--output-dir", type=str, required=True, help="")

    parser.add_argument("--debug", action="store_true", help="Print debug messages")

    return parser.parse_args()


def main(args):
    if args.debug:
        logger = setup_logging(level="DEBUG")
    else:
        logger = setup_logging(level="INFO")

    # Extract inputs
    model_name = args.submodel_yamls[0].split("/")[-1].split(".")[0].split("_")[0]
    submodels = {}
    wilson_coeffs = {}
    for submodel_path in args.submodel_yamls:
        submodel_name = submodel_path.split("/")[-1].split(".")[0].split("_")[-1]
        with open(submodel_path, "r") as f:
            submodels[submodel_name] = yaml.safe_load(f)
    submodel_names = list(submodels.keys())
    logger.debug(f"Submodels: {submodels}")
    for k, v in submodels.items():
        wilson_coeffs[k] = list(v.keys())
    logger.info(f"Wilson coefficients: {wilson_coeffs}")
    all_wilson_coeffs = []
    for k, v in wilson_coeffs.items():
        all_wilson_coeffs += v
    decays_dct, production_dct_of_dcts, edges_dct = refactor_predictions_multichannel(
        args.prediction_dir, args.channels
    )
    mus_dct = {}
    for channel in args.channels:
        mus_dct[channel] = list(production_dct_of_dcts[channel].keys())

    C_diff_inv_dct = {}

    for channel in args.channels:
        logger.info(f"Processing channel {channel} for C_diff_inv")
        # Build C_diff
        # c_diff = np.random.rand(len(production_dct), len(production_dct)) # debug
        me = MatricesExtractor(mus_dct[channel])
        me.extract_from_robusthesse(robusthesse_paths[channel])
        C_diff = me.matrices["hessian_covariance"]
        C_diff_inv_dct[channel] = linalg.inv(C_diff)

    C_diff_inv = block_diag(
        [C_diff_inv_dct[channel] for channel in args.channels]
    ).toarray()

    rotation_matrices = {}
    eigenvalues = []
    for submodel_name in submodel_names:
        P_dct = {}
        for channel in args.channels:
            # Build P
            lol = []
            for mu in mus_dct[channel]:
                row = []
                for wc in wilson_coeffs[submodel_name]:
                    row.append(
                        get_p_ij(
                            mu,
                            wc,
                            production_dct_of_dcts[channel],
                            decays_dct,
                            channel,
                            how="A",
                        )
                    )
                lol.append(row)
            P_dct[channel] = np.array(lol)
            logger.info(
                f"P shape {P_dct[channel].shape} for channel {channel} in submodel {submodel_name}"
            )
        # concatenate Ps along y
        P = np.concatenate([P_dct[channel] for channel in args.channels], axis=0)
        logger.debug(f"Concatenated P shape {P.shape} in submodel {submodel_name}")

        # eigenvalue decomposition
        C_smeft_inv = np.dot(P.T, np.dot(C_diff_inv, P))
        logger.info(
            f"C_smeft_inv shape {C_smeft_inv.shape} in submodel {submodel_name}"
        )
        l, v = linalg.eig(C_smeft_inv)

        # order by decreasing eigenvalues
        idx = l.argsort()[::-1]
        l = l[idx]
        eigenvalues.append(l)
        v = v[:, idx]
        # v = v.T[idx] # jonno

        # define rotation matrix
        rot = linalg.inv(v)
        rotation_matrices[submodel_name] = rot
        logger.info(f"Rotation matrix shape {rot.shape} in submodel {submodel_name}")

    rot = block_diag(list(rotation_matrices.values())).toarray()
    logger.info(f"Final rotation matrix shape {rot.shape}")
    eigenvalues = np.concatenate(eigenvalues)
    ev_names = []
    for submodel_name in submodel_names:
        if len(wilson_coeffs[submodel_name]) == 1:
            ev_names.append(wilson_coeffs[submodel_name][0])
        else:
            for i, wc in enumerate(wilson_coeffs[submodel_name]):
                ev_names.append(f"EV{submodel_name}{i}")

    # plot rotation matrix
    output_dir = os.path.join(
        args.output_dir, args.prediction_dir.split("/")[-1] + "-" + model_name
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for is_full in [True, False]:
        plot_rotation_matrix(
            rot,
            eigenvalues,
            all_wilson_coeffs,
            args.channels,
            output_dir,
            full=is_full,
            suffix="-groups",
            ev_names=ev_names,
        )

    base_output_dir = f"{args.prediction_dir}_rotated{model_name}{''.join(args.channels)}{''.join(submodel_names)}"

    for channel in args.channels:
        mus = mus_dct[channel]
        production_dct = production_dct_of_dcts[channel]
        edges = edges_dct[channel]

        # create and dump equations (both prod and decay) with eigenvectors as coefficients
        production_dct_rot = {}
        for mu, edge in zip(mus, edges[:-1]):
            production_dct_rot[str(edge)] = rotate_back(
                production_dct[mu], rot, all_wilson_coeffs, ev_names
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
        decays_dct_rot[dec] = rotate_back(
            decays_dct[dec], rot, all_wilson_coeffs, ev_names
        )
    logger.info(f"Rotated decays dct: {decays_dct_rot}")

    with open(f"{base_output_dir}/decay.json", "w") as f:
        json.dump(decays_dct_rot, f, indent=4)
    logger.info(f"Dumped rotated decays to {base_output_dir}/decay.json")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
