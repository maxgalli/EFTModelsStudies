import argparse
import os
import json
import ROOT
from itertools import product

from differential_combination_postprocess.scan import DifferentialSpectrum
from differential_combination_postprocess.matrix import MatricesExtractor
from differential_combination_postprocess.utils import (
    setup_logging,
    extract_from_yaml_file,
)
from get_mus import all_pois


def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--input-file",
        type=str,
        default="inputs",
        help="ROOT file containing corr matrix, usually starting with 'multidimfit_'",
    )

    parser.add_argument(
        "--observable",
        type=str,
        required=True,
        help="Observable to be used",
        choices=all_pois.keys(),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where output files will be stored",
    )

    parser.add_argument("--channel", type=str, required=True, help="Channel to be used")

    parser.add_argument("--debug", action="store_true", help="Print debug messages")

    return parser.parse_args()


def main(args):
    if args.debug:
        logger = setup_logging(level="DEBUG")
    else:
        logger = setup_logging(level="INFO")

    pois = all_pois[args.observable][args.channel]

    me = MatricesExtractor(pois)
    me.extract_from_roofitresult(args.input_file, "fit_mdf")
    for matrix_name, output in zip(
        ["rfr_correlation", "rfr_covariance"], ["correlation", "covariance"]
    ):
        matrix_values = {}
        for poi in pois:
            matrix_values[poi] = {}
        for pair, value in zip(
            product(pois, repeat=2), me.matrices[matrix_name].flatten()
        ):
            matrix_values[pair[0]][pair[1]] = float(value)
        logger.info(f"Matrix values: {matrix_values}")
        with open(f"{args.output_dir}/{output}_matrix.json", "w") as f:
            json.dump(matrix_values, f, indent=4)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
