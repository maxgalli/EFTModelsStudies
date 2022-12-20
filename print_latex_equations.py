import argparse
import yaml
from pylatex import LongTable, NoEscape
from utils import refactor_predictions_multichannel, max_to_matt
from differential_combination_postprocess.utils import setup_logging
from differential_combination_postprocess.cosmetics import SMEFT_parameters_labels


def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--prediction-dir", type=str, required=True, help="")

    parser.add_argument("--submodel-yaml", type=str, required=True, help="")

    parser.add_argument(
        "--channels", nargs="+", type=str, required=True, default=[], help=""
    )

    parser.add_argument("--output", type=str, default="equations", help="")

    parser.add_argument(
        "--fit-model",
        choices=["full", "linearised"],
        default="full",
        help="What type of model to fit",
    )

    parser.add_argument("--debug", action="store_true", help="Print debug messages")

    return parser.parse_args()


def get_linearised_string(production_dct, decays_dct, pois, channel):
    string = "1 "
    strings = []
    for i, poi in enumerate(pois):
        coeff = 0
        look_for = f"A_{poi}"
        if look_for in production_dct:
            coeff += production_dct[look_for]
        matt_channel_name = max_to_matt[channel]
        if look_for in decays_dct[matt_channel_name]:
            coeff += decays_dct[matt_channel_name][look_for]
        if look_for in decays_dct["tot"]:
            coeff -= decays_dct["tot"][look_for]
        if coeff != 0:
            # round coeff to 3 decimals and convert to string
            coeff_str = f"{coeff:.3f}"
            start = "+" if coeff > 0 else ""
            poi_name = SMEFT_parameters_labels[poi]
            poi_name = poi_name.replace("$", "")
            string += f"{start}{coeff_str} {poi_name}"
        if (i + 1) % 4 == 0 or i == len(pois) - 1:
            string = "$ " + string + " $"
            strings.append(string)
            string = ""
    return strings


channel_tex = {
    "hgg": r"$H\rightarrow\gamma\gamma$",
    "hzz": r"$H\rightarrow ZZ$",
    "hww": r"$H\rightarrow WW$",
    "htt": r"$H\rightarrow\tau\tau$",
    "hbbvbf": r"$H\rightarrow b\bar{b}$",
    "httboost": r"$H\rightarrow\tau\tau \mathrm{boost.}$",
}


def main(args):
    if args.debug:
        logger = setup_logging(level="DEBUG")
    else:
        logger = setup_logging(level="INFO")

    fit_model = args.fit_model

    decays_dct, production_dct_of_dcts, edges_dct = refactor_predictions_multichannel(
        args.prediction_dir, args.channels
    )
    logger.debug(f"production_dct_of_dcts keys: {list(production_dct_of_dcts.keys())}")
    for k in production_dct_of_dcts.keys():
        logger.debug(
            f"production_dct_of_dcts[{k}] keys: {list(production_dct_of_dcts[k].keys())}"
        )
    logger.debug(f"decays_dct keys: {list(decays_dct.keys())}")

    with open(args.submodel_yaml) as f:
        dct = yaml.load(f)
        pois = list(dct.keys())
    logger.debug(f"POIs: {pois}")

    if fit_model == "linearised":
        table = LongTable("|l|l|")
        table.add_hline()
        table.add_row(("Bin", "Scalng equation"))
        table.add_hline()

        for channel in args.channels:
            production_dct = production_dct_of_dcts[channel]
            for bin in production_dct:
                production_dct_bin = production_dct[bin]
                bin_name = bin.replace("r_smH_PTH_", "")
                bin_name = bin_name.replace("_", "-")
                bin_name = f"{channel_tex[channel]}, {bin_name}"

                scaling_equation_parts = get_linearised_string(
                    production_dct_bin, decays_dct, pois, channel
                )
                for i, part in enumerate(scaling_equation_parts):
                    if i == 0:
                        table.add_row((NoEscape(bin_name), NoEscape(part)))
                    else:
                        table.add_row(("", NoEscape(part)))
                table.add_hline()

        table.generate_tex(args.output)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
