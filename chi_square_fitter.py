import argparse
import os
import json
import yaml
import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize as scipy_minimize
import logging
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")

from differential_combination_postprocess.utils import setup_logging


def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--mu-json",
        type=str,
        required=True,
        help="Path to the json file containing the mu values",
    )

    parser.add_argument(
        "--corr-json",
        type=str,
        required=True,
        help="Path to the json file containing the correlation matrix",
    )

    parser.add_argument("--prediction-dir", type=str, required=True, help="")

    parser.add_argument("--submodel-yaml", type=str, required=True, help="")

    parser.add_argument("--channel", type=str, required=True, help="")

    parser.add_argument("--output-dir", type=str, required=True, help="")

    parser.add_argument("--debug", action="store_true", help="Print debug messages")

    return parser.parse_args()


def chi_square(y0, y, yerr):
    residuals = y0 - y
    return residuals.dot(inv(yerr)).dot(residuals)


def corr_to_cov(corr, standard_deviations):
    # see https://blogs.sas.com/content/iml/2010/12/10/converting-between-correlation-and-covariance-matrices.html#:~:text=You%20can%20use%20similar%20operations,reciprocals%20of%20the%20standard%20deviations.
    dsd = np.diag(standard_deviations)
    return dsd.dot(corr).dot(dsd)


def extract_coeffs(pois, dct):
    """
    pois is a list of dicts with shape {'name': name, 'value': value}
    return 1D array with lenght len(self.mus)
    """
    mu = 1
    for poi in pois:
        # linear
        try:
            mu += dct[f"A_{poi['name']}"] * poi["value"]
        except KeyError:
            pass
        # quadratic
        try:
            mu += dct[f"B_{poi['name']}_2"] * poi["value"] ** 2
        except KeyError:
            pass
        # mixed
        for poi in pois:
            try:
                mu += (
                    dct[f"B_{poi['name']}_{poi['name']}"] * poi["value"] * poi["value"]
                )
            except KeyError:
                pass
    return mu


def get_mu_prediction(pois, mus, production_dct, decays_dct, channel):
    """
    pois is a list of dicts with shape {'name': name, 'value': value}
    """
    pred_mus = []
    br_num = extract_coeffs(pois, decays_dct[channel])
    br_den = extract_coeffs(pois, decays_dct["tot"])
    for mu in mus:
        prod_term = extract_coeffs(pois, production_dct[mu])
        pred_mus.append(prod_term * br_num / br_den)
    return np.array(pred_mus)


class EFTFitter:
    def __init__(
        self,
        mus_dict,
        correlation_matrix,
        submodel_dict,
        production_coeffs_dict,
        decay_coeffs_dict,
        channel,
    ):
        """
        """
        self.mus = list(mus_dict.keys())
        self.pois_dict = submodel_dict
        self.production_coeffs_dict = production_coeffs_dict
        self.decay_coeffs_dict = decay_coeffs_dict
        self.channel = channel
        if channel == "hgg":
            self.channel = "gamgam"

        self.y0 = np.array([mus_dict[mu]["bestfit"] for mu in self.mus])
        self.y0_asimov = np.ones(len(self.y0))
        y0_stdvs = np.array(
            [
                0.5 * (mus_dict[mu]["Up01Sigma"] + mus_dict[mu]["Down01Sigma"])
                for mu in self.mus
            ]
        )
        y0_stdvs_asimov = np.array(
            [
                0.5 * (mus_dict[mu]["Up01SigmaExp"] + mus_dict[mu]["Down01SigmaExp"])
                for mu in self.mus
            ]
        )
        self.y_err = corr_to_cov(correlation_matrix, y0_stdvs)
        self.y_err_asimov = corr_to_cov(correlation_matrix, y0_stdvs_asimov)

    def minimize(self, params_to_float, params_to_freeze, y0, y_err):
        """
        Has to return a list of dicts {'name': name, 'value': value}
        """

        def to_minimize(params):
            params_to_float_passed = [
                {"name": name, "value": value}
                for name, value in zip(params_to_float, params)
            ]
            params_to_freeze_passed = [
                {"name": name, "value": dct["val"]}
                for name, dct in params_to_freeze.items()
            ]
            params_to_pass = params_to_float_passed + params_to_freeze_passed
            mus = get_mu_prediction(
                params_to_pass,
                self.mus,
                self.production_coeffs_dict,
                self.decay_coeffs_dict,
                self.channel,
            )
            return chi_square(y0, mus, y_err)

        res = scipy_minimize(
            fun=to_minimize,
            x0=[v["val"] for v in params_to_float.values()],
            method="TNC",
            bounds=[(v["min"], v["max"]) for v in params_to_float.values()],
        )

        ret = [
            {"name": name, "value": value}
            for name, value in zip(params_to_float, res.x)
        ]

        return ret

    def scan(self, how, expected=False, points=100):
        y0 = self.y0 if not expected else self.y0_asimov
        y_err = self.y_err if not expected else self.y_err_asimov
        result = {}
        for poi_name, poi_info in self.pois_dict.items():
            poi_values = np.linspace(poi_info["min"], poi_info["max"], points)
            chi_s = []
            if how == "profiled":
                pois_to_float = {
                    pn: pv for pn, pv in self.pois_dict.items() if pn != poi_name
                }
            for poi_value in poi_values:
                if how == "fixed":
                    dcts = [{"name": poi_name, "value": poi_value}]
                elif how == "profiled":
                    pois_to_freeze = {poi_name: {"val": poi_value}}
                    dcts = self.minimize(pois_to_float, pois_to_freeze, y0, y_err)
                    dcts.append({"name": poi_name, "value": poi_value})
                y = get_mu_prediction(
                    dcts,
                    self.mus,
                    self.production_coeffs_dict,
                    self.decay_coeffs_dict,
                    self.channel,
                )
                chi = chi_square(y0, y, y_err)
                chi_s.append(chi)
            chi_s = np.array(chi_s)
            chi_s -= chi_s.min()
            result[poi_name] = {"values": poi_values, "chi_square": chi_s}

        return result


def main(args):
    if args.debug:
        logger = setup_logging(level="DEBUG")
    else:
        logger = setup_logging(level="INFO")

    with open(args.mu_json) as f:
        mus_dict = json.load(f)
    with open(args.corr_json) as f:
        dct = json.load(f)
        list_of_lists = []
        for key, value in dct.items():
            list_of_lists.append(list(value.values()))
        correlation_matrix = np.array(list_of_lists)
    with open(args.submodel_yaml) as f:
        submodel_dict = yaml.load(f)
    decays_file = f"{args.prediction_dir}/decay.json"
    with open(decays_file, "r") as f:
        decays_dct = json.load(f)
    logger.debug(f"decays_dct: {decays_dct}")
    production_file = (
        f"{args.prediction_dir}/differentials/{args.channel}/ggH_SMEFTatNLO_pt_gg.json"
    )
    with open(production_file, "r") as f:
        tmp_production_dct = json.load(f)
    dict_keys = list(tmp_production_dct.keys())
    sorted_keys = sorted(dict_keys, key=lambda x: float(x))
    tmp_production_dct = {k: tmp_production_dct[k] for k in sorted_keys}
    edges = [float(k) for k in sorted_keys] + [10000.0]
    logger.debug(f"edges: {edges}")
    production_dct = {}
    for edge, next_edge in zip(edges[:-1], edges[1:]):
        production_dct[
            "r_smH_PTH_{}_{}".format(
                str(edge).replace(".0", ""), str(next_edge).replace(".0", "")
            )
        ] = tmp_production_dct[str(edge)]
    if args.channel == "hgg":
        key_to_remove = "r_smH_PTH_450_10000"
        key_dct = production_dct[key_to_remove]
        production_dct.pop(key_to_remove)
        production_dct["r_smH_PTH_GT450"] = key_dct
    logger.debug(f"production_dct: {production_dct}")
    logger.debug(f"production_dct keys: {list(production_dct.keys())}")

    fitter = EFTFitter(
        mus_dict,
        correlation_matrix,
        submodel_dict,
        production_dct,
        decays_dct,
        args.channel,
    )

    results = {}
    results["expected_fixed"] = fitter.scan(how="fixed", expected=True)
    results["expected_profiled"] = fitter.scan(how="profiled", expected=True)
    logger.info(f"results: {results}")

    pois = list(submodel_dict.keys())
    for poi in pois:
        fig, ax = plt.subplots()
        for result_name, result in results.items():
            ax.plot(result[poi]["values"], result[poi]["chi_square"], label=result_name)
        ax.set_xlabel(poi)
        ax.set_ylabel("chi_square")
        ax.set_ylim(0, 10)
        ax.legend()
        logger.info(f"Saving plots for poi {poi}")
        fig.savefig(f"{args.output_dir}/{poi}.png")
        fig.savefig(f"{args.output_dir}/{poi}.pdf")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
