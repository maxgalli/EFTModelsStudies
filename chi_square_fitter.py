import argparse
import os
import json
import yaml
import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize as scipy_minimize
from scipy.sparse import block_diag
import logging
import matplotlib.pyplot as plt
import mplhep as hep
from itertools import combinations, product
from scipy import interpolate
from scipy.interpolate import griddata

hep.style.use("CMS")

from utils import (
    refactor_predictions_multichannel,
    ggH_production_files,
    max_to_matt,
    mus_paths,
    corr_matrices_observed,
    corr_matrices_expected,
)
from differential_combination_postprocess.utils import setup_logging


def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--prediction-dir", type=str, required=True, help="")

    parser.add_argument("--submodel-yaml", type=str, required=True, help="")

    parser.add_argument(
        "--channels", nargs="+", type=str, required=True, default=[], help=""
    )

    parser.add_argument("--output-dir", type=str, required=True, help="")

    parser.add_argument("--suffix", type=str, required=False, default="", help="")

    parser.add_argument("--linear", action="store_true", help="Fit only linear terms")

    parser.add_argument(
        "--quadratic", action="store_true", help="Fit only quadratic terms"
    )

    parser.add_argument(
        "--skip2D", action="store_true", help="Skip 2D plots of the fit"
    )

    parser.add_argument("--debug", action="store_true", help="Print debug messages")

    return parser.parse_args()


def chi_square(y0, y, yerr):
    residuals = y0 - y
    return residuals.dot(inv(yerr)).dot(residuals)


def corr_to_cov(corr, standard_deviations):
    # see https://blogs.sas.com/content/iml/2010/12/10/converting-between-correlation-and-covariance-matrices.html#:~:text=You%20can%20use%20similar%20operations,reciprocals%20of%20the%20standard%20deviations.
    dsd = np.diag(standard_deviations)
    return dsd.dot(corr).dot(dsd)


def extract_coeffs(pois, dct, linear_only=False, quadratic_only=False):
    """
    pois is a list of dicts with shape {'name': name, 'value': value} in case of print_coeffs=False, otherwise it is just a list of strings
    return 1D array with lenght len(self.mus)
    """
    mu = 1
    for poi in pois:
        # logging.debug(f"Extract coeff for {poi['name']}")
        # linear
        if not quadratic_only:
            try:
                mu += dct[f"A_{poi['name']}"] * poi["value"]
                # logging.debug("Linear term: {}".format(dct[f"A_{poi['name']}"]))
            except KeyError:
                pass
        # quadratic
        if not linear_only:
            try:
                mu += dct[f"B_{poi['name']}_2"] * poi["value"] ** 2
                # logging.debug("Quadratic term: {}".format(dct[f"B_{poi['name']}_2"]))
            except KeyError:
                pass
    # mixed
    prods = product(pois, pois)
    if len(pois) > 1:
        for prod in prods:
            poi1 = prod[0]
            poi2 = prod[1]
            try:
                mu += (
                    dct[f"B_{poi1['name']}_{poi2['name']}"]
                    * poi1["value"]
                    * poi2["value"]
                )
                # logging.debug(
                #    "Mixed term: {}".format(dct[f"B_{poi['name']}_{poi['name']}"])
                # )
            except KeyError:
                pass
    return mu


def extract_coeffs_string(pois, dct, linear_only=False, quadratic_only=False):
    """
    pois is a list of dicts with shape {'name': name, 'value': value} in case of print_coeffs=False, otherwise it is just a list of strings
    return 1D array with lenght len(self.mus)
    """
    mu = "1"
    for poi in pois:
        # logging.debug(f"Extract coeff for {poi['name']}")
        # linear
        if not quadratic_only:
            try:
                mu += " + {} * {}".format(dct[f"A_{poi}"], poi)
            except KeyError:
                pass
        # quadratic
        if not linear_only:
            try:
                mu += " + {} * {}^2".format(dct[f"B_{poi}_2"], poi)
            except KeyError:
                pass
    # mixed
    prods = product(pois, pois)
    if len(pois) > 1:
        for prod in prods:
            poi1 = prod[0]
            poi2 = prod[1]
            try:
                mu += " + {} * {} * {}".format(dct[f"B_{poi1}_{poi2}"], poi1, poi2)
            except KeyError:
                pass
    return mu


def get_mu_prediction(
    pois, mus, production_dct, decays_dct, linear_only=False, quadratic_only=False
):
    """
    pois is a list of dicts with shape {'name': name, 'value': value}
    """
    pred_mus = []
    for mu in mus:
        channel = mu.split("_")[-1]
        channel = max_to_matt[channel]
        prod_term = extract_coeffs(
            pois,
            production_dct[mu],
            linear_only=linear_only,
            quadratic_only=quadratic_only,
        )
        br_num = extract_coeffs(
            pois,
            decays_dct[channel],
            linear_only=linear_only,
            quadratic_only=quadratic_only,
        )
        br_den = extract_coeffs(
            pois,
            decays_dct["tot"],
            linear_only=linear_only,
            quadratic_only=quadratic_only,
        )
        pred_mus.append(prod_term * br_num / br_den)

    return np.array(pred_mus)


def get_mu_prediction_string(
    pois, mus, production_dct, decays_dct, linear_only=False, quadratic_only=False
):
    pred_mus = {}
    for mu in mus:
        channel = mu.split("_")[-1]
        channel = max_to_matt[channel]
        prod_term = extract_coeffs_string(
            pois,
            production_dct[mu],
            linear_only=linear_only,
            quadratic_only=quadratic_only,
        )
        br_num = extract_coeffs_string(
            pois,
            decays_dct[channel],
            linear_only=linear_only,
            quadratic_only=quadratic_only,
        )
        br_den = extract_coeffs_string(
            pois,
            decays_dct["tot"],
            linear_only=linear_only,
            quadratic_only=quadratic_only,
        )
        pred_mus[mu] = {}
        pred_mus[mu]["prod"] = prod_term
        pred_mus[mu]["br_num"] = br_num
        pred_mus[mu]["br_den"] = br_den

    return pred_mus


class EFTFitter:
    def __init__(
        self,
        mus_dict,
        obs_correlation_matrix,
        exp_correlation_matrix,
        submodel_dict,
        production_coeffs_dict,
        decay_coeffs_dict,
        linear=False,
        quadratic=False,
    ):
        """
        """
        self.mus = list(mus_dict.keys())
        self.pois_dict = submodel_dict
        self.production_coeffs_dict = production_coeffs_dict
        self.decay_coeffs_dict = decay_coeffs_dict
        self.linear = linear
        self.quadratic = quadratic

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
        self.y_err = corr_to_cov(obs_correlation_matrix, y0_stdvs)
        self.y_err_asimov = corr_to_cov(exp_correlation_matrix, y0_stdvs_asimov)

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

    def scan(self, how, expected=False, points=500, skip2d=False):
        logging.info(f"Scanning {how} with expected={expected}")
        y0 = self.y0 if not expected else self.y0_asimov
        y_err = self.y_err if not expected else self.y_err_asimov
        result = {}
        # 1D scans
        for poi_name, poi_info in self.pois_dict.items():
            logging.info(f"Scanning {poi_name}")
            poi_values = np.linspace(poi_info["min"], poi_info["max"], points)
            chi_s = []
            if how == "profiled":
                pois_to_float = {
                    pn: pv for pn, pv in self.pois_dict.items() if pn != poi_name
                }
            for i, poi_value in enumerate(poi_values):
                if how == "fixed":
                    dcts = [{"name": poi_name, "value": poi_value}]
                elif how == "profiled":
                    pois_to_freeze = {poi_name: {"val": poi_value}}
                    dcts = self.minimize(pois_to_float, pois_to_freeze, y0, y_err)
                    dcts.append({"name": poi_name, "value": poi_value})
                logging.debug(f"Scanning point {poi_value}, ({i}/{len(poi_values)})")
                y = get_mu_prediction(
                    dcts,
                    self.mus,
                    self.production_coeffs_dict,
                    self.decay_coeffs_dict,
                    linear_only=self.linear,
                    quadratic_only=self.quadratic,
                )
                chi = chi_square(y0, y, y_err)
                chi_s.append(chi)
            chi_s = np.array(chi_s)
            chi_s -= chi_s.min()
            result[poi_name] = {"values": poi_values, "chi_square": chi_s}

        # 2D scans
        if not skip2d:
            if how == "profiled" and len(self.pois_dict) < 3:
                logging.info("Skipping 2D scan")
                return result

            if len(self.pois_dict) > 1:
                pois_combinations = list(combinations(self.pois_dict.keys(), 2))
                points = int(np.sqrt(points))
                for poi1, poi2 in pois_combinations:
                    logging.info(f"Scanning {poi1} vs {poi2}")
                    # poi values will now be a pair
                    poi_values_pairs = (
                        np.mgrid[
                            self.pois_dict[poi1]["min"] : self.pois_dict[poi1][
                                "max"
                            ] : points
                            * 1j,
                            self.pois_dict[poi2]["min"] : self.pois_dict[poi2][
                                "max"
                            ] : points
                            * 1j,
                        ]
                        .reshape(2, -1)
                        .T
                    )
                    chi_s = []
                    if how == "profiled":
                        pois_to_float = {
                            pn: pv
                            for pn, pv in self.pois_dict.items()
                            if pn not in [poi1, poi2]
                        }
                    for poi_values_pair in poi_values_pairs:
                        if how == "fixed":
                            dcts = [
                                {"name": poi1, "value": poi_values_pair[0]},
                                {"name": poi2, "value": poi_values_pair[1]},
                            ]
                        elif how == "profiled":
                            pois_to_freeze = {
                                poi1: {"val": poi_values_pair[0]},
                                poi2: {"val": poi_values_pair[1]},
                            }
                            dcts = self.minimize(
                                pois_to_float, pois_to_freeze, y0, y_err
                            )
                            dcts.append({"name": poi1, "value": poi_values_pair[0]})
                            dcts.append({"name": poi2, "value": poi_values_pair[1]})
                        y = get_mu_prediction(
                            dcts,
                            self.mus,
                            self.production_coeffs_dict,
                            self.decay_coeffs_dict,
                            linear_only=self.linear,
                            quadratic_only=self.quadratic,
                        )
                        chi = chi_square(y0, y, y_err)
                        chi_s.append(chi)
                    chi_s = np.array(chi_s)
                    chi_s -= chi_s.min()
                    result[f"{poi1}_{poi2}"] = {
                        "values1": poi_values_pairs[:, 0],
                        "values2": poi_values_pairs[:, 1],
                        "chi_square": chi_s,
                    }

        return result

    def print_scaling_equations(self):
        pois = list(self.pois_dict.keys())
        predictions_dct = get_mu_prediction_string(
            pois,
            self.mus,
            self.production_coeffs_dict,
            self.decay_coeffs_dict,
            linear_only=self.linear,
            quadratic_only=self.quadratic,
        )
        print(json.dumps(predictions_dct, indent=4))


def main(args):
    if args.debug:
        logger = setup_logging(level="DEBUG")
    else:
        logger = setup_logging(level="INFO")
    os.makedirs(args.output_dir, exist_ok=True)

    mus_dct_of_dcts = {}
    for channel in args.channels:
        with open(mus_paths[channel], "r") as f:
            mus_dct_of_dcts[channel] = json.load(f)
    logger.debug(f"Loaded mus {mus_dct_of_dcts}")

    corr_matrices_obs_dct = {}
    for channel in args.channels:
        with open(corr_matrices_observed[channel], "r") as f:
            dct = json.load(f)
            list_of_lists = []
            for key, value in dct.items():
                list_of_lists.append(list(value.values()))
            corr_matrices_obs_dct[channel] = np.array(list_of_lists)
    logger.debug(f"Loaded corr_matrices_obs {corr_matrices_obs_dct}")

    corr_matrices_exp_dct = {}
    for channel in args.channels:
        with open(corr_matrices_expected[channel], "r") as f:
            dct = json.load(f)
            list_of_lists = []
            for key, value in dct.items():
                list_of_lists.append(list(value.values()))
            corr_matrices_exp_dct[channel] = np.array(list_of_lists)
    logger.debug(f"Loaded corr_matrices_exp {corr_matrices_exp_dct}")

    with open(args.submodel_yaml) as f:
        submodel_dict = yaml.load(f)
    submodel_name = args.submodel_yaml.split("/")[-1].split(".")[0]
    decays_dct, production_dct_of_dcts, edges_dct = refactor_predictions_multichannel(
        args.prediction_dir, args.channels
    )
    logger.debug(f"production_dct: {production_dct_of_dcts}")
    logger.debug(f"production_dct keys: {list(production_dct_of_dcts.keys())}")

    # Now refactor in order to make everything work in EFTFitter
    mus_dict = {}
    for channel in args.channels:
        dct = mus_dct_of_dcts[channel]
        for poi in dct:
            mus_dict[f"{poi}_{channel}"] = dct[poi]
    corr_mat_obs = block_diag(
        [corr_matrices_obs_dct[channel] for channel in args.channels]
    ).toarray()
    corr_mat_exp = block_diag(
        [corr_matrices_exp_dct[channel] for channel in args.channels]
    ).toarray()
    production_dct = {}
    for channel in args.channels:
        dct = production_dct_of_dcts[channel]
        for poi in dct:
            production_dct[f"{poi}_{channel}"] = dct[poi]
    logger.debug("After refactoring")
    logger.debug(f"mus_dict: {mus_dict}")
    logger.debug(f"corr_mat_obs shape: {corr_mat_obs.shape}")
    logger.debug(f"corr_mat_exp shape: {corr_mat_exp.shape}")
    logger.debug(f"production_dct: {production_dct}")

    fitter = EFTFitter(
        mus_dict,
        corr_mat_obs,
        corr_mat_exp,
        submodel_dict,
        production_dct,
        decays_dct,
        args.linear,
        args.quadratic,
    )

    logger.info("First printing scaling equations")
    fitter.print_scaling_equations()

    results = {}
    results["expected_fixed"] = fitter.scan(
        how="fixed", expected=True, skip2d=args.skip2D
    )
    results["expected_profiled"] = fitter.scan(
        how="profiled", expected=True, skip2d=args.skip2D
    )
    # results["observed_fixed"] = fitter.scan(how="fixed", expected=False)
    # results["observed_profiled"] = fitter.scan(how="profiled", expected=False)
    logger.debug(f"results: {results}")

    # plot
    oned_styles = {
        "line": {
            "expected_fixed": {"color": "blue", "linestyle": "solid"},
            "expected_profiled": {"color": "green", "linestyle": "solid"},
            "observed_fixed": {"color": "red", "linestyle": "solid"},
            "observed_profiled": {"color": "orange", "linestyle": "solid"},
        },
        "points": {
            "expected_fixed": {
                "color": "blue",
                "marker": "o",
                "fillstyle": "none",
                "markersize": 5,
            },
            "expected_profiled": {
                "color": "green",
                "marker": "o",
                "fillstyle": "full",
                "markersize": 5,
            },
            "observed_fixed": {
                "color": "red",
                "marker": "o",
                "fillstyle": "none",
                "markersize": 5,
            },
            "observed_profiled": {
                "color": "orange",
                "marker": "o",
                "fillstyle": "full",
                "markersize": 5,
            },
        },
    }
    twod_styles = {
        "expected_fixed": {"color": "blue"},
        "expected_profiled": {"color": "green"},
        "observed_fixed": {"color": "red"},
        "observed_profiled": {"color": "orange"},
    }
    pois = list(submodel_dict.keys())
    for poi in pois:
        fig, ax = plt.subplots()
        for result_name, result in results.items():
            logger.info(f"Plotting for poi: {poi}, result_name: {result_name}")
            func = interpolate.interp1d(
                result[poi]["values"], result[poi]["chi_square"], kind="cubic"
            )
            x = np.linspace(
                result[poi]["values"].min(), result[poi]["values"].max(), 1000
            )
            y = func(x)
            ax.plot(x, y, linewidth=1, **oned_styles["line"][result_name])
            # plot dots
            ax.plot(
                result[poi]["values"],
                result[poi]["chi_square"],
                label=result_name,
                **oned_styles["points"][result_name],
            )
        ax.set_xlabel(poi)
        ax.set_ylabel("$\Delta\chi^2$")
        ax.set_ylim(-0.5, 8)
        ax.set_xlim(result[poi]["values"].min(), result[poi]["values"].max())
        # horizontal dashed line at 1 and 4
        ax.axhline(y=0, color="black", linestyle="dashed")
        ax.axhline(y=1, color="black", linestyle="dashed")
        ax.axhline(y=4, color="black", linestyle="dashed")
        ax.legend()
        logger.info(f"Saving plots for poi {poi}")
        fig.savefig(
            f"{args.output_dir}/{poi}_{''.join(args.channels)}_{args.suffix}.png"
        )
        fig.savefig(
            f"{args.output_dir}/{poi}_{''.join(args.channels)}_{args.suffix}.pdf"
        )
    if not args.skip2D:
        pois_pairs = list(combinations(pois, 2))
        for poi1, poi2 in pois_pairs:
            fig, ax = plt.subplots()
            for result_name, result in results.items():
                logger.info(
                    f"Plotting for poi: {poi1}, {poi2}, result_name: {result_name}"
                )
                try:
                    x, y = np.mgrid[
                        result[f"{poi1}_{poi2}"]["values1"]
                        .min() : result[f"{poi1}_{poi2}"]["values1"]
                        .max() : 100j,
                        result[f"{poi1}_{poi2}"]["values2"]
                        .min() : result[f"{poi1}_{poi2}"]["values2"]
                        .max() : 100j,
                    ]
                    z = griddata(
                        (
                            result[f"{poi1}_{poi2}"]["values1"],
                            result[f"{poi1}_{poi2}"]["values2"],
                        ),
                        result[f"{poi1}_{poi2}"]["chi_square"],
                        (x, y),
                        method="cubic",
                    )
                    cs = ax.contour(
                        x,
                        y,
                        z,
                        levels=[1.0, 4.0],
                        # levels=[0.5, 2.0],
                        linewidths=2,
                        linestyles=["solid", "dashed"],
                        colors=[
                            twod_styles[result_name]["color"],
                            twod_styles[result_name]["color"],
                        ],
                    )
                    levels = ["$1\sigma$", "$2\sigma$"]
                    for i, cl in enumerate(levels):
                        try:
                            cs.collections[i].set_label(f"{result_name} {cl}")
                        except IndexError:
                            logger.warning(
                                f"IndexError for {result_name} {cl}, i: {i}, len(cs.collections): {len(cs.collections)}"
                            )
                except KeyError:
                    pass
            ax.set_xlabel(poi1)
            ax.set_ylabel(poi2)
            ax.legend()
            logger.info(f"Saving plots for pois {poi1} vs {poi2}")
            fig.savefig(
                f"{args.output_dir}/{poi1}-{poi2}_{''.join(args.channels)}_{args.suffix}.png"
            )
            fig.savefig(
                f"{args.output_dir}/{poi1}-{poi2}_{''.join(args.channels)}_{args.suffix}.pdf"
            )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
