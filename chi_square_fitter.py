import argparse
from cmath import log
import os
import json
import yaml
import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize as scipy_minimize
from scipy.sparse import block_diag
from scipy.optimize import curve_fit
import logging
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from iminuit import Minuit

matplotlib.use("AGG")
import mplhep as hep
from itertools import combinations, product
from scipy import interpolate
from scipy.interpolate import griddata
import dask
from dask.distributed import Client, LocalCluster, get_client, wait
from dask_jobqueue import SLURMCluster

hep.style.use("CMS")

from utils import (
    refactor_predictions_multichannel,
    ggH_production_files,
    max_to_matt,
    mus_paths,
    corr_matrices_observed,
    corr_matrices_expected,
    print_corr_matrix,
)
from differential_combination_postprocess.utils import setup_logging


import warnings
from iminuit.util import IMinuitWarning

warnings.filterwarnings(action="ignore", category=IMinuitWarning)


def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--prediction-dir", type=str, required=True, help="")

    parser.add_argument("--submodel-yaml", type=str, required=True, help="")

    parser.add_argument(
        "--channels", nargs="+", type=str, required=True, default=[], help=""
    )

    parser.add_argument("--output-dir", type=str, required=True, help="")

    parser.add_argument("--suffix", type=str, required=False, default="", help="")

    parser.add_argument(
        "--fit-model",
        choices=["full", "linear", "linearised"],
        default="full",
        help="What type of model to fit",
    )

    parser.add_argument(
        "--skip2D", action="store_true", help="Skip 2D plots of the fit"
    )

    parser.add_argument(
        "--multiprocess",
        action="store_true",
        help="Use multiprocessing to speed up the fit",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use dask jobqueue to speed up the fit",
    )

    parser.add_argument(
        "--with-observed",
        action="store_true",
        help="Use observed data along with expected",
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


def extract_coeffs(pois, dct, fit_model="full"):
    """
    pois is a list of dicts with shape {'name': name, 'value': value} in case of print_coeffs=False, otherwise it is just a list of strings
    return 1D array with lenght len(self.mus)
    """
    if fit_model == "linearised":
        mu = 0
    else:
        mu = 1
    for poi in pois:
        # logging.debug(f"Extract coeff for {poi['name']}")
        # linear
        try:
            mu += dct[f"A_{poi['name']}"] * poi["value"]
            # logging.debug("Linear term: {}".format(dct[f"A_{poi['name']}"]))
        except KeyError:
            pass
        # quadratic
        if fit_model == "full":
            try:
                mu += dct[f"B_{poi['name']}_2"] * poi["value"] ** 2
                # logging.debug("Quadratic term: {}".format(dct[f"B_{poi['name']}_2"]))
            except KeyError:
                pass
    # mixed
    if fit_model == "full":
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


def extract_coeffs_string(pois, dct, fit_model="full"):
    """
    pois is a list of dicts with shape {'name': name, 'value': value} in case of print_coeffs=False, otherwise it is just a list of strings
    return 1D array with lenght len(self.mus)
    """
    if fit_model == "linearised":
        mu = ""
    else:
        mu = "1"
    for poi in pois:
        # logging.debug(f"Extract coeff for {poi['name']}")
        # linear
        try:
            mu += " + {} * {}".format(dct[f"A_{poi}"], poi)
        except KeyError:
            pass
        # quadratic
        if fit_model == "full":
            try:
                mu += " + {} * {}^2".format(dct[f"B_{poi}_2"], poi)
            except KeyError:
                pass
    # mixed
    if fit_model == "full":
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


def get_mu_prediction(pois, mus, production_dct, decays_dct, fit_model="full"):
    """
    pois is a list of dicts with shape {'name': name, 'value': value}
    """
    pred_mus = []
    for mu in mus:
        channel = mu.split("_")[-1]
        channel = max_to_matt[channel]
        prod_term = extract_coeffs(pois, production_dct[mu], fit_model=fit_model)
        br_num = extract_coeffs(pois, decays_dct[channel], fit_model=fit_model)
        br_den = extract_coeffs(pois, decays_dct["tot"], fit_model=fit_model)
        if fit_model == "linearised":
            pred_mus.append(1 + prod_term + br_num - br_den)
        else:
            pred_mus.append(prod_term * br_num / br_den)

    return np.array(pred_mus)


def get_mu_prediction_string(pois, mus, production_dct, decays_dct, fit_model="full"):
    pred_mus = {}
    for mu in mus:
        channel = mu.split("_")[-1]
        channel = max_to_matt[channel]
        prod_term = extract_coeffs_string(pois, production_dct[mu], fit_model=fit_model)
        br_num = extract_coeffs_string(pois, decays_dct[channel], fit_model=fit_model)
        br_den = extract_coeffs_string(pois, decays_dct["tot"], fit_model=fit_model)
        if fit_model == "linearised":
            pred_mus[mu] = "1 + {} + {} - ({})".format(prod_term, br_num, br_den)
        else:
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
        fit_model="full",
    ):
        """
        """
        self.mus = list(mus_dict.keys())
        self.pois_dict = submodel_dict
        self.production_coeffs_dict = production_coeffs_dict
        self.decay_coeffs_dict = decay_coeffs_dict
        self.fit_model = fit_model

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
        logging.debug(f"Covariance matrix observed: {self.y_err}")
        logging.debug(f"Covariance matrix expected: {self.y_err_asimov}")

    def minimize(self, params_to_float, params_to_freeze, y0, y_err, return_corr=False):
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

        if return_corr:
            m = Minuit(to_minimize, [v["val"] for v in params_to_float.values()])
            m.limits = [(v["min"], v["max"]) for v in params_to_float.values()]
            m.migrad()
            m.hesse()

            return m.covariance.correlation()
        else:
            # res = scipy_minimize(
            #    fun=to_minimize,
            #    x0=[v["val"] for v in params_to_float.values()],
            #    method="TNC",
            #    # method="SLSQP",
            #    bounds=[(v["min"], v["max"]) for v in params_to_float.values()],
            # )

            # ret = [
            #    {"name": name, "value": value}
            #    for name, value in zip(params_to_float, res.x)
            # ]

            m = Minuit(to_minimize, [v["val"] for v in params_to_float.values()])
            m.limits = [(v["min"], v["max"]) for v in params_to_float.values()]
            m.migrad()
            m.hesse()
            # print("Initial values: ", [v["val"] for v in params_to_float.values()])
            # print(m.params)
            # print(m.valid)

            ret = [
                {"name": name, "value": par.value}
                for name, par in zip(params_to_float, m.params)
            ]
            return ret

    def compute_correlation_matrix(self, expected=False):
        logging.info("Computing covariance matrix")
        pois_to_float = self.pois_dict
        pois_to_freeze = {}
        y0 = self.y0 if not expected else self.y0_asimov
        y_err = self.y_err if not expected else self.y_err_asimov
        corr = self.minimize(pois_to_float, pois_to_freeze, y0, y_err, return_corr=True)

        return corr

    def scan(self, how, expected=False, points=100, multiprocess=False):
        logging.info(f"Scanning {how} with expected={expected}")
        y0 = self.y0 if not expected else self.y0_asimov
        y_err = self.y_err if not expected else self.y_err_asimov
        result = {}

        if multiprocess:
            client = get_client()
        # 1D scans
        for poi_name, poi_info in self.pois_dict.items():
            logging.info(f"Scanning {poi_name}")
            poi_values = np.linspace(poi_info["min"], poi_info["max"], points)
            chi_s = []
            if how == "profiled":
                pois_to_float = {
                    pn: pv for pn, pv in self.pois_dict.items() if pn != poi_name
                }

            def get_chi_to_parallelize(how, i, poi_value):
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
                    fit_model=self.fit_model,
                )
                chi = chi_square(y0, y, y_err)

                return chi

            for i, poi_value in enumerate(poi_values):
                if multiprocess:
                    chi_s.append(
                        client.submit(get_chi_to_parallelize, how, i, poi_value)
                    )
                else:
                    chi_s.append(get_chi_to_parallelize(how, i, poi_value))

            result[poi_name] = {"values": poi_values, "chi_square": chi_s}

        return result

    def scan2D(self, how, expected=False, points=55, multiprocess=False):
        logging.info(f"Scanning {how} with expected={expected}")
        y0 = self.y0 if not expected else self.y0_asimov
        y_err = self.y_err if not expected else self.y_err_asimov
        result = {}

        if multiprocess:
            client = get_client()

        if how == "profiled" and len(self.pois_dict) < 3:
            logging.info("Skipping 2D scan")
            return result

        if len(self.pois_dict) > 1:
            pois_combinations = list(combinations(self.pois_dict.keys(), 2))
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

                def get_chi_to_parallelize_2D(how, poi_values_pair):
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
                        dcts = self.minimize(pois_to_float, pois_to_freeze, y0, y_err)
                        dcts.append({"name": poi1, "value": poi_values_pair[0]})
                        dcts.append({"name": poi2, "value": poi_values_pair[1]})
                    y = get_mu_prediction(
                        dcts,
                        self.mus,
                        self.production_coeffs_dict,
                        self.decay_coeffs_dict,
                        fit_model=self.fit_model,
                    )
                    chi = chi_square(y0, y, y_err)
                    return chi

                for poi_values_pair in poi_values_pairs:
                    if multiprocess:
                        chi_s.append(
                            client.submit(
                                get_chi_to_parallelize_2D, how, poi_values_pair
                            )
                        )
                    else:
                        chi_s.append(get_chi_to_parallelize_2D(how, poi_values_pair))

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
            fit_model=self.fit_model,
        )
        print(json.dumps(predictions_dct, indent=4))


def main(args):
    if args.debug:
        logger = setup_logging(level="DEBUG")
    else:
        logger = setup_logging(level="INFO")
    # os.makedirs(args.output_dir, exist_ok=True)

    fit_model = args.fit_model

    multiple_workers = False
    if args.multiprocess:
        multiple_workers = True
        logger.info("Using multiprocessing")
        cluster = LocalCluster()
        client = Client(cluster)
        cluster.scale(18)
        logger.info("Waiting for workers to be ready")
        client.wait_for_workers(10)
    elif args.distributed:
        multiple_workers = True
        logger.info("Using distributed")
        cluster = SLURMCluster(
            # queue="short",
            queue="standard",
            walltime="10:00:00",
            cores=1,
            processes=1,
            memory="6G",
            log_directory="slurm_logs",
            local_directory="slurm_logs",
        )
        cluster.adapt(minimum=10, maximum=50)
        client = Client(cluster)
        logger.info(cluster.job_script())
        logger.info("Waiting for workers to be ready")
        client.wait_for_workers(10)

    mus_dct_of_dcts = {}
    for channel in args.channels:
        with open(mus_paths[channel], "r") as f:
            mus_dct_of_dcts[channel] = json.load(f)
        # hbbvbf does not have prediction in last bin, so we remove it
        if channel == "hbbvbf":
            mus_dct_of_dcts[channel].pop("r_smH_PTH_800_1200")
    logger.debug(f"Loaded mus {mus_dct_of_dcts}")

    corr_matrices_obs_dct = {}
    for channel in args.channels:
        with open(corr_matrices_observed[channel], "r") as f:
            dct = json.load(f)
            list_of_lists = []
            for key, value in dct.items():
                list_of_lists.append(list(value.values()))
            corr_matrices_obs_dct[channel] = np.array(list_of_lists)
        # remove last row and column in hbbvbf
        if channel == "hbbvbf":
            corr_matrices_obs_dct[channel] = corr_matrices_obs_dct[channel][:-1, :-1]
    logger.debug(f"Loaded corr_matrices_obs {corr_matrices_obs_dct}")

    corr_matrices_exp_dct = {}
    for channel in args.channels:
        with open(corr_matrices_expected[channel], "r") as f:
            dct = json.load(f)
            list_of_lists = []
            for key, value in dct.items():
                list_of_lists.append(list(value.values()))
            corr_matrices_exp_dct[channel] = np.array(list_of_lists)
        # remove last row and column in hbbvbf
        if channel == "hbbvbf":
            corr_matrices_exp_dct[channel] = corr_matrices_exp_dct[channel][:-1, :-1]
    logger.debug(f"Loaded corr_matrices_exp {corr_matrices_exp_dct}")

    with open(args.submodel_yaml) as f:
        submodel_dict = yaml.load(f)
    submodel_name = args.submodel_yaml.split("/")[-1].split(".")[0]
    decays_dct, production_dct_of_dcts, edges_dct = refactor_predictions_multichannel(
        args.prediction_dir, args.channels
    )
    logger.debug(f"production_dct: {production_dct_of_dcts}")
    logger.debug(f"production_dct keys: {list(production_dct_of_dcts.keys())}")
    output_dir = os.path.join(
        args.output_dir, args.prediction_dir.split("/")[-1] + "-" + submodel_name
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

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
        fit_model,
    )

    logger.info("First printing scaling equations")
    fitter.print_scaling_equations()

    results = {}
    results["expected_fixed"] = fitter.scan(how="fixed", expected=True)
    if not args.skip2D:
        results["expected_fixed"] = {
            **results["expected_fixed"],
            **fitter.scan2D(how="fixed", expected=True),
        }
    results["expected_profiled"] = fitter.scan(
        how="profiled", expected=True, points=150, multiprocess=multiple_workers
    )
    if not args.skip2D:
        results["expected_profiled"] = {
            **results["expected_profiled"],
            **fitter.scan2D(
                how="profiled", expected=True, multiprocess=multiple_workers
            ),
        }
    if args.with_observed:
        results["observed_fixed"] = fitter.scan(how="fixed", expected=False)
        if not args.skip2D:
            results["observed_fixed"] = {
                **results["observed_fixed"],
                **fitter.scan2D(how="fixed", expected=False),
            }
        results["observed_profiled"] = fitter.scan(how="profiled", expected=False)
        if not args.skip2D:
            results["observed_profiled"] = {
                **results["observed_profiled"],
                **fitter.scan2D(how="profiled", expected=False),
            }
    if multiple_workers:
        results = client.gather(results)
    for k, v in results.items():
        for kk, vv in v.items():
            vv["chi_square"] = np.array(vv["chi_square"])
            vv["chi_square"] -= np.min(vv["chi_square"])
    logger.debug(f"results: {results}")

    if multiple_workers:
        client.close()
        cluster.close()

    # compute correlation matrix
    postfit_corr_matrix_exp = fitter.compute_correlation_matrix(expected=True)
    if args.with_observed:
        postfit_corr_matrix_obs = fitter.compute_correlation_matrix(expected=False)

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
                # "linestyle": "none",
            },
            "expected_profiled": {
                "color": "green",
                "marker": "o",
                "fillstyle": "full",
                "markersize": 5,
                # "linestyle": "none",
            },
            "observed_fixed": {
                "color": "red",
                "marker": "o",
                "fillstyle": "none",
                "markersize": 5,
                # "linestyle": "none",
            },
            "observed_profiled": {
                "color": "orange",
                "marker": "o",
                "fillstyle": "full",
                "markersize": 5,
                # "linestyle": "none",
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
            x = np.linspace(
                result[poi]["values"].min(), result[poi]["values"].max(), 1000
            )
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
            f"{output_dir}/{poi}_{''.join(args.channels)}_{fit_model}_{args.suffix}.png"
        )
        fig.savefig(
            f"{output_dir}/{poi}_{''.join(args.channels)}_{fit_model}_{args.suffix}.pdf"
        )
        plt.close(fig)
    # plot matrices
    print(postfit_corr_matrix_exp)
    print_corr_matrix(
        postfit_corr_matrix_exp,
        pois,
        f"{output_dir}/corr_matrix_postfit_exp_{''.join(args.channels)}_{fit_model}_{args.suffix}.png",
    )
    if args.with_observed:
        print(postfit_corr_matrix_obs)
        print_corr_matrix(
            postfit_corr_matrix_obs,
            pois,
            f"{output_dir}/corr_matrix_postfit_obs_{''.join(args.channels)}_{fit_model}_{args.suffix}.png",
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
                        levels=[2.295748928898636, 6.180074306244173],
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
                f"{output_dir}/{poi1}-{poi2}_{''.join(args.channels)}_{fit_model}_{args.suffix}.png"
            )
            fig.savefig(
                f"{output_dir}/{poi1}-{poi2}_{''.join(args.channels)}_{fit_model}_{args.suffix}.pdf"
            )
            plt.close(fig)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
