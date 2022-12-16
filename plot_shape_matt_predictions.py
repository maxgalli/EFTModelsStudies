import argparse
import json
from math import prod
from turtle import left
import yaml
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as itr
import mplhep as hep
import itertools
from itertools import combinations

from utils import (
    refactor_predictions,
    refactor_predictions_multichannel,
    sm_prediction_files,
    max_to_matt,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--prediction-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--config-file", type=str, default="config.yaml")
    parser.add_argument(
        "--channels", nargs="+", type=str, required=True, default=[], help=""
    )
    parser.add_argument(
        "--skip-spectra", action="store_true", help="Skip plotting spectra"
    )
    parser.add_argument(
        "--fit-model",
        choices=["full", "linear", "linearised"],
        default="full",
        help="What type of model to fit",
    )

    return parser.parse_args()


def get_prediction(arr, mass, weights=None, interPRepl=None, massRepl=None):
    # Defined by Thomas in https://github.com/threiten/HiggsAnalysis-CombinedLimit/blob/d5d9ef377a7c69a8d4eaa366b47e7c81931e71d9/test/plotBinnedSigStr.py#L236
    # To be used with weights [1, 2.3, 1]
    nBins = len(arr)
    masses = [120.0, 125.0, 130.0]
    if weights is None:
        weights = np.ones(arr.shape[1])
    splines = []
    if arr.shape[1] == 1:
        if interPRepl is None or massRepl is None:
            raise ValueError(
                "If only one masspoint is given, interPRepl and massRepl must be provided!"
            )
        for i in range(nBins):
            splines.append(
                itr.UnivariateSpline(masses, interPRepl[i, :], w=weights, k=2)
            )

        return np.array(
            [
                splines[i](mass) - interPRepl[i, masses.index(massRepl)] + arr[i, 0]
                for i in range(nBins)
            ]
        )

    for i in range(nBins):
        splines.append(itr.UnivariateSpline(masses, arr[i, :], w=weights, k=2))

    return np.array([splines[i](mass) for i in range(nBins)])


def mu(pois, coeff_prod, coeff_decay, coeff_tot, fit_model="full"):
    """
    pois: numpy array of pois to be tested
    others can be lists
    """
    if fit_model == "linearised":
        return 1 + pois * (coeff_prod[0] + coeff_decay[0] - coeff_tot[0])
    else:
        if fit_model == "linear":
            prod = 1 + pois * coeff_prod[0]
            decay = 1 + pois * coeff_decay[0]
            tot = 1 + pois * coeff_tot[0]
        else:
            prod = 1 + pois * coeff_prod[0] + pois ** 2 * coeff_prod[1]
            decay = 1 + pois * coeff_decay[0] + pois ** 2 * coeff_decay[1]
            tot = 1 + pois * coeff_tot[0] + pois ** 2 * coeff_tot[1]

        return prod * (decay / tot)


def get_coeffs(poi, production_dct, decays_dct, channel):
    dec = max_to_matt[channel]
    tot = "tot"
    decay_coeffs = [
        decays_dct[dec][f"A_{poi}"] if f"A_{poi}" in decays_dct[dec] else 0.0,
        decays_dct[dec][f"B_{poi}_2"] if f"B_{poi}_2" in decays_dct[dec] else 0.0,
    ]
    tot_coeffs = [
        decays_dct[tot][f"A_{poi}"] if f"A_{poi}" in decays_dct[tot] else 0.0,
        decays_dct[tot][f"B_{poi}_2"] if f"B_{poi}_2" in decays_dct[tot] else 0.0,
    ]
    production_coeffs = {}
    for k in production_dct:
        prod_coeff = [
            production_dct[k][f"A_{poi}"] if f"A_{poi}" in production_dct[k] else 0.0,
            production_dct[k][f"B_{poi}_2"]
            if f"B_{poi}_2" in production_dct[k]
            else 0.0,
        ]
        production_coeffs[k] = prod_coeff

    return production_coeffs, decay_coeffs, tot_coeffs


def main(args):
    pois_file = args.config_file
    pois_dct = yaml.safe_load(open(pois_file))
    pois = list(pois_dct.keys())

    decays_dct, production_dct, edges = refactor_predictions_multichannel(
        args.prediction_dir, args.channels
    )
    print(production_dct)

    bin_names = {}
    for k, v in production_dct.items():
        bin_names[k] = [s.replace("r_smH_PTH_", "") for s in list(v.keys())]

    hep.style.use("CMS")
    # plot one figure per POI with the parabolas for each bin
    print(f"Plotting parabolas with fit_model = {args.fit_model}...")
    channel_linestyles = {
        "hgg": "-",
        "hzz": "--",
        "hww": "-.",
        "htt": ":",
        "hbbvbf": ":",
        "httboost": "-.",
    }
    # shades of different colors for each channel
    channel_colors = {}
    try:
        channel_colors["hgg"] = [
            plt.get_cmap("Reds")(i) for i in np.linspace(0.1, 1, len(bin_names["hgg"]))
        ]
    except KeyError:
        pass
    try:
        channel_colors["hzz"] = [
            plt.get_cmap("Blues")(i) for i in np.linspace(0.1, 1, len(bin_names["hzz"]))
        ]
    except KeyError:
        pass
    try:
        channel_colors["hww"] = [
            plt.get_cmap("Purples")(i)
            for i in np.linspace(0.1, 1, len(bin_names["hww"]))
        ]
    except KeyError:
        pass
    try:
        channel_colors["htt"] = [
            plt.get_cmap("Greens")(i)
            for i in np.linspace(0.1, 1, len(bin_names["htt"]))
        ]
    except KeyError:
        pass
    try:
        channel_colors["hbbvbf"] = [
            plt.get_cmap("Oranges")(i)
            for i in np.linspace(0.1, 1, len(bin_names["hbbvbf"]))
        ]
    except KeyError:
        pass
    try:
        channel_colors["httboost"] = [
            plt.get_cmap("Greys")(i)
            for i in np.linspace(0.1, 1, len(bin_names["httboost"]))
        ]
    except KeyError:
        pass

    for poi in pois:
        print(f"Plotting parabolas for {poi}")
        poi_range = np.linspace(pois_dct[poi]["min"], pois_dct[poi]["max"], 101)
        fig, ax = plt.subplots()
        for channel in args.channels:
            production_coeffs, decay_coeffs, tot_coeffs = get_coeffs(
                poi, production_dct[channel], decays_dct, channel
            )
            for n, k in enumerate(production_coeffs):
                ax.plot(
                    poi_range,
                    mu(
                        poi_range,
                        production_coeffs[k],
                        decay_coeffs,
                        tot_coeffs,
                        fit_model=args.fit_model,
                    ),
                    label=f"{channel} {bin_names[channel][n]}",
                    linestyle=channel_linestyles[channel],
                    color=channel_colors[channel][n],
                )
        ax.legend(loc="upper center", prop={"size": 10}, ncol=4)
        ax.set_xlabel(poi)
        ax.set_ylabel("$\mu$")
        ax.plot(0, 1, marker="P", color="grey", markersize=8)
        ax.text(
            0.05,
            0.95,
            "$\sigma_{SM}$",
            va="center",
            ha="center",
            fontsize=10,
            color="grey",
        )

        # mplhep boilerplate
        hep.cms.label(loc=0, data=True, llabel="Work in Progress", lumi=138, ax=ax)
        fig.savefig(
            f"{args.output_dir}/{poi}_{'-'.join(args.channels)}_{args.fit_model}_1D.pdf"
        )
        fig.savefig(
            f"{args.output_dir}/{poi}_{'-'.join(args.channels)}_{args.fit_model}_1D.png"
        )
        plt.close(fig)

    if not args.skip_spectra:
        for channel in args.channels:
            # plot one figure per POI with the spectrum
            print(f"Plotting spectra for channel {channel}...")
            mass = 125.38
            weights = [1.0, 2.3, 1.0]
            hgg_br = 0.0023
            prediction_file = sm_prediction_files[channel]

            with open(prediction_file, "rb") as f:
                obs = pkl.load(f)
                sm_prediction = get_prediction(obs, mass, weights)
                sm_prediction = sm_prediction / hgg_br

            for poi in pois:
                fig, ax = plt.subplots()
                poi_range = np.linspace(pois_dct[poi]["min"], pois_dct[poi]["max"], 4)
                production_coeffs, decay_coeffs, tot_coeffs = get_coeffs(
                    poi, production_dct[channel], decays_dct, channel
                )
                fig, (ax, rax) = plt.subplots(
                    nrows=2, ncols=1, gridspec_kw={"height_ratios": (3, 1)}, sharex=True
                )
                mus_stack = []
                mus_ref_stack = []
                for n, k in enumerate(production_coeffs):
                    mus = mu(poi_range, production_coeffs[k], decay_coeffs, tot_coeffs)
                    mus_ref = mu(
                        np.array(0), production_coeffs[k], decay_coeffs, tot_coeffs
                    )
                    mus_stack.append(mus)
                    mus_ref_stack.append(mus_ref)
                mus_stack = np.vstack(mus_stack)
                mus_ref_stack = np.vstack(mus_ref_stack)
                hist_ref = sm_prediction * mus_ref_stack[:, 0]
                ax.stairs(
                    hist_ref, range(len(edges[channel])), label="SM", color="black"
                )
                rax.stairs(
                    hist_ref / hist_ref,
                    range(len(edges[channel])),
                    label="SM",
                    color="black",
                )
                for n, i in enumerate(poi_range):
                    hist = sm_prediction * mus_stack[:, n]
                    hist_ratio = hist / hist_ref
                    print(f"Yields for {poi} = {i}: {list(hist)}")
                    print(f"Mus for {poi} = {i}: {list(mus_stack[:, n])}")
                    ax.stairs(
                        hist,
                        range(len(edges[channel])),
                        label="{} = {:.2f}".format(poi, i),
                    )
                    rax.stairs(
                        hist_ratio,
                        range(len(edges[channel])),
                        label="{} = {:.2f}".format(poi, i),
                    )
                ax.legend(loc="lower left", prop={"size": 10}, ncol=1)
                ax.set_yscale("log")
                rax.set_xlabel("$p_{T}$")
                ax.set_ylabel("$\sigma_{SM} \cdot \mu$")
                rax.set_ylabel("SMEFT/SM")
                rax.set_xticks(range(len(edges[channel])))
                rax.set_xticklabels(edges[channel])
                ax.tick_params(axis="x", which="major", labelsize=10)
                ax.tick_params(axis="x", which="minor", bottom=False, top=False)
                rax.tick_params(axis="x", which="major", labelsize=10)
                rax.tick_params(axis="x", which="minor", bottom=False, top=False)
                rax.set_xlim(0, len(edges[channel]) - 1)
                hep.cms.label(
                    loc=0, data=True, llabel="Work in Progress", lumi=138, ax=ax
                )
                fig.savefig(f"{args.output_dir}/{poi}_{channel}_spectrum.pdf")
                fig.savefig(f"{args.output_dir}/{poi}_{channel}_spectrum.png")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
