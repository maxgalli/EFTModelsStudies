import argparse
import json
import yaml
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as itr
import mplhep as hep

hep.style.use("CMS")


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")

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


def mu(pois, coeff_prod, coeff_decay, coeff_tot):
    """
    pois: numpy array of pois to be tested
    others can be lists
    """
    prod = 1 + pois * coeff_prod[0] + pois ** 2 * coeff_prod[1]
    decay = 1 + pois * coeff_decay[0] + pois ** 2 * coeff_decay[1]
    tot = 1 + pois * coeff_tot[0] + pois ** 2 * coeff_tot[1]

    return prod * (decay / tot)


def get_coeffs(poi, pois_range, production_dct, decays_dct):
    dec = "gamgam"
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


def plot_spectrum():
    pass


def main(args):
    pois_file = f"{args.input_dir}/pois.yml"
    pois_dct = yaml.safe_load(open(pois_file))
    pois = list(pois_dct.keys())
    decays_file = f"{args.input_dir}/decay.json"
    with open(decays_file, "r") as f:
        decays_dct = json.load(f)
    production_file = f"{args.input_dir}/differentials/hgg/ggH_SMEFTatNLO_pt_gg.json"
    with open(production_file, "r") as f:
        production_dct = json.load(f)
    dict_keys = list(production_dct.keys())
    sorted_keys = sorted(dict_keys, key=lambda x: float(x))
    production_dct = {k: production_dct[k] for k in sorted_keys}
    edges = [float(k) for k in sorted_keys] + [10000.0]

    bin_names = []
    for l, r in zip(sorted_keys[:-1], sorted_keys[1:]):
        ln = l.replace(".", "p")
        rn = r.replace(".", "p")
        bin_names.append(f"{ln}_{rn}")
    bin_names.append("GT{}".format(sorted_keys[-1].replace(".", "p")))

    # plot one figure per POI with the parabolas for each bin
    print("Plotting parabolas...")
    for poi in pois:
        poi_range = np.linspace(pois_dct[poi]["min"], pois_dct[poi]["max"], 101)
        production_coeffs, decay_coeffs, tot_coeffs = get_coeffs(
            poi, poi_range, production_dct, decays_dct
        )
        fig, ax = plt.subplots()
        for n, k in enumerate(production_coeffs):
            ax.plot(
                poi_range,
                mu(poi_range, production_coeffs[k], decay_coeffs, tot_coeffs),
                label=bin_names[n],
            )
        ax.legend(loc="upper center", prop={"size": 10}, ncol=4)
        ax.set_xlabel(poi)
        ax.set_ylabel("$\mu$")
        # mplhep boilerplate
        hep.cms.label(loc=0, data=True, llabel="Work in Progress", lumi=138, ax=ax)
        fig.savefig(f"{args.output_dir}/{poi}.pdf")
        fig.savefig(f"{args.output_dir}/{poi}.png")

    # plot one figure per POI with the spectrum
    print("Plotting spectrum...")
    mass = 125.38
    weights = [1.0, 2.3, 1.0]
    hgg_br = 0.0023
    prediction_file = "predictions/theoryPred_Pt_18_fullPS.pkl"

    with open(prediction_file, "rb") as f:
        obs = pkl.load(f)
        sm_prediction = get_prediction(obs, mass, weights)
        sm_prediction = sm_prediction / hgg_br

    for poi in pois:
        fig, ax = plt.subplots()
        poi_range = np.linspace(pois_dct[poi]["min"], pois_dct[poi]["max"], 5)
        production_coeffs, decay_coeffs, tot_coeffs = get_coeffs(
            poi, poi_range, production_dct, decays_dct
        )
        fig, ax = plt.subplots()
        mus_stack = []
        for n, k in enumerate(production_coeffs):
            mus = mu(poi_range, production_coeffs[k], decay_coeffs, tot_coeffs)
            mus_stack.append(mus)
        mus_stack = np.vstack(mus_stack)
        for n, i in enumerate(poi_range):
            hist = sm_prediction * mus_stack[:, n]
            ax.stairs(hist, range(len(edges)), label=f"{poi} = {i}")
        ax.legend()
        ax.set_yscale("log")
        ax.set_xlabel("$p_{T}$")
        ax.set_ylabel("$\sigma_{SM} \cdot \mu$")
        ax.set_xticks(range(len(edges)))
        ax.set_xticklabels(edges)
        ax.tick_params(axis="x", which="major", labelsize=10)
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.set_xlim(0, len(edges) - 1)
        hep.cms.label(loc=0, data=True, llabel="Work in Progress", lumi=138, ax=ax)
        fig.savefig(f"{args.output_dir}/{poi}_spectrum.pdf")
        fig.savefig(f"{args.output_dir}/{poi}_spectrum.png")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
