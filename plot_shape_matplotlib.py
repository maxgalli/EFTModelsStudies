import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
from copy import deepcopy
import itertools

colors = ("red", "orange", "green", "blue", "indigo", "purple", "brown", "cyan", "pink", "olive")



def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--production", type=str, required=True)
    parser.add_argument("--observable", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")

    return parser.parse_args()


def plot_histogram_with_error_bands(ax, edges, histogram, label=None, color=None, error_band=None):
    """
    """
    ax.stairs(
        histogram,
        edges,
        edgecolor=color,
        label=label
    )

    if error_band:
        ax.fill_between(
            error_band["x"],
            error_band["down"],
            error_band["up"],
            color=color,
            alpha=0.5,
        )

    return ax


def compute_deviation_histogram(nominal_histo, production_json, coeff_values):
    """ Compute histogram from EFT deviations
    production_json: json file output of EFT2Obs
    coeff_values: dictionary of coefficients for a certain configuration, e.g.
    {
        "cu": 0,
        "cd": 0,
        "cl": 0.1,
        "cww": 0,
    }

    bands is needed to construct the error bands using pyplot Rectanlges
    the structure is {"x": [], "down": [], "up": []} and it is used to fill
    plt.fill_between
    """
    add_bins = []
    add_errs = []
    for deviations in production_json["bins"]:
        to_add = 0
        to_add_err = 0
        for dev in deviations:
            val = dev[0]
            err = dev[1]
            pars = dev[2:]
            for par in pars:
                val *= coeff_values[par]
                err *= coeff_values[par]
            to_add += val
            to_add_err += err ** 2
        add_bins.append(to_add)
        add_errs.append(to_add_err)

    # print("Will add to {}".format(dev_hist))
    add_bins = np.array(add_bins) * nominal_histo
    # print("Bins to be added {}".format(add_bins))
    full_hist = nominal_histo + add_bins
    full_errs = np.sqrt(add_errs) * full_hist
    full_hist_nonegative = np.array([val if val > 0 else 0 for val in full_hist])
    bands = {}
    x = np.array(production_json["edges"])
    y = (np.ones_like(x).T * full_hist).T
    full_errs_mat = (np.ones_like(x).T * full_errs).T
    down = y - full_errs_mat
    up = y + full_errs_mat
    bands["x"] = x.flatten()
    bands["y"] = y.flatten()
    bands["down"] = down.flatten()
    bands["up"] = up.flatten()

    return full_hist_nonegative, bands


# Parabola plots (one for each parameter inside each bin)
def parabola(x, a, b):
    return a * x ** 2 + b * x


def main(args):
    n_tests = 6
    coeff_test_values = {
        "cu": {
            "min": -0.1,
            "max": 0.1,
        },
        "cd": {
            "min": -0.1,
            "max": 0.1,
        },
        "cl": {
            "min": -0.1,
            "max": 0.1,
        },
        "cww": {
            "min": -0.01,
            "max": 0.01,
        },
        "cb": {
            "min": -0.1,
            "max": 0.1,
        },
        "chw": {
            "min": -0.01,
            "max": 0.01,
        },
        "chb": {
            "min": -0.1,
            "max": 0.1,
        },
        "ca": {
            "min": -0.1,
            "max": 0.1,
        },
        "cg": {
            "min": -0.1,
            "max": 0.1,
        },
    }
    output_dir = args.output_dir
    
    args = parse_arguments()

    with open(args.production, "r") as f:
        production_json = json.load(f)

    # Nominal (SM) histogram
    edges = np.array(
        [edges[0] for edges in production_json["edges"]] + [production_json["edges"][-1][1]]
    )
    # print("Edges {}".format(edges))
    bin_widths = np.array([edges[1] - edges[0] for edges in production_json["edges"]])
    # print("Bin widths {}".format(bin_widths))
    areas = np.array(production_json["areas"])
    nominal_histo = areas / bin_widths
    parameters = production_json["parameters"]
    print("Nominal histogram {}".format(nominal_histo))
    print("Edges {}".format(edges))

    # Observable shape plots (one per Wilson coefficient)
    for parameter in parameters:
        fig, (main_ax, ratio_ax) = plt.subplots(
            nrows=2, 
            ncols=1, 
            sharex=True, 
            gridspec_kw={"height_ratios": [3, 1]}
        )

        # Nominal (SM) histogram
        main_ax = plot_histogram_with_error_bands(
            main_ax, edges, nominal_histo, label="SM", color="black")
        ratio_ax = plot_histogram_with_error_bands(
            ratio_ax, edges, nominal_histo/nominal_histo, label="SM", color="black")

        # Now create and plot one histogram per Wilson coefficient variation
        try:
            parameter_variations = np.linspace(
                coeff_test_values[parameter]["min"], 
                coeff_test_values[parameter]["max"], 
                n_tests
                )
        except KeyError:
            raise KeyError(f"No entry for {parameter} in {coeff_test_values}.\nImplement it and try again")

        colors_iterator = itertools.cycle(colors)
        for variation in parameter_variations:
            variation_color = next(colors_iterator)
            coeff_values = {k: 0 for k in parameters}
            coeff_values[parameter] = variation
            nominal_histo_copy = deepcopy(nominal_histo)
            dev_hist, error_bands = compute_deviation_histogram(nominal_histo_copy, production_json, coeff_values)

            main_ax = plot_histogram_with_error_bands(
                main_ax, edges, dev_hist, label="{}={:.3f}".format(parameter, variation), 
                color=variation_color, error_band=error_bands
                )
            ratio_error_bands = {}
            ratio_error_bands["x"] = error_bands["x"]
            ratio_error_bands["down"] = error_bands["down"] / error_bands["y"]
            ratio_error_bands["up"] = error_bands["up"] / error_bands["y"]
            ratio_ax = plot_histogram_with_error_bands(
                ratio_ax, edges, dev_hist/nominal_histo, label="{}={:.3f}".format(parameter, variation), 
                color=variation_color,
                error_band=ratio_error_bands
                )

        # mplhep boilerplate
        hep.cms.label(
            loc=0, 
            data=True, 
            llabel="Work in Progress", 
            lumi=35.9, 
            ax=main_ax
            )

        # Set limits and stuff
        main_ax.set_ylim(bottom=0)
        main_ax.legend(loc="upper right")
        main_ax.set_xlim(left=edges[0], right=edges[-1])
        ratio_ax.set_ylim(bottom=0, top=4)

        fig.savefig("{}/{}_{}.pdf".format(output_dir, args.observable, parameter))
        fig.savefig("{}/{}_{}.png".format(output_dir, args.observable, parameter))

    # Parabolas
    for parameter_idx, parameter in enumerate(parameters):
        fig, ax = plt.subplots()
        colors_iterator = itertools.cycle(colors)
        for bin_idx, bin in enumerate(nominal_histo):
            deviations = production_json["bins"][bin_idx]
            edges = production_json["edges"][bin_idx]
            range_str = f"{edges[0]}-{edges[1]}"
            color = next(colors_iterator)
            # Build parabolic plot
            for dev in deviations:
                if len(dev) == 3 and dev[2] == parameter:
                    b = dev[0]
                if len(dev) == 4 and dev[2] == parameter and dev[3] == parameter:
                    a = abs(dev[0])
                try:
                    x = np.linspace(coeff_test_values[parameter]["min"], coeff_test_values[parameter]["max"], 1000)
                    y = parabola(x, a, b)
                    ax.plot(x, y, color=color, label=f"{args.observable}_{range_str}")
                    del a, b
                except NameError:
                    continue

        ax.set_xlabel(parameter)
        ax.set_xlim(
            coeff_test_values[parameter]["min"], 
            coeff_test_values[parameter]["max"]
            )
        ax.legend(loc='upper center', prop={'size': 10}, ncol=4)

        fig.savefig("{}/sf_{}_{}.pdf".format(output_dir, args.observable, parameter))
        fig.savefig("{}/sf_{}_{}.png".format(output_dir, args.observable, parameter))


if __name__ == "__main__":
    args = parse_arguments()
    main(args)