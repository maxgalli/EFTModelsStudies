import argparse
import json
import numpy as np
import pandas as pd
from copy import deepcopy

from bokeh.io import curdoc
from bokeh.models import Slider, ColumnDataSource, Band
from bokeh.layouts import column, row
from bokeh.plotting import figure


def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--production", type=str, required=True)

    parser.add_argument("--br-num", type=str)

    parser.add_argument("--br-den", type=str)

    return parser.parse_args()


""" 
Main starts here
"""
args = parse_arguments()

with open(args.production) as f:
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

# See https://gist.github.com/bhishanpdl/43f5ddad264c1a712979ddc63dd3d6ee
dev_hist = deepcopy(nominal_histo)
arr_df = pd.DataFrame({"count": dev_hist, "left": edges[:-1], "right": edges[1:]})
arr_df["f_count"] = ["%d" % count for count in arr_df["count"]]
arr_df["f_interval"] = [
    "%d to %d " % (left, right) for left, right in zip(arr_df["left"], arr_df["right"])
]
arr_src = ColumnDataSource(arr_df)
# print("Nominal histogram {}".format(nominal_histo))

# For the band too
x = np.array(production_json["edges"])
y = (np.ones_like(x).T * dev_hist).T
x = x.flatten()
y = y.flatten()
arr_df_err = pd.DataFrame(
    {"x": x, "y": y, "lower": y - np.zeros_like(x), "upper": y + np.zeros_like(x)}
)
arr_err_src = ColumnDataSource(arr_df_err.reset_index())

# Plot
p = figure(width=1200, height=600)

# SM
p.quad(
    top=nominal_histo,
    bottom=0,
    left=edges[:-1],
    right=edges[1:],
    color="white",
    line_color="black",
    fill_alpha=0.0,
    legend_label="SM",
)

# Deviation
p.quad(
    top="count",
    bottom=0,
    left="left",
    right="right",
    source=arr_src,
    color="white",
    line_color="blue",
    fill_alpha=0.0,
    legend_label="SMEFT",
)

# band
band = Band(
    base="x",
    lower="lower",
    upper="upper",
    source=arr_err_src,
    level="underlay",
    fill_alpha=1.0,
    line_width=1,
    line_color="lightblue",
    fill_color="lightblue",
)

p.legend.location = "top_right"
p.add_layout(band)

# Set up widgets (one slider for each parameter)
widgets = {}
for par in production_json["parameters"]:
    widgets[par] = Slider(title=par, value=0.0, start=-1, end=1, step=0.005)


def update_dev_histo(attrname, old, new):
    # Get current slider values
    widgets_curr = {k: widgets[k].value for k in widgets}
    # print("Current widgets values {}".format(widgets_curr))

    # Not exactly nice, but reflects what is done in EFT2Obs
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
                val *= widgets_curr[par]
                err *= widgets_curr[par]
            to_add += val
            to_add_err += err ** 2
        add_bins.append(to_add)
        add_errs.append(to_add_err)

    # print("Will add to {}".format(dev_hist))
    add_bins = np.array(add_bins) * dev_hist
    # print("Bins to be added {}".format(add_bins))
    full_hist = dev_hist + add_bins
    # print("Full histogram {}".format(full_hist))
    full_errs = np.sqrt(add_errs) * full_hist
    full_hist_norm_nonegative = np.array([val if val > 0 else 0 for val in full_hist])
    arr_src.data["count"] = full_hist_norm_nonegative
    x = np.array(production_json["edges"])
    y = (np.ones_like(x).T * full_hist).T
    full_errs_mat = (np.ones_like(x).T * full_errs).T
    down = y - full_errs_mat
    up = y + full_errs_mat
    arr_err_src.data["x"] = x.flatten()
    arr_err_src.data["y"] = y.flatten()
    arr_err_src.data["lower"] = down.flatten()
    arr_err_src.data["upper"] = up.flatten()


for w in widgets:
    widgets[w].on_change("value", update_dev_histo)

# Set up layouts and add to document
inputs = column(*widgets.values())

# Boilerplate for interactive plot
curdoc().add_root(row(inputs, p, width=800))
