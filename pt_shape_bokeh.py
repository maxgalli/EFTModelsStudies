import argparse
import json
import numpy as np
import pandas as pd
from copy import deepcopy

from bokeh.io import curdoc
from bokeh.models import Slider, ColumnDataSource, Band
from bokeh.layouts import column, row, grid
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
min_parameter = -0.1
max_parameter = 0.1
widgets = {}
for par in production_json["parameters"]:
    widgets[par] = Slider(title=par, value=0.0, start=min_parameter, end=max_parameter, step=0.005)

# Parabola plots (one for each parameter inside each bin)
def parabola(x, a, b):
    return a * x ** 2 + b * x

parabola_plots = [] # will be a list of dictionaries of the form {'cu': figure, 'cl': figure, 'cd': figure}
parabola_coeffs = []
for deviations, edges in zip(production_json["bins"], production_json["edges"]):
    parabola_plot = {}
    parabola_coeff = {}
    for par in production_json["parameters"]:
        fig = figure(title=f"{edges[0]}_{edges[1]}", width=300, height=200)
        fig.xaxis.axis_label = par
        # Build parabolic plot
        for dev in deviations:
            if len(dev) == 3 and dev[2] == par:
                b = dev[0]
            if len(dev) == 4 and dev[2] == par and dev[3] == par:
                a = abs(dev[0])
        # If it fails, it means that the parameter is not in any of the deviations of the bin
        # We then discard it
        try:
            x = np.linspace(min_parameter, max_parameter, 1000)
            y = parabola(x, a, b)
            fig.line(x, y, line_width=2, line_color="black")
            parabola_plot[par] = fig
            parabola_coeff[par] = {"a": a, "b": b}
            del a, b
        except NameError:
            continue
    parabola_plots.append(parabola_plot)
    parabola_coeffs.append(parabola_coeff)

# Little circles to put on the parabolas, which will be updated when the sliders are moved
circle_sources = []
for row_dictionary in parabola_plots:
    circle_row_dictionary = {}
    for par, fig in row_dictionary.items():
        circle_row_dictionary[par] = ColumnDataSource(data=dict(x=[0], y=[0]))
        fig.circle('x', 'y', source=circle_row_dictionary[par], size=10, color="red")
    circle_sources.append(circle_row_dictionary)
#print("circle sources {}".format(circle_sources))


def update_dev_histo(attrname, old, new):
    """ This is the function which is called every time the slider is moved
    """
    # Get current slider values
    widgets_curr = {k: widgets[k].value for k in widgets}
    #print("Current widgets values {}".format(widgets_curr))

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

    # Update little circles
    for dct_circle, dct_coeffs in zip(circle_sources, parabola_coeffs):
        for par, source in dct_circle.items():
            x = widgets_curr[par]
            a = dct_coeffs[par]["a"]
            b = dct_coeffs[par]["b"]
            source.data = dict(x=[x], y=[parabola(x, a, b)])


for w in widgets:
    widgets[w].on_change("value", update_dev_histo)

# Set up layouts and add to document + boilerplate for interactive plot
inputs = column(*widgets.values())
curdoc().add_root(row(inputs, p, sizing_mode="fixed"))
parabola_grid = grid([list(parabola_plot.values()) for parabola_plot in parabola_plots], sizing_mode="fixed")
curdoc().add_root(parabola_grid)
