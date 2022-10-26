import numpy as np
import json


robusthesse_paths = {
    "hgg": "input/SMEFT/expected/Hgg/robustHesseAsimovBestFit.root",
    "hzz": "input/SMEFT/expected/HZZ/robustHesse_POSTFIT_HZZ.root",
}

ggH_production_files = {
    "hgg": "{}/differentials/hgg/ggH_SMEFTatNLO_pt_gg.json",
    "hzz": "{}/differentials/hzz/ggH_SMEFTatNLO_pt_h.json",
}

max_to_matt = {"hgg": "gamgam", "hzz": "ZZ"}

mus_paths = {"hgg": "input/SMEFT/mus_Hgg.json", "hzz": "input/SMEFT/mus_HZZ.json"}
corr_matrices_observed = {
    "hgg": "input/SMEFT/observed/Hgg/correlation_matrix.json",
    "hzz": "input/SMEFT/observed/HZZ/correlation_matrix.json",
}
corr_matrices_expected = {
    "hgg": "input/SMEFT/expected/Hgg/correlation_matrix.json",
    "hzz": "input/SMEFT/expected/HZZ/correlation_matrix.json",
}


def refactor_predictions(prediction_dir, channel):
    decays_file = f"{prediction_dir}/decay.json"
    with open(decays_file, "r") as f:
        decays_dct = json.load(f)
    production_file = ggH_production_files[channel].format(prediction_dir)
    with open(production_file, "r") as f:
        tmp_production_dct = json.load(f)
    dict_keys = list(tmp_production_dct.keys())
    sorted_keys = sorted(dict_keys, key=lambda x: float(x))
    tmp_production_dct = {k: tmp_production_dct[k] for k in sorted_keys}
    edges = [float(k) for k in sorted_keys] + [10000.0]
    production_dct = {}
    for edge, next_edge in zip(edges[:-1], edges[1:]):
        production_dct[
            "r_smH_PTH_{}_{}".format(
                str(edge).replace(".0", ""), str(next_edge).replace(".0", "")
            )
        ] = tmp_production_dct[str(edge)]
    if channel == "hgg":
        key_to_remove = "r_smH_PTH_450_10000"
        key_dct = production_dct[key_to_remove]
        production_dct.pop(key_to_remove)
        production_dct["r_smH_PTH_GT450"] = key_dct
    elif channel == "hzz":
        key_to_remove = "r_smH_PTH_200_10000"
        key_dct = production_dct[key_to_remove]
        production_dct.pop(key_to_remove)
        production_dct["r_smH_PTH_GT200"] = key_dct

    return decays_dct, production_dct, edges


def refactor_predictions_multichannel(prediction_dir, channels):
    production_dct = {}
    edges = {}
    for channel in channels:
        decays_dct, production_dct[channel], edges[channel] = refactor_predictions(
            prediction_dir, channel
        )
    return decays_dct, production_dct, edges
