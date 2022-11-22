import numpy as np
import json


robusthesse_paths = {
    "hgg": "input/SMEFT/expected/Hgg/robustHesseAsimovBestFit.root",
    "hzz": "input/SMEFT/expected/HZZ/robustHesse_POSTFIT_HZZ.root",
    "htt": "input/SMEFT/expected/Htt/robustHesseTest.root",
    "hww": "input/SMEFT/expected/HWW/robustHesse_POSTFIT_HWW.root",
    "hbbvbf": "input/SMEFT/expected/HbbVBF/robustHesse_POSTFIT_HbbVBF.root",
    "httboost": "input/SMEFT/expected/HttBoost/robustHesse_POSTFIT_HttBoost.root",
}

ggH_production_files = {
    "hgg": "{}/differentials/hgg/ggH_SMEFTatNLO_pt_gg.json",
    "hzz": "{}/differentials/hzz/ggH_SMEFTatNLO_pt_h.json",
    "htt": "{}/differentials/htt/ggH_SMEFTatNLO_pt_h.json",
    "hww": "{}/differentials/hww/ggH_SMEFTatNLO_pt_h.json",
    "hbbvbf": "{}/differentials/hbbvbf/ggH_SMEFTatNLO_pt_h.json",
    "httboost": "{}/differentials/httboost/ggH_SMEFTatNLO_pt_h.json",
}

max_to_matt = {
    "hgg": "gamgam",
    "hzz": "ZZ",
    "htt": "tautau",
    "hww": "WW",
    "hbbvbf": "bb",
    "httboost": "tautau",
}

mus_paths = {
    "hgg": "input/SMEFT/mus_Hgg.json",
    "hzz": "input/SMEFT/mus_HZZ.json",
    "htt": "input/SMEFT/mus_Htt.json",
    "hww": "input/SMEFT/mus_HWW.json",
    "hbbvbf": "input/SMEFT/mus_HbbVBF.json",
    "httboost": "input/SMEFT/mus_HttBoost.json",
}
corr_matrices_observed = {
    "hgg": "input/SMEFT/observed/Hgg/correlation_matrix.json",
    "hzz": "input/SMEFT/observed/HZZ/correlation_matrix.json",
    "htt": "input/SMEFT/observed/Htt/correlation_matrix.json",
    "hww": "input/SMEFT/observed/HWW/correlation_matrix.json",
    "hbbvbf": "input/SMEFT/observed/HbbVBF/correlation_matrix.json",
    "httboost": "input/SMEFT/observed/HttBoost/correlation_matrix.json",
}
corr_matrices_expected = {
    "hgg": "input/SMEFT/expected/Hgg/correlation_matrix.json",
    "hzz": "input/SMEFT/expected/HZZ/correlation_matrix.json",
    "htt": "input/SMEFT/expected/Htt/correlation_matrix.json",
    "hww": "input/SMEFT/expected/HWW/correlation_matrix.json",
    "hbbvbf": "input/SMEFT/expected/HbbVBF/correlation_matrix.json",
    "httboost": "input/SMEFT/expected/HttBoost/correlation_matrix.json",
}
sm_prediction_files = {
    "hgg": "predictions/theoryPred_Pt_18_fullPS.pkl",
    "hzz": "predictions/theoryPred_Pt_18_fullPS_HZZ.pkl",
    "htt": "predictions/theoryPred_Pt_18_fullPS_Htt.pkl",
    "hww": "predictions/theoryPred_Pt_18_fullPS_HWW.pkl",
    "hbbvbf": "predictions/theoryPred_Pt_18_fullPS_HbbVBF.pkl",
    "httboost": "predictions/theoryPred_Pt_18_fullPS_HttBoost.pkl",
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
    if channel in ["hgg", "htt"]:
        key_to_remove = "r_smH_PTH_450_10000"
        key_dct = production_dct[key_to_remove]
        production_dct.pop(key_to_remove)
        production_dct["r_smH_PTH_GT450"] = key_dct
    elif channel in ["hzz", "hww"]:
        key_to_remove = "r_smH_PTH_200_10000"
        key_dct = production_dct[key_to_remove]
        production_dct.pop(key_to_remove)
        production_dct["r_smH_PTH_GT200"] = key_dct
    elif channel == "httboost":
        key_to_remove = "r_smH_PTH_600_10000"
        key_dct = production_dct[key_to_remove]
        production_dct.pop(key_to_remove)
        production_dct["r_smH_PTH_GT600"] = key_dct
    elif channel == "hbbvbf":
        # this is we don't add manually 800 to the prediction JSON
        # key_to_remove = "r_smH_PTH_675_10000"
        # key_dct = production_dct[key_to_remove]
        # production_dct.pop(key_to_remove)
        # production_dct["r_smH_PTH_675_800"] = key_dct
        key_to_remove = "r_smH_PTH_800_10000"
        key_dct = production_dct[key_to_remove]
        production_dct.pop(key_to_remove)
        production_dct["r_smH_PTH_800_1200"] = key_dct

    return decays_dct, production_dct, edges, sorted_keys


def refactor_predictions_multichannel(prediction_dir, channels):
    production_dct = {}
    edges = {}
    for channel in channels:
        decays_dct, production_dct[channel], edges[channel], _ = refactor_predictions(
            prediction_dir, channel
        )
    return decays_dct, production_dct, edges
