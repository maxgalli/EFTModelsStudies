from statistics import covariance
import ROOT
import json
import uproot
import numpy as np

from differential_combination_postprocess.matrix import MatricesExtractor


if __name__ == "__main__":
    pois = {
        "Hgg": [
            "r_smH_PTH_0_5",
            "r_smH_PTH_5_10",
            "r_smH_PTH_10_15",
            "r_smH_PTH_15_20",
            "r_smH_PTH_20_25",
            "r_smH_PTH_25_30",
            "r_smH_PTH_30_35",
            "r_smH_PTH_35_45",
            "r_smH_PTH_45_60",
            "r_smH_PTH_60_80",
            "r_smH_PTH_80_100",
            "r_smH_PTH_100_120",
            "r_smH_PTH_120_140",
            "r_smH_PTH_140_170",
            "r_smH_PTH_170_200",
            "r_smH_PTH_200_250",
            "r_smH_PTH_250_350",
            "r_smH_PTH_350_450",
            "r_smH_PTH_GT450",
        ],
        "HZZ": [
            "r_smH_PTH_0_15",
            "r_smH_PTH_15_30",
            "r_smH_PTH_30_45",
            "r_smH_PTH_45_80",
            "r_smH_PTH_80_120",
            "r_smH_PTH_120_200",
            "r_smH_PTH_GT200",
        ],
    }

    files = {
        # "observed": {
        #    "Hgg": "multidimfit_POSTFIT_Hgg.root",
        #    "HZZ": "multidimfit_POSTFIT_HZZ.root",
        # },
        "asimov": {
            "Hgg": "multidimfit_POSTFIT_ASIMOV_Hgg.root",
            "HZZ": "multidimfit_POSTFIT_ASIMOV_HZZ.root",
        }
    }

    combine_output_files = {
        "observed": {
            "Hgg": "higgsCombine_POSTFIT_Hgg.MultiDimFit.mH125.38.root",
            "HZZ": "higgsCombine_POSTFIT_HZZ.MultiDimFit.mH125.38.root",
        }
    }

    decay_channels = list(pois.keys())

    for cat in files:  # "observed", "asimov"
        dc_files = files[cat]
        matrix_dct_to_dump = {}
        matrices = {}
        for dc, file in dc_files.items():
            me = MatricesExtractor(pois[dc])
            me.extract_from_roofitresult(file, "fit_mdf")
            matrices[dc] = me.matrices["rfr_correlation"]
        for dc in matrices:
            l_pois = [f"{dc}_{poi}" for poi in pois[dc]]
            for l_poi, row in zip(l_pois, matrices[dc]):
                matrix_dct_to_dump[l_poi] = {}
                for dc_int in matrices:
                    ll_pois = [f"{dc_int}_{poi}" for poi in pois[dc_int]]
                    for ll_poi, val in zip(ll_pois, row):
                        if dc == dc_int:
                            matrix_dct_to_dump[l_poi][ll_poi] = val
                        else:
                            matrix_dct_to_dump[l_poi][ll_poi] = 0.0

        # Dump the matrix
        with open(f"correlation_matrix_{cat}.json", "w") as f:
            json.dump(matrix_dct_to_dump, f, indent=4)
        
    mus = {}
    for dc in decay_channels:
        # Get best fit
        f = uproot.open(combine_output_files["observed"][dc])
        t = f["limit"]
        best_fit = t["limit"].array()
        for poi in pois[dc]:
            mus[f"{dc}_{poi}"] = {}
            mus[f"{dc}_{poi}"]["bestfit"] = best_fit[poi][0]
        # uncertainties
        for cat in files:
            me = MatricesExtractor(pois[dc])
            me.extract_from_roofitresult(files[cat][dc], "fit_mdf")
            cov_matrix = me.matrices["rfr_covariance"]
            stdevs = np.sqrt(np.diag(cov_matrix))
            for poi, stdev in zip(pois[dc], stdevs):
                mus[f"{dc}_{poi}"]["Up01Sigma{}".format('' if cat == "observed" else "Exp")] = stdev / 2
                mus[f"{dc}_{poi}"]["Down01Sigma{}".format('' if cat == "observed" else "Exp")] = stdev / 2
    with open("mus.json", "w") as f:
        json.dump(mus, f, indent=4) 