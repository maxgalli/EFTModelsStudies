"""
Takes as input a JSON file like the ones dumped in pca.py and dump a Latex table 
with e.g.:
    EV0 = x*chg + y*chb
"""
import argparse
import json
from pylatex import LongTable, NoEscape

from differential_combination_postprocess.cosmetics import SMEFT_parameters_labels


def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--input-file", type=str, required=True, help="")

    parser.add_argument("--threshold", type=float, default=0.009, help="")

    parser.add_argument("--output", type=str, default="equations_ev", help="")

    return parser.parse_args()


def make_single_equation(ev, dct, threshold):
    strings = []
    lim_parts_per_line = 4
    tot = len([v for v in dct.values() if abs(v) > threshold])
    string = ""
    i = 0
    for k, v in dct.items():
        if abs(v) > threshold:
            start = "+" if v > 0 else ""
            wc_name = SMEFT_parameters_labels[k].replace("$", "")
            string += f"{start}{v:.3f} \cdot {wc_name}"
            if (i + 1) % lim_parts_per_line == 0 or i == tot - 1:
                string = "$ " + string + " $"
                strings.append(string)
                string = ""
            i += 1
    return strings


def main(args):
    with open(args.input_file) as f:
        dct = json.load(f)
    table = LongTable("c|l")
    table.add_hline()
    table.add_row(("Eigenvector", "Linear combination"))
    table.add_hline()
    for ev, dct in dct.items():
        parts = make_single_equation(ev, dct, args.threshold)
        for i, part in enumerate(parts):
            if i == 0:
                table.add_row((NoEscape(SMEFT_parameters_labels[ev]), NoEscape(part)))
            else:
                table.add_row((NoEscape(""), NoEscape(part)))
        table.add_hline()

    table.generate_tex(args.output)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
