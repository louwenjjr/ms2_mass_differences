#!/usr/bin/env python
"""
Author: Joris Louwen

Loads AllPositive dataset and classes file (per inchikey) and select spectra
with certain class. Outputs pickle file of matchms.SpectrumType
"""

import pickle
import argparse
import time
import os
import numpy as np
from mass_differences.utils import read_classes_txt


def get_commands() -> argparse.Namespace:
    """
    Returns argparse.ArgumentParser.parse_args object for command line options
    """
    parser = argparse.ArgumentParser(description="Reads in AllPositive dataset\
        and classes file (per inchikey) and select spectra with certain class.\
        Outputs pickle file of matchms.SpectrumType")
    parser.add_argument("-i", "--input_file", metavar="<.pickle>", help="Path\
        to AllPositive dataset (cleaned spectra)", required=True)
    parser.add_argument("-o", "--output_file", metavar="<.pickle>",
                        help="location of output folder, default: ./",
                        default="./class_spectra.pickle")
    parser.add_argument("-c", "--classes_file", metavar="<.txt>", help="File\
        with inchikeys linked to class annotation")
    parser.add_argument("-s", "--selected_class", metavar="<str>", help="Name\
        (between '') of class you want to select on, default: Amino acids, \
        peptides, and analogues",
                        default="Amino acids, peptides, and analogues")
    parser.add_argument("--column", metavar="<int>", help="Column index of\
        chosen class (0 based), default: 5 (CF subclass)", default=5, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    cmd = get_commands()
    start = time.time()
    print("Start")

    print("\nReading spectra")
    input_spectrums = pickle.load(open(cmd.input_file, 'rb'))
    print(f"Read {len(input_spectrums)} spectra")
    class_dict = read_classes_txt(cmd.classes_file)
    print(f"\nRead classes for {len(class_dict)} inchikeys")

    print(f"\nSelecting spectra with the class {cmd.selected_class}")
    select_inchikeys = []
    for inchikey, comp_classes in class_dict.items():
        curr_class = comp_classes[cmd.column-1]  # first column is inchikey
        if cmd.selected_class in curr_class:
            select_inchikeys.append(inchikey)

    selected_spectrums = []
    for spec in input_spectrums:
        spec_inchikey = spec.metadata.get("inchikey")
        if spec_inchikey in select_inchikeys:
            selected_spectrums.append(spec)
    print(f"Selected {len(selected_spectrums)} spectra")

    out_file = cmd.output_file
    if not out_file.endswith(".pickle"):
        out_file.append(".pickle")

    print("\nWriting output to", out_file)
    pickle.dump(selected_spectrums, open(out_file, "wb"))

    end = time.time()
    print(f"\nFinished in {end - start:.3f} s")
