#!/usr/bin/env python
"""
Author: Joris Louwen

Loads data from notebook 2 and calculates jaccard similarity between mass differences and fragments/neutral losses.
The resulting matrix will be written to a file in output folder.
"""

import pickle
import os
import argparse
from multiprocessing import Pool
from typing import List


def get_commands():
    """Returns argparse.ArgumentParser.parse_args object for command line options
    """
    parser = argparse.ArgumentParser(description="Loads data from notebook 2 in this repo\
        and calculates jaccard similarity between mass differences and fragments/neutral\
        losses in a multithreading way. Resulting matrix will be written to output_file.")
    parser.add_argument("-m", "--mds", metavar=".pickle", help="pickle file containing\
        list of list of mass difference occurrences (str)")
    parser.add_argument("-f", "--fragments", metavar=".pickle", help="pickle file containing\
        list of list of fragment/neutral loss occurrences (str)")
    parser.add_argument("-o", "--output_file", metavar="<file>", help="location of output\
        file")
    return parser.parse_args()


def jaccard_list_occurrences(list_1: List[str], list_2: List[str]) -> float:
    """Return jaccard similarity (intersection/union) of the input lists
    
    Parameters
    ------------
    list_1:
        List of words (str)
    list_2:
        List of words (str)
    """
    set_1 = set(list_1)
    set_2 = set(list_2)
    jac_sim = len(set_1 & set_2) / len(set_1 | set_2)
    return jac_sim


def calculate_row_jaccard(md_occ_list: List[str], all_fragment_occ_list: List[List[str]]) -> List[float]:
    """For one mass difference, calc Jaccard similarity to all fragments/neutral losses
    
    Parameters
    -----------
    md_occ_list:
        List of spectra names in which a MD occurs
    all_fragment_occ_list:
        List of list of spectra names of spectra occurrences for all fragments/neutral losses
    """
    jaccard_sims = []
    for frag_occ_list in all_fragment_occ_list:
        jaccard_sims.append(jaccard_list_occurrences(md_occ_list, frag_occ_list))
    return jaccard_sims


if __name__ == "__main__":
    cmd = get_commands()
