#!/usr/bin/env python
"""
Author: Joris Louwen

Loads AllPositive dataset and recreates Spec2Vec figures/metrics with Spec2Vec
embeddings trained with Mass differences added as features.
"""

import pickle
import argparse
import time
import numpy as np
import scipy.sparse as sp
from typing import List, Tuple


def get_commands() -> argparse.Namespace:
    """
    Returns argparse.ArgumentParser.parse_args object for command line options
    """
    parser = argparse.ArgumentParser(description="Reads in AllPositive dataset\
        and recreates Spec2Vec metrics with mass differences added as features\
        in the SpectrumDocuments.")
    parser.add_argument("-i", "--input_file", metavar="<.json>", help="Path to\
        AllPositive dataset (cleaned spectra)")
    parser.add_argument("-o", "--output_dir", metavar="<dir>",
                        help="location of output folder, default: ./",
                        default="./")
    parser.add_argument("-m", "--mds", metavar="<.txt>", help="Text file\
        containing list of mass differences to use, default: use all mass\
        differences found with other parameters", default=False)
    parser.add_argument("-s", "--s2v_embedding", metavar="<.model>", help="Use\
        an existing Spec2Vec embedding instead of training a new one, default:\
        False", default=False)
    parser.add_argument("-e", "--existing_md_embedding", metavar="<.model>",
                        help="Use an existing Spec2Vec embedding that includes\
        mass differences as features instead of training a new\
        one, default: False", default=False)
    parser.add_argument("-l", "--lower_intensity_cutoff", metavar=None,
                        help="Minimum intensity for peaks to be included in\
                        mass difference selection, default: 0.05", type=float,
                        default=0.05)
    parser.add_argument("-b", "--binning_precision", metavar=None,
                        help="Number of decimals to bin on, default: 2",
                        type=int, default=2)
    parser.add_argument("-p", "--punish_intensities", metavar=None,
                        help="Toggle to punish intensities of mass differences",
                        action="store_true", default=False, type=bool)
    parser.add_argument("--max_mds_per_peak", metavar=None, help="Limit the\
        maximum number of mass differences that can be derived from one peak,\
        default: 30", default=30, type=int)
    parser.add_argument("--multiply_intensities", metavar=None, help="Turn on\
        this flag to multiply intensities of two parent peaks to get the mass\
        difference intensity", default=False, type=bool, action="store_true")
    return parser.parse_args()

