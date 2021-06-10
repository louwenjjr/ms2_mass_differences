#!/usr/bin/env python
import os
from typing import List, Dict, Union


def read_mds(file_path: str, n_decimals: int = 2) -> List[str]:
    """Read white listed mass differences, return as list of str (words)

    Parameters
    ----------
    file_path:
        Path to MDs file
    n_decimals:
        Number of decimals to bin on
    """
    mass_differences = set()
    if os.path.exists(file_path):
        with open(file_path) as inf:
            inf.readline()  # header
            for line in inf:
                # deal with: in Xing et al MDs are present as both pos and neg
                md = abs(float(line.strip().split('\t')[0]))
                mass_differences.add(f"{md:.{n_decimals}f}")
    else:
        raise ValueError("file supplied with --mds does not exist")
    return list(mass_differences)


def read_classes_txt(file_path: str, header: bool = True) -> \
        Dict[str, List[List[str]]]:
    """Read classes file to dict of inchikey: [[smiles], [class_x]]

    Parameters
    ----------
    file_path:
        Path to classes file
    header:
        Whether or not there is a header in the file, default True
    """
    classes_dict = {}
    if os.path.exists(file_path):
        with open(file_path) as inf:
            if header:
                inf.readline()  # remove header
            for line in inf:
                line = line.strip().split('\t')
                inchikey = line.pop(0)
                info = [elem.split("; ") for elem in line]
                classes_dict[inchikey] = info
    else:
        raise ValueError("classes file doest not exist")
    return classes_dict
