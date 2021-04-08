#!/usr/bin/env python
import os
from typing import List


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
