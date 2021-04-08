#!/usr/bin/env python
from typing import List
import numpy as np
import pandas as pd
from matchms.typing import SpectrumType


def select_query_spectra(spectrums: List[SpectrumType],
                         min_copies_in_data: int = 2,
                         num_spectra: int = 1000):
    """

    Parameters
    ----------
    """
    # get inchikey occurrences
    inchikeys = []
    for spec in spectrums:
        inchikeys.append(spec.get("inchikey"))
    inchikeys_pd = pd.Series([x for x in inchikeys if x])

    # get inchikeys that occur at least min_copies_in_data times
    suitable_inchikeys = pd.DataFrame(
        inchikeys_pd.str[:14].value_counts()[
            inchikeys_pd.str[:14].value_counts().values >= min_copies_in_data])
    suitable_inchikeys.reset_index(level=suitable_inchikeys.index.names,
                                   inplace=True)
    suitable_inchikeys.columns = (['inchikey14', 'occurences'])

    # Important: sort values to make it reproducible (same occurences have
    # random order otherwise!)
    suitable_inchikeys = suitable_inchikeys.sort_values(
        ['occurences', 'inchikey14'], ascending=False)

    # select num_spectra inchikeys as queries
    np.random.seed(42)  # to make it reproducible
    selection = np.random.choice(suitable_inchikeys.shape[0], num_spectra,
                                 replace=False)
    selected_inchikeys = suitable_inchikeys['inchikey14'].values[selection]
    selected_spectra = []
    # include all even empty ones to get the IDs right!
    inchikeys_pd = pd.Series([x for x in inchikeys])
    # select one spectrum per inchikey
    np.random.seed(42)  # to make it reproducible
    for inchikey in selected_inchikeys:
        matches = inchikeys_pd[inchikeys_pd.str[:14] == inchikey].index.values
        selected_spectra.append(int(np.random.choice(matches, 1)[0]))
    return selected_spectra
