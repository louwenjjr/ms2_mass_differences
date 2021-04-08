#!/usr/bin/env python
import numpy as np
from typing import Union, List, Tuple
from matchms.Spikes import Spikes
from matchms.typing import SpectrumType


def get_mass_differences(spectrum_in: SpectrumType, cutoff: int = 36,
                         n_max: int = 100) -> Union[Spikes, None]:
    """Returns Spikes with top X (100) mass differences and intensities

    Parameters:
    spectrum_in:
        Spectrum in matchms.Spectrum format
    cutoff:
        Mass cutoff for mass difference (default like Xing et al.)
    n_max:
        Maximum amount of mass differences to select, ranked on intensity
        (default like Xing et al.)
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()
    peaks_mz, peaks_intensities = spectrum.peaks
    mass_diff_mz = []
    mass_diff_intensities = []
    for i, (mz_i, int_i) in enumerate(
            zip(peaks_mz[:-1], peaks_intensities[:-1])):
        for mz_j, int_j in zip(peaks_mz[i + 1:], peaks_intensities[i + 1:]):
            mz_diff = mz_j - mz_i
            if mz_diff > cutoff:
                mass_diff_mz.append(mz_diff)
                mass_diff_intensities.append(np.mean([int_i, int_j]))
    mass_diff_mz = np.array(mass_diff_mz)
    mass_diff_intensities = np.array(mass_diff_intensities)
    idx = mass_diff_intensities.argsort()[-n_max:]
    idx_sort_by_mz = mass_diff_mz[idx].argsort()
    mass_diff_peaks = Spikes(mz=mass_diff_mz[idx][idx_sort_by_mz],
                             intensities=mass_diff_intensities[idx][
                                 idx_sort_by_mz])
    return mass_diff_peaks
