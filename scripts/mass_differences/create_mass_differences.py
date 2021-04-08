#!/usr/bin/env python
import numpy as np
from typing import Union, List, Tuple
from collections import defaultdict
from matchms.Spikes import Spikes
from matchms.typing import SpectrumType


def get_mass_differences(spectrum_in: SpectrumType, multiply: bool = False,
                         max_mds_per_peak: int = 30, cutoff: int = 36,
                         n_max: int = 100) -> Union[Spikes, None]:
    """Returns Spikes with top X mass differences and intensities

    Parameters
    ----------
    spectrum_in:
        Spectrum in matchms.Spectrum format
    multiply:
        Multiply parent peak intensities instead of taking the mean
    max_mds_per_peak:
        Maximum amount of MDs that can originate from one peak, ranked on
        intensity. The minimum is 2 (with this implementation)
    cutoff:
        Mass cutoff for mass difference (default like Xing et al.)
    n_max:
        Maximum amount of mass differences to select, ranked on intensity
        (default like Xing et al.)
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()
    peaks_mz_ori, peaks_intensities_ori = spectrum.peaks
    # sort on intensities to allow for max_mds_per_peak selection
    sort_idx = peaks_intensities_ori.argsort()[::-1]
    peaks_intensities = peaks_intensities_ori[sort_idx]
    peaks_mz = peaks_mz_ori[sort_idx]

    # for every peak, calculate MDs to all other peaks
    mass_diff_mz = []
    mass_diff_intensities = []
    used_mz_dict = {mz_val: 0 for mz_val in peaks_mz}  # keep track of used mz
    for i, (mz_i, int_i) in enumerate(
            zip(peaks_mz[:-1], peaks_intensities[:-1])):
        cur = used_mz_dict[mz_i]  # number of uses of this peak
        allowed = max_mds_per_peak - cur  # still allowed uses
        for mz_j, int_j in zip(peaks_mz[i + 1: i + 1 + allowed],
                               peaks_intensities[i + 1: i + 1 + allowed]):
            # update used peaks dict
            used_mz_dict[mz_i] += 1
            used_mz_dict[mz_j] += 1
            # calculate mass difference
            mz_diff = mz_j - mz_i
            if mz_diff > cutoff:
                mass_diff_mz.append(mz_diff)
                if multiply:
                    new_intensity = int_i * int_j
                else:
                    new_intensity = np.mean([int_i, int_j])
                mass_diff_intensities.append(new_intensity)
    # sort on mz
    mass_diff_mz = np.array(mass_diff_mz)
    mass_diff_intensities = np.array(mass_diff_intensities)
    idx = mass_diff_intensities.argsort()[-n_max:]
    idx_sort_by_mz = mass_diff_mz[idx].argsort()
    mass_diff_peaks = Spikes(mz=mass_diff_mz[idx][idx_sort_by_mz],
                             intensities=mass_diff_intensities[idx][
                                 idx_sort_by_mz])
    return mass_diff_peaks


def get_md_documents(mass_differences: List[Spikes],
                     n_decimals: int = 2) -> List[
        List[Tuple[str, List[float], int]]]:
    """Bin mass differences and return them as list of 'documents' with words

    Parameters
    ----------
    mass_differences:
        List of Spikes
    n_decimals:
        Number of decimals to bin on
    Returns
    -------
    md_documents:
        List of 'documents' which are a tuple of (md, [intensities], count)
    """
    md_documents = []
    for mz, intensities in mass_differences:
        mz_round_strs = [f"{mz_i:.{n_decimals}f}" for mz_i in mz]
        summary = defaultdict(list)
        for mz_round_str, intensity in zip(mz_round_strs, intensities):
            summary[mz_round_str].append(intensity)
        info_tup = [(key, sorted(vals, reverse=True), len(vals))
                    for key, vals in summary.items()]
        info_tup.sort(key=lambda x: x[2], reverse=True)
        md_documents.append(info_tup)
    return md_documents
