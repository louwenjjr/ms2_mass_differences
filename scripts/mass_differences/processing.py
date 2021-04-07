#!/usr/bin/env python
import numpy as np
from typing import Union, List, Tuple
from matchms.filtering import normalize_intensities
from matchms.filtering import select_by_mz
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_relative_intensity
from matchms.filtering import reduce_to_number_of_peaks
from matchms.filtering import add_losses
from matchms.Spikes import Spikes
from matchms.typing import SpectrumType


def remove_precursor_mz_peak(spectrum_in: SpectrumType) -> SpectrumType:
    """Remove the peak for precursor_mz in the spectrum (if it exists)

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    prec_mz = spectrum.get("precursor_mz")
    before_len = len(spectrum.peaks)
    if prec_mz:  # precursor_mz exists
        mzs, intensities = spectrum.peaks.clone()
        prec_mz_i = [i for i, mz in enumerate(mzs) if mz == prec_mz]
        if prec_mz_i:  # precursor_mz peak exists -> remove it
            new_mzs = np.delete(mzs, prec_mz_i)
            new_intensities = np.delete(intensities, prec_mz_i)
            new_spikes = Spikes(mz=new_mzs, intensities=new_intensities)
            spectrum.peaks = new_spikes
            after_len = len(spectrum.peaks)
            assert after_len == before_len - 1, \
                "Expected only one peak to have been removed"

    return spectrum


def post_process_normal(spectrum_in: SpectrumType, min_peaks: int = 10) \
        -> Union[SpectrumType, None]:
    """Normal processing of spectra for Spec2Vec

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    min_peaks:
        Minimum number of peaks to pass the spectrum (otherwise -> None)
    """
    if spectrum_in is None:
        return None

    s = spectrum_in.clone()
    s = normalize_intensities(s)
    s = select_by_mz(s, mz_from=0, mz_to=1000)
    s = require_minimum_number_of_peaks(s, n_required=min_peaks)
    s = reduce_to_number_of_peaks(s, n_required=min_peaks, ratio_desired=0.5)
    if s is None:
        return None
    # remove low peaks unless less than 10 peaks are left
    s_remove_low_peaks = select_by_relative_intensity(s, intensity_from=0.001)
    if len(s_remove_low_peaks.peaks) >= 10:
        s = s_remove_low_peaks
    # add losses to normally processed spectra
    s = add_losses(s, loss_mz_from=5.0, loss_mz_to=200.0)
    return s


def post_process_md(spectrum_in: SpectrumType, low_int_cutoff: float = 0.05,
                    min_peaks: int = 10,
                    max_peaks: int = 30) -> Union[SpectrumType, None]:
    """Processing of spectra that are used for mass difference extraction

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    low_int_cutoff:
        Lower intensity cutoff for the peaks selected for MD
    min_peaks:
        Minimum number of peaks to pass the spectrum (otherwise -> None)
    max_peaks:
        Maximum number of peaks allowed in the spectrum (ranked on intensity)
    """
    if spectrum_in is None:
        return None

    s = spectrum_in.clone()
    # remove precurzor_mz from spectra so neutral losses don't end up in MDs
    s = remove_precursor_mz_peak(s)
    s = normalize_intensities(s)
    s = select_by_mz(s, mz_from=0, mz_to=1000)
    s = require_minimum_number_of_peaks(s, n_required=min_peaks)
    s = reduce_to_number_of_peaks(s, n_required=min_peaks, ratio_desired=0.5)
    if s is None:
        return None
    # remove low peaks unless less than 10 peaks are left
    s_remove_low_peaks = select_by_relative_intensity(s, intensity_from=0.001)
    if len(s_remove_low_peaks.peaks) >= 10:
        s = s_remove_low_peaks
    # do an additional removal step with a different intensity cutoff
    s_second_peak_removal = select_by_relative_intensity(
        s, intensity_from=low_int_cutoff)
    if len(s_second_peak_removal.peaks) >= 10:
        s = s_second_peak_removal

    # reduce to top30 peaks
    s = reduce_to_number_of_peaks(s, n_required=min_peaks, n_max=max_peaks)
    return s


def post_process_classical(spectrum_in: SpectrumType, min_peaks: int = 10) \
        -> Union[SpectrumType, None]:
    """Processing of spectra for calculating classical scores

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    min_peaks:
        Minimum number of peaks to pass the spectrum (otherwise -> None)
    """
    if spectrum_in is None:
        return None

    s = spectrum_in.clone()
    s = normalize_intensities(s)
    s = select_by_mz(s, mz_from=0, mz_to=1000)
    s = require_minimum_number_of_peaks(s, n_required=min_peaks)
    s = select_by_relative_intensity(s, intensity_from=0.01, intensity_to=1.0)
    return s


def processing_master(spectrums: List[SpectrumType],
                      low_int_cutoff: float = 0.05) -> Tuple[
        List[SpectrumType], List[SpectrumType], List[SpectrumType]]:
    """
    Returns tuple of processed spectra for: MD selection, Spec2Vec, classical

    Parameters
    ----------
    spectrums:
        List of input spectra to be processed.
    low_int_cutoff:
        Lower intensity cutoff for the peaks selected for MD
    """
    spectrums_top30 = []
    spectrums_processed = []
    spectrums_classical = []
    minimum_peaks = 10
    maximum_peaks = 30
    for spec in spectrums:
        s_top30 = post_process_md(spec, low_int_cutoff=low_int_cutoff,
                                  min_peaks=minimum_peaks,
                                  max_peaks=maximum_peaks)
        s_normal = post_process_normal(spec, min_peaks=minimum_peaks)
        s_classical = post_process_classical(spec, min_peaks=minimum_peaks)
        if s_top30 is not None and s_normal is not None and\
                s_classical is not None:
            spectrums_top30.append(s_top30)
            spectrums_processed.append(s_normal)
            spectrums_classical.append(s_classical)
    return spectrums_top30, spectrums_processed, spectrums_classical
