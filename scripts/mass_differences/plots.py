#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def mds_per_spec_plot(non_md_words: List[int], md_words: List[int],
                      output_dir: str):
    """Make scatter plot of MD words vs non MD words in spectra

    Parameters
    ----------
    non_md_words:
        Paired list with md_words, non md words in the i-th spectrum
    md_words:
        Paired list with non_md_words, md words in the i-th spectrum
    output_dir:
        Directory to save the plot in
    """
    plt.scatter(non_md_words, md_words, s=0.05)
    plt.xlabel('Fragments + neutral losses per spectrum')
    plt.ylabel('MDs per spectrum')
    plt.savefig(os.path.join(
        output_dir,
        'MDs_per_spectrum.svg'))


def md_words_frac_plot(md_words, total_words, output_dir):
    """Make scatter plot of MD words vs non MD words in spectra

    Parameters
    ----------
    md_words:
        MD words in the i-th spectrum, paired list with total_words
    total_words:
        Total words in the i-th spectrum, paired list with md_words
    output_dir:
        Directory to save the plot in
    """
    md_tot_frac = [md_l / tot_l for md_l, tot_l in zip(md_words, total_words)]
    plt.hist(md_tot_frac, bins=100)
    plt.xlabel('Fraction of MDs in spectrum document')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(
        output_dir,
        'MD_fraction_per_spectrum.svg'))


def md_intensity_dist(non_md_avg_intensity: List[np.ndarray],
                      md_avg_intensity: List[np.ndarray], output_dir: str):
    """Make scatter plot of MD words vs non MD words in spectra

    Parameters
    ----------
    non_md_avg_intensity:
        Average non MD intensities in the i-th spectrum, paired list with
        md_avg_intensity
    md_avg_intensity:
        Average MD intensities in the i-th spectrum, paired list with
        non_md_avg_intensity
    output_dir:
        Directory to save the plot in
    """
    plt.scatter(non_md_avg_intensity, md_avg_intensity, s=0.03)
    plt.xlabel('Average intensity of fragments + neutral losses per spectrum')
    plt.ylabel('Average intensity of MDs per spectrum')
    plt.ylim((0, 1))
    plt.savefig(os.path.join(
        output_dir,
        'MD_intensities_distribution.svg'))


def true_false_pos_plot(test_matches_cos: List[List[np.ndarray]],
                        test_matches_s2v: List[List[np.ndarray]],
                        test_matches_s2v_mds: List[List[np.ndarray]],
                        output_dir: str,
                        min_match: int = 6):
    """
    Make plot of true vs false positives of cos, s2v and s2v+md library matches

    Parameters
    ----------
    test_matches_cos:
        For all thresholds the true, false positives and not founds.
    test_matches_s2v:
        For all thresholds the true, false positives and not founds.
    test_matches_s2v_mds:
        For all thresholds the true, false positives and not founds.
    output_dir:
        Directory to save the plot in
    min_match:
        The number of minimum matches used in the cosine library matching
    """
    test_matches_cosine_arr = np.array(test_matches_cos)
    test_matches_s2v_arr = np.array(test_matches_s2v)
    test_matches_s2v_mds_arr = np.array(test_matches_s2v_mds)

    thresholds = np.arange(0, 1, 0.05)
    label_picks = [0, 4, 8, 12, 14, 15, 16, 17, 18, 19]

    plt.figure(figsize=(7, 6))
    plt.style.use('ggplot')
    num_max = np.sum(test_matches_cosine_arr[0, :])

    plt.plot(test_matches_s2v_arr[:, 1] / num_max,
             test_matches_s2v_arr[:, 0] / num_max,
             'o-', label='Spec2Vec')
    plt.plot(test_matches_s2v_mds_arr[:, 1] / num_max,
             test_matches_s2v_mds_arr[:, 0] / num_max,
             'o-', label='Spec2Vec + MDs')
    plt.plot(test_matches_cosine_arr[:, 1] / num_max,
             test_matches_cosine_arr[:, 0] / num_max,
             'o-', color='black',
             label='cosine (min match = {})'.format(min_match))
    for i, threshold in enumerate(thresholds):
        if i in label_picks:
            plt.annotate(">{:.2}".format(threshold),
                         (test_matches_s2v_arr[i, 1] / num_max,
                          test_matches_s2v_arr[i, 0] / num_max),
                         textcoords="offset points", xytext=(2, -10),
                         fontsize=12)
            plt.annotate(">{:.2}".format(threshold),
                         (test_matches_s2v_mds_arr[i, 1] / num_max,
                          test_matches_s2v_mds_arr[i, 0] / num_max),
                         textcoords="offset points", xytext=(2, -10),
                         fontsize=12)
            plt.annotate(">{:.2}".format(threshold),
                         (test_matches_cosine_arr[i, 1] / num_max,
                          test_matches_cosine_arr[i, 0] / num_max),
                         textcoords="offset points", xytext=(2, -10),
                         fontsize=12)

    plt.title('true/false positives per query')
    plt.legend(fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('false positives rate', fontsize=16)
    plt.ylabel('true positive rate', fontsize=16)
    # plt.xlim([0, 0.3])
    plt.savefig(os.path.join(
        output_dir,
        'library_matching_true_false_positives_labeled.svg'))
    plt.close()


def accuracy_vs_retrieval_plot(
        test_matches_cos: List[List[np.ndarray]],
        test_matches_s2v: List[List[np.ndarray]],
        test_matches_s2v_mds: List[List[np.ndarray]],
        output_dir: str,
        min_match: int = 2):
    """
    Make plot of accuracy vs retrieval of cos, s2v and s2v+md library matches

    Parameters
    ----------
    test_matches_cos:
        For all thresholds the true, false positives and not founds.
    test_matches_s2v:
        For all thresholds the true, false positives and not founds.
    test_matches_s2v_mds:
        For all thresholds the true, false positives and not founds.
    output_dir:
        Directory to save the plot in
    min_match:
        The number of minimum matches used in the cosine library matching
    """
    test_matches_cosine_arr = np.array(test_matches_cos)
    # test_matches_cosine_arr = np.array(test_matches_min6)
    test_matches_s2v_arr = np.array(test_matches_s2v)
    test_matches_s2v_mds_arr = np.array(test_matches_s2v_mds)

    thresholds = np.arange(0, 1, 0.05)
    label_picks = [0, 4, 8, 10, 12, 14, 15, 16, 17, 18, 19]

    accuracy_s2v = 100 * test_matches_s2v_arr[:, 0] / (
                test_matches_s2v_arr[:, 0] + test_matches_s2v_arr[:, 1])
    accuracy_s2v_mds = 100 * test_matches_s2v_mds_arr[:, 0] / (
                test_matches_s2v_mds_arr[:, 0] + test_matches_s2v_mds_arr[:,
                                                 1])
    accuracy_cosine = 100 * test_matches_cosine_arr[:, 0] / (
                test_matches_cosine_arr[:, 0] + test_matches_cosine_arr[:, 1])

    retrieval_s2v = (test_matches_s2v_arr[:, 1] + test_matches_s2v_arr[:,
                                                  0]) / 1000
    retrieval_s2v_mds = (test_matches_s2v_mds_arr[:,
                         1] + test_matches_s2v_mds_arr[:, 0]) / 1000
    retrieval_cosine = (test_matches_cosine_arr[:,
                        1] + test_matches_cosine_arr[:, 0]) / 1000

    plt.figure(figsize=(7, 6))
    plt.style.use('ggplot')
    plt.plot(retrieval_s2v, accuracy_s2v, 'o-', label='Spec2Vec')
    plt.plot(retrieval_s2v_mds, accuracy_s2v_mds, 'o-', label='Spec2Vec + MDs')
    plt.plot(retrieval_cosine, accuracy_cosine, 'o-', color="black",
             label='cosine (min match = {})'.format(min_match))

    for i, threshold in enumerate(thresholds):
        if i in label_picks:
            plt.annotate(">{:.2}".format(threshold),
                         (retrieval_s2v[i], accuracy_s2v[i]),
                         textcoords="offset points", xytext=(2, 5),
                         fontsize=12)
            plt.annotate(">{:.2}".format(threshold),
                         (retrieval_s2v_mds[i], accuracy_s2v_mds[i]),
                         textcoords="offset points", xytext=(2, 5),
                         fontsize=12)
            plt.annotate(">{:.2}".format(threshold),
                         (retrieval_cosine[i], accuracy_cosine[i]),
                         textcoords="offset points", xytext=(2, 5),
                         fontsize=12)

    plt.title('accuracy vs retrieval')
    plt.legend(fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylim([74, 90])
    plt.xlabel('retrieval (hits per query spectrum)', fontsize=16)
    plt.ylabel('accuracy (% correct hits)', fontsize=16)
    plt.savefig(os.path.join(
        output_dir,
        f'library_matching_accuracy_vs_retrieval_minmatch{min_match}.svg'))
