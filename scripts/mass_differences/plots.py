#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List


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
