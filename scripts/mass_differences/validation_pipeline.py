#!/usr/bin/env python
import numpy as np
import pandas as pd
from typing import List, Tuple
from matchms.typing import SpectrumType
from spec2vec import SpectrumDocument


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


def library_matching_metrics(
        documents_query_classical: List[SpectrumDocument],
        documents_library_classical: List[SpectrumDocument],
        found_matches_classical: List[pd.DataFrame],
        documents_query_processed: List[SpectrumDocument],
        documents_library_processed: List[SpectrumDocument],
        found_matches_processed: List[pd.DataFrame],
        documents_query_processed_with_mds: List[SpectrumDocument],
        documents_library_processed_with_mds: List[SpectrumDocument],
        found_matches_processed_with_mds: List[pd.DataFrame]) -> Tuple[
        List[List[np.ndarray]], List[List[np.ndarray]], List[List[np.ndarray]],
        List[List[np.ndarray]]]:
    """
    From all the different library matching results, calc true/false/NA matches
    """
    min_match = 2
    cosine_thresholds = np.arange(0, 1, 0.05)

    test_matches_min2 = []
    for threshold in cosine_thresholds:
        # print(f"Checking matches for cosine score > {threshold:.2f}")
        test_matches = []

        for ID in range(len(documents_query_classical)):
            if len(found_matches_classical[ID]) > 0:
                # Scenario 1: mass + sort by cosine
                df_select = found_matches_classical[ID][
                    (found_matches_classical[ID]['mass_match'] == 1)
                    & (found_matches_classical[ID]['cosine_score'] > threshold)
                    & (found_matches_classical[ID][
                           'cosine_matches'] >= min_match)]

                if df_select.shape[0] > 0:
                    best_match_ID = df_select.sort_values(
                        by=['cosine_score'], ascending=False).index[0]
                    inchikey_selected = documents_library_classical[
                                            best_match_ID]._obj.get(
                        "inchikey")[:14]
                    inchikey_query = documents_query_classical[ID]._obj.get(
                        "inchikey")[:14]

                    best_bet = 1 * (inchikey_selected == inchikey_query)
                else:
                    best_bet = -1  # meaning: not found
                test_matches.append(best_bet)

        # Make arrays from lists:
        test_arr = np.array(test_matches)

        test_matches_min2.append([np.sum(test_arr == 1), np.sum(test_arr == 0),
                                  np.sum(test_arr == -1)])

    min_match = 6
    test_matches_min6 = []
    for threshold in cosine_thresholds:
        # print(f"Checking matches for cosine score > {threshold:.2f}")
        test_matches = []

        for ID in range(len(documents_query_classical)):
            if len(found_matches_classical[ID]) > 0:
                # Scenario 1: mass + sort by cosine
                df_select = found_matches_classical[ID][
                    (found_matches_classical[ID]['mass_match'] == 1)
                    & (found_matches_classical[ID]['cosine_score'] > threshold)
                    & (found_matches_classical[ID][
                           'cosine_matches'] >= min_match)]

                if df_select.shape[0] > 0:
                    best_match_ID = df_select.sort_values(
                        by=['cosine_score'], ascending=False).index[0]
                    inchikey_selected = documents_library_classical[
                                            best_match_ID]._obj.get(
                        "inchikey")[:14]
                    inchikey_query = documents_query_classical[ID]._obj.get(
                        "inchikey")[:14]

                    best_bet = 1 * (inchikey_selected == inchikey_query)
                else:
                    best_bet = -1  # meaning: not found
                test_matches.append(best_bet)

        # Make arrays from lists:
        test_arr = np.array(test_matches)

        test_matches_min6.append([np.sum(test_arr == 1), np.sum(test_arr == 0),
                                  np.sum(test_arr == -1)])

    test_matches_s2v = []
    for threshold in cosine_thresholds:
        # print(f"Checking matches for spec2vec score > {threshold:.2f}")
        test_matches = []

        for ID in range(len(documents_query_processed)):

            # Scenario 2: mass + sort by Spec2Vec
            df_select = found_matches_processed[ID][
                (found_matches_processed[ID]['mass_match'] == 1)
                & (found_matches_processed[ID]['s2v_score'] > threshold)]
            if df_select.shape[0] > 0:
                best_match_ID = df_select.sort_values(
                    by=['s2v_score'], ascending=False).index[0]
                inchikey_selected = documents_library_processed[
                                        best_match_ID]._obj.get(
                    "inchikey")[:14]
                inchikey_query = documents_query_processed[ID]._obj.get(
                    "inchikey")[:14]

                best_bet = 1 * (inchikey_selected == inchikey_query)
            else:
                best_bet = -1  # meaning: not found
            test_matches.append(best_bet)

        # Make arrays from lists:
        test_arr = np.array(test_matches)

        test_matches_s2v.append([np.sum(test_arr == 1), np.sum(test_arr == 0),
                                 np.sum(test_arr == -1)])

    test_matches_s2v_mds = []

    cosine_thresholds = np.arange(0, 1, 0.05)

    for threshold in cosine_thresholds:
        # print(f"Checking matches for spec2vec score > {threshold:.2f}")
        test_matches = []

        for ID in range(len(documents_query_processed_with_mds)):

            # Scenario 2: mass + sort by Spec2Vec
            df_select = found_matches_processed_with_mds[ID][
                (found_matches_processed_with_mds[ID]['mass_match'] == 1)
                & (found_matches_processed_with_mds[ID][
                       's2v_score'] > threshold)]
            if df_select.shape[0] > 0:
                best_match_ID = df_select.sort_values(
                    by=['s2v_score'], ascending=False).index[0]
                inchikey_selected = documents_library_processed_with_mds[
                                        best_match_ID]._obj.get("inchikey")[
                                    :14]
                inchikey_query = documents_query_processed_with_mds[
                                     ID]._obj.get("inchikey")[:14]

                best_bet = 1 * (inchikey_selected == inchikey_query)
            else:
                best_bet = -1  # meaning: not found
            test_matches.append(best_bet)

        # Make arrays from lists:
        test_arr = np.array(test_matches)

        test_matches_s2v_mds.append(
            [np.sum(test_arr == 1), np.sum(test_arr == 0),
             np.sum(test_arr == -1)])
    return test_matches_min2, test_matches_min6, test_matches_s2v, \
        test_matches_s2v_mds
