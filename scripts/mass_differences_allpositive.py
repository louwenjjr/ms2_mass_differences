#!/usr/bin/env python
"""
Author: Joris Louwen

Loads AllPositive dataset and recreates Spec2Vec figures/metrics with Spec2Vec
embeddings trained with Mass differences added as features.
"""

import pickle
import argparse
import time
import os
import gensim
import numpy as np
import matplotlib.pyplot as plt
from mass_differences.processing import processing_master
from mass_differences.create_mass_differences import get_mass_differences
from mass_differences.create_mass_differences import get_md_documents
from mass_differences.create_mass_differences import convert_md_tup
from mass_differences.utils import read_mds
from mass_differences.validation_pipeline import select_query_spectra
from mass_differences.validation_pipeline import library_matching_metrics
from mass_differences.library_search import library_matching
from mass_differences.plots import true_false_pos_plot
from mass_differences.plots import accuracy_vs_retrieval_plot
from spec2vec import SpectrumDocument
from spec2vec.model_building import train_new_word2vec_model
from copy import deepcopy


def get_commands() -> argparse.Namespace:
    """
    Returns argparse.ArgumentParser.parse_args object for command line options
    """
    parser = argparse.ArgumentParser(description="Reads in AllPositive dataset\
        and recreates Spec2Vec metrics with mass differences added as features\
        in the SpectrumDocuments.")
    parser.add_argument("-i", "--input_file", metavar="<.pickle>", help="Path\
        to AllPositive dataset (cleaned spectra)", required=True)
    parser.add_argument("-o", "--output_dir", metavar="<dir>",
                        help="location of output folder, default: ./",
                        default="./")
    parser.add_argument("-m", "--mds", metavar="<.txt>", help="Text file\
        containing list of mass differences to use. Should be a tab delim file\
        with mass differences in first column. A header is expected! Default:\
        use all mass differences found with other parameters", default=False)
    parser.add_argument("-s", "--s2v_embedding", metavar="<.model>", help="Use\
        an existing Spec2Vec embedding instead of training a new one, default:\
        False", default=False)
    parser.add_argument("-e", "--existing_md_embedding", metavar="<.model>",
                        help="Use an existing Spec2Vec embedding that includes\
        mass differences as features instead of training a new\
        one, default: False", default=False)
    parser.add_argument("-l", "--lower_intensity_cutoff",
                        help="Minimum intensity for peaks to be included in\
                        mass difference selection, default: 0.05", type=float,
                        default=0.05)
    parser.add_argument("-b", "--binning_precision",
                        help="Number of decimals to bin on, default: 2",
                        type=int, default=2)
    parser.add_argument("-p", "--punish_intensities",
                        help="Toggle to punish intensities of mass\
                        differences",
                        action="store_true", default=False)
    parser.add_argument("--max_mds_per_peak", help="Limit the\
        maximum number of mass differences that can be derived from one peak,\
        default: 30", default=30, type=int)
    parser.add_argument("--multiply_intensities", help="Turn on\
        this flag to multiply intensities of two parent peaks to get the mass\
        difference intensity", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cmd = get_commands()
    start = time.time()
    print("Start")
    if not os.path.isdir(cmd.output_dir):
        os.mkdir(cmd.output_dir)
    print("Writing output to", cmd.output_dir)

    if not cmd.mds:
        white_listed_mds = False
        print("\nNo MDs found as input, all UNFILTERED found mass differences\
            will be used!")
    else:
        print("\nRead list of mass difference to use")
        white_listed_mds = read_mds(cmd.mds, n_decimals=cmd.binning_precision)

    input_spectrums = pickle.load(open(cmd.input_file, 'rb'))
    processing_res = processing_master(
        input_spectrums, low_int_cutoff=cmd.lower_intensity_cutoff)
    print(f"\n{len(processing_res[0])} remaining MD processed spectra.")
    print(f"as a check: {len(processing_res[1])} remaining spectra in "
          f"normally processed data for s2v.")
    print(f"as a check: {len(processing_res[2])} remaining spectra in "
          f"classically processed data for cosine.")

    mass_differences = []
    for spec in processing_res[0]:
        mass_differences.append(get_mass_differences(spec))  # list of Spikes

    md_documents = get_md_documents(mass_differences,
                                    n_decimals=cmd.binning_precision)
    print(f"\n{len(md_documents)} remaining MD documents (spectra).")
    print("An example:", md_documents[-1])

    # validation pipeline
    spectrums_top30, spectrums_processed, spectrums_classical = processing_res
    # select query spectra
    print("\nSelecting query spectra")
    selected_spectra = select_query_spectra(spectrums_top30)

    # train new embedding for 'normal' Spec2Vec
    documents_library_processed = [SpectrumDocument(s, n_decimals=2) for i, s
                                   in enumerate(spectrums_processed) if
                                   i not in selected_spectra]
    documents_library_classical = [SpectrumDocument(s, n_decimals=2) for i, s
                                   in enumerate(spectrums_classical) if
                                   i not in selected_spectra]
    if not cmd.s2v_embedding:
        model_file = os.path.join(cmd.output_dir,
                                  "spec2vec_librarymatching.model")
        print("\nTraining new 'normal' Spec2Vec model at", model_file)
        model = train_new_word2vec_model(documents_library_processed,
                                         [15], model_file)  # 15 iterations
    else:
        model_file = cmd.s2v_embedding
        print("Loading existing 'normal' Spec2Vec model from", model_file)
        model = gensim.models.Word2Vec.load(model_file)
    print("Normal Spec2Vecmodel:", model)

    # train new embedding for Spec2Vec + MDs
    documents_library_mds = [md_doc for i, md_doc in enumerate(md_documents) if
                             i not in selected_spectra]
    documents_library_processed_with_mds = []
    set_chosen_mds = set(white_listed_mds)
    c_multiply = True  # multiply intensities with sqrt of count
    for doc, md_doc in zip(documents_library_processed, documents_library_mds):
        new_doc = deepcopy(doc)  # make sure original doc is not affected

        processed_mds = [
            convert_md_tup(md,
                           count_multiplier=c_multiply,
                           punish=cmd.punish_intensities)
            for md in md_doc if md[0] in set_chosen_mds]
        if processed_mds:
            md_words, md_intensities = zip(*processed_mds)
            new_doc.words.extend(md_words)
            new_doc.weights.extend(md_intensities)
        assert len(new_doc.words) == len(new_doc.weights)

        documents_library_processed_with_mds.append(new_doc)

    if not cmd.existing_md_embedding:
        model_file_mds = os.path.join(
            cmd.output_dir, "spec2vec_librarymatching_added_MDs.model")
        print("\nTraining new 'Spec2Vec model with MDs at", model_file_mds)
        model_mds = train_new_word2vec_model(
            documents_library_processed_with_mds, [15], model_file_mds)
    else:
        model_file_mds = cmd.existing_md_embedding
        print("\nLoading existing Spec2Vec model with MDs from",
              model_file_mds)
        model_mds = gensim.models.Word2Vec.load(model_file_mds)
    print("MDs Spec2Vec model:", model_mds)

    # library matching
    documents_query_processed = [
        SpectrumDocument(spectrums_processed[i], n_decimals=2) for i in
        selected_spectra]
    documents_query_classical = [
        SpectrumDocument(spectrums_classical[i], n_decimals=2) for i in
        selected_spectra]

    found_matches_processed = library_matching(
        documents_query_processed,
        documents_library_processed,
        model,
        presearch_based_on=[
           "precursor_mz",
           "spec2vec-top20"],
        include_scores=["cosine",
                       "modcosine"],
        ignore_non_annotated=True,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=50.0,
        cosine_tol=0.005,
        mass_tolerance=1.0,
        mass_tolerance_type="ppm")
    found_matches_classical = library_matching(
        documents_query_classical,
        documents_library_classical,
        model,
        presearch_based_on=[
           "precursor_mz"],
        include_scores=["cosine",
                       "modcosine"],
        ignore_non_annotated=True,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=50.0,
        cosine_tol=0.005,
        mass_tolerance=1.0,
        mass_tolerance_type="ppm")
    # library matching for MDs
    documents_query_mds = [md_doc for i, md_doc in enumerate(md_documents) if
                           i in selected_spectra]
    documents_query_processed_with_mds = []
    set_chosen_mds = set(white_listed_mds)
    for doc, md_doc in zip(documents_query_processed, documents_query_mds):
        new_doc = deepcopy(doc)  # make sure original doc is not affected

        processed_mds = [convert_md_tup(md) for md in md_doc if
                         md[0] in set_chosen_mds]
        if processed_mds:
            md_words, md_intensities = zip(*processed_mds)
            new_doc.words.extend(md_words)
            new_doc.weights.extend(md_intensities)
        assert len(new_doc.words) == len(new_doc.weights)

        documents_query_processed_with_mds.append(new_doc)

    found_matches_processed_with_mds = library_matching(
        documents_query_processed_with_mds,
        documents_library_processed_with_mds,
        model_mds,
        presearch_based_on=["precursor_mz", "spec2vec-top20"],
        include_scores=["cosine", "modcosine"],
        ignore_non_annotated=True,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=50.0,
        cosine_tol=0.005,
        mass_tolerance=1.0,
        mass_tolerance_type="ppm")

    print("\nMaking metrics plots")
    test_matches_min2, test_matches_min6, test_matches_s2v, \
        test_matches_s2v_mds = library_matching_metrics(
            documents_query_classical, documents_library_classical,
            found_matches_classical, documents_query_processed,
            documents_library_processed, found_matches_processed,
            documents_query_processed_with_mds,
            documents_library_processed_with_mds,
            found_matches_processed_with_mds)

    # make plots
    true_false_pos_plot(test_matches_min6, test_matches_s2v,
                        test_matches_s2v_mds, cmd.output_dir, min_match=6)

    accuracy_vs_retrieval_plot(
        test_matches_min2, test_matches_s2v, test_matches_s2v_mds,
        cmd.output_dir, min_match=2)

    end = time.time()
    print(f"\nFinished in {end - start:.3f} s")
