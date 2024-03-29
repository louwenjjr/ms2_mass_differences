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
from mass_differences.processing import get_ids_for_unique_inchikeys
from mass_differences.create_mass_differences import get_mass_differences
from mass_differences.create_mass_differences import get_md_documents
from mass_differences.create_mass_differences import \
    create_md_spectrum_documents
from mass_differences.utils import read_mds
from mass_differences.validation_pipeline import select_query_spectra
from mass_differences.validation_pipeline import library_matching_metrics
from mass_differences.validation_pipeline import md_distribution_metrics
from mass_differences.library_search import library_matching
from mass_differences.plots import true_false_pos_plot
from mass_differences.plots import accuracy_vs_retrieval_plot
from mass_differences.plotting_functions import plot_precentile
from spec2vec import SpectrumDocument, Spec2Vec
from spec2vec.model_building import train_new_word2vec_model
from matchms.similarity import ModifiedCosine


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
    parser.add_argument("-w", "--white_listed_mds", metavar="<.txt>", help="\
        Text file containing list of white listed mass differences to use.\
        These mass differences will be added regardless of other parameters\
        such as requiring in-spectrum count of 2. Should be a tab delim file\
        with mass differences in first column. A header is expected! Default:\
        use all mass differences from --mds with other filtering options",
                        default=False)
    parser.add_argument("-s", "--s2v_embedding", metavar="<.model>", help="Use\
        an existing Spec2Vec embedding instead of training a new one, default:\
        False", default=False)
    parser.add_argument("-e", "--existing_md_embedding", metavar="<.model>",
                        help="Use an existing Spec2Vec embedding that includes\
        mass differences as features instead of training a new\
        one, default: False", default=False)
    parser.add_argument("-t", "--tanimoto_scores_inchikeys",
                        metavar="<.pickle>", help="Pickled pd.DataFrame of\
        tanimoto scores between all unique inchikeys in data, row and columns\
        should be inchikeys. Default: False - no unique inchikey figure will\
        be made", default=False)
    parser.add_argument("-l", "--lower_intensity_cutoff",
                        help="Minimum intensity for peaks to be included in\
                        mass difference selection, default: 0.05", type=float,
                        default=0.05)
    parser.add_argument("-b", "--binning_precision",
                        help="Number of decimals to bin on, default: 2",
                        type=int, default=2)
    parser.add_argument("-p", "--punish_intensities", help="Toggle to punish\
        intensities of mass differences", action="store_true", default=False)
    parser.add_argument("--require_in_count", help="Require mass differences\
        to occur X times in spectrum to be taken into account, with the\
        exception of --white_listed_mds, default: 1",
                        default=1, type=int)
    parser.add_argument("--max_mds_per_peak", help="Limit the\
        maximum number of mass differences that can be derived from one peak,\
        default: 30", default=30, type=int)
    parser.add_argument("--multiply_intensities", help="Turn on\
        this flag to multiply intensities of two parent peaks to get the mass\
        difference intensity", default=False, action="store_true")
    parser.add_argument("--no_count_benefit", help="Toggle to not multiply\
        intensity of a mass difference by the sqrt(in-spectrum count)",
                        default=True, action="store_false")
    parser.add_argument("--library_matching", action="store_true",
                        help="Toggle to do library matching based on 1,000\
        taken out query spectra", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    cmd = get_commands()
    start = time.time()
    print("Start")
    if not os.path.isdir(cmd.output_dir):
        os.mkdir(cmd.output_dir)
    print("Writing output to", cmd.output_dir)

    if not cmd.mds:
        mds_to_use = False
        print("\nNo MDs found as input, all UNFILTERED found mass differences\
            will be used!")
    else:
        print("\nRead list of mass difference to use")
        mds_to_use = read_mds(cmd.mds, n_decimals=cmd.binning_precision)

    if not cmd.white_listed_mds:
        white_listed_mds = []
        print("\nNo white listed MDs found as input, all mass differences\
            will be subjected to other selection criteria.")
    else:
        print("\nRead list of mass difference to use")
        white_listed_mds = read_mds(cmd.white_listed_mds,
                                    n_decimals=cmd.binning_precision)

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
    all_unfiltered_mds = set()
    for md_doc in md_documents:
        for md in md_doc:
            all_unfiltered_mds.add(md[0])
    print(f"\n{len(all_unfiltered_mds)} unfiltered MDs present")
    print(f"{len(md_documents)} remaining MD documents (spectra).")
    print("An example:", md_documents[-1])
    if not mds_to_use:
        mds_to_use = all_unfiltered_mds

    # validation pipeline
    spectrums_top30, spectrums_processed, spectrums_classical = processing_res

    # make SpectrumDocuments
    documents_processed = [SpectrumDocument(s, n_decimals=2) for i, s
                           in enumerate(spectrums_processed)]
    documents_classical = [SpectrumDocument(s, n_decimals=2) for i, s
                           in enumerate(spectrums_classical)]
    # create md SpectrumDocuments
    set_white_listed_mds = set(white_listed_mds)
    set_chosen_mds = set(mds_to_use)
    c_multiply = cmd.no_count_benefit  # multiply intensities with sqrt(count)
    md_spectrum_documents = create_md_spectrum_documents(
        md_documents, spectrums_processed, set_white_listed_mds,
        set_chosen_mds, c_multiply, cmd.punish_intensities,
        cmd.require_in_count)
    spec_docs_mds_file = os.path.join(cmd.output_dir,
                                      'spectrum_documents_MDs.pickle')
    print("Saving spectrum documents with MDs at:", spec_docs_mds_file)
    with open(spec_docs_mds_file, 'wb') as outf:
        pickle.dump(md_spectrum_documents, outf)

    print("\nCalculating metrics + metrics plots")
    all_used_mds = md_distribution_metrics(
        md_spectrum_documents, cmd.output_dir)
    used_mds_out = os.path.join(cmd.output_dir, 'used_mds.txt')
    with open(used_mds_out, 'w') as outf:
        for used_md in all_used_mds:
            outf.write(f"{used_md}\n")
    print(f"{len(all_used_mds)} MDs are used in data, saved at" +
          f" {used_mds_out}")

    # similarity calculations (1st UniqueInchikey figure in Spec2Vec paper)
    print("\nCalculating similarity between all unique compound pairs")
    uniq_ids = get_ids_for_unique_inchikeys(spectrums_processed)
    print(f"{len(uniq_ids)} unique inchikeys present in data" +
          "(1st 14 characters)")
    uniq_documents_processed = [documents_processed[i] for i in uniq_ids]
    uniq_spectrums_classical = [spectrums_classical[i] for i in uniq_ids]
    uniq_documents_mds = [md_spectrum_documents[i] for i in uniq_ids]
    unique_inchikeys_14 = [
        s.get("inchikey", "")[:14] for s in uniq_spectrums_classical]
    # # normal s2v similarities
    # sims_out = os.path.join(
    #     cmd.output_dir,
    #     'similarities_unique_inchikey_spec2vec_librarymodel.npy')
    # if not os.path.exists(sims_out):
    #     spec2vec_similarity = Spec2Vec(model, intensity_weighting_power=0.5)
    #     similarity_matrix = spec2vec_similarity.matrix(
    #         uniq_documents_processed, uniq_documents_processed,
    #         is_symmetric=True)
    #     np.save(sims_out, similarity_matrix)
    # else:
    #     similarity_matrix = np.load(sims_out)

    # classical mod cosine similarities
    mod_cos_sims_out = os.path.join(
        cmd.output_dir,
        "similarities_unique_inchikey_mod_cosine.npy")
    if not os.path.exists(mod_cos_sims_out):
        similarity_measure = ModifiedCosine(tolerance=0.005, mz_power=0,
                                            intensity_power=1.0)
        mod_cos_similarity = similarity_measure.matrix(
            uniq_spectrums_classical, uniq_spectrums_classical,
            is_symmetric=True)
        np.save(mod_cos_sims_out, mod_cos_similarity)
    else:
        mod_cos_similarity = np.load(mod_cos_sims_out)

    # # md s2v similarities
    # md_sims_out = os.path.join(
    #     cmd.output_dir,
    #     'similarities_unique_inchikey_mds_spec2vec_librarymodel.npy')
    # if not os.path.exists(md_sims_out):
    #     md_spec2vec_similarity = Spec2Vec(model_mds,
    #                                       intensity_weighting_power=0.5)
    #     md_similarity_matrix = md_spec2vec_similarity.matrix(
    #         uniq_documents_mds, uniq_documents_mds, is_symmetric=True)
    #     np.save(md_sims_out, md_similarity_matrix)
    # else:
    #     md_similarity_matrix = np.load(md_sims_out)

    # same similarities calculation but for models built on unique inchikey
    unique_inchi_model_file = os.path.join(
        cmd.output_dir, "spec2vec_unique_inchikey.model")
    if not os.path.exists(unique_inchi_model_file):
        print("\nTraining new 'normal' Spec2Vec model on all unique inchikey" +
              " spectra at", unique_inchi_model_file)
        unique_inchi_model = train_new_word2vec_model(
            uniq_documents_processed, [50], unique_inchi_model_file)
    else:
        print("Loading existing 'normal' unique inchikey Spec2Vec model from",
              unique_inchi_model_file)
        unique_inchi_model = gensim.models.Word2Vec.load(
            unique_inchi_model_file)
    print("Normal unique inchikey Spec2Vecmodel:", unique_inchi_model)

    # train new embedding for Spec2Vec + MDs library for unique inchikey
    unique_inchi_model_mds_file = os.path.join(
        cmd.output_dir, "spec2vec_unique_inchikey_added_mds.model")
    if not os.path.exists(unique_inchi_model_mds_file):
        print("\nTraining new Spec2Vec model with MDs on all unique inchikey" +
              " spectra at", unique_inchi_model_mds_file)
        unique_inchi_model_mds = train_new_word2vec_model(
            uniq_documents_mds, [50], unique_inchi_model_mds_file)
    else:
        print(
            "\nLoading existing unique inchikey Spec2Vec model with MDs from",
            unique_inchi_model_mds_file)
        unique_inchi_model_mds = gensim.models.Word2Vec.load(
            unique_inchi_model_mds_file)
    print("MDs unique inchikey Spec2Vec model:", unique_inchi_model_mds)

    # normal s2v similarities unique inchikey model
    sims_unique_model_out = os.path.join(
        cmd.output_dir,
        'similarities_unique_inchikey_spec2vec_unique_inchikey.npy')
    if not os.path.exists(sims_unique_model_out):
        spec2vec_ui_similarity = Spec2Vec(unique_inchi_model,
                                          intensity_weighting_power=0.5,
                                          allowed_missing_percentage=60)
        similarity_ui_matrix = spec2vec_ui_similarity.matrix(
            uniq_documents_processed, uniq_documents_processed,
            is_symmetric=True)
        np.save(sims_unique_model_out, similarity_ui_matrix)
    else:
        similarity_ui_matrix = np.load(sims_unique_model_out)

    # md s2v similarities unique inchikey model
    md_sims_unique_out = os.path.join(
        cmd.output_dir,
        'similarities_unique_inchikey_mds_spec2vec_unique_inchikey.npy')
    if not os.path.exists(md_sims_unique_out):
        md_spec2vec_ui_similarity = Spec2Vec(unique_inchi_model_mds,
                                             intensity_weighting_power=0.5,
                                             allowed_missing_percentage=60)
        md_similarity_ui_matrix = md_spec2vec_ui_similarity.matrix(
            uniq_documents_mds, uniq_documents_mds, is_symmetric=True)
        np.save(md_sims_unique_out, md_similarity_ui_matrix)
    else:
        md_similarity_ui_matrix = np.load(md_sims_unique_out)

    # plot unique inchikey figure
    if cmd.tanimoto_scores_inchikeys:
        # get matrix of tanimoto scores based data order
        tan_df = pickle.load(open(cmd.tanimoto_scores_inchikeys, 'rb'))
        cols_dict = {in14: i for i, in14 in enumerate(tan_df.columns)}
        # get only the inchikeys for which there are tanimoto scores (slices)
        ui_14_tan_inds, unique_inchikeys_14_tan = zip(
            *[(i, in14) for i, in14 in enumerate(unique_inchikeys_14) if in14
              in cols_dict])
        chosen_inds = [cols_dict[in14] for in14 in unique_inchikeys_14_tan if
                       in14 in cols_dict]
        # slice tan matrix to only include existing inchikeys
        slice1 = np.take(tan_df.values, chosen_inds, 0)
        tan_matrix = np.take(slice1, chosen_inds, 1)

        # make mod cosine min match 10
        mod_arr_min10_file = os.path.join(
            cmd.output_dir,
            "similarities_unique_inchikey_mod_cosine_min10.npy")
        if not os.path.exists(mod_arr_min10_file):
            mod_cos_similarity_min10 = []
            mod_cos_similarity_matches = []
            for row in mod_cos_similarity:
                vals, matches = zip(*row)
                mod_cos_similarity_min10.append(list(vals))
                mod_cos_similarity_matches.append(list(matches))
            mod_cos_similarity_min10_arr = np.array(mod_cos_similarity_min10)
            matches_arr = np.array(mod_cos_similarity_matches)
            mod_cos_similarity_min10_arr[matches_arr < 10] = 0
            np.save(mod_arr_min10_file, mod_cos_similarity_min10_arr)
        else:
            mod_cos_similarity_min10_arr = np.load(mod_arr_min10_file)

        # slice s2v/mod cosine matrices to only include existing tan inchikeys
        mod_slice = np.take(mod_cos_similarity_min10_arr, ui_14_tan_inds, 0)
        mod_cos_similarity_min10_final = np.take(mod_slice, ui_14_tan_inds, 1)
        similarity_ui_matrix_slice = np.take(similarity_ui_matrix,
                                             ui_14_tan_inds, 0)
        similarity_ui_matrix_final = np.take(similarity_ui_matrix_slice,
                                             ui_14_tan_inds, 1)
        md_similarity_ui_matrix_slice = np.take(md_similarity_ui_matrix,
                                                ui_14_tan_inds, 0)
        md_similarity_ui_matrix_final = np.take(md_similarity_ui_matrix_slice,
                                                ui_14_tan_inds, 1)

        # plot
        percentile_spec2vec_ui = plot_precentile(
            tan_matrix,
            similarity_ui_matrix_final,
            num_bins=100, show_top_percentile=0.1,
            ignore_diagonal=True)
        percentile_spec2vec_md_ui = plot_precentile(
            tan_matrix,
            md_similarity_ui_matrix_final,
            num_bins=100, show_top_percentile=0.1,
            ignore_diagonal=True)
        percentile_mod_cosine_tol0005 = plot_precentile(
            tan_matrix,
            mod_cos_similarity_min10_final,
            num_bins=100, show_top_percentile=0.1,
            ignore_diagonal=True)
        percentile_tanimoto = plot_precentile(
            tan_matrix,
            tan_matrix,
            num_bins=100, show_top_percentile=0.1,
            ignore_diagonal=True)

        # Compare all:
        num_bins = 100
        show_top_percentile = 0.1

        plt.style.use('seaborn-white')  # ('ggplot')
        fig, ax = plt.subplots(figsize=(7, 6))

        x_percentiles = (show_top_percentile /
                         num_bins * (1 + np.arange(num_bins)))[::-1]

        plt.plot(x_percentiles, percentile_tanimoto,
                 color='gray',
                 label='1) Tanimoto score (best theoretically possible)')

        plt.plot(x_percentiles, percentile_spec2vec_ui,
                 color='black',
                 label=
                 '2) Spec2Vec - trained on UniqueInchikey (50 iterations)')

        plt.plot(x_percentiles, percentile_spec2vec_md_ui,
                 ":", color='blue',
                 label='3) Spec2Vec with MDs - trained on UniqueInchikey' +
                       ' (50 iterations)')

        plt.plot(x_percentiles, percentile_mod_cosine_tol0005,
                 color='crimson',
                 label='4) Modified cosine score (tol = 0.005, min_match = 10)')

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # plt.xticks(np.arange(0, 0.11, step=0.02), ('0.00%', '0.02%', '0.04%', '0.06%', '0.08%', '0.10%'))
        plt.xticks(np.linspace(0, show_top_percentile, 5),
                   ["{:.3f}%".format(x) for x in
                    np.linspace(0, show_top_percentile, 5)])
        plt.legend()
        plt.xlabel("Top percentile of spectral similarity score g(s1,s2)",
                   fontsize=13)
        plt.ylabel(
            "Mean molecular similarity f(m1,m2) \n (within respective percentile)",
            fontsize=13)
        plt.grid(True)
        plt.xlim(0, 0.1)
        plt.ylim(0, 1.02)  # (0.38, 0.92)
        plt.savefig(os.path.join(
            cmd.output_dir,
            'Benchmarking_top_percentil_comparison.pdf'))
        plt.close(fig)

    # library matching
    if cmd.library_matching:
        # select query spectra
        print("\nSelecting query spectra")
        selected_spectra = select_query_spectra(spectrums_top30)

        # divide in query and library
        documents_library_processed_with_mds = []
        documents_query_processed_with_mds = []
        documents_library_processed = []
        documents_query_processed = []
        documents_library_classical = []
        documents_query_classical = []
        for i, (doc_proc, doc_clas, doc_md) in enumerate(
                zip(documents_processed, documents_classical,
                    md_spectrum_documents)
        ):
            if i in selected_spectra:
                documents_query_processed.append(doc_proc)
                documents_query_classical.append(doc_clas)
                documents_query_processed_with_mds.append(doc_md)
            else:
                documents_library_processed.append(doc_proc)
                documents_library_classical.append(doc_clas)
                documents_library_processed_with_mds.append(doc_md)

        # train new embedding for 'normal' Spec2Vec library
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

        # train new embedding for Spec2Vec + MDs library
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

        print(
            "\nPerforming library matching with 1,000 randomly chosen queries")
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
            allowed_missing_percentage=60.0,
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
            allowed_missing_percentage=60.0,
            cosine_tol=0.005,
            mass_tolerance=1.0,
            mass_tolerance_type="ppm")

        # library matching for MDs
        found_matches_processed_with_mds = library_matching(
            documents_query_processed_with_mds,
            documents_library_processed_with_mds,
            model_mds,
            presearch_based_on=["precursor_mz", "spec2vec-top20"],
            include_scores=["cosine", "modcosine"],
            ignore_non_annotated=True,
            intensity_weighting_power=0.5,
            allowed_missing_percentage=60.0,
            cosine_tol=0.005,
            mass_tolerance=1.0,
            mass_tolerance_type="ppm")

        all_lib_matching_metrics = library_matching_metrics(
            documents_query_classical, documents_library_classical,
            found_matches_classical, documents_query_processed,
            documents_library_processed, found_matches_processed,
            documents_query_processed_with_mds,
            documents_library_processed_with_mds,
            found_matches_processed_with_mds)
        # save metrics to be used in other plots
        lib_metrics_file = os.path.join(cmd.output_dir,
                                        'lib_matching_metrics.pickle')
        with open(lib_metrics_file, 'wb') as outf:
            pickle.dump(all_lib_matching_metrics, outf)

        test_matches_min2, test_matches_min6, test_matches_s2v, \
        test_matches_s2v_mds = all_lib_matching_metrics

        # make plots
        true_false_pos_plot(test_matches_min6, test_matches_s2v,
                            test_matches_s2v_mds, cmd.output_dir, min_match=6)

        accuracy_vs_retrieval_plot(
            test_matches_min2, test_matches_s2v, test_matches_s2v_mds,
            cmd.output_dir, min_match=2)

    end = time.time()
    print(f"\nFinished in {end - start:.3f} s")
