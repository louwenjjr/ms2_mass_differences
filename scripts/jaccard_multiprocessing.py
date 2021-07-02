#!/usr/bin/env python
"""
Author: Joris Louwen

Loads data from notebook 2 and calculates jaccard similarity between mass
differences and fragments/neutral losses.
The resulting matrix will be written to a file in output folder.

The files from notebook 2 are called:
"gnps_positive_ionmode_cleaned_by_matchms_and_lookups_mass_difference_
occurrence.pickle"
"gnps_positive_ionmode_cleaned_by_matchms_and_lookups_fragments_
occurrences.pickle"
"""

import os
import pickle
import argparse
import time
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from multiprocessing import Pool
from typing import List, Tuple
from functools import partial
from math import ceil


def get_commands() -> argparse.Namespace:
    """
    Returns argparse.ArgumentParser.parse_args object for command line options
    """
    parser = argparse.ArgumentParser(description="Loads either spectrum\
        documents containing mass differences as words (md@...), or data from\
        notebook 2 in this repo. Calculates jaccard similarity between mass\
        differences and fragments/neutral losses in a multithreading way.\
        Resulting matrix will be written to output_file.")
    parser.add_argument("-s", "--spectrum_documents", metavar="<.pickle>",
                        help="pickle file of spectrum documents also\
        containing mass differences as 'md@...' in .words", required=False)
    parser.add_argument("-m", "--mds", metavar="<.pickle>", help="pickle file\
        containing list of list of mass difference occurrences (tuple)",
                        required=False)
    parser.add_argument("-f", "--fragments", metavar="<.pickle>", help="pickle\
        file containing list of list of fragment/neutral loss occurrences\
        (tuple)", required=False)
    parser.add_argument("-o", "--output_file", metavar="<file>",
                        help="location of output file (default:\
                        jaccard_matrix.npz)", default="jaccard_matrix.npz")
    parser.add_argument("-c", "--cores", help="Cores to use (default: 20)",
                        default=20, type=int)
    parser.add_argument("-l", "--lower_cutoff", type=float, help="lower\
        cutoff for selecting MDs based on max jaccard score, default: 0.1",
                        default=0.1)
    parser.add_argument("-u", "--upper_cutoff", type=float, help="upper\
            cutoff for selecting MDs based on max jaccard score, default: 0.5",
                        default=0.5)
    return parser.parse_args()


def jaccard_list_occurrences(list_1: List[str], list_2: List[str]) -> float:
    """Return jaccard similarity (intersection/union) of the input lists
    
    Parameters
    ------------
    list_1:
        List of words (str)
    list_2:
        List of words (str)
    """
    set_1 = set(list_1)
    set_2 = set(list_2)
    jac_sim = len(set_1 & set_2) / len(set_1 | set_2)
    return jac_sim


def calculate_row_jaccard(md_occ_list: List[str],
                          all_fragment_occ_list: List[List[str]]) \
        -> Tuple[str, List[float]]:
    """
    For one mass difference, calc Jaccard similarity to all fragments/n_losses
    
    Parameters
    -----------
    md_occ_list:
        List of [MD_name, spectra names] in which a MD occurs, first element
        is MD name
    all_fragment_occ_list:
        List of list of spectra names of spectra occurrences for all fragments/
        neutral losses

    Returns
    -----------
    jaccard_sims:
        List of [MD_name, jaccard_similarities], first element is str, rest are
        floats
    """
    md_name = md_occ_list.pop(0)
    jaccard_sims = []
    for frag_occ_list in all_fragment_occ_list:
        jaccard_sims.append(
            jaccard_list_occurrences(md_occ_list, frag_occ_list))
    return md_name, jaccard_sims


def main():
    """Main functionality of this script
    """
    start = time.time()
    print("\nStart")
    cmd = get_commands()
    input_checked = False

    # read pickled input files
    print("Reading input files")
    if cmd.mds and cmd.fragments:
        input_checked = True
        with open(cmd.mds, 'rb') as inf:
            md_occ = pickle.load(inf)
        with open(cmd.fragments, 'rb') as inf:
            fragment_occ = pickle.load(inf)

    # read pickled spectrum documents if provided
    if cmd.spectrum_documents:
        input_checked = True
        # dict of {frag: [[spectra_names], [intensities]]}
        per_fragment_spec_occ_dict = {}
        per_md_spec_occ_dict = {}
        spectrum_documents_mds = pickle.load(
            open(cmd.spectrum_documents, 'rb'))
        for i, doc in enumerate(spectrum_documents_mds):
            spec_name = str(i)
            for word, intensity in zip(doc.words, doc.weights):
                if word.startswith("md@"):  # mass difference
                    if word in per_md_spec_occ_dict:
                        per_md_spec_occ_dict[word][0].append(spec_name)
                        per_md_spec_occ_dict[word][1].append(intensity)
                    else:
                        per_md_spec_occ_dict[word] = []
                        per_md_spec_occ_dict[word].append([spec_name])
                        per_md_spec_occ_dict[word].append([intensity])
                else:  # fragment/neutral loss
                    if word in per_fragment_spec_occ_dict:
                        per_fragment_spec_occ_dict[word][0].append(spec_name)
                        per_fragment_spec_occ_dict[word][1].append(intensity)
                    else:
                        per_fragment_spec_occ_dict[word] = []
                        per_fragment_spec_occ_dict[word].append([spec_name])
                        per_fragment_spec_occ_dict[word].append([intensity])
        md_occ = [(key, val[0], val[1]) for key, val in
                  per_md_spec_occ_dict.items()]
        fragment_occ = [(key, val[0], val[1]) for key, val in
                        per_fragment_spec_occ_dict.items()]

    if not input_checked:
        raise ValueError("Supply either --spectrum_documents or" +
                         " --mds and --fragments")
    # get only the occurrences
    just_md_occ = [[tup[0]] + tup[1] for tup in md_occ]
    column_names = []  # collect column names (fragments + neutral losses)
    just_fragment_occ = []
    for tup in fragment_occ:
        column_names.append(tup[0])
        just_fragment_occ.append(tup[1])

    # calc jaccard with multiprocessing, in chunks of chunk_len rows
    print("\nStart with calculations")
    chunk_len = 7500
    num_chunks = ceil(len(just_md_occ) / chunk_len)
    row_names = []
    all_sparse_jacc_chunks = []
    for chunk_num in range(num_chunks):
        print(f"\nData chunk number {chunk_num}/{num_chunks - 1}")
        current_chunk = just_md_occ[
                        chunk_len * chunk_num: chunk_len * (chunk_num + 1)]
        pool = Pool(processes=cmd.cores)
        jaccard_sims = pool.imap_unordered(
            partial(
                calculate_row_jaccard,
                all_fragment_occ_list=just_fragment_occ),
            current_chunk, chunksize=100)
        pool.close()
        pool.join()

        # append all rows in a sparse csr matrix and save rownames
        print("  constructing sparse matrix")
        sparse_jacc_chunk = None
        for i, row in enumerate(jaccard_sims):
            row_name, row_vals = row
            row_names.append(row_name)
            curr = sp.csr_matrix(row_vals, dtype=np.float64)
            if i == 0:
                sparse_jacc_chunk = curr
            else:
                sparse_jacc_chunk = sp.vstack([sparse_jacc_chunk, curr])
        del (pool, jaccard_sims)  # clear memory
        all_sparse_jacc_chunks.append(sparse_jacc_chunk)

    # add all sparse chunks together in one sparse matrix and save
    output_file = cmd.output_file
    if not output_file.endswith(".npz"):
        output_file += ".npz"
    print("\nAdding all chunks together and saving to output:", output_file)

    sparse_jacc_matrix = sp.vstack(all_sparse_jacc_chunks)
    np.savez(output_file, rows=row_names, columns=column_names,
             sparse_jaccard=sparse_jacc_matrix)

    # open matrix and save a tab delim of md\tmax_jaccard_value
    out_base = os.path.split(output_file)[0]
    out_max = os.path.join(out_base,
                           "jaccard_max_for_every_MD.txt")
    print(f"\nSaving max value for each MD to {out_max}")
    maxes = sparse_jacc_matrix.max(axis=1).toarray()
    maxes = [m[0] for m in maxes]  # unlist
    with open(out_max, 'w') as outf:
        for md, mx in zip(row_names, maxes):
            outf.write(f"{md}\t{mx}\n")

    # select mds based on cutoffs on jaccard score (default 0.1-0.5)
    cut_scores, cut_names = zip(
        *[(sc, name) for sc, name in zip(maxes, row_names) if
          cmd.lower_cutoff <= sc[0] <= cmd.upper_cutoff])
    cutoff_md_vals = [name.strip("md@") for name in cut_names]

    # make plots based on max jacc score
    plt.hist(maxes, bins=np.arange(0, 1.05, 0.05))
    out_scores = os.path.join(out_base,
                              "jaccard_max_for_every_MD_hist.png")
    plt.savefig(out_scores)
    plt.close()
    plt.hist(cut_scores, bins=np.arange(0, 1.05, 0.05))
    out_cut_scores = os.path.join(out_base,
                                  "jaccard_max_for_every_MD_0.1-0.5_hist.png")
    plt.savefig(out_cut_scores)
    plt.close()

    # save selected mds
    out_sel_mds = os.path.join(out_base, 'selected_MDs_0.1-0.5_jaccard.txt')
    with open(out_sel_mds, 'w') as outf:
        for md in cutoff_md_vals:
            outf.write(f"{md}\n")

    end = time.time()
    print("Time elapsed (hours): ", (end - start) / 3600)


if __name__ == "__main__":
    main()
