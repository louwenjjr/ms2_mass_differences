#!/usr/bin/env python
"""
Author: Joris Louwen

Loads data from notebook 2 and calculates jaccard similarity between mass differences and fragments/neutral losses.
The resulting matrix will be written to a file in output folder.

The files from notebook 2 are called:
gnps_positive_ionmode_cleaned_by_matchms_and_lookups_mass_difference_occurrence.pickle
gnps_positive_ionmode_cleaned_by_matchms_and_lookups_fragments_occurrences.pickle
"""

import pickle
import os
import argparse
import time
from multiprocessing import Pool
from typing import List, Union
from functools import partial
from math import ceil


def get_commands():
    """Returns argparse.ArgumentParser.parse_args object for command line options
    """
    parser = argparse.ArgumentParser(description="Loads data from notebook 2 in this repo\
        and calculates jaccard similarity between mass differences and fragments/neutral\
        losses in a multithreading way. Resulting matrix will be written to output_file.")
    parser.add_argument("-m", "--mds", metavar="<.pickle>", help="pickle file containing\
        list of list of mass difference occurrences (tuple)", required=True)
    parser.add_argument("-f", "--fragments", metavar="<.pickle>", help="pickle file containing\
        list of list of fragment/neutral loss occurrences (tuple)", required=True)
    parser.add_argument("-o", "--output_file", metavar="<file>", help="location of output\
        file (default: jaccard_matrix.csv)", default="jaccard_matrix.csv")
    parser.add_argument("-c", "--cores", help="Cores to use (default: 20)", default=20,
        type=int)
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
                          all_fragment_occ_list: List[List[str]]) -> List[Union[str,float]]:
    """For one mass difference, calc Jaccard similarity to all fragments/neutral losses
    
    Parameters
    -----------
    md_occ_list:
        List of [MD_name, spectra names] in which a MD occurs, first element is MD name
    all_fragment_occ_list:
        List of list of spectra names of spectra occurrences for all fragments/neutral losses

    Returns
    -----------
    jaccard_sims:
        List of [MD_name, jaccard_similarities], first element is str, rest are floats
    """
    md_name = md_occ_list.pop(0)
    jaccard_sims = [md_name]
    for frag_occ_list in all_fragment_occ_list:
        jaccard_sims.append(jaccard_list_occurrences(md_occ_list, frag_occ_list))
    return jaccard_sims


def main():
    """Main functionality of this script"""
    start = time.time()
    print("\nStart")
    cmd = get_commands()

    # read pickled input files
    print("Reading input files, creating output file: ", cmd.output_file)
    with open(cmd.mds, 'rb') as inf:
        md_occ = pickle.load(inf)
    with open(cmd.fragments, 'rb') as inf:
        fragment_occ = pickle.load(inf)

    # get only the occurrences
    just_md_occ = [[tup[0]]+tup[1] for tup in md_occ]
    just_fragment_occ = [tup[1] for tup in fragment_occ]

    # create/overwrite output file with header
    with open(cmd.output_file, 'w') as outf:
        outf.write(",{}\n".format(",".join([tup[0] for tup in fragment_occ])))

    # calc jaccard with multiprocessing, in chuncks of 10,000 rows
    print("\nStart with calculations")
    chunk_len = 7500
    num_chunks = ceil(len(just_md_occ)/chunk_len)
    for chunk_num in range(num_chunks):
        print(f"\nData chunk number {chunk_num}")
        current_chunk = just_md_occ[chunk_len*chunk_num: chunk_len*(chunk_num+1)]
        pool = Pool(processes=cmd.cores)
        jaccard_sims = pool.imap_unordered(partial(calculate_row_jaccard,
            all_fragment_occ_list=just_fragment_occ), current_chunk, chunksize=100)
        pool.close()
        pool.join()

        # write to output file
        print("Writing current chunck to file")
        with open(cmd.output_file, 'a') as outf:
            for j_sims in jaccard_sims:
                outf.write("{}\n".format(",".join(map(str, j_sims))))
        del(pool, jaccard_sims)

    end = time.time()
    print("Time elapsed (hours): ", (end-start)/3600)

if __name__ == "__main__":
    main()
