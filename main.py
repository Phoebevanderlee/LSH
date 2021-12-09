import random
import time
import scipy
from scipy.special import comb

import numpy as np

from LSH import convert_binary, minhash, lsh, common_count
from data_prep import load


def main():
    

    identify_common_count = False
    run_lsh = True
    write_result = True

    thresholds = [x / 100 for x in range(5, 100, 5)]
    bootstraps = 5
    random.seed(0)

    file_path = "/Users/phoebevanderlee/python-workspace/mypython/TVs-all-merged.json"
    result_path = "/Users/phoebevanderlee/python-workspace/results/"

    start_time = time.time()

    data_list, duplicates = load(file_path)

    if identify_common_count:
        common_count(data_list)

    if run_lsh:
        if write_result:
            with open(result_path + "results.csv", 'w') as out:
                out.write(
                    "t,comparisons,pq,pc,f1")

        for t in thresholds:
            print("t = ", t)

            # Declare statistics
            results_old = np.zeros(4)

            for run in range(bootstraps):
                data_sample, duplicates_sample = bootstrap(data_list, duplicates)
                comparisons_old_run, pq_old_run, pc_old_run, f1_old_run = do_lsh_old(data_sample, duplicates_sample, t)
                results_old += np.array([comparisons_old_run, pq_old_run, pc_old_run, f1_old_run])

            # Compute average statistics over all bootstraps
            statistics_old = results_old / bootstraps

            if write_result:
                with open(result_path + "results.csv", 'a') as out:
                    out.write(str(t))
                    for stat in statistics_old:
                        out.write("," + str(stat))
                    out.write("\n")

    end_time = time.time()
    print("Run time:", end_time - start_time, "seconds")


def do_lsh_old(data_list, duplicates, t):


    binary_vec = convert_binary(data_list)
    n = round(round(0.5 * len(binary_vec)) / 100) * 100
    signature = minhash(binary_vec, n)
    candidates = lsh(signature, t)

    # Compute number of comparisons and fractions
    comparisons = np.sum(candidates) / 2
    comparison_frac = comparisons / comb(len(data_list), 2)

    # Matrix of correctly specified duplicates, element (i, j) is equal to one if item i and item j are duplicates
    correct = np.where(duplicates + candidates == 2, 1, 0)
    n_correct = np.sum(correct) / 2

    # Compute Pair Quality
    pq = n_correct / comparisons

    # Compute Pair Completeness
    pc = n_correct / (np.sum(duplicates) / 2)

    # Compute F_1 measure
    f1 = 2 * pq * pc / (pq + pc)

    return comparison_frac, pq, pc, f1


def bootstrap(data_list, duplicates):


    # Compute indices to be included in the bootstrap
    indices = [random.randint(x, len(data_list) - 1) for x in [0] * len(data_list)]

    # Collect samples
    data_sample = [data_list[index] for index in indices]
    duplicates_sample = np.take(np.take(duplicates, indices, axis=0), indices, axis=1)
    return data_sample, duplicates_sample


if __name__ == '__main__':
    main()