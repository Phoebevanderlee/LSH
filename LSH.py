import random
import re
import sys
from sympy import nextprime
import numpy as np

def convert_binary(data):
 
    model_words = dict()
    binary_vec = []

    # Loop through all tv descriptions to find model words in titles
    for i in range(len(data)):
        item = data[i]
        mw_title = re.findall("([a-zA-Z0-9]*(?:(?:[0-9]+[^0-9, ]+)|(?:[^0-9, ]+[0-9]+))[a-zA-Z0-9]*)", item["title"])
        item_mw = mw_title

        # Find model words in the key-value pairs
        features = item["featuresMap"]
        for key in features:
            value = features[key]

        
            mw_decimal = re.findall("(?:(?:([0-9]+(?:\.[0-9]+))[a-zA-Z]+)|([0-9](?:\.[0-9]+)))", value)
            for decimal in mw_decimal:
                for group in decimal:
                    if group != "":
                        item_mw.append(group)

        # Update the binary vector product representation for identified model words
        for mw in item_mw:
            if mw in model_words:
                # Set index for model word to one
                row = model_words[mw]
                binary_vec[row][i] = 1
            else:
                # Add model word to the binary vector, and set index to one
                binary_vec.append([0] * len(data))
                binary_vec[len(binary_vec) - 1][i] = 1

                # Add model word to the dictionary
                model_words[mw] = len(binary_vec) - 1
    return binary_vec



def minhash(binary_vec, n):
 
    random.seed(1)

    # Identify number of rows and columns
    r = len(binary_vec)
    c = len(binary_vec[0])
    binary_vec = np.array(binary_vec)

  
    k = nextprime(r - 1)

    # Generate n random hash functions
    hash_params = np.empty((n, 2))
    for i in range(n):
        # Generate a, b, and k.
        a = random.randint(1, k - 1)
        b = random.randint(1, k - 1)
        hash_params[i, 0] = a
        hash_params[i, 1] = b

    signature = np.full((n, c), np.inf)

    # Compute signature matrix
    for row in range(1, r + 1):
        # Compute each of the n random hashes for each row
        e = np.ones(n)
        row_vec = np.full(n, row)
        x = np.stack((e, row_vec), axis=1)
        row_hash = np.sum(hash_params * x, axis=1) % k

        for i in range(n):
            # Update column j if and only if it contains a one and its current value is larger than the hash value for
            # the signature matrix row i
            updates = np.where(binary_vec[row - 1] == 0, np.inf, row_hash[i])
            signature[i] = np.where(updates < signature[i], row_hash[i], signature[i])
    return signature.astype(int)


def lsh(signature, t):
   

    n = len(signature)

    # Compute the approximate number of bands and rows from the threshold t, using that n = r * b, and t is
    # approximately (1/b)^(1/r)
    r_best = 1
    b_best = 1
    best = 1
    for r in range(1, n + 1):
        for b in range(1, n + 1):
            if r * b == n:
                # Valid pair.
                approximation = (1 / b) ** (1 / r)
                if abs(approximation - t) < abs(best - t):
                    best = approximation
                    r_best = r
                    b_best = b

    candidates = np.zeros((len(signature[0]), len(signature[0])))
    for band in range(b_best):
        buckets = dict()
        start_row = r_best * band  # Inclusive
        end_row = r_best * (band + 1)  # Exclusive
        strings = ["".join(signature[start_row:end_row, column].astype(str)) for column in range(len(signature[0]))]
        ints = [int(string) for string in strings]
        hashes = [integer % sys.maxsize for integer in ints]

        # Add all item hashes to the correct bucket
        for item in range(len(hashes)):
            hash_value = hashes[item]
            if hash_value in buckets:

                # All items in this bucket might be duplicates of this item
                for candidate in buckets[hash_value]:
                    candidates[item, candidate] = 1
                    candidates[candidate, item] = 1
                buckets[hash_value].append(item)
            else:
                buckets[hash_value] = [item]
    return candidates.astype(int)


def common_count(data):

    feature_count = dict()

    # Identify common count features
    for i in range(len(data)):
        item = data[i]
        features = item["featuresMap"]

        for key in features:
            value = features[key]

            count = re.match("^[0-9]+$", value)
            if count is not None:
                if key in feature_count:
                    feature_count[key] += 1
                else:
                    feature_count[key] = 1

    count_list = [(v, k) for k, v in feature_count.items()]
    count_list.sort(reverse=True)
    for feature in count_list:
        print(feature[1], feature[0])