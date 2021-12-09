import json
from random import shuffle

import numpy as np

def load(file_path):

    # Load data into dictionary
    with open(file_path, "r") as file:
        data = json.load(file)

    # Replace inconsistent units of measurement by last value of list
    inch = ["Inch", "inches", "\"", "-inch", "-Inch", " inch", "inch"]
    hz = ["Hertz", "hertz", "Hz", "HZ", " hz", "-hz", " Hz", "hz"]
    to_replace = [inch, hz]
    replacements = dict()
    for replace_list in to_replace:
        replacement = replace_list[-1]
        values = replace_list[0:-1]
        for value in values:
            replacements[value] = replacement

    # Clean data
    clean_list = []
    for model in data:
        for occurence in data[model]:
            # Clean title
            for value in replacements:
                occurence["title"] = occurence["title"].replace(value, replacements[value])

            # Clean features map
            features = occurence["featuresMap"]
            for key in features:
                for value in replacements:
                    features[key] = features[key].replace(value, replacements[value])
            clean_list.append(occurence)

    # Shuffle items
    shuffle(clean_list)

    # Computation of binary matrix, element (i,j) is one if duplicate
    duplicates = np.zeros((len(clean_list), len(clean_list)))
    for i in range(len(clean_list)):
        model_i = clean_list[i]["modelID"]
        for j in range(i + 1, len(clean_list)):
            model_j = clean_list[j]["modelID"]
            if model_i == model_j:
                duplicates[i][j] = 1
                duplicates[j][i] = 1
    return clean_list, duplicates.astype(int)