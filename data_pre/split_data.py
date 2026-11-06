import json
import os
import random


def split_dataset_into_splits(path_to_dataset, output_path):
    with open(path_to_dataset, 'r') as f:
        dataset = json.load(f)

    # shuffle the keys of the dictionary
    keys = list(dataset.keys())
    random.shuffle(keys)

    # calculate the number of samples for each split
    num_samples = len(dataset)
    num_train = int(num_samples * 0.8)
    num_val = int(num_samples * 0.1)

    # create three new dictionaries for training, validation, and testing
    train_dict = dict()
    val_dict = dict()
    test_dict = dict()

    # iterate over the shuffled keys and add each key-value pair to the appropriate dictionary
    for i, key in enumerate(keys):
        if i < num_train:
            train_dict[key] = dataset[key]
        elif i < num_train + num_val:
            val_dict[key] = dataset[key]
        else:
            test_dict[key] = dataset[key]

    combined_dict = {
        "train": train_dict,
        "val": val_dict,
        "test": test_dict
    }
    with open(output_path, mode="w", encoding="utf8") as f:
        json.dump(combined_dict, f, indent=4)

    #os.makedirs("features", exist_ok=True)
    # save the resulting dictionaries
    #with open("data/intoxicated-speech-detection/dataset/Functionals_train.json", mode="w", encoding="utf8") as f:
        #json.dump(train_dict, f)
    #with open("data/intoxicated-speech-detection/dataset/Functionals_valid.json", mode="w", encoding="utf8") as f:
        #json.dump(val_dict, f)
    #with open("data/intoxicated-speech-detection/dataset/Functionals_test.json", mode="w", encoding="utf8") as f:
        #json.dump(test_dict, f)

split_dataset_into_splits("data/intoxicated-speech-detection/data_pre/ALL_feature_extraction.json", "data/intoxicated-speech-detection/dataset/ALL_Functionals.json")
