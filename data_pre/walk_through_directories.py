import os
import json


# Function to count the top-level keys in a JSON file
def count_top_level_keys(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        print(len(data.keys()))

def walk_throgh_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} audios in '{dirpath}'.")

def count_sober_conditions(data):
    count = 0
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'condition' and value == 'sober':
                count += 1
            count += count_sober_conditions(value)
    elif isinstance(data, list):
        for item in data:
            count += count_sober_conditions(item)
    return count


def count_intoxicated_conditions(data):
    count = 0
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'condition' and value == 'intoxicated':
                count += 1
            count += count_intoxicated_conditions(value)
    elif isinstance(data, list):
        for item in data:
            count += count_intoxicated_conditions(item)
    return count

# Function to count 'sober' conditions in a JSON file
def count_sober_conditions_in_json(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        return count_sober_conditions(data)

def count_intoxicated_conditions_in_json(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        return count_intoxicated_conditions(data)

# 9420 audios/ 960
# walk_throgh_dir("data/intoxicated-speech-detection/intoxicated_audios_BIS30")


# 19440 audios/ 2676
#walk_throgh_dir("data/intoxicated-speech-detection/Audios/sober_audios_BIS30")

# 23088 / 15312
# count_top_level_keys("data/intoxicated-speech-detection/dataset/Functionals_train.json")

#2886 / 1914
#count_top_level_keys("data/intoxicated-speech-detection/dataset/Functionals_valid_30_withBak.json")

#2886 / 1914
#count_top_level_keys("data/intoxicated-speech-detection/dataset/Functionals_test_30_withBak.json")

json_file_path = "data/intoxicated-speech-detection/dataset/Functionals_train_30_withBak.json"

# 15534 / train_sober: 7782 / valid_sober: 980 / test_sober: 958
num_sober_conditions = count_sober_conditions_in_json(json_file_path)
print(f"The number of 'sober' conditions in the JSON file is: {num_sober_conditions}")

# 7554 / train_intoxicated: 7530 / valid_intoxicated: 934 / test_intoxicated: 956
#num_intoxicated_conditions = count_intoxicated_conditions_in_json(json_file_path)
#print(f"The number of 'intoxicated' conditions in the JSON file is: {num_intoxicated_conditions}")

#data = json.load(open(json_file_path, 'r'))
#print(next(iter(data.items())))

"valid_intoxicaed = 962"
"valid_sober = 1924"
"train_intoxicated = 7554"
"train_sober = 7554"

"train_sober = 15534/ train_intoxicated = 7554"
"valid_sober = 1924/ valid_intoxicated = 962"
"test_sober = 1982/ test_intoxicated = 904"
