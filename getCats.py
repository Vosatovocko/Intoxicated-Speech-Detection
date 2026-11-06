import pandas as pd
import json
import os
import re
from pathlib import Path

# First we inspect the SPEAEXT file to see the categories which we can identify from it. The ones we are intrested in are:
# sex, age, acc, wei and hei to make the BMI, smo and drh, although ces, ges, wea and bak are alo to be inspected
# in the json annotation afterwards, since these values are proper to the specific sessions and not to the speaker.
speaext_table = 'data/source/ALC/TABLE/SPEAEXT.TBL'

# Read the file into a DataFrame, specifying the delimiter and header
df_speaext_table = pd.read_csv(speaext_table, delimiter='\t')  # Adjust delimiter and header as needed

# Sort the DataFrame by a specified column, e.g., 'column_name'
df_speaext_table_sorted = df_speaext_table.sort_values(by='SCD')

# print(df_speaext_table_sorted.to_string(index=True))

# Initialize dictionaries
sex_dict = {'M': [], 'F': []}
age_dict = {'21-35': [], '36-49': [], '50-64': []}
acc_dict = {'Hochdeutsch': [], 'Mitteldeutsch': [], 'Niederdeutsch': []}
smo_dict = {'Y': [], 'N': []}
drh_dict = {'light': [], 'moderate': [], 'heavy': []}
bmi_dict = {'Underweight': [], 'Normal weight': [], 'Overweight': []}

# Initialize dictionaries
sex_dict = {'M': [], 'F': []}
age_dict = {'21-35': [], '36-49': [], '50-64': []}
acc_dict = {'Hochdeutsch': [], 'Mitteldeutsch': [], 'Niederdeutsch': []}
smo_dict = {'Y': [], 'N': []}
drh_dict = {'light': [], 'moderate': [], 'heavy': []}
bmi_dict = {'Underweight': [], 'Normal weight': [], 'Overweight': []}

def create_dicts_from_df(df):

    # Fill dictionaries
    for index, row in df.iterrows():
        scd = row['SCD']

        # Sex dictionary
        sex_dict[row['SEX']].append(scd)

        # Age dictionary
        age = row['AGE']
        if 21 <= age <= 35:
            age_dict['21-35'].append(scd)
        elif 36 <= age <= 49:
            age_dict['36-49'].append(scd)
        elif 50 <= age <= 64:
            age_dict['50-64'].append(scd)

        # Dialect dictionary
        acc = row['ACC']
        if acc in ['BY', 'BW']:
            acc_dict['Hochdeutsch'].append(scd)
        elif acc in ['HE', 'SN', 'NW', 'RP', 'SL']:
            acc_dict['Mitteldeutsch'].append(scd)
        elif acc in ['NI', 'BE', 'MV', 'SH']:
            acc_dict['Niederdeutsch'].append(scd)

        # Smoking dictionary
        smo_dict[row['SMO']].append(scd)

        # Drinking dictionary
        drh = row['DRH']
        if drh in drh_dict:
            drh_dict[drh].append(scd)

        # Calculate BMI
        bmi = row['WEI'] / ((row['HEI'] / 100) ** 2)
        if bmi < 18.5:
            bmi_dict['Underweight'].append(scd)
        elif 18.5 <= bmi < 25:
            bmi_dict['Normal weight'].append(scd)
        else:
            bmi_dict['Overweight'].append(scd)

    return sex_dict, age_dict, acc_dict, smo_dict, drh_dict, bmi_dict


create_dicts_from_df(df_speaext_table_sorted)

# Function to get the speakers that match the characteristics
def filter_speakers(*args):
# Dictionary for results
    result_dict = {key: [] for key in args}

    # Populate the result lists with the corresponding speaker numbers
    for key in args:
        if key in sex_dict:
            result_dict[key] = sex_dict[key]
        elif key in age_dict:
            result_dict[key] = age_dict[key]
        elif key in acc_dict:
            result_dict[key] = acc_dict[key]
        elif key in smo_dict:
            result_dict[key] = smo_dict[key]
        elif key in drh_dict:
            result_dict[key] = drh_dict[key]
        elif key in bmi_dict:
            result_dict[key] = bmi_dict[key]

    # Find intersection of all lists
    intersection = set(result_dict[args[0]])
    for key in args[1:]:
        intersection &= set(result_dict[key])

    # Convert to sorted list
    return sorted(list(intersection))

# Pads a list of numbers with zeros
def pad_numbers(numbers):
    return [str(number).zfill(3) for number in numbers]

# Search directories for files corresponding to the speaker number
def search_files(list_of_numbers, base_path):
    padded_numbers = pad_numbers(list_of_numbers)
    matching_directories = []

    for directory in os.listdir(base_path):
        dir_path = os.path.join(base_path, directory)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith(".json") and file[:3] in padded_numbers:
                    matching_directories.append(directory[3:])
                    break  # Only need the first matching file
    return matching_directories

def filter_keys_by_session(input_file, session_numbers):
    # regex for the wav files
    pattern = re.compile(r'^\d{3}(\d{4})\d{3}_[mh]_\d{2}\.wav$')

    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Dict for filtered results
    filtered_data = {}

    # Iterate through the keys in the dictionary
    for key, value in data.items():
        # Match key with regex pattern
        match = pattern.match(key)
        if match:
            # Extract the session number from the matched groups
            session_number = match.group(1)
            # Check if the session number is in the provided list
            if session_number in session_numbers:
                # Add the key-value pair to the filtered results
                filtered_data[key] = value

    # Write the wav file names to a json file
    #with open(output_file, 'w') as outfile:
    #    json.dump(filtered_data, outfile, indent=4)
    return filtered_data

# Example usage
numbers = filter_speakers('F') #'M', '50-64', 'Hochdeutsch'
base_path = 'data/source/ALC/'  # Path with the dirs
session_numbers = search_files(numbers, base_path)
input_file = 'data/intoxicated-speech-detection/dataset/Functionals_valid_30_withBak.json'

print(filter_keys_by_session(input_file, session_numbers))
