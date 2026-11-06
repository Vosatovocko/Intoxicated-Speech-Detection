import audiofile
import opensmile
from pathlib import Path
import json


def extract_features_opensmile_Functionals(audio):
    # the audio data should always be represented as a 2-dimensional array.
    signal, sampling_rate = audiofile.read(audio, always_2d=True,)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,)
    features_functionals = smile.process_signal(signal, sampling_rate)
    return features_functionals.to_dict('list')


def extract_features_opensmile_LLDs(audio):
    signal, sampling_rate = audiofile.read(audio, always_2d=True,)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,)
    features_LLDs = smile.process_signal(signal, sampling_rate)
    return features_LLDs.to_dict('list')  # dicr structure with lists in it


def extract_all(source_dir_intoxicated, source_dir_sober, out_file):
    features_dict = {}
    for wav_file in Path(source_dir_intoxicated).glob("*.wav"):
        key = wav_file.name
        features_dict[key] = {}
        features_dict[key]["condiction"] = "intoxicated"
        features_dict[key]["features"] = extract_features_opensmile_LLDs(wav_file)
    for wav_file in Path(source_dir_sober).glob("*.wav"):
        key = wav_file.name
        features_dict[key] = {}
        features_dict[key]["condiction"] = "sober"
        features_dict[key]["features"] = extract_features_opensmile_LLDs(wav_file)
    with open(out_file, 'w') as out:
        json.dump(features_dict, out, sort_keys=True)



# print(extract_features_opensmile_Functionals("data/source/ALC/ses1006/0061006025_m_00.wav"))
# print(extract_features_opensmile_LLDs("data/source/ALC/ses1006/0061006025_m_00.wav"))
source_dir_intoxicated = "data/intoxicated-speech-detection/intoxicated_audios"
source_dir_sober = "data/intoxicated-speech-detection/sober_audios"
out_file = "ALL_LLDs_feature_extraction.json"
extract_all(source_dir_intoxicated, source_dir_sober, out_file)
