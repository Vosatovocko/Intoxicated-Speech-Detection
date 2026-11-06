from pathlib import Path
import re
import shutil


def extract_files(source_dir, regex_pattern_a, regex_pattern_na, destination_dir):
    source_dir = Path(source_dir)
    # Create directories
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, # create parent directories if needed
                   exist_ok=True) # if binary directory already exists, don't error.
    category_name_a = Path("intoxicated_audios_BIS30") # create drunk save path
    category_name_a.mkdir(parents=True,
                          exist_ok=True)
    intoxicated_audios_save_path = destination_dir / category_name_a
    category_name_na = Path("sober_audios_BIS30") # create sober save path
    category_name_na.mkdir(parents=True,
                           exist_ok=True)
    sober_audios_save_path = destination_dir / category_name_na



    for sub_dir in source_dir.glob("**/"):
        # Check if the directory matches the regex pattern
        if sub_dir.is_dir() and re.match(regex_pattern_a, sub_dir.name):
            for wav_file in sub_dir.glob("*.wav"):
                extract_drunk_audios_path = intoxicated_audios_save_path / wav_file.name
                shutil.copy(wav_file, extract_drunk_audios_path)
        elif sub_dir.is_dir() and re.match(regex_pattern_na, sub_dir.name):
            for wav_file in sub_dir.glob("*.wav"):
                extract_sober_audios_path = sober_audios_save_path / wav_file.name
                shutil.copy(wav_file, extract_sober_audios_path)


source_dir = "data/source/ALC"
regex_pattern_a = r'^ses(10|30)\d*$'
regex_pattern_na = r'^ses(20|40)\d*$'
# destination_dir can be changed.
destination_dir = "data/intoxicated-speech-detection"
extract_files(source_dir,regex_pattern_a, regex_pattern_na, destination_dir)
