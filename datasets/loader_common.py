########################################################################
# import python-library
########################################################################
# default
import glob
import argparse
import random
import sys
import os
import itertools
import re
import pathlib

# additional
import numpy as np
import librosa
import librosa.core
import librosa.feature
import yaml
import urllib.request
import urllib.error
import zipfile

########################################################################


########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""
import logging

logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.0"
########################################################################


########################################################################
# load parameter.yaml
########################################################################
# def yaml_load():
#     with open("baseline.yaml") as stream:
#         param = yaml.safe_load(stream)
#     return param

########################################################################


########################################################################
# file I/O
########################################################################
# wav file input
def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


########################################################################


########################################################################
# feature extractor
########################################################################
def file_to_vectors(file_name,
                    n_mels=64,
                    n_frames=5,
                    n_fft=1024,
                    hop_length=512,
                    power=2.0,
                    fmax=None,
                    fmin=None,
                    win_length=None):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # generate melspectrogram using librosa
    y, sr = file_load(file_name, mono=True)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                        sr=sr,
                                                        n_fft=n_fft,
                                                        hop_length=hop_length,
                                                        n_mels=n_mels,
                                                        power=power,
                                                        fmax=fmax,
                                                        fmin=fmin,
                                                        win_length=win_length)

    # convert melspectrogram to log mel energies
    log_mel_spectrogram = 20.0 / power * np.log10(np.maximum(mel_spectrogram, sys.float_info.epsilon))

    # calculate total vector size
    n_vectors = len(log_mel_spectrogram[0, :]) - n_frames + 1

    # skip too short clips
    if n_vectors < 1:
        return np.empty((0, dims))

    # generate feature vectors by concatenating multiframes
    vectors = np.zeros((n_vectors, dims))
    for t in range(n_frames):
        vectors[:, n_mels * t : n_mels * (t + 1)] = log_mel_spectrogram[:, t : t + n_vectors].T

    return vectors


########################################################################


########################################################################
# get directory paths according to mode
########################################################################
def select_dirs(param, mode):
    """
    param : dict
        baseline.yaml data

    return :
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode:
        logger.info("load_directory <- development")
        query = os.path.abspath("{base}/*".format(base=param["dev_directory"]))
    else:
        logger.info("load_directory <- evaluation")
        query = os.path.abspath("{base}/*".format(base=param["eval_directory"]))
    dirs = sorted(glob.glob(query))
    dirs = [f for f in dirs if os.path.isdir(f)]
    return dirs


########################################################################


########################################################################
# get machine IDs
########################################################################
def get_section_names(target_dir,
                      dir_name,
                      ext="wav"):
    """
    target_dir : str
        base directory path
    dir_name : str
        sub directory name
    ext : str (default="wav)
        file extension of audio files

    return :
        section_names : list [ str ]
            list of section names extracted from the names of audio files
    """
    # create test files
    query = os.path.abspath("{target_dir}/{dir_name}/*.{ext}".format(target_dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(query))
    # extract section names
    section_names = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('section_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return section_names


########################################################################


########################################################################
# get the list of wave file paths
########################################################################
def file_list_generator(target_dir,
                        section_name,
                        unique_section_names,
                        dir_name,
                        mode,
                        train,
                        prefix_normal="normal",
                        prefix_anomaly="anomaly",
                        ext="wav"):
    """
    target_dir : str
        base directory path
    section_name : str
        section name of audio file in <<dir_name>> directory
    dir_name : str
        sub directory name
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files

    return :
        if the mode is "development":
            files : list [ str ]
                audio file list
            labels : list [ boolean ]
                label info. list
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            files : list [ str ]
                audio file list
    """
    logger.info("target_dir : {}".format(target_dir + "_" + section_name))
    condition_array = np.eye(len(unique_section_names))

    # development
    if mode:
        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{ext}".format(target_dir=target_dir,
                                                                                                     dir_name=dir_name,
                                                                                                     section_name=section_name,
                                                                                                     prefix_normal=prefix_normal,
                                                                                                     ext=ext))
        normal_files = sorted(glob.glob(query))
        if len(normal_files) == 0:
            normal_files = sorted(
                glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                        dir_name=dir_name,
                                                                                        prefix_normal=prefix_normal,
                                                                                        id_name=section_name,
                                                                                        ext=ext)))
        normal_labels = np.zeros(len(normal_files))

        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{ext}".format(target_dir=target_dir,
                                                                                                     dir_name=dir_name,
                                                                                                     section_name=section_name,
                                                                                                     prefix_normal=prefix_anomaly,
                                                                                                     ext=ext))
        anomaly_files = sorted(glob.glob(query))
        if len(anomaly_files) == 0:
            anomaly_files = sorted(
                glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                        dir_name=dir_name,
                                                                                        prefix_anomaly=prefix_anomaly,
                                                                                        id_name=section_name,
                                                                                        ext=ext)))
        anomaly_labels = np.ones(len(anomaly_files))

        files = np.concatenate((normal_files, anomaly_files), axis=0)
        labels = np.concatenate((normal_labels, anomaly_labels), axis=0)

        logger.info("#files : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n========================================")

    # evaluation
    else:
        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*.{ext}".format(target_dir=target_dir,
                                                                                                     dir_name=dir_name,
                                                                                                     section_name=section_name,
                                                                                                     ext=ext))
        files = sorted(glob.glob(query))
        if train:
            normal_files = sorted(glob.glob(query))
            labels = np.zeros(len(normal_files))
        else:
            labels = None
        logger.info("#files : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n=========================================")

    condition = []
    for _, file in enumerate(files):
        for j, unique_section_name in enumerate(unique_section_names):
            if unique_section_name in file:
                condition.append(condition_array[j])

    return files, labels, condition
########################################################################

def download_raw_data(
    target_dir,
    machine_type,
    data_type,
    download_path_yaml,
    dataset,
):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    with open(download_path_yaml, "r") as f:
        file_url = yaml.safe_load(f)[dataset]

    for i in np.arange(len(file_url[machine_type][data_type])):
        zip_file_path = f"{target_dir}/{dataset}{machine_type}_{data_type}_{i}.zip"
        try:
            print("Downloading...")
            with urllib.request.urlopen(file_url[machine_type][data_type][i]) as download_file:
                data = download_file.read()
                with open(zip_file_path, mode='wb') as save_file:
                    save_file.write(data)
        except urllib.error.URLError as e:
            print(e)

        if os.path.exists(zip_file_path):
            print("unzip...")
            with zipfile.ZipFile(zip_file_path) as obj_zip:
                obj_zip.extractall(target_dir)

    return

########################################################################
# get machine type and section id in yaml
########################################################################
YAML_PATH = {
    "legacy":"datasets/machine_type_legacy.yaml",
    "DCASE2023T2_dev":"datasets/machine_type_2023_dev.yaml",
    "DCASE2023T2_eval":"datasets/machine_type_2023_eval.yaml",
}

def get_machine_type_dict(dataset_name, mode=True):
    if dataset_name in ["DCASE2020T2", "DCASE2022T2"]:
        yaml_path = YAML_PATH["legacy"]
    elif dataset_name == "DCASE2023T2" and not mode:
        yaml_path = YAML_PATH["DCASE2023T2_eval"]
    else: 
        yaml_path = YAML_PATH["DCASE2023T2_dev"]
    
    with open(yaml_path, "r") as f:
        machine_type_dict = yaml.safe_load(f)
        return machine_type_dict[dataset_name]