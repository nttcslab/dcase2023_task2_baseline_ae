import os
import pickle
import torch
from tqdm import tqdm
import numpy as np
import fasteners
from pathlib import Path
import time
import datetime

from datasets import loader_common as com

class DCASE202XT2Loader(torch.utils.data.Dataset):
    def __init__(self,
                root:str,
                dataset_name,
                section_keyword,
                machine_type:str="ToyCar",
                section_ids=[],
                train=True,
                n_mels=128,
                frames=5,
                frame_hop_length=1,
                n_fft=1024,
                hop_length=512,
                fmax=None,
                fmin=None,
                win_length=None,
                power=2.0,
                data_type = "dev",
                source_domain="mix",
                use_id = [],
                is_auto_download=False,
    ):
        super().__init__()

        self.use_id = use_id
        self.section_ids = section_ids
        self.machine_type = machine_type

        target_dir = os.getcwd()+"/"+root+"raw/"+machine_type
        dir_name = "train" if train else "test"
        
        self.mode = data_type == "dev"
        if train:
            dir_name = "train"
        elif os.path.exists("{target_dir}/{dir_name}".format(target_dir=target_dir,dir_name="test_rename")):
            dir_name = "test_rename"
            self.mode = True
        else:
            dir_name = "test"

        self.pickle_dir = os.path.abspath(
            "{dir}/processed/{machine_type}/{dir_name}".format(
                dir=root,
                machine_type=machine_type,
                dir_name=dir_name
            )
        )
        if not (fmax or fmin):
            fmin_max = ""
        else:
            fmin_max = f"_f{fmin}-{fmax}"
        self.log_melspectrogram_dir = os.path.abspath(
            "{dir}/mels{n_mels}_fft{n_fft}_hop{hop_length}{fmin_max}".format(
                dir=self.pickle_dir,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                fmin_max=fmin_max
            )
        )

        target_dir = os.path.abspath("{}/".format(target_dir))
        if is_auto_download:
            # download DCASE2022T2 dataset
            com.download_raw_data(
                target_dir=target_dir,
                dir_name=dir_name,
                machine_type=machine_type,
                data_type=data_type,
                dataset=dataset_name,
                root=root
            )
        elif not os.path.exists(f"{target_dir}/{dir_name}"):
            raise FileNotFoundError(f"{target_dir}/{dir_name} is not directory and do not use auto download. \nplease download dataset or using auto download.")

        print("dataset dir is exists.")

        # get section names from wave file names
        section_names = [f"{section_keyword}_{section_id}" for section_id in section_ids]
        unique_section_names = np.unique(section_names)
        n_sections = len(unique_section_names)
        
        # generate dataset
        print("============== DATASET_GENERATOR ==============")
        if not os.path.exists(self.log_melspectrogram_dir):
            os.makedirs(self.log_melspectrogram_dir, exist_ok=True)
        pickle_name = section_keyword
        for section_id in section_ids:
            pickle_name = f"{pickle_name}_{section_id}"
        pickle_name = f"{pickle_name}_{source_domain}_TF{frames}-{frame_hop_length}_mel{n_fft}-{hop_length}"
        pickle_path = os.path.abspath(f"{self.log_melspectrogram_dir}/{pickle_name}.pickle")

        self.load_pre_process(
            pickle_name=pickle_name,
            target_dir=target_dir,
            pickle_path=pickle_path,
            n_sections=n_sections,
            unique_section_names=unique_section_names,
            dir_name=dir_name,
            train=train,
            n_mels=n_mels,
            frames=frames,
            frame_hop_length=frame_hop_length,
            n_fft=n_fft,
            hop_length=hop_length,
            power=power,
            fmax=fmax,
            fmin=fmin,
            win_length=win_length,
        )
        if len(self.use_id) > 0:
            idx_list = [i for i, n in enumerate(np.argmax(self.condition, axis=1)) if int(section_ids[n]) in self.use_id]
        else:
            idx_list = list(range(len(self.condition)))
        idx_list_2 = [i//self.n_vectors_ea_file for i in idx_list[::self.n_vectors_ea_file]]
        self.data = self.data[idx_list]
        if len(self.y_true) > 0:
            self.y_true = self.y_true[idx_list_2]
        self.condition = self.condition[idx_list]
        self.basenames = [self.basenames[i] for i in idx_list_2]

        # getitem method setting
        self.frame_idx_list = list(range(len(self.data)))
        self.dataset_len = len(self.frame_idx_list)
        self.getitem_fn = self.default_item

    def load_pre_process(
            self,
            pickle_name,
            target_dir,
            pickle_path,
            n_sections,
            unique_section_names,
            dir_name,
            train,
            n_mels,
            frames,
            frame_hop_length,
            n_fft,
            hop_length,
            power,
            fmax,
            fmin,
            win_length,
        ):
 
        pickle_lockfile_path = os.path.abspath(f"{self.log_melspectrogram_dir}/{pickle_name}_lockfile")
        if os.path.isfile(pickle_path) and not os.path.isfile(pickle_lockfile_path):
            print("Setup has already been completed.")
            print(f"{pickle_path} is exists.")
            self.load_pickle(pickle_path=pickle_path)
            return
        else:
            pickle_lock = fasteners.InterProcessReaderWriterLock(pickle_lockfile_path)
            dataset_dir_lockfile_path = com.get_lockfile_path(target_dir=target_dir)
            is_enabled_dataset_dir_lock = os.path.isfile(dataset_dir_lockfile_path)
            if is_enabled_dataset_dir_lock:
                dataset_dir_lock = fasteners.InterProcessReaderWriterLock(dataset_dir_lockfile_path)
            try:
                pickle_lock.acquire_write_lock()
            except:
                print(f"{datetime.datetime.now()}\tcan not lock {pickle_lockfile_path}.")
                print(f"{pickle_path} is exists.")
                self.load_pickle(pickle_path=pickle_path)                
                return
            
            if os.path.exists(pickle_path):
                print(f"{pickle_path} is exists.")
                com.release_write_lock(
                    lock=pickle_lock,
                    lock_file_path=pickle_lockfile_path
                )
                self.load_pickle(pickle_path=pickle_path)
                return
            
            if is_enabled_dataset_dir_lock:
                try:
                    dataset_dir_lock.acquire_read_lock()
                except:
                    print(f"can not lock {dataset_dir_lockfile_path}.")
                    is_enabled_dataset_dir_lock = False
            
            # number of wave files in each section
            # required for calculating y_pred for each wave file
            n_files_ea_section = []

            self.data = np.empty((0, frames * n_mels), float)
            self.y_true = np.empty(0, float)
            self.condition = np.empty((0, n_sections), float)
            self.basenames = []
            for section_idx, section_name in enumerate(unique_section_names):

                # get file list for all sections
                # all values of y_true are zero in training
                files, y_true, condition = com.file_list_generator(target_dir=target_dir,
                                                        section_name=section_name,
                                                        unique_section_names=unique_section_names,
                                                        dir_name=dir_name,
                                                        mode=self.mode,
                                                        train=train)
                n_files_ea_section.append(len(files))
                for file in files:
                    self.basenames.append(os.path.basename(file))

                data_ea_section = file_list_to_data(files,
                                        msg=f"generate {self.machine_type} section {self.section_ids[section_idx]} {dir_name} dataset",
                                        n_mels=n_mels,
                                        n_frames=frames,
                                        n_hop_frames=frame_hop_length,
                                        n_fft=n_fft,
                                        hop_length=hop_length,
                                        power=power,
                                        fmax=fmax,
                                        fmin=fmin,
                                        win_length=win_length)
                self.data = np.append(self.data, data_ea_section, axis=0)
                if self.mode or train:
                    self.y_true = np.append(self.y_true, y_true, axis=0)
                for i in range(len(data_ea_section) // len(files)):
                    self.condition = np.append(self.condition, condition, axis=0)
            
            if is_enabled_dataset_dir_lock:
                com.release_read_lock(
                    lock=dataset_dir_lock,
                    lock_file_path=dataset_dir_lockfile_path
                )
            
            # number of all files
            n_all_files = sum(n_files_ea_section)
            # number of vectors for each wave file
            self.n_vectors_ea_file = int(self.data.shape[0] / n_all_files)

            # save dataset to pickle
            with open(pickle_path, 'wb') as f:
                pickle.dump((
                    self.data,
                    self.y_true,
                    self.condition,
                    self.n_vectors_ea_file,
                    self.basenames
                ), f, protocol=pickle.HIGHEST_PROTOCOL)

            count = 0
            while not com.is_enabled_pickle(pickle_path=pickle_path):
                assert count < 10
                print(f"{datetime.datetime.now()}\tpickle is not ready yet.")
                time.sleep(count+1)
                count+=1
                print(f"retry check pickle.\ncount : {count}")
            
            com.release_write_lock(
                lock=pickle_lock,
                lock_file_path=pickle_lockfile_path,
            )

    def load_pickle(self, pickle_path, pickle_lock=None, pickle_lock_file=None, retry_count=0, retry_lim=20, retry_delay_time=1):
        assert retry_count < retry_lim
        if pickle_lock:
            try:
                print("wait setting dataset")
                pickle_lock.acquire_read_lock()
            except:
                self.load_pickle(pickle_path=pickle_path)
                return
            com.release_read_lock(
                lock=pickle_lock,
                lock_file_path=pickle_lock_file,
            )
        print(f"load pickle : {pickle_path}")
        try:
            with open(pickle_path, 'rb') as f:
                self.data, self.y_true, self.condition, self.n_vectors_ea_file, self.basenames = pickle.load(f)
        except FileNotFoundError:
            print(f"{datetime.datetime.now()}\tFileNotFoundError: can not load pickle.")
            time.sleep(retry_delay_time)
            print(f"retry load pickle : {retry_count+1}/{retry_lim}")
            self.load_pickle(
                pickle_path=pickle_path,
                pickle_lock=pickle_lock,
                pickle_lock_file=pickle_lock_file,
                retry_count=retry_count+1,
                retry_lim=retry_lim,
                retry_delay_time=retry_delay_time+1,
            )

    def __getitem__(self, index):
        """
        Returns:
            Tensor: input data
            Tensor: anomaly label
            Tensor: one-hot vector for conditioning (section id)
            int: start index
            str: file basename
        """
        return self.getitem_fn(index)

    def default_item(self, index):
        data = self.data[index]
        y_true = self.y_true[index//self.n_vectors_ea_file] if len(self.y_true) > 0 else -1
        condition = self.condition[index]
        basename = self.basenames[index//self.n_vectors_ea_file]
        return data, y_true, condition, basename, index

    def __len__(self):
        return self.dataset_len

def file_list_to_data(file_list,
                        msg="calc...",
                        n_mels=64,
                        n_frames=5,
                        n_hop_frames=1,
                        n_fft=1024,
                        hop_length=512,
                        power=2.0,
                        fmax=None,
                        fmin=None,
                        win_length=None):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        data for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vectors = com.file_to_vectors(file_list[idx],
                                                n_mels=n_mels,
                                                n_frames=n_frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power,
                                                fmax=fmax,
                                                fmin=fmin,
                                                win_length=win_length)
        vectors = vectors[: : n_hop_frames, :]
        if idx == 0:
            data = np.zeros((len(file_list) * vectors.shape[0], dims), float)
        if len(file_list) * vectors.shape[0] > data.shape[0]:
            tmp_data = np.zeros((len(file_list) * vectors.shape[0], dims), float)
            tmp_data[:data.shape[0], :] = data
            data = tmp_data
        data[vectors.shape[0] * idx : vectors.shape[0] * (idx + 1), :] = vectors

    return data
