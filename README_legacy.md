# Legacy support

You can use DCASE2020 task2, DCASE2021 task2, and DCASE2022 task2 dataset for inputs.

## Description

Legacy support script are similar to main scripts. These are in `tools` directory.

- Helper scripts
  - tools/data_download_2020.sh
    - This script downloads development data and evaluation data files and put them into `data/dcase2020t2/dev_data/raw/` and `data/dcase2020t2/eval_data/raw/`.
    - Rename evaluation data after download dataset to evaluation and calculate AUC score. Renamed data is stored in `data/dcase2020t2/eval_data/raw/test_rename`
  - tools/data_download_2021.sh
    - This script downloads development data and evaluation data files and put them into `data/dcase2021t2/dev_data/raw/` and `data/dcase2021t2/eval_data/raw/`.
    - Merge `source_test` and `target_test` into `test` to be treated like any other.
    - Rename evaluation data after download dataset to evaluation and calculate AUC score. Renamed data is stored in `data/dcase2021t2/eval_data/raw/test_rename`
  - tools/data_download_2022.sh
    - This script downloads development data and evaluation data files and put them into `data/dcase2022t2/dev_data/raw/` and `data/dcase2022t2/eval_data/raw/`.
    - Rename evaluation data after download dataset to evaluation and calculate AUC score. Renamed data is stored in `data/dcase2022t2/eval_data/raw/test_rename`

- tools/01_train_legacy.sh
  - DCASE2020 task2 mode:
    - "Development" mode:
      - This script trains a model for each machine type for each section ID by using the directory `data/dcase2020t2/dev_data/raw/<machine_type>/train/<section_id>`
    - "Evaluation" mode:
      - This script trains a model for each machine type for each section ID by using the directory `data/dcase2020t2/eval_data/raw/<machine_type>/train/<section_id>`.
  - tools/DCASE2021 task2 mode:
    - "Development" mode:
      - This script trains a model for each machine type for each section ID by using the directory `data/dcase2021t2/dev_data/raw/<machine_type>/train/<section_id>`
    - "Evaluation" mode:
      - This script trains a model for each machine type for each section ID by using the directory `data/dcase2021t2/eval_data/raw/<machine_type>/train/<section_id>`.
  - DCASE2022 task2 mode:
    - "Development" mode:
      - This script trains a model for each machine type for each section ID by using the directory `data/dcase2022t2/dev_data/raw/<machine_type>/train/<section_id>`
    - "Evaluation" mode:
      - This script trains a model for each machine type for each section ID by using the directory `data/dcase2022t2/eval_data/raw/<machine_type>/train/<section_id>`.
  
- tools/02a_test_legacy.sh (Use MSE as a score function for the Simple Autoencoder mode)
  - DCASE2020 task2 mode:
    - "Development" mode:
      - This script makes a csv file for each section including the anomaly scores for each wav file in the directories `data/dcase2020t2/dev_data/raw/<machine_type>/test/`.
      - The csv files will be stored in the directory `results/`.
      - It also makes a csv file including AUC, pAUC, precision, recall, and F1-score for each section.
    - "Evaluation" mode:
      - This script makes a csv file for each section including the anomaly scores for each wav file in the directories `data/dcase2020t2/eval_data/raw/<machine_type>/test/`. (These directories will be made available with the "evaluation dataset".)
      - The csv files are stored in the directory `results/`.
      - If `test_rename` directory is available, makes a csv file including AUC, pAUC, precision, recall, and F1-score for each section.
  - DCASE2021 task2 mode:
    - "Development" mode:
      - This script makes a csv file for each section including the anomaly scores for each wav file in the directories `data/dcase2021t2/dev_data/raw/<machine_type>/test/`.
      - The csv files will be stored in the directory `results/`.
      - It also makes a csv file including AUC, pAUC, precision, recall, and F1-score for each section.
    - "Evaluation" mode:
      - This script makes a csv file for each section including the anomaly scores for each wav file in the directories `data/dcase2021t2/eval_data/raw/<machine_type>/test/`. (These directories will be made available with the "evaluation dataset".)
      - The csv files are stored in the directory `results/`.
      - If `test_rename` directory is available, makes a csv file including AUC, pAUC, precision, recall, and F1-score for each section.
  - DCASE2022 task2 mode:
    - "Development" mode:
      - This script makes a csv file for each section including the anomaly scores for each wav file in the directories `data/dcase2022t2/dev_data/raw/<machine_type>/test/`.
      - The csv files will be stored in the directory `results/`.
      - It also makes a csv file including AUC, pAUC, precision, recall, and F1-score for each section.
    - "Evaluation" mode:
      - This script makes a csv file for each section including the anomaly scores for each wav file in the directories `data/dcase2022t2/eval_data/raw/<machine_type>/test/`. (These directories will be made available with the "evaluation dataset".)
      - The csv files are stored in the directory `results/`.
      - If `test_rename` directory is available, makes a csv file including AUC, pAUC, precision, recall, and F1-score for each section.

- tools/02b_test_legacy.sh (Use Mahalanobis distance as a score function for the Selective Mahalanobis mode)
    - "Development" mode:
      - This script makes a csv file for each section including the anomaly scores for each wav file in the directories `data/dcase2021t2/dev_data/raw/<machine_type>/test/`.
      - The csv files will be stored in the directory `results/`.
      - It also makes a csv file including AUC, pAUC, precision, recall, and F1-score for each section.
    - "Evaluation" mode:
      - This script makes a csv file for each section including the anomaly scores for each wav file in the directories `data/dcase2021t2/eval_data/raw/<machine_type>/test/`. (These directories will be made available with the "evaluation dataset".)
      - The csv files are stored in the directory `results/`.
      - If `test_rename` directory is available, makes a csv file including AUC, pAUC, precision, recall, and F1-score for each section.
  - DCASE2022 task2 mode:
    - "Development" mode:
      - This script makes a csv file for each section including the anomaly scores for each wav file in the directories `data/dcase2022t2/dev_data/raw/<machine_type>/test/`.
      - The csv files will be stored in the directory `results/`.
      - It also makes a csv file including AUC, pAUC, precision, recall, and F1-score for each section.
    - "Evaluation" mode:
      - This script makes a csv file for each section including the anomaly scores for each wav file in the directories `data/dcase2022t2/eval_data/raw/<machine_type>/test/`. (These directories will be made available with the "evaluation dataset".)
      - The csv files are stored in the directory `results/`.
      - If `test_rename` directory is available, makes a csv file including AUC, pAUC, precision, recall, and F1-score for each section.
- 03_summarize_results.sh
  - This script summarize results into a csv file.
  - Use the same as when summarize DCASE2023T2 results.

## Usage

Legacy scripts in `tools` directory can be executed regardless of the current directory.

### 1. Download datasets
  + DCASE2020T2 task2
    + "Development dataset"
      + Download "dev\_data_<machine_type>.zip" from [https://zenodo.org/record/3678171](https://zenodo.org/record/3678171).
    + "Additional training dataset", i.e., the evaluation dataset for training
      + Download "eval\_data_<machine_type>_train.zip" from [https://zenodo.org/record/3727685](https://zenodo.org/record/3727685).
    + "Evaluation dataset", i.e., the evaluation dataset for test
      + Download "eval\_data_<machine_type>_test.zip" from [https://zenodo.org/record/3841772](https://zenodo.org/record/3841772).
  + DCASE2021T2 task2
    + "Development dataset"
      + Download "dev\_data_<machine_type>.zip" from [https://zenodo.org/record/4562016](https://zenodo.org/record/4562016).
    + "Additional training dataset", i.e., the evaluation dataset for training
      + Download "eval\_data_<machine_type>_train.zip" from [https://zenodo.org/record/4660992](https://zenodo.org/record/4660992).
    + "Evaluation dataset", i.e., the evaluation dataset for test
      + Download "eval\_data_<machine_type>_test.zip" from [https://zenodo.org/record/4884786](https://zenodo.org/record/4884786).
  + DCASE2022T2 task2
    + "Development dataset"
      + Download "dev\_data_<machine_type>.zip" from [https://zenodo.org/record/6355122](https://zenodo.org/record/6355122).
    + "Additional training dataset", i.e., the evaluation dataset for training
      + Download "eval\_data_<machine_type>_train.zip" from [https://zenodo.org/record/6462969](https://zenodo.org/record/6462969).
    + "Evaluation dataset", i.e., the evaluation dataset for test
      + Download "eval\_data_<machine_type>_test.zip" from [https://zenodo.org/record/6586456](https://zenodo.org/record/6586456).

### 3. Unzip the downloaded files and make the directory structure as the followings:

Legacy dataset directory structure is same as DCASE2023 task2. These parent directories are the result of downloaded all dataset.

- dcase2023_task2_baseline_ae
  - /data
    - /dcase2020t2
    - /dcase2021t2
    - /dcase2022t2
    - /dcase2023t2

[learn more of directory structure](#directory-structure-of-downloaded-dataset).

### 4. Change parameters

Change parameters using `baseline.yaml` in the same as [DCASE2023 mode](./README.md#4-change-parameters).

### 5. Run the training script

Run the training script `01_train_legacy.sh`. this script differs from `01_train.sh` in that it uses two options. First option is using dataset name. An example is `DCASE2020T2`. Second option chooses whether to use dev data or eval data. Use the options `DCASE2020T2` and `-d` for the development dataset `data/dcase2020t2/dev_data/<machine_type>/raw/train/`.

`01_train_legacy.sh` can used wherever the current directory. Specify by relative path.

```dotnetcli
# if your current directory is dcase2023_task2_baseline_ae.
$ bash tools/01_train.sh DCASE2020T2 -d
```

- First parameters
  - `DCASE2020T2`
  - `DCASE2021T2`
  - `DCASE2022T2`
- Second parameters
  - `-d`
  - `-e`

Others are the same as in [01_train.sh](./README.md#5-run-the-training-script-for-the-development-dataset).

### 6. Run the test script

### 6.1. Testing with the Simple Autoencoder mode

Run the training script `02a_test_legacy.sh`. this script differs from `02a_test.sh` in that it uses two options. First option is using dataset name. An example is `DCASE2020T2`. Second option chooses whether to use dev data or eval data. Use the options `DCASE2020T2` and `-d` for the development dataset `data/dcase2020t2/dev_data/<machine_type>/raw/test/`.

```dotnetcli
# if your current directory is dcase2023_task2_baseline_ae.
$ bash tools/02a_test_legacy.sh DCASE2020T2 -d
```
- First parameters
  - `DCASE2020T2`
  - `DCASE2021T2`
  - `DCASE2022T2`
- Second parameters
  - `-d`
  - `-e`

Others are the same as in [02a_test_legacy.sh](./README.md#61-testing-with-the-simple-autoencoder-mode).

### 6.2. Testing with the Selective Mahalanobis mode

Run the training script `02b_test_legacy.sh`. this script differs from `02b_test.sh` in that it uses two options. First option is using dataset name. An example is `DCASE2020T2`. Second option chooses whether to use dev data or eval data. Use the options `DCASE2020T2` and `-d` for the development dataset `data/dcase2020t2/dev_data/<machine_type>/raw/test/`.

```dotnetcli
# if your current directory is dcase2023_task2_baseline_ae.
$ bash tools/02b_test_legacy.sh DCASE2020T2 -d
```
- First parameters
  - `DCASE2020T2`
  - `DCASE2021T2`
  - `DCASE2022T2`
- Second parameters
  - `-d`
  - `-e`

Others are the same as in [02a_test_legacy.sh](./README.md#61-testing-with-the-simple-autoencoder-mode).

### 7. Check results

You can check the anomaly scores in the csv files `anomaly_score_<machine_type>_section_<section_index>_test.csv` in the directory `results/`.
Each anomaly score corresponds to a wav file in the directories. If you were learning with `DCASE2020T2` then `data/dcase2023t2/dev_data/<machine_type>/test/`.

Also, anomaly detection results based on the corresponding threshold can be checked in the csv files `decision_result_<machine_type>_section_<section_index>_test.csv`.
In addition, you can check performance indicators such as AUC, pAUC, precision, recall, and F1-score.

### 8. Summarize results

After the executed `02a_test_legacy.sh`, `02b_test_legacy.sh`, or both. Run the summarize script `03_summarize_results.sh` with options are same as `01_train_legacy.sh`, `02a_test_legacy.sh` and `02_b_test_legacy.sh`.

```dotnetcli
# Summarize development dataset
$ 03_summarize_results.sh DCASE2020T2 -d

# Summarize evaluation dataset
$ 03_summarize_results.sh DCASE2020T2 -e
```

- First parameters
  - `DCASE2020T2`
  - `DCASE2021T2`
  - `DCASE2022T2`
  - `DCASE2023T2`
- Second parameters
  - `-d`
  - `-e`

If you want to change summarize results directory or export directory, edit `03_summarize_results.sh`.

## Directory structure of downloaded dataset

Note that wav file's parent directory. At that time dataset directory is `dev_data` and `eval_data`, but in this repository it is `data/dcase202xt2/dev_data/raw` and `data/dcase202xt2/eval_data/raw`.

### DCASE2020 task2

  - dcase2023_task2_baseline_ae
    - /data/dcase2020t2/dev_data/raw
        - /ToyCar
            - /train (Only normal data for all Machine IDs are included.)
                - /normal_id_01_00000000.wav
                - ...
                - /normal_id_01_00000999.wav
                - /normal_id_02_00000000.wav
                - ...
                - /normal_id_04_00000999.wav
            - /test (Normal and anomaly data for all Machine IDs are included.)
                - /normal_id_01_00000000.wav
                - ...
                - /normal_id_01_00000349.wav
                - /anomaly_id_01_00000000.wav
                - ...
                - /anomaly_id_01_00000263.wav
                - /normal_id_02_00000000.wav
                - ...
                - /anomaly_id_04_00000264.wav
        - /ToyConveyor (The other Machine Types have the same directory structure as ToyCar.)
        - /fan
        - /pump
        - /slider
        - /valve
    - /data/dcase2020t2/eval_data/raw
        - /ToyCar
            - /train (Unzipped "additional training dataset". Only normal data for all Machine IDs are included.)
                - /normal_id_05_00000000.wav
                - ...
                - /normal_id_05_00000999.wav
                - /normal_id_06_00000000.wav
                - ...
                - /normal_id_07_00000999.wav
            - /test (Unzipped "evaluation dataset". Normal and anomaly data for all Machine IDs are included, but there is no label about normal or anomaly.)
                - /id_05_00000000.wav
                - ...
                - /id_05_00000514.wav
                - /id_06_00000000.wav
                - ...
                - /id_07_00000514.wav
            - test_rename/ (convert from test directory using `tools/rename.py`)
                - /normal_id_05_00000000.wav
                - ...
                - /normal_id_05_00000249.wav
                - /anomaly_id_05_00000000.wav
                - ...
                - /anomaly_id_05_00000264.wav
                - /normal_id_06_00000000.wav
                - ...
                - /anomaly_id_07_00000264.wav
        - /ToyConveyor (The other machine types have the same directory structure as ToyCar.)
        - /fan
        - /pump
        - /slider
        - /valve

### DCASE2021 task2

- dcase2023_task2_baseline_ae
  - /data/dcase2021t2/dev_data/raw
      - /fan
          - /train (Normal data in the **source** and **target** domains for all sections are included.)
              - /section_00_source_train_normal_0000_<attribute>.wav
              - ...
              - /section_00_source_train_normal_0999_<attribute>.wav
              - /section_00_target_train_normal_0000_<attribute>.wav
              - /section_00_target_train_normal_0001_<attribute>.wav
              - /section_00_target_train_normal_0002_<attribute>.wav
              - /section_01_source_train_normal_0000_<attribute>.wav
              - ...
              - /section_02_target_train_normal_0002_<attribute>.wav
          - /source_test (Normal and anomaly data in the **source** domain for all sections are included.)
          - /target_test (Normal and anomaly data in the **target** domain for all sections are included.)
          - /test (Normal and anomaly data in the **source** and **target** domains for all sections are included)
              - /section_00_source_test_normal_0000.wav
              - ...
              - /section_00_source_test_normal_0099.wav
              - /section_00_source_test_anomaly_0000.wav
              - ...
              - /section_00_source_test_anomaly_0099.wav
              - /section_01_source_test_normal_0000.wav
              - ...
              - /section_02_source_test_anomaly_0099.wav
              - /section_00_target_test_normal_0000.wav
              - ...
              - /section_00_target_test_normal_0099.wav
              - /section_00_target_test_anomaly_0000.wav
              - ...
              - /section_00_target_test_anomaly_0099.wav
              - /section_01_target_test_normal_0000.wav
              - ...
              - /section_02_target_test_anomaly_0099.wav
      - /gearbox (The other machine types have the same directory structure as fan.)
      - /pump
      - /slider
      - /valve
      - /ToyCar
      - /ToyTrain
  - /data/dcase2021t2/eval_data/raw
      - /fan
          - /train (Unzipped additional training dataset. Normal data in the **source** and **target** domains for all sections are included.)
              - /section_03_source_train_normal_0000_<attribute>.wav
              - ...
              - /section_03_source_train_normal_0999_<attribute>.wav
              - /section_03_target_train_normal_0000_<attribute>.wav
              - /section_03_target_train_normal_0001_<attribute>.wav
              - /section_03_target_train_normal_0002_<attribute>.wav
              - /section_04_source_train_normal_0000_<attribute>.wav
              - ...
              - /section_05_target_train_normal_0002_<attribute>.wav
          - /source_test (Unzipped evaluation dataset. Normal and anomaly data in the **source** domain for all sections are included.)
          - /target_test (Unzipped evaluation dataset. Normal and anomaly data in the **target** domain for all sections are included.)
          - /test (Unzipped evaluation dataset. Normal and anomaly data in the **source** and **target** domains for all sections are included)
              - /section_03_source_test_0000.wav
              - ...
              - /section_03_source_test_0199.wav
              - /section_04_source_test_0000.wav
              - ...
              - /section_05_source_test_0199.wav
              - /section_03_target_test_0000.wav
              - ...
              - /section_03_target_test_0199.wav
              - /section_04_target_test_0000.wav
              - ...
              - /section_05_target_test_0199.wav
          - test_rename/ (convert from test directory using `tools/rename.py`)
              - /section_03_source_test_normal_0000.wav
              - ...
              - /section_03_source_test_normal_0099.wav
              - /section_03_source_test_anomaly_0000.wav
              - ...
              - /section_03_source_test_anomaly_0099.wav
              - /section_04_source_test_normal_0000.wav
              - ...
              - /section_05_source_test_anomaly_0099.wav
              - /section_03_target_test_normal_0000.wav
              - ...
              - /section_03_target_test_normal_0099.wav
              - /section_03_target_test_anomaly_0000.wav
              - ...
              - /section_03_target_test_anomaly_0099.wav
              - /section_04_target_test_normal_0000.wav
              - ...
              - /section_05_target_test_anomaly_0099.wav
      - /gearbox (The other machine types have the same directory structure as fan.)
      - /pump
      - /slider
      - /valve
      - /ToyCar
      - /ToyTrain

### DCASE2022 task2

- dcase2023_task2_baseline_ae
  - /data/dcase2022t2/dev_data/raw
    - /fan
        - /train (only normal clips)  
            - /section_00_source_train_normal_0000_<attribute>.wav  
            - ...  
            - /section_00_source_train_normal_0989_<attribute>.wav  
            - /section_00_target_train_normal_0000_<attribute>.wav  
            - ...  
            - /section_00_target_train_normal_0009_<attribute>.wav  
            - /section_01_source_train_normal_0000_<attribute>.wav  
            - ...  
            - /section_02_target_train_normal_0009_<attribute>.wav  
        - /test 
            - /section_00_source_test_normal_0000_<attribute>.wav    
            - ...  
            - /section_00_source_test_normal_0049_<attribute>.wav    
            - /section_00_source_test_anomaly_0000_<attribute>.wav  
            - ...  
            - /section_00_source_test_anomaly_0049_<attribute>.wav  
            - /section_00_target_test_normal_0000_<attribute>.wav
            - ...  
            - /section_00_target_test_normal_0049_<attribute>.wav 
            - /section_00_target_test_anomaly_0000_<attribute>.wav  
            - ...  
            - /section_00_target_test_anomaly_0049_<attribute>.wav 
            - /section_01_source_test_normal_0000_<attribute>.wav
            - ...  
            - /section_02_target_test_anomaly_0049_<attribute>.wav
        - attributes_00.csv (attribute csv for section 00)
        - attributes_01.csv (attribute csv for section 01)
        - attributes_02.csv (attribute csv for section 02)      
    - /gearbox (The other machine types have the same directory structure as fan.)  
    - /bearing
    - /slider (`slider` means "slide rail")
    - /ToyCar  
    - /ToyTrain  
    - /valve  
  - /data/dcase2022t2/eval_data/raw
    - /fan  
        - /train (after launch of the additional training dataset)  
            - /section_03_source_train_normal_0000_<attribute>.wav  
            - ...  
            - /section_03_source_train_normal_0989_<attribute>.wav  
            - /section_03_target_train_normal_0000_<attribute>.wav  
            - ...  
            - /section_03_target_train_normal_0009_<attribute>.wav  
            - /section_04_source_train_normal_0000_<attribute>.wav  
            - ...  
            - /section_05_target_train_normal_0009_<attribute>.wav  
        - /test (after launch of the evaluation dataset)  
            - /section_03_test_0000.wav  
            - ...  
            - /section_03_test_0199.wav  
            - /section_04_test_0000.wav  
            - ...  
            - /section_05_test_0199.wav  
        - test_rename/ (convert from test directory using `tools/rename.py`)
            - /section_03_source_test_normal_0000_<attribute>.wav    
            - ...  
            - /section_03_source_test_normal_0049_<attribute>.wav    
            - /section_03_source_test_anomaly_0000_<attribute>.wav  
            - ...  
            - /section_03_source_test_anomaly_0049_<attribute>.wav  
            - /section_03_target_test_normal_0000_<attribute>.wav
            - ...  
            - /section_03_target_test_normal_0049_<attribute>.wav 
            - /section_03_target_test_anomaly_0000_<attribute>.wav  
            - ...  
            - /section_03_target_test_anomaly_0049_<attribute>.wav 
            - /section_04_source_test_normal_0000_<attribute>.wav
            - ...  
            - /section_05_target_test_anomaly_0049_<attribute>.wav
        - attributes_03.csv (attribute csv for train data in section 03)
        - attributes_04.csv (attribute csv for train data in section 04)
        - attributes_05.csv (attribute csv for train data in section 05) 
    - /gearbox (The other machine types have the same directory structure as fan.)  
    - /bearing  
    - /slider (`slider` means "slide rail")
    - /ToyCar  
    - /ToyTrain  
    - /valve  