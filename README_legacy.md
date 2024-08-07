# Legacy support

This version supports reading the datasets from DCASE2020 task2, DCASE2021 task2, DCASE2022 task2, DCASE2023 task2 and DCASE2024 task2 dataset for inputs.

## Description

Legacy-support scripts are similar to the main scripts. These are in `tools` directory.

- Helper scripts
  - tools/data\_download\_2020.sh
    - This script downloads development and evaluation data files into `data/dcase2020t2/dev_data/raw/` and `data/dcase2020t2/eval_data/raw/`.
    - Rename evaluation data after downloading the dataset to evaluate and calculate AUC score. Renamed data is stored in `data/dcase2020t2/eval_data/raw/test_rename`
  - tools/data\_download\_2021.sh
    - This script downloads development data and evaluation data files and puts them into `data/dcase2021t2/dev_data/raw/` and `data/dcase2021t2/eval_data/raw/`.
    - Merge `source_test` and `target_test` into `test` to be treated like any other.
    - Rename evaluation data after downloading the dataset to evaluate and calculate the AUC score. Renamed data is stored in `data/dcase2021t2/eval_data/raw/test_rename`
  - tools/data\_download\_2022.sh
    - This script downloads development data and evaluation data files and puts them into `data/dcase2022t2/dev_data/raw/` and `data/dcase2022t2/eval_data/raw/`.
    - Rename evaluation data after downloading the dataset to evaluate and calculate AUC score. Renamed data is stored in `data/dcase2022t2/eval_data/raw/test_rename`
  - tools/data\_download\_2023.sh
    - This script downloads development data and evaluation data files and puts them into `data/dcase2023t2/dev_data/raw/` and `data/dcase2023t2/eval_data/raw/`.
    - Rename evaluation data after downloading the dataset to evaluate and calculate AUC score. Renamed data is stored in `data/dcase2023t2/eval_data/raw/test_rename`
  - tools/data\_download\_2024.sh
    - This script downloads development data and evaluation data files and puts them into `data/dcase2024t2/dev_data/raw/` and `data/dcase2024t2/eval_data/raw/`.
    - Rename evaluation data after downloading the dataset to evaluate and calculate AUC score. Renamed data is stored in `data/dcase2024t2/eval_data/raw/test_rename`


- tools/01\_train\_legacy.sh
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
  - DCASE2023 task2 mode:
    - "Development" mode:
      - This script trains a model for each machine type for each section ID by using the directory `data/dcase2023t2/dev_data/raw/<machine_type>/train/<section_id>`
    - "Evaluation" mode:
      - This script trains a model for each machine type for each section ID by using the directory `data/dcase2023t2/eval_data/raw/<machine_type>/train/<section_id>`.
  - DCASE2024 task2 mode:
    - "Development" mode:
      - This script trains a model for each machine type for each section ID by using the directory `data/dcase2024t2/dev_data/raw/<machine_type>/train/<section_id>`
    - "Evaluation" mode:
      - This script trains a model for each machine type for each section ID by using the directory `data/dcase2024t2/eval_data/raw/<machine_type>/train/<section_id>`.

  
- tools/02a\_test\_legacy.sh (Use MSE as a score function for the Simple Autoencoder mode)
  - DCASE2020 task2 mode:
    - "Development" mode:
      - This script generates a CSV file for each section, including the anomaly scores for each WAV file in the directories `data/dcase2020t2/dev_data/raw/<machine_type>/test/`.
      - The generated CSV files will be stored in the directory `results/`.
      - The generated CSV file contains AUC, pAUC, precision, recall, and F1-score for each section.
    - "Evaluation" mode:
      - This script generates a CSV file for each section, including the anomaly scores for each WAV file in the directories `data/dcase2020t2/eval_data/raw/<machine_type>/test/`. (These directories will be made available with the "evaluation dataset".)
      - The generated CSV files are stored in the directory `results/`.
      - If `test_rename` directory is available, this script generates a CSV file that contains AUC, pAUC, precision, recall, and F1-score for each section.
  - DCASE2021 task2 mode:
    - "Development" mode:
      - This script generates a CSV file for each section, including the anomaly scores for each wav file in the directories `data/dcase2021t2/dev_data/raw/<machine_type>/test/`.
      - The generated CSV files will be stored in the directory `results/`.
      - This script also generates a CSV file that contains AUC, pAUC, precision, recall, and F1-score for each section.
    - "Evaluation" mode:
      - This script generates a CSV file for each section, including the anomaly scores for each wav file in the directories `data/dcase2021t2/eval_data/raw/<machine_type>/test/`. (These directories will be made available with the "evaluation dataset".)
      - The generated CSV files are stored in the directory `results/`.
      - If `test_rename` directory is available, this script generates a CSV file including AUC, pAUC, precision, recall, and F1-score for each section.
  - DCASE2022 task2 mode:
    - "Development" mode:
      - This script generates a CSV file for each section, including the anomaly scores for each wav file in the directories `data/dcase2022t2/dev_data/raw/<machine_type>/test/`.
      - The generated CSV files will be stored in the directory `results/`.
      - It also generates a CSV file including AUC, pAUC, precision, recall, and F1-score for each section.
    - "Evaluation" mode:
      - This script generates a CSV file for each section, including the anomaly scores for each wav file in the directories `data/dcase2022t2/eval_data/raw/<machine_type>/test/`. (These directories will be made available with the "evaluation dataset".)
      - The generated CSV files are stored in the directory `results/`.
      - If `test_rename` directory is available, this script generates a CSV file including AUC, pAUC, precision, recall, and F1-score for each section.
  - DCASE2023 task2 mode:
    - "Development" mode:
      - This script generates a CSV file for each section, including the anomaly scores for each wav file in the directories `data/dcase2023t2/dev_data/raw/<machine_type>/test/`.
      - The generated CSV files will be stored in the directory `results/`.
      - It also generates a CSV file including AUC, pAUC, precision, recall, and F1-score for each section.
    - "Evaluation" mode:
      - This script generates a CSV file for each section, including the anomaly scores for each wav file in the directories `data/dcase2023t2/eval_data/raw/<machine_type>/test/`. (These directories will be made available with the "evaluation dataset".)
      - The generated CSV files are stored in the directory `results/`.
      - If `test_rename` directory is available, this script generates a CSV file including AUC, pAUC, precision, recall, and F1-score for each section.
  - DCASE2024 task2 mode:
    - "Development" mode:
      - This script generates a CSV file for each section, including the anomaly scores for each wav file in the directories `data/dcase2024t2/dev_data/raw/<machine_type>/test/`.
      - The generated CSV files will be stored in the directory `results/`.
      - It also generates a CSV file including AUC, pAUC, precision, recall, and F1-score for each section.
    - "Evaluation" mode:
      - This script generates a CSV file for each section, including the anomaly scores for each wav file in the directories `data/dcase2024t2/eval_data/raw/<machine_type>/test/`. (These directories will be made available with the "evaluation dataset".)
      - The generated CSV files are stored in the directory `results/`.
      - If `test_rename` directory is available, this script generates a CSV file including AUC, pAUC, precision, recall, and F1-score for each section.

- tools/02b\_test\_legacy.sh (Use Mahalanobis distance as a score function for the Selective Mahalanobis mode)
    - "Development" mode:
      - This script generates a CSV file for each section, including the anomaly scores for each wav file in the directories `data/dcase2021t2/dev_data/raw/<machine_type>/test/`.
      - The CSV files will be stored in the directory `results/`.
      - It also generates a CSV file including AUC, pAUC, precision, recall, and F1-score for each section.
    - "Evaluation" mode:
      - This script generates a CSV file for each section, including the anomaly scores for each wav file in the directories `data/dcase2021t2/eval_data/raw/<machine_type>/test/`. (These directories will be made available with the "evaluation dataset".)
      - The CSV files are stored in the directory `results/`.
      - If `test_rename` directory is available, this script generates a CSV file including AUC, pAUC, precision, recall, and F1-score for each section.
  - DCASE2022 task2 mode:
    - "Development" mode:
      - This script generates a CSV file for each section, including the anomaly scores for each wav file in the directories `data/dcase2022t2/dev_data/raw/<machine_type>/test/`.
      - The CSV files will be stored in the directory `results/`.
      - It also makes a csv file including AUC, pAUC, precision, recall, and F1-score for each section.
    - "Evaluation" mode:
      - This script generates a CSV file for each section, including the anomaly scores for each wav file in the directories `data/dcase2022t2/eval_data/raw/<machine_type>/test/`. (These directories will be made available with the "evaluation dataset".)
      - The generated CSV files are stored in the directory.
      - This script also generates a CSV file, containing AUC, pAUC, precision, recall, and F1-score for each section.
  - DCASE2023 task2 mode:
    - "Development" mode:
      - This script generates a CSV file for each section, including the anomaly scores for each wav file in the directories `data/dcase2023t2/dev_data/raw/<machine_type>/test/`.
      - The CSV files will be stored in the directory `results/`.
      - It also makes a csv file including AUC, pAUC, precision, recall, and F1-score for each section.
    - "Evaluation" mode:
      - This script generates a CSV file for each section, including the anomaly scores for each wav file in the directories `data/dcase2023t2/eval_data/raw/<machine_type>/test/`. (These directories will be made available with the "evaluation dataset".)
      - The generated CSV files are stored in the directory.
      - This script also generates a CSV file, containing AUC, pAUC, precision, recall, and F1-score for each section.
  - DCASE2024 task2 mode:
    - "Development" mode:
      - This script generates a CSV file for each section, including the anomaly scores for each wav file in the directories `data/dcase2024t2/dev_data/raw/<machine_type>/test/`.
      - The CSV files will be stored in the directory `results/`.
      - It also makes a csv file including AUC, pAUC, precision, recall, and F1-score for each section.
    - "Evaluation" mode:
      - This script generates a CSV file for each section, including the anomaly scores for each wav file in the directories `data/dcase2024t2/eval_data/raw/<machine_type>/test/`. (These directories will be made available with the "evaluation dataset".)
      - The generated CSV files are stored in the directory.
      - This script also generates a CSV file, containing AUC, pAUC, precision, recall, and F1-score for each section.
- 03_summarize_results.sh
  - This script summarizes results into a csv file.
  - Use the same as when summarizing DCASE2023T2 and DCASE2024T2 results.

## Usage

Legacy scripts in `tools` directory can be executed regardless of the current directory.

### 1. Download datasets
  + DCASE2020T2
    + "Development dataset"
      + Download "dev\_data\_\<machine\_type\>\.zip" from [https://zenodo.org/record/3678171](https://zenodo.org/record/3678171).
    + "Additional training dataset", i.e., the evaluation dataset for training
      + Download "eval\_data\_<machine\_type\>\_train.zip" from [https://zenodo.org/record/3727685](https://zenodo.org/record/3727685).
    + "Evaluation dataset", i.e., the evaluation dataset for test
      + Download "eval\_data\_\<machine\_type\>\_test.zip" from [https://zenodo.org/record/3841772](https://zenodo.org/record/3841772).
  + DCASE2021T2
    + "Development dataset"
      + Download "dev\_data\_\<machine\_type\>.zip" from [https://zenodo.org/record/4562016](https://zenodo.org/record/4562016).
    + "Additional training dataset", i.e., the evaluation dataset for training
      + Download "eval\_data\_\<machine\_\type\>_train.zip" from [https://zenodo.org/record/4660992](https://zenodo.org/record/4660992).
    + "Evaluation dataset", i.e., the evaluation dataset for test
      + Download "eval\_data\_\<machine\_type>\_test.zip" from [https://zenodo.org/record/4884786](https://zenodo.org/record/4884786).
  + DCASE2022T2
    + "Development dataset"
      + Download "dev\_data\_\<machine\_type\>.zip" from [https://zenodo.org/record/6355122](https://zenodo.org/record/6355122).
    + "Additional training dataset", i.e., the evaluation dataset for training
      + Download "eval\_data\_\<machine\_type\>\_train.zip" from [https://zenodo.org/record/6462969](https://zenodo.org/record/6462969).
    + "Evaluation dataset", i.e., the evaluation dataset for test
      + Download "eval\_data\_\<machine\_type\>\_test.zip" from [https://zenodo.org/record/6586456](https://zenodo.org/record/6586456).
  + DCASE2023T2
    + "Development Dataset"
      + Download "dev\_data_<machine_type>.zip" from [https://zenodo.org/record/7882613](https://zenodo.org/record/7882613).
    + "Additional Training Dataset", i.e., the evaluation dataset for training 
      + Download "eval\_data_<machine_type>_train.zip" from [https://zenodo.org/record/7830345](https://zenodo.org/record/7830345).
    + "Evaluation Dataset", i.e., the evaluation dataset for test
      + Download "eval\_data_<machine_type>_test.zip" from [https://zenodo.org/record/7860847](https://zenodo.org/record/7860847).
  + DCASE2024T2
    + "Development Dataset"
      + Download "dev\_data_<machine_type>.zip" from [https://zenodo.org/records/10902294](https://zenodo.org/records/10902294).
    + "Additional Training Dataset", i.e., the evaluation dataset for training 
      + Download "eval\_data_<machine_type>_train.zip" from [https://zenodo.org/records/11259435](https://zenodo.org/records/11259435).
    + "Evaluation Dataset", i.e., the evaluation dataset for test
      + Download "eval\_data_<machine_type>_test.zip" from [https://zenodo.org/records/11363076](https://zenodo.org/records/11363076).


### 3. Unzip the downloaded files and make the directory structure as follows:

The legacy dataset directory structure is the same as DCASE2023 task2. These parent directories are the result of all the downloaded datasets.

- dcase2023\_task2\_baseline\_ae
  - /data
    - /dcase2020t2
    - /dcase2021t2
    - /dcase2022t2
    - /dcase2023t2
    - /dcase2024t2

[learn more about directory structure](#directory-structure-of-downloaded-dataset).

### 4. Change parameters

Change parameters using `baseline.yaml` in the same as [DCASE2024 mode](./README.md#4-change-parameters).

### 4.1. Enable Auto-download dataset

If you haven't yet downloaded the dataset yourself nor you have not run the download script (example, `data_download_2020.sh`) then you may want to use the auto download.
To enable the auto-downloading, set the parameter `--is_auto_download` (default: `False`) `True` in `baseline.yaml`. If `--is_auto_download` is `True`, then auto-download is executed.

### 5. Run the training script

Run the training script `01_train_legacy.sh`. this script differs from `01_train.sh` in using two options. The first option is using the dataset name. An example is `DCASE2020T2`. The second option chooses whether to use dev data or eval data. Use the options `DCASE2020T2` and `-d` for the development dataset `data/dcase2020t2/dev_data/<machine_type>/raw/train/`.

`01_train_legacy.sh` can be used wherever the current directory. Specify it in a relative path.

```dotnetcli
# if your current directory is dcase2023_task2_baseline_ae.
$ bash tools/01_train.sh DCASE2020T2 -d
```

- First parameters
  - `DCASE2020T2`
  - `DCASE2021T2`
  - `DCASE2022T2`
  - `DCASE2023T2`
  - `DCASE2024T2`
- Second parameters
  - `-d`
  - `-e`

Others are the same as in [01_train.sh](./README.md#5-run-the-training-script-for-the-development-dataset).

### 6. Run the test script

### 6.1. Testing with the Simple Autoencoder mode

Run the training script `02a_test_legacy.sh`. this script differs from `02a_test.sh` in using two options. The first option is using the dataset name. An example is `DCASE2020T2`. The second option chooses whether to use dev data or eval data. Use the options `DCASE2020T2` and `-d` for the development dataset `data/dcase2020t2/dev_data/<machine_type>/raw/test/`.

```dotnetcli
# if your current directory is dcase2023_task2_baseline_ae.
$ bash tools/02a_test_legacy.sh DCASE2020T2 -d
```
- First parameters
  - `DCASE2020T2`
  - `DCASE2021T2`
  - `DCASE2022T2`
  - `DCASE2023T2`
  - `DCASE2024T2`
- Second parameters
  - `-d`
  - `-e`

Others are the same as in [02a_test_legacy.sh](./README.md#61-testing-with-the-simple-autoencoder-mode).

### 6.2. Testing with the Selective Mahalanobis mode

Run the training script `02b_test_legacy.sh`. this script differs from `02b_test.sh` in using two options. The first option is using the dataset name. An example is `DCASE2020T2`. The second option chooses whether to use dev data or eval data. Use the options `DCASE2020T2` and `-d` for the development dataset `data/dcase2020t2/dev_data/<machine_type>/raw/test/`.

```dotnetcli
# if your current directory is dcase2023_task2_baseline_ae.
$ bash tools/02b_test_legacy.sh DCASE2020T2 -d
```
- First parameters
  - `DCASE2020T2`
  - `DCASE2021T2`
  - `DCASE2022T2`
  - `DCASE2023T2`
  - `DCASE2024T2`
- Second parameters
  - `-d`
  - `-e`

Others are the same as in [02a_test_legacy.sh](./README.md#61-testing-with-the-simple-autoencoder-mode).

### 7. Check results

You can check the anomaly scores in the csv files `anomaly_score_<machine_type>_section_<section_index>_test.csv` in the directory `results/`.
Each anomaly score corresponds to a wav file in the directories. If you were learning with `DCASE2020T2` then `data/dcase2020t2/dev_data/<machine_type>/test/`.

Also, anomaly detection results based on the corresponding threshold can be checked in the CSV files `decision_result_<machine_type>_section_<section_index>_test.csv`.
In addition, you can check performance indicators such as AUC, pAUC, precision, recall, and F1-score.

### 8. Summarize results

After the executed `02a_test_legacy.sh`, `02b_test_legacy.sh`, or both. Run the summarize script `03_summarize_results.sh` with options that are the same as `01_train_legacy.sh`, `02a_test_legacy.sh` and `02_b_test_legacy.sh`.

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
  - `DCASE2024T2`
- Second parameters
  - `-d`
  - `-e`

If you want to change, summarize results directory or export directory, edit `03_summarize_results.sh`.

## Directory structure of the downloaded dataset

Note that the wav file's parent directory. At that time dataset directory is `dev_data` and `eval_data`, but in this repository, it is `data/dcase202xt2/dev_data/raw` and `data/dcase202xt2/eval_data/raw`.

### DCASE2020 task2

  - dcase2023\_task2\_baseline\_ae
    - /data/dcase2020t2/dev\_data/raw
        - /ToyCar
            - /train (Only normal data for all Machine IDs are included.)
                - /normal\_id\_01\_00000000.wav
                - ...
                - /normal\_id\_01\_00000999.wav
                - /normal\_id\_02\_00000000.wav
                - ...
                - /normal\_id\_04\_00000999.wav
            - /test (Normal and anomaly data for all Machine IDs are included.)
                - /normal\_id\_01\_00000000.wav
                - ...
                - /normal\_id\_01\_00000349.wav
                - /anomaly\_id\_01\_00000000.wav
                - ...
                - /anomaly\_id\_01\_00000263.wav
                - /normal\_id\_02\_00000000.wav
                - ...
                - /anomaly\_id\_04\_00000264.wav
        - /ToyConveyor (The other Machine Types have the same directory structure as ToyCar.)
        - /fan
        - /pump
        - /slider
        - /valve
    - /data/dcase2020t2/eval\_data/raw
        - /ToyCar
            - /train (Unzipped "additional training dataset". Only normal data for all Machine IDs are included.)
                - /normal\_id\_05\_00000000.wav
                - ...
                - /normal\_id\_05\_00000999.wav
                - /normal\_id\_06\_00000000.wav
                - ...
                - /normal\_id\_07\_00000999.wav
            - /test (Unzipped "evaluation dataset". Normal and anomaly data for all Machine IDs are included, but there is no label about normal or anomaly.)
                - /id\_05\_00000000.wav
                - ...
                - /id\_05\_00000514.wav
                - /id\_06\_00000000.wav
                - ...
                - /id\_07\_00000514.wav
            - test_rename/ (convert from test directory using `tools/rename.py`)
                - /normal\_id\_05\_00000000.wav
                - ...
                - /normal\_id\_05\_00000249.wav
                - /anomaly\_id\_05\_00000000.wav
                - ...
                - /anomaly\_id\_05\_00000264.wav
                - /normal\_id\_06\_00000000.wav
                - ...
                - /anomaly\_id\_07\_00000264.wav
        - /ToyConveyor (The other machine types have the same directory structure as ToyCar.)
        - /fan
        - /pump
        - /slider
        - /valve

### DCASE2021 task2

- dcase2023\_task2\_baseline\_ae
  - /data/dcase2021t2/dev\_data/raw
      - /fan
          - /train (Normal data in the **source** and **target** domains for all sections are included.)
              - /section\_00\_source\_train\_normal\_0000\_\<attribute\>.wav
              - ...
              - /section\_00\_source\_train\_normal\_0999\_\<attribute\>.wav
              - /section\_00\_target\_train\_normal\_0000\_\<attribute\>.wav
              - /section\_00\_target\_train\_normal\_0001\_\<attribute\>.wav
              - /section\_00\_target\_train\_normal\_0002\_\<attribute\>.wav
              - /section\_01\_source\_train\_normal\_0000\_\<attribute\>.wav
              - ...
              - /section\_02\_target\_train\_normal\_0002\_\<attribute\>.wav
          - /source\_test (Normal and anomaly data in the **source** domain for all sections are included.)
          - /target\_test (Normal and anomaly data in the **target** domain for all sections are included.)
          - /test (Normal and anomaly data in the **source** and **target** domains for all sections are included)
              - /section\_00\_source\_test\_normal\_0000.wav
              - ...
              - /section\_00\_source\_test\_normal\_0099.wav
              - /section\_00\_source\_test\_anomaly\_0000.wav
              - ...
              - /section\_00\_source\_test\_anomaly\_0099.wav
              - /section\_01\_source\_test\_normal\_0000.wav
              - ...
              - /section\_02\_source\_test\_anomaly\_0099.wav
              - /section\_00\_target\_test\_normal\_0000.wav
              - ...
              - /section\_00\_target\_test\_normal\_0099.wav
              - /section\_00\_target\_test\_anomaly\_0000.wav
              - ...
              - /section\_00\_target\_test\_anomaly\_0099.wav
              - /section\_01\_target\_test\_normal\_0000.wav
              - ...
              - /section\_02\_target\_test\_anomaly\_0099.wav
      - /gearbox (The other machine types have the same directory structure as fan.)
      - /pump
      - /slider
      - /valve
      - /ToyCar
      - /ToyTrain
  - /data/dcase2021t2/eval\_data/raw
      - /fan
          - /train (Unzipped additional training dataset. Normal data in the **source** and **target** domains for all sections are included.)
              - /section\_03\_source\_train\_normal\_0000\_\<attribute\>.wav
              - ...
              - /section\_03\_source\_train\_normal\_0999\_\<attribute\>.wav
              - /section\_03\_target\_train\_normal\_0000\_\<attribute\>.wav
              - /section\_03\_target\_train\_normal\_0001\_\<attribute\>.wav
              - /section\_03\_target\_train\_normal\_0002\_\<attribute\>.wav
              - /section\_04\_source\_train\_normal\_0000\_\<attribute\>.wav
              - ...
              - /section\_05\_target\_train\_normal\_0002\_\<attribute\>.wav
          - /source_test (Unzipped evaluation dataset. Normal and anomaly data in the **source** domain for all sections are included.)
          - /target\_test (Unzipped evaluation dataset. Normal and anomaly data in the **target** domain for all sections are included.)
          - /test (Unzipped evaluation dataset. Normal and anomaly data in the **source** and **target** domains for all sections are included)
              - /section\_03\_source\_test\_0000.wav
              - ...
              - /section\_03\_source\_test\_0199.wav
              - /section\_04\_source\_test\_0000.wav
              - ...
              - /section\_05\_source\_test\_0199.wav
              - /section\_03\_target\_test\_0000.wav
              - ...
              - /section\_03\_target\_test\_0199.wav
              - /section\_04\_target\_test\_0000.wav
              - ...
              - /section\_05\_target\_test\_0199.wav
          - test_rename/ (convert from test directory using `tools/rename.py`)
              - /section\_03\_source\_test\_normal\_0000.wav
              - ...
              - /section\_03\_source\_test\_normal\_0099.wav
              - /section\_03\_source\_test\_anomaly\_0000.wav
              - ...
              - /section\_03\_source\_test\_anomaly\_0099.wav
              - /section\_04\_source\_test\_normal\_0000.wav
              - ...
              - /section\_05\_source\_test\_anomaly\_0099.wav
              - /section\_03\_target\_test\_normal\_0000.wav
              - ...
              - /section\_03\_target\_test\_normal\_0099.wav
              - /section\_03\_target\_test\_anomaly\_0000.wav
              - ...
              - /section\_03\_target\_test\_anomaly\_0099.wav
              - /section\_04\_target\_test\_normal\_0000.wav
              - ...
              - /section\_05\_target\_test\_anomaly\_0099.wav
      - /gearbox (The other machine types have the same directory structure as fan.)
      - /pump
      - /slider
      - /valve
      - /ToyCar
      - /ToyTrain

### DCASE2022 task2

- dcase2023\_task2\_baseline\_ae
  - /data/dcase2022t2/dev\_data/raw
    - /fan
        - /train (only normal clips)  
            - /section\_00\_source\_train\_normal\_0000\_\<attribute\>.wav  
            - ...  
            - /section\_00\_source\_train\_normal\_0989\_\<attribute\>.wav  
            - /section\_00\_target\_train\_normal\_0000\_\<attribute\>.wav  
            - ...  
            - /section\_00\_target\_train\_normal\_0009\_\<attribute\>.wav  
            - /section\_01\_source\_train\_normal\_0000\_\<attribute\>.wav  
            - ...  
            - /section\_02\_target\_train\_normal\_0009\_\<attribute\>.wav  
        - /test 
            - /section\_00\_source\_test\_normal\_0000\_\<attribute\>.wav    
            - ...  
            - /section\_00\_source\_test\_normal\_0049\_\<attribute\>.wav    
            - /section\_00\_source\_test\_anomaly\_0000\_\<attribute\>.wav  
            - ...  
            - /section\_00\_source\_test\_anomaly\_0049\_\<attribute\>.wav  
            - /section\_00\_target\_test\_normal\_0000\_\<attribute\>.wav
            - ...  
            - /section\_00\_target\_test\_normal\_0049\_\<attribute\>.wav 
            - /section\_00\_target\_test\_anomaly\_0000\_\<attribute\>.wav  
            - ...  
            - /section\_00\_target\_test\_anomaly\_0049\_\<attribute\>.wav 
            - /section\_01\_source\_test\_normal\_0000\_\<attribute\>.wav
            - ...  
            - /section\_02\_target\_test\_anomaly\_0049\_\<attribute\>.wav
        - attributes\_00.csv (attribute csv for section 00)
        - attributes\_01.csv (attribute csv for section 01)
        - attributes\_02.csv (attribute csv for section 02)      
    - /gearbox (The other machine types have the same directory structure as fan.)  
    - /bearing
    - /slider (`slider` means "slide rail")
    - /ToyCar  
    - /ToyTrain  
    - /valve  
  - /data/dcase2022t2/eval\_data/raw
    - /fan  
        - /train (after the launch of the additional training dataset)  
            - /section\_03\_source\_train\_normal\_0000\_\<attribute\>.wav  
            - ...  
            - /section\_03\_source\_train\_normal\_0989\_\<attribute\>.wav  
            - /section\_03\_target\_train\_normal\_0000\_\<attribute\>.wav  
            - ...  
            - /section\_03\_target\_train\_normal\_0009\_\<attribute\>.wav  
            - /section\_04\_source\_train\_normal\_0000\_\<attribute\>.wav  
            - ...  
            - /section\_05\_target\_train\_normal\_0009\_\<attribute\>.wav  
        - /test (after the launch of the evaluation dataset)  
            - /section\_03\_test\_0000.wav  
            - ...  
            - /section\_03\_test\_0199.wav  
            - /section\_04\_test\_0000.wav  
            - ...  
            - /section\_05\_test\_0199.wav  
        - test_rename/ (convert from test directory using `tools/rename.py`)
            - /section\_03\_source\_test\_normal\_0000\_\<attribute\>.wav    
            - ...  
            - /section\_03\_source\_test\_normal\_0049\_\<attribute\>.wav    
            - /section\_03\_source\_test\_anomaly\_0000\_\<attribute\>.wav  
            - ...  
            - /section\_03\_source\_test\_anomaly\_0049\_\<attribute\>.wav  
            - /section\_03\_target\_test\_normal\_0000\_\<attribute\>.wav
            - ...  
            - /section\_03\_target\_test\_normal\_0049\_\<attribute\>.wav 
            - /section\_03\_target\_test\_anomaly\_0000\_\<attribute\>.wav  
            - ...  
            - /section\_03\_target\_test\_anomaly\_0049\_\<attribute\>.wav 
            - /section\_04\_source\_test\_normal\_0000\_\<attribute\>.wav
            - ...  
            - /section\_05\_target\_test\_anomaly\_0049\_\<attribute\>.wav
        - attributes\_03.csv (attribute csv for train data in section 03)
        - attributes\_04.csv (attribute csv for train data in section 04)
        - attributes\_05.csv (attribute csv for train data in section 05) 
    - /gearbox (The other machine types have the same directory structure as fan.)  
    - /bearing  
    - /slider (`slider` means "slide rail")
    - /ToyCar  
    - /ToyTrain  
    - /valve

### DCASE2023 task2
- dcase2023\_task2\_baseline\_ae
  - /data/dcase2023t2/dev\_data/raw
    - /bearing
      - /train (only normal clips)
        - /section\_00\_source\_train\_normal\_0000\_\<attribute\>.wav
        - ...
        - /section\_00\_source\_train\_normal\_0989\_\<attribute\>.wav
        - /section\_00\_target\_train\_normal\_0000\_\<attribute\>.wav
        - ...
        - /section\_00\_target\_train\_normal\_0009\_\<attribute\>.wav
      - test/
        - /section\_00\_source\_test\_normal\_0000\_\<attribute\>.wav
        - ...
        - /section\_00\_source\_test\_normal\_0049\_\<attribute\>.wav
        - /section\_00\_source\_test\_anomaly\_0000\_\<attribute\>.wav
        - ...
        - /section\_00\_source\_test\_anomaly\_0049\_\<attribute\>.wav
        - /section\_00\_target\_test\_normal\_0000\_\<attribute\>.wav 
        - ...
        - /section\_00\_target\_test\_normal\_0049\_\<attribute\>.wav 
        - /section\_00\_target\_test\_anomaly\_0000\_\<attribute\>.wav 
        - ...
        - /section\_00\_target\_test\_anomaly\_0049\_\<attribute\>.wav
      - attributes\_00.csv (attributes CSV for section 00)
    - /fan (The other machine types have the same directory structure as fan.)
    - /gearbox
    - /slider (`slider` means "slide rail")
    - /ToyCar
    - /ToyTrain
    - /valve
  - /data/dcase2023t2/eval\_data/raw/
    - /bandsaw
      - /train (after launch of the additional training dataset)
        - /section\_00\_source\_train\_normal\_0000\_\<attribute\>.wav
        - ...
        - /section\_00\_source\_train\_normal\_0989\_\<attribute\>.wav
        - /section\_00\_target\_train\_normal\_0000\_\<attribute\>.wav
        - ...
        - /section\_00\_target\_train\_normal\_0009\_\<attribute\>.wav
      - /test (after launch of the evaluation dataset)
        - /section\_00\_test\_0000.wav
        - ...
        - /section\_00\_test\_0199.wav
      - /test_rename (convert from test directory using `tools/rename.py`)
        - /section\_00\_source\_test\_normal\_0000\_\<attribute\>.wav
        - ...
        - /section\_00\_source\_test\_normal\_0049\_\<attribute\>.wav
        - /section\_00\_source\_test\_anomaly\_0000\_\<attribute\>.wav
        - ...
        - /section\_00\_source\_test\_anomaly\_0049\_\<attribute\>.wav
        - /section\_00\_target\_test\_normal\_0000\_\<attribute\>.wav 
        - ...
        - /section\_00\_target\_test\_normal\_0049\_\<attribute\>.wav 
        - /section\_00\_target\_test\_anomaly\_0000\_\<attribute\>.wav 
        - ...
        - /section\_00\_target\_test\_anomaly\_0049\_\<attribute\>.wav
      - attributes\_00.csv (attributes CSV for section 00)
    - /grinder
    - /shaker
    - /ToyDrone
    - /ToyNscale
    - /ToyTank
    - /Vacuum

### DCASE2024 task2
- dcase2023\_task2\_baseline\_ae
  - /data/dcase2024t2/dev\_data/raw
    - /bearing
      - /train (only normal clips)
        - /section\_00\_source\_train\_normal\_0001\_\<attribute\>.wav
        - ...
        - /section\_00\_source\_train\_normal\_0990\_\<attribute\>.wav
        - /section\_00\_target\_train\_normal\_0001\_\<attribute\>.wav
        - ...
        - /section\_00\_target\_train\_normal\_0010\_\<attribute\>.wav
      - test/
        - /section\_00\_source\_test\_normal\_0001\_\<attribute\>.wav
        - ...
        - /section\_00\_source\_test\_normal\_0050\_\<attribute\>.wav
        - /section\_00\_source\_test\_anomaly\_0001\_\<attribute\>.wav
        - ...
        - /section\_00\_source\_test\_anomaly\_0050\_\<attribute\>.wav
        - /section\_00\_target\_test\_normal\_0001\_\<attribute\>.wav 
        - ...
        - /section\_00\_target\_test\_normal\_0050\_\<attribute\>.wav 
        - /section\_00\_target\_test\_anomaly\_0001\_\<attribute\>.wav 
        - ...
        - /section\_00\_target\_test\_anomaly\_0050\_\<attribute\>.wav
      - attributes\_00.csv (attributes CSV for section 00)
    - /fan (The other machine types have the same directory structure as fan.)
    - /gearbox
    - /slider (`slider` means "slide rail")
    - /ToyCar
    - /ToyTrain
    - /valve
  - /data/dcase2024t2/eval\_data/raw/
    - /3DPrinter
      - /train (after launch of the additional training dataset)
        - /section\_00\_source\_train\_normal\_0001\_\<attribute\>.wav
        - ...
        - /section\_00\_source\_train\_normal\_0990\_\<attribute\>.wav
        - /section\_00\_target\_train\_normal\_0001\_\<attribute\>.wav
        - ...
        - /section\_00\_target\_train\_normal\_0010\_\<attribute\>.wav
      - /test (after launch of the evaluation dataset)
        - /section\_00\_test\_0001.wav
        - ...
        - /section\_00\_test\_0200.wav
      - /test_rename (convert from test directory using `tools/rename.py`)
        - /section\_00\_source\_test\_normal\_\<0000\~0200\>\_\<attribute\>.wav
        - ...
        - /section\_00\_source\_test\_anomaly\_\<0000\~0200\>\_\<attribute\>.wav
        - ...
        - /section\_00\_target\_test\_normal\_\<0000\~0200\>\_\<attribute\>.wav 
        - ...
        - /section\_00\_target\_test\_anomaly\_\<0000\~0200\>\_\<attribute\>.wav 
        - ...
      - attributes\_00.csv (attributes CSV for section 00)
    - /AirCompressor
    - /BrushlessMotor
    - /HairDryer
    - /HoveringDrone
    - /RoboticArm
    - /Scanner
    - /ToothBrush
    - /ToyCircuit

## Ground truth attribute

### Public ground truth of evaluation dataset

The following code was used to calculate the official score. Among these is evaluation datasets ground truth.

- [dcase2020_task2_evaluator](https://github.com/y-kawagu/dcase2020_task2_evaluator)
- [dcase2021_task2_evaluator](https://github.com/y-kawagu/dcase2021_task2_evaluator)
- [dcase2022_task2_evaluator](https://github.com/Kota-Dohi/dcase2022_evaluator)
- [dcase2023_task2_evaluator](https://github.com/nttcslab/dcase2023_task2_evaluator)
- [dcase2024_task2_evaluator](https://github.com/nttcslab/dcase2024_task2_evaluator)

### Ground truth for evaluation datasets in this repository

This repository have evaluation data's ground truth csv. this csv is using to rename evaluation datasets.
You can calculate AUC and other score if add ground truth to evaluation datasets file name. *Usually, rename function is executed along with [download script](#description) and [auto download function](#41-enable-auto-download-dataset).

- [DCASE2020 task2](datasets/eval_data_list_2020.csv)
- [DCASE2021 task2](datasets/eval_data_list_2021.csv)
- [DCASE2022 task2](datasets/eval_data_list_2022.csv)
- [DCASE2023 task2](datasets/eval_data_list_2023.csv)
- [DCASE2024 task2](datasets/eval_data_list_2024.csv)

### Ground truth attributes

Attribute information is hidden by default for the following machine types:

- DCASE2024 Task2
  - dev data
    - gearbox
    - slider
    - ToyTrain
  - eval data
    - AirCompressor
    - BrushlessMotor
    - HoveringDrone
    - ToothBrush

You can view the hidden attributes in the following directory:

- [DCASE2024 task2 Ground truth Attributes](datasets/ground_truth_attributes)

## Citation

If you use this system, please cite all papers that correspond to the datasets you use.

### [DCASE2020 task2](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds#citation)

+ Yuma Koizumi, Shoichiro Saito, Hisashi Uematsu, Noboru Harada, and Keisuke Imoto. ToyADMOS: a dataset of miniature-machine operating sounds for anomalous sound detection. In Proceedings of IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 308 E12. November 2019. [URL](https://ieeexplore.ieee.org/document/8937164).
+ Harsh Purohit, Ryo Tanabe, Takeshi Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi. MIMII Dataset: sound dataset for malfunctioning industrial machine investigation and inspection. In Proceedings of the Detection and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE2019), 209 E13. November 2019. [URL](http://dcase.community/documents/workshop2019/proceedings/DCASE2019Workshop_Purohit_21.pdf).
+ Yuma Koizumi, Yohei Kawaguchi, Keisuke Imoto, Toshiki Nakamura, Yuki Nikaido, Ryo Tanabe, Harsh Purohit, Kaori Suefusa, Takashi Endo, Masahiro Yasuda, and Noboru Harada. Description and discussion on DCASE2020 challenge task2: unsupervised anomalous sound detection for machine condition monitoring. In Proceedings of the Detection and Classification of Acoustic Scenes and Events 2020 Workshop (DCASE2020), 81 E5. November 2020. [URL](http://dcase.community/documents/workshop2020/proceedings/DCASE2020Workshop_Koizumi_3.pdf).

### [DCASE2021 task2](https://dcase.community/challenge2021/task-unsupervised-detection-of-anomalous-sounds#citation)

+ Ryo Tanabe, Harsh Purohit, Kota Dohi, Takashi Endo, Yuki Nikaido, Toshiki Nakamura, and Yohei Kawaguchi. MIMII DUE: sound dataset for malfunctioning industrial machine investigation and inspection with domain shifts due to changes in operational and environmental conditions. IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), pages 21 E5, 2021. doi:10.1109/WASPAA52581.2021.9632802. [URL](https://ieeexplore.ieee.org/document/9632802).
+ Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, and Shoichiro Saito. ToyADMOS2: another dataset of miniature-machine operating sounds for anomalous sound detection under domain shift conditions. In Proceedings of the 6th Detection and Classification of Acoustic Scenes and Events 2021 Workshop (DCASE2021), 1 E. Barcelona, Spain, November 2021. [URL](https://dcase.community/documents/workshop2021/proceedings/DCASE2021Workshop_Harada_6.pdf).
+ Yohei Kawaguchi, Keisuke Imoto, Yuma Koizumi, Noboru Harada, Daisuke Niizumi, Kota Dohi, Ryo Tanabe, Harsh Purohit, and Takashi Endo. Description and discussion on dcase 2021 challenge task 2: unsupervised anomalous detection for machine condition monitoring under domain shifted conditions. In Proceedings of the 6th Detection and Classification of Acoustic Scenes and Events 2021 Workshop (DCASE2021), 186 E90. Barcelona, Spain, November 2021. [URL](https://dcase.community/documents/workshop2021/proceedings/DCASE2021Workshop_Kawaguchi_61.pdf).

### [DCASE2022 task2](https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring#citation)

+ Kota Dohi, Tomoya Nishida, Harsh Purohit, Ryo Tanabe, Takashi Endo, Masaaki Yamamoto, Yuki Nikaido, and Yohei Kawaguchi. MIMII DG: sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task. In Proceedings of the 7th Detection and Classification of Acoustic Scenes and Events 2022 Workshop (DCASE2022), 1 E. Nancy, France, November 2022. [URL](https://dcase.community/documents/workshop2022/proceedings/DCASE2022Workshop_Dohi_62.pdf).
+ Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, and Shoichiro Saito. ToyADMOS2: another dataset of miniature-machine operating sounds for anomalous sound detection under domain shift conditions. In Proceedings of the 6th Detection and Classification of Acoustic Scenes and Events 2021 Workshop (DCASE2021), 1 E. Barcelona, Spain, November 2021. [URL](https://dcase.community/documents/workshop2021/proceedings/DCASE2021Workshop_Harada_6.pdf).
+ Kota Dohi, Keisuke Imoto, Noboru Harada, Daisuke Niizumi, Yuma Koizumi, Tomoya Nishida, Harsh Purohit, Ryo Tanabe, Takashi Endo, Masaaki Yamamoto, and Yohei Kawaguchi. Description and discussion on DCASE 2022 challenge task 2: unsupervised anomalous sound detection for machine condition monitoring applying domain generalization techniques. In Proceedings of the 7th Detection and Classification of Acoustic Scenes and Events 2022 Workshop (DCASE2022), 1 E. Nancy, France, November 2022. [URL](https://dcase.community/documents/workshop2022/proceedings/DCASE2022Workshop_Dohi_63.pdf).

### [DCASE2023 task2](https://dcase.community/challenge2023/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring#citation)

+ Kota Dohi, Keisuke Imoto, Noboru Harada, Daisuke Niizumi, Yuma Koizumi, Tomoya Nishida, Harsh Purohit, Ryo Tanabe, Takashi Endo, and Yohei Kawaguchi. Description and discussion on DCASE 2023 challenge task 2: first-shot unsupervised anomalous sound detection for machine condition monitoring. In arXiv e-prints: 2305.07828, 2023. [URL](https://arxiv.org/pdf/2305.07828.pdf).
+ Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, and Shoichiro Saito. ToyADMOS2: another dataset of miniature-machine operating sounds for anomalous sound detection under domain shift conditions. In Proceedings of the Detection and Classification of Acoustic Scenes and Events Workshop (DCASE), 1 E. Barcelona, Spain, November 2021. [URL](https://dcase.community/documents/workshop2021/proceedings/DCASE2021Workshop_Harada_6.pdf).
+ Kota Dohi, Tomoya Nishida, Harsh Purohit, Ryo Tanabe, Takashi Endo, Masaaki Yamamoto, Yuki Nikaido, and Yohei Kawaguchi. MIMII DG: sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task. In Proceedings of the 7th Detection and Classification of Acoustic Scenes and Events 2022 Workshop (DCASE2022). Nancy, France, November 2022. [URL](https://dcase.community/documents/workshop2022/proceedings/DCASE2022Workshop_Dohi_62.pdf).
+ Noboru Harada, Daisuke Niizumi, Yasunori Ohishi, Daiki Takeuchi and Masahiro Yasuda, "First-Shot Anomaly Sound Detection for Machine Condition Monitoring: A Domain Generalization Baseline," 2023 31st European Signal Processing Conference (EUSIPCO), Helsinki, Finland, 2023, pp. 191-195, doi: 10.23919/EUSIPCO58844.2023.10289721. [URL](https://ieeexplore.ieee.org/document/10289721).

### [DCASE2024 task2](https://dcase.community/challenge2024/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring#citation)

+ Tomoya Nishida, Noboru Harada, Daisuke Niizumi, Davide Albertini, Roberto Sannino, Simone Pradolini, Filippo Augusti, Keisuke Imoto, Kota Dohi, Harsh Purohit, Takashi Endo, and Yohei Kawaguchi. Description and discussion on DCASE 2024 challenge task 2: first-shot unsupervised anomalous sound detection for machine condition monitoring. In arXiv e-prints: 2406.07250, 2024. [URL](https://arxiv.org/pdf/2406.07250.pdf)
+ Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, and Shoichiro Saito. ToyADMOS2: another dataset of miniature-machine operating sounds for anomalous sound detection under domain shift conditions. In Proceedings of the Detection and Classification of Acoustic Scenes and Events Workshop (DCASE), 1–5. Barcelona, Spain, November 2021. [URL](https://dcase.community/documents/workshop2021/proceedings/DCASE2021Workshop_Harada_6.pdf)
+ Kota Dohi, Tomoya Nishida, Harsh Purohit, Ryo Tanabe, Takashi Endo, Masaaki Yamamoto, Yuki Nikaido, and Yohei Kawaguchi. MIMII DG: sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task. In Proceedings of the 7th Detection and Classification of Acoustic Scenes and Events 2022 Workshop (DCASE2022). Nancy, France, November 2022. [URL](https://dcase.community/documents/workshop2022/proceedings/DCASE2022Workshop_Dohi_62.pdf)
+ Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, and Masahiro Yasuda. First-shot anomaly detection for machine condition monitoring: a domain generalization baseline. Proceedings of 31st European Signal Processing Conference (EUSIPCO), pages 191–195, 2023. [URL](https://eurasip.org/Proceedings/Eusipco/Eusipco2023/pdfs/0000191.pdf)