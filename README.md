# Anomalous Sound Detection
## DCASE 2023 Challenge Task 2 Baseline Auto Encoder: dcase2023_task2_baseline_ae

This is audoencoder based baseline for the [DCASE2023 Challenge Task 2](https://dcase.community/challenge2023/).

This source code is an example implementation of the baseline Auto Encoder of DCASE 2023 Challenge Task 2: First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring.

This baseline implementation is based on the previous baseline, dcase2022_baseline_ae. The model parameter settings of this baseline AE are almost equivalent to those of the dcase2022\_task2\_baseline\_ae. 

Difference between the previouse dcase2022_baseline_ae and this version are as follows:

- The dcase2022_baseline_ae was implemented with Keras; however, this version is written in PyTorch.
- Date folder structure is updated to support DCASE 2023 Challenge Task 2 data sets.
- The system uses the MSE loss as a loss function for training but for testing, there are two score functions depending on the testing modes (i.e., MSE for the Simple Autoencoder mode, and Mahalanobis distance for the Selective Mahalanobis mode).
  
## Description

This system consists of three main scripts (01_train.sh, 02a_test.sh, and 02b_test.sh) with some helper scripts:

- Helper scripts
  - data_download_2023dev.sh
    - "Development dataset":
      - This script downloads development data files and put them into "data/dcase2023t2/dev_data/raw/train/" and "data/dcase2023t2/dev_data/raw/test/".
  - data_download_2023add.sh **Newly added!!**
    - "Additional train dataset for Evaluation":
      - This script downloads evaluation data files and put them into "data/dcase2023t2/eval_data/raw/train". 
  - data_download_2023eval.sh **Newly added!!**
    - "Additional test dataset for Evaluation"
      - This script downloads evaluation data files and put them into "data/dcase2023t2/eval_data/raw/test". 

- 01_train.sh
  - "Development" mode:
    - This script trains a model for each machine type for each section ID by using the directory `data/dcase2023t2/dev_data/raw/<machine_type>/train/<section_id>`
  - "Evaluation" mode:
    - This script trains a model for each machine type for each section ID by using the directory `data/dcase2023t2/eval_data/raw/<machine_type>/train/<section_id>`.
  
- 02a_test.sh (Use MSE as a score function for the Simple Autoencoder mode)
  - "Development" mode:
    - This script makes a csv file for each section including the anomaly scores for each wav file in the directories `data/dcase2023t2/dev_data/raw/<machine_type>/test/`.
    - The csv files will be stored in the directory `results/`.
    - It also makes a csv file including AUC, pAUC, precision, recall, and F1-score for each section.
  
  - "Evaluation" mode:
    - This script makes a csv file for each section inclusing the anomaly scores for each wav file in the directories `data/dcase2023t2/eval_data/raw/<machine_type>/test/`. (These directories will be made available with the "evaluation datset".)
    - The csv files are stored in the directory `results/`.

- 02b_test.sh (Use Mahalanobis distance as a score function for the Selective Mahalanobis mode)
  - "Development" mode:
    - This script makes a csv file for each section including the anomaly scores for each wav file in the directories `data/dcase2023t2/dev_data/raw/<machine_type>/test/`.
    - The csv files will be stored in the directory `results/`.
    - It also makes a csv file including AUC, pAUC, precision, recall, and F1-score for each section.
  
  - "Evaluation" mode:
    - This script makes a csv file for each section inclusing the anomaly scores for each wav file in the directories `data/dcase2023t2/eval_data/raw/<machine_type>/test/`. (These directories will be made available with the "evaluation datset".)
    - The csv files are stored in the directory `results/`.

- 03_summarize_results.sh
  - This script summarize results into a csv file.

## Usage

### 1. Clone repository
   
Clone this repository from GitHub.

### 2. Download datasets

We will launch the datasets in three stages. Therefore, please download the datasets in each stage:

  + "Development dataset"
    + ~~Download dev\_data_<machine_type>.zip from [https://zenodo.org/record/7690148](https://zenodo.org/record/7690148).~~ 
    + Download "dev\_data_<machine_type>.zip" from [https://zenodo.org/record/7690157](https://zenodo.org/record/7690157).
  + "Additional training dataset", i.e., the evaluation dataset for training **New!**
    + After April 15, 2023, download additional training dataset
    + Download "eval\_data_<machine_type>_train.zip" from [https://zenodo.org/record/7830345](https://zenodo.org/record/7830345).
  + "Evaluation dataset", i.e., the evaluation dataset for test **New!**
    + After May 1, 2023, download evaluation dataset.
    + Download "eval\_data_<machine_type>_test.zip" from [https://zenodo.org/record/7860847](https://zenodo.org/record/7860847).

### 3. Unzip the downloaded files and make the directory structure as the followings:
   
  + dcase2023_task2_baseline_ae
    + data/dcase2023t2/dev_data/raw/
      + fan/
        + train (only normal clips)
          + section_00_source_train_normal_0000_.wav
          + ...
          + section_00_source_train_normal_0989_.wav
          + section_00_target_train_normal_0000_.wav
          + ...
          + section_00_target_train_normal_0009_.wav
        + test/
          + section_00_source_test_normal_0000_.wav
          + ...
          + section_00_source_test_normal_0049_.wav
          + section_00_source_test_anomaly_0000_.wav
          + ...
          + section_00_source_test_anomaly_0049_.wav
          + section_00_target_test_normal_0000_.wav 
          + ...
          + section_00_target_test_normal_0049_.wav 
          + section_00_target_test_anomaly_0000_.wav 
          + ...
          + section_00_target_test_anomaly_0049_.wav
        + attributes_00.csv (attributes csv for section 00)
     + gearbox/ (The other machine types have the same directory structure as fan.)
   + data/dcase2023t2/eval_data/raw/
     + <machine_type0_of_additional_dataset>/
        + train/ (after launch of the additional training dataset)
          + section_00_source_train_normal_0000_.wav
          + ...
          + section_00_source_train_normal_0989_.wav
          + section_00_target_train_normal_0000_.wav
          + ...
          + section_00_target_train_normal_0009_.wav
        + test/
          + section_00_source_test_normal_0000_.wav
          + ...
          + section_00_source_test_normal_0049_.wav
          + section_00_source_test_anomaly_0000_.wav
          + ...
          + section_00_source_test_anomaly_0049_.wav
          + section_00_target_test_normal_0000_.wav 
          + ...
          + section_00_target_test_normal_0049_.wav 
          + section_00_target_test_anomaly_0000_.wav 
          + ...
          + section_00_target_test_anomaly_0049_.wav
        + test/ (after launch of the evaluation dataset)
          + section_00_test_0000.wav
          + ...
          + section_00_test_0199.wav
        + attributes_00.csv (attributes csv for section 00)
     + <machine_type1_of_additional_dataset>/ (The other machine types have the same directory structure as <machine_type0_of_additional_dataset>/.)

### 4. Change parameters

You can change parameters for feature extraction and model definition by editing `baseline.yaml`.
Note that if values is specified with command line option, it will overwrite the parameter settings in `baseline.yaml`.

### 5. Run the training script (for the development dataset)

Run the training script 01_train.sh. Use the option -d for the development dataset `data/dcase2023t2/dev_data/<machine_tyep>/raw/train/`.

```dotnetcli
$ 01_train.sh -d
```

The two operating modes of this baseline implementation, the simple Autoencoder and the Selective Mahalanobis AE modes, share the common training process. By running the script `01_train.sh`, all the model parameter for the simple Autoencoder and the selective Mahalanobis AE will be trained at the same time.
After the parameter update of the Autoencoder at the last epoch specified by either the yaml file or the command line option, the covariance matrixes for the Mahalanobis distance calculation will be set.

### 6. Run the test script (for the development dataset)

### 6.1. Testing with the Simple Audoencoder mode
Run the test script `02a_test.sh`. Use the option `-d` for the development dataset `data/dcase2023t2/dev_data/<machine_tyep>/raw/test/`.

```dotnetcli
$ 02a_test.sh -d
```

The options for `02a_test.sh` are the same as those for `01_train.sh`. `02a_test.sh` calculates an anomaly score for each wav file in the directories `data/dcase2023t2/dev_data/raw/<machine_type>/test/` or `data/dcase2023t2/dev_data/raw/<machine_type>/source_test/` and `data/dcase2023t2/dev_data/raw/<machine_type>/target_test/`.
A csv file for each section including the anomaly scores will be stored in the directory `results/`. If the mode is "development", the script also outputs another csv file including AUC, pAUC, precision, recall and F1-score for each section.


### 6.2. Testing with the Selective Mahalanobis mode
Run the test script `02a_test.sh`. Use the option `-d` for the development dataset `data/dcase2023t2/dev_data/<machine_tyep>/raw/test/`.

```dotnetcli
$ 02b_test.sh -d
```

The options for `02b_test.sh` are the same as those for `01_train.sh`. `02b_test.sh` calculates an anomaly score for each wav file in the directories `data/dcase2023t2/dev_data/raw/<machine_type>/test/` or `data/dcase2023t2/dev_data/raw/<machine_type>/source_test/` and `data/dcase2023t2/dev_data/raw/<machine_type>/target_test/`.
A csv file for each section including the anomaly scores will be stored in the directory `results/`. If the mode is "development", the script also outputs another csv file including AUC, pAUC, precision, recall and F1-score for each section.


### 7. Check results

You can check the anomaly scores in the csv files `anomaly_score_<machine_type>_section_<section_index>_test.csv` in the directory `results/`.
Each anomaly score corresponds to a wav file in the directories `data/dcase2023t2/dev_data/<machine_type>/test/`.

`anomaly_score_<machine_type>_section_00_test.csv`

```dotnetcli
section_00_source_test_normal_0000_car_A2_spd_28V_mic_1_noise_1.wav,0.3084583878517151
section_00_source_test_normal_0001_car_A2_spd_28V_mic_1_noise_1.wav,0.31289517879486084
section_00_source_test_normal_0002_car_A2_spd_28V_mic_1_noise_1.wav,0.4160425364971161
section_00_source_test_normal_0003_car_A2_spd_28V_mic_1_noise_1.wav,0.25631701946258545

```

Also, anomaly detection results based on the corresponding threshold can be checked in the csv files `decision_result_<machine_type>_section_<section_index>_test.csv`:

`decision_result_<machine_type>_section_<section_index>_test.csv`

```dotnetcli
section_00_source_test_normal_0000_car_A2_spd_28V_mic_1_noise_1.wav,0
section_00_source_test_normal_0001_car_A2_spd_28V_mic_1_noise_1.wav,0
section_00_source_test_normal_0002_car_A2_spd_28V_mic_1_noise_1.wav,0
section_00_source_test_normal_0003_car_A2_spd_28V_mic_1_noise_1.wav,0
...
```

In addition, you can check performance indicatiors such as AUC, pAUC, precision, recall, and F1-score:

`result.csv`

```dotnetcli
section,AUC (source),AUC (target),pAUC,pAUC (source),pAUC (target),precision (source),precision (target),recall (source),recall (target),F1 score (source),F1 score (target)
00,0.88,0.5078,0.5063157894736842,0.5536842105263158,0.4926315789473684,0.0,0.0,0.0,0.0,0.0,0.
arithmetic mean,00,0.88,0.5078,0.5063157894736842,0.5536842105263158,0.4926315789473684,0.0,0.0,0.0,0.0,0.0,0.
harmonic mean,00,0.88,0.5078,0.5063157894736842,0.5536842105263158,0.4926315789473684,0.0,0.0,0.0,0.0,0.0,0.
```

### 8. Run training script for the additional training dataset (after April 15, 2023)

After the additional training dataset is launched, download and unzip it. Move it to `data/dcase2023t2/eval_data/raw/<machine_type>/train/`. Run the training script `01_train.sh` with the option `-e`.

```dotnetcli
$ 01_train.sh -e
```

Models are trained by using the additional training dataset `data/dcase2023t2/raw/eval_data/<machine_type>/train/`.

### 9. Run test script for the evaluation dataset (after May 1, 2023)

### 9.1. Testing with the Simple Autoencoder mode

After the evaluation dataset for test is launched, download and unzip it. Move it to `data/dcase2023t2/eval_data/raw/<machine_type>/test/`. Run the test script `02a_test.sh` with the option `-e`.

```dotnetcli
$ 02a_test.sh -e
```

Anomaly scores are calculated usijng the evaluation dataset, i.e., `data/dcase2023t2/eval_data/raw/<machine_type>/test/`. The anomaly scores are stored as csv files in the directory `results/`. You can submit the csv files for the challenge. From the submitted csv files, we will calculate AUC, pAUC, and your ranking.

### 9.2. Testing with the Selective Mahalanobis mode

After the evaluation dataset for test is launched, download and unzip it. Move it to `data/dcase2023t2/eval_data/raw/<machine_type>/test/`. Run the test script `02b_test.sh` with the option `-e`.

```dotnetcli
$ 02b_test.sh -e
```

Anomaly scores are calculated usijng the evaluation dataset, i.e., `data/dcase2023t2/eval_data/raw/<machine_type>/test/`. The anomaly scores are stored as csv files in the directory `results/`. You can submit the csv files for the challenge. From the submitted csv files, we will calculate AUC, pAUC, and your ranking.


## Dependency

We develop the source code on Ubuntu 18.04.6 LTS.

### Software package

- Python == 3.10.8
- cuda == 11.6
- libsndfile1

### Python packages

- Pytorch == 1.13.1
- torchvision == 0.14.1
- numpy == 1.22.3
- pyYAML == 6.0
- scipy == 1.10.1
- librosa == 0.9.2
- matplotlib == 3.7.0
- tqdm == 4.63
- seaborn == 0.12.2

## Citation

If you use this system, please cite all the following four papers:

+ Kota Dohi, Keisuke Imoto, Noboru Harada, Daisuke Niizumi, Yuma Koizumi, Tomoya Nishida, Harsh Purohit, Takashi Endo, Masaaki Yamamoto, and Yohei Kawaguchi, "Description and discussion on DCASE 2022 challenge task 2: unsupervised anomalous sound detection for machine condition monitoring applying domain generalization techniques," in Proc. DCASE 2022 Workshop, 2022. [URL](https://dcase.community/documents/workshop2022/proceedings/DCASE2022Workshop_Dohi_63.pdf)
+ Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, Shoichiro Saito, "ToyADMOS2: Another Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection under Domain Shift Conditions," in Proc. DCASE 2022 Workshop, 2022. [URL](https://dcase.community/documents/workshop2021/proceedings/DCASE2021Workshop_Harada_6.pdf)
+ Kota Dohi, Tomoya Nishida, Harsh Purohit, Ryo Tanabe, Takashi Endo, Masaaki Yamamoto, Yuki Nikaido, and Yohei Kawaguchi, "MIMII DG: sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task," in Proc. DCASE 2022 Workshop, 2022. [URL](https://dcase.community/documents/workshop2022/proceedings/DCASE2022Workshop_Dohi_62.pdf)
+ Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, "First-Shot Anomaly Detection for Machine Condition Monitoring: A Domain Generalization Baseline," in arXiv e-prints: 2303.00455, 2023. [URL](https://arxiv.org/abs/2303.00455)




