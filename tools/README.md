# Utility tools

<<<<<<< Updated upstream
## Description for plot script
=======
## Description for plot tools
>>>>>>> Stashed changes

- concat_divided_roc.py
    - This script is called by export_results.sh.
    - Integrate results for each sectionID into results for each machine ID.
- export_results.py
    - This script is called by export_result.sh.
    - Summarize the results in a CSV file.
- export_results.sh
    - This script is called by 03_summarize_results.sh.
    - Integrate, summarize, and extract results.
- extract_results.py
    - This script is called by export_results.sh
    - This is used after export_results.py to extract arithmetic mean and harmonic mean results and summarize all results into a csv.
- plot_anm_score.py
    - Export boxplots of anomaly scores.
- plot_common.py
    - Plot figure utility.
- plot_loss_curve.py
    - Plot loss curve in training.
- plot_time_frequency.py
    - Export time frequency image.

<<<<<<< Updated upstream
## Description for legacy dataset to using
=======
## Tools for legacy dataset
>>>>>>> Stashed changes

Those script are in `tools/legacy`. 

- 01_train_legacy.sh
<<<<<<< Updated upstream
    - Run DCASE2020T2 and DCASE2022T2 for training.
    - Use dev data and eval data.
- 02a_test_legacy.sh (Use MSE as a score function for the Simple Autoencoder mode)
    - Run DCASE2020T2 and DCASE2022T2 for evaluation.
    - Use dev data and eval data.
- 02b_test_legacy.sh (Use Mahalanobis distance as a score function for the Selective Mahalanobis mode)
    - Run DCASE2020T2 and DCASE2022T2 for evaluation.
    - Use dev data and eval data.
=======
    - Run this script for training with DCASE2020T2 or DCASE2022T2 dataset.
    - Specify dev data or eval data.
    - Usage:
    ```$ 01_train_legacy.sh [-d|-e] [DCASE2020T2|DCASE2022T2]```

- 02a_test_legacy.sh (Use MSE as a score function for the Simple Autoencoder mode)
    - Run this script for evaluation with DCASE2020T2 or DCASE2022T2 dataset.
    - Specify dev data or eval data.
    - Usage:
    ```$ 02a_test_legacy.sh [-d|-e] [DCASE2020T2|DCASE2022T2]```

- 02b_test_legacy.sh (Use Mahalanobis distance as a score function for the Selective Mahalanobis mode)
    - Run this script for evaluation with DCASE2020T2 or DCASE2022T2 dataset.
    - Specift dev data and eval data.
    - Usage:
    ```$ 02b_test_legacy.sh [-d|-e] [DCASE2020T2|DCASE2022T2]```

>>>>>>> Stashed changes
- data_download_2020.sh
    - Download DCASE2020 Challenge Task 2 data.
    - Copy renamed eval data 'test' to 'test_rename' after download wav file.
- data_download_2022.sh
    - Download DCASE2022 Challenge Task 2 data.
    - Copy renamed eval data 'test' to 'test_rename' after download wav file.
- rename_eval_2020.sh
    - Copy renamed eval data 'test' to 'test_rename'.
- rename_eval_2022.sh
    - Copy renamed eval data 'test' to 'test_rename'.
- rename_eval_legacy_wav.py
    - Copy renamed eval data 'test' to 'test_rename'.
    - This script is called by data_download_2020.sh, data_download_2022, rename_eval_2020.sh, rename_eval_2022.sh and loader_common.py.
