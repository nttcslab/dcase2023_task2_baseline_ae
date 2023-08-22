# Utility tools

## Description

- concat_divided_roc.py
    - This script is called by export_results.sh.
    - Integrate results for each sectionID into results for each machine ID.
- data_download_2020.sh
    - Download DCASE2020 Challenge Task 2 data.
- data_download_2022.sh
    - Download DCASE2022 Challenge Task 2 data.
- export_results.py
    - This script is called by export_result.sh.
    - Summarize the results in a CSV file.
- export_results.sh
    - This scriput is called by 03_summarize_results.sh.
    - Integrate, summarize, and extract results.
- extract_results.py
    - This script is called by export_results.sh
    - This is used after export_results.py to extract arithmetic mean and harmonic mean results and summarize all results into a csv.
- job_ae.sh
    - Run DCASE2020T2 and DCASE2022T2 for training and evaluation.
- plot_anm_score.py
    - Export boxplots of anomaly scores.
- plot_common.py
    - Plot figure utility.
- plot_loss_curve.py
    - Plot loss curve in trainning.
- plot_time_frequency.py
    - Export time frequency image.
