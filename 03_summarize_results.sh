TZ=JST-9 date

echo $0 $*

dataset=$1
dev_eval=$2
echo -e "\tdataset = '$dataset'"
echo -e "\tdev_eval = '$dev_eval'"
echo

# check args
args_flag=0
args_flag_dataset=0
if [ "${dataset}" != "DCASE2020T2" ] \
    && [ "${dataset}" != "DCASE2021T2" ] \
    && [ "${dataset}" != "DCASE2022T2" ] \
    && [ "${dataset}" != "DCASE2023T2" ] \
    && [ "${dataset}" != "DCASE2024T2" ] \
    && [ "${dataset}" != "DCASE2025T2" ]
then
    args_flag=1
    args_flag_dataset=1
fi

args_flag_dev_eval=0
if [ "${dev_eval}" != "-d" ] \
    && [ "${dev_eval}" != "-e" ] \
    && [ "${dev_eval}" != "--dev" ] \
    && [ "${dev_eval}" != "--eval" ]
then
    args_flag=1
    args_flag_dev_eval=1
fi

if [ $args_flag -eq 1 ]
then
    echo "$0: argument error"
    echo -e "usage\t: $0 ['DCASE2020T2' | 'DCASE2021T2' | 'DCASE2022T2' | 'DCASE2023T2' | 'DCASE2024T2' | 'DCASE2025T2' ] ['-d' | '--dev' | '-e' | '--eval']"

    if [ $args_flag_dataset -eq 1 ]
    then
        echo -e "\tdataset: invalid choice '$dataset'"
        echo -e "\tchoice from ['DCASE2020T2' | 'DCASE2021T2' | 'DCASE2022T2' | 'DCASE2023T2' | 'DCASE2024T2' | 'DCASE2025T2' ]."
        echo -e "\t\tDCASE2020T2\t: Use DCASE2020 Task2 datasets. "
        echo -e "\t\tDCASE2021T2\t: Use DCASE2021 Task2 datasets. "
        echo -e "\t\tDCASE2022T2\t: Use DCASE2022 Task2 datasets. "
        echo -e "\t\tDCASE2023T2\t: Use DCASE2023 Task2 datasets. "
        echo -e "\t\tDCASE2024T2\t: Use DCASE2024 Task2 datasets. "
        echo -e "\t\tDCASE2025T2\t: Use DCASE2025 Task2 datasets. "
        echo 
    fi

    if [ $args_flag_dev_eval -eq 1 ]
    then
        echo -e "\tdev_eval: invalid choice '$dev_eval'"
        echo -e "\tchoice from ['-d' | '--dev' | '-e' | '--eval']."
        echo -e "\t\t-d, --dev\t: Using Development dataset. "
        echo -e "\t\t-e, --eval\t: Using Additional training dataset and Evaluation dataset. "
        echo
    fi

    echo -e "example\t: $ bash $0 DCASE2020T2 -d"
    exit 1
fi

if [ "${dev_eval}" = "-d" ] \
    || [ "${dev_eval}" = "--dev" ]
then
    dev_eval_dir_name="dev_data"
fi

if [ "${dev_eval}" = "-e" ] \
    || [ "${dev_eval}" = "--eval" ]
then
    dev_eval_dir_name="eval_data"
fi

# parameters
model="DCASE2023T2-AE"
dir_name="summarize"
float_format=None
# float_format="%.4f"
summarize_dir_list=" \
results/${dev_eval_dir_name}/baseline_MAHALA \
results/${dev_eval_dir_name}/baseline_MSE \
"

# summarize all data
export_dir="results/${dev_eval_dir_name}/baseline/${dir_name}/${dataset}"
echo ${dataset} ${export_dir} ${float_format} ${dev_eval} ${summarize_dir_list}
bash tools/export_results.sh ${dataset} ${export_dir} ${float_format} ${dev_eval} ${summarize_dir_list}
