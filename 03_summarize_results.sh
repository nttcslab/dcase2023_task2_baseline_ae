TZ=JST-9 date

echo $0 $*

dataset=$1
echo -e "\tdataset = '$dataset'"
echo

if [ "${dataset}" != "DCASE2020T2" ] \
    && [ "${dataset}" != "DCASE2022T2" ] \
    && [ "${dataset}" != "DCASE2023T2" ]
then
    echo -e "\tdataset: invalid choice '$dataset'"
    echo -e "\tchoice from ['DCASE2023T2' | 'DCASE2020T2' | 'DCASE2022T2']."
    echo -e "\t\tDCASE2023T2\t: Use DCASE2023 Task2 datasets. "
    echo -e "\t\tDCASE2022T2\t: Use DCASE2022 Task2 datasets. "
    echo -e "\t\tDCASE2020T2\t: Use DCASE2020 Task2 datasets. "
    echo 
    echo -e "example\t: $ bash $0 DCASE2023T2"
    exit 1
fi

model="DCASE2023T2-AE"
dir_name="summarize"
float_format=None
# float_format="%.4f"
summarize_dir_list=" \
results/dev_data/baseline_MAHALA \
results/dev_data/baseline_MSE \
"

# summarize all data
export_dir="results/dev_data/baseline/${dir_name}/${dataset}"
bash tools/export_results.sh $dataset $export_dir $float_format $summarize_dir_list
