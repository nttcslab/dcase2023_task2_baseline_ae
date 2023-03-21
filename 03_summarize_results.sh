dataset=$1

if [ "${dataset}" = "" ]
then
    echo "incorrect argument"
    echo "please set option argument 'DCASE2023T2' or 'DCASE2022T2' or 'DCASE2020T2"
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
