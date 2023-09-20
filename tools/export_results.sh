# results output dir
arg=("$@")
dataset=${arg[0]}
export_dir=${arg[1]}
float_format=${arg[2]}
dev_eval=${arg[3]}

echo "export_dir : ${export_dir}"

for ((i=4; i<$#; i++)) do
id_single_dirs="${id_single_dirs} ${arg[i]}"
done

# aggregate results by machine type
for dir in $id_single_dirs
do
echo $dir
python3 tools/concat_divided_roc.py \
--parent_dir ${dir} \
--export_dir ${export_dir} \
--dataset ${dataset} \
${dev_eval} \

done

# summarize all results
python3 tools/export_results.py "${export_dir}/" --dataset=${dataset} ${dev_eval}
# extract the AUC mean
python3 tools/extract_results.py "${export_dir}/" --dataset=${dataset} --float_format=${float_format} ${dev_eval}
echo "===================================================="