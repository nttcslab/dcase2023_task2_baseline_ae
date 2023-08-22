parent_dir=data
ROOT_DIR=$(cd $(dirname $0); pwd)/
mkdir -p "${ROOT_DIR}/${parent_dir}/dcase2023t2/eval_data/raw"

# download eval data
cd "${parent_dir}/dcase2023t2/eval_data/raw"
for machine_type in bandsaw grinder shaker ToyDrone ToyNscale ToyTank Vacuum; do
wget "https://zenodo.org/record/7860847/files/eval_data_${machine_type}_test.zip"
unzip "eval_data_${machine_type}_test.zip"
done

# Adds reference labels to test data.
python ${ROOT_DIR}/tools/rename_eval_wav.py --dataset_parent_dir=${parent_dir} --dataset_type=DCASE2023T2
