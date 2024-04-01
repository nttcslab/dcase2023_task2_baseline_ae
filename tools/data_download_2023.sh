parent_dir=data
ROOT_DIR=$(cd $(dirname $0); pwd)/../
mkdir -p "${ROOT_DIR}/${parent_dir}/dcase2023t2/dev_data/raw"
mkdir -p "${ROOT_DIR}/${parent_dir}/dcase2023t2/eval_data/raw"

# download dev data
cd "${ROOT_DIR}/data/dcase2023t2/dev_data/raw"
for machine_type in bearing fan gearbox slider ToyCar ToyTrain valve; do
wget "https://zenodo.org/record/7882613/files/dev_${machine_type}.zip"
unzip "dev_${machine_type}.zip"
done

# download eval data
cd -
cd "${ROOT_DIR}/data/dcase2023t2/eval_data/raw"
for machine_type in bandsaw grinder shaker ToyDrone ToyNscale ToyTank Vacuum; do
wget "https://zenodo.org/record/7830345/files/eval_data_${machine_type}_train.zip"
unzip "eval_data_${machine_type}_train.zip"

wget "https://zenodo.org/record/7860847/files/eval_data_${machine_type}_test.zip"
unzip "eval_data_${machine_type}_test.zip"
done

# Adds reference labels to test data.
python ${ROOT_DIR}/tools/rename_eval_wav.py --dataset_parent_dir=${parent_dir} --dataset_type=DCASE2023T2
