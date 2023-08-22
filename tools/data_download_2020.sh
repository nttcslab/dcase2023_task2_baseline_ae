parent_dir=data
ROOT_DIR=$(cd $(dirname $0); pwd)/../
mkdir -p "${ROOT_DIR}/${parent_dir}/dcase2020t2/dev_data/raw"
mkdir -p "${ROOT_DIR}/${parent_dir}/dcase2020t2/eval_data/raw"

# download dev data
cd "${ROOT_DIR}/${parent_dir}/dcase2020t2/dev_data/raw"
for machine_type in ToyCar ToyConveyor fan valve slider pump; do
wget "https://zenodo.org/record/3678171/files/dev_data_${machine_type}.zip"
unzip "dev_data_${machine_type}.zip"
done

# download eval data
cd -
cd "${ROOT_DIR}/${parent_dir}/dcase2020t2/eval_data/raw"
for machine_type in ToyCar ToyConveyor fan valve slider pump; do
wget "https://zenodo.org/record/3727685/files/eval_data_train_${machine_type}.zip"
unzip "eval_data_train_${machine_type}.zip"

wget "https://zenodo.org/record/3841772/files/eval_data_test_${machine_type}.zip"
unzip "eval_data_test_${machine_type}.zip"
done

# Adds reference labels to test data.
python ${ROOT_DIR}/tools/rename_eval_wav.py --dataset_parent_dir=${parent_dir} --dataset_type=DCASE2020T2
