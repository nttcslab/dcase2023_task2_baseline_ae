parent_dir=data
ROOT_DIR=$(cd $(dirname $0); pwd)/../
mkdir -p "${ROOT_DIR}/${parent_dir}/dcase2021t2/dev_data/raw"
mkdir -p "${ROOT_DIR}/${parent_dir}/dcase2021t2/eval_data/raw"

# download dev data
cd "${ROOT_DIR}/${parent_dir}/dcase2021t2/dev_data/raw"
for machine_type in fan gearbox pump slider ToyCar ToyTrain valve; do
wget "https://zenodo.org/record/4562016/files/dev_data_${machine_type}.zip"
unzip "dev_data_${machine_type}.zip"
mkdir -p ${ROOT_DIR}/data/dcase2021t2/dev_data/raw/${machine_type}/test
tsp scp -C ${ROOT_DIR}/data/dcase2021t2/dev_data/raw/${machine_type}/source_test/* data/dcase2021t2/dev_data/raw/${machine_type}/test/
tsp scp -C ${ROOT_DIR}/data/dcase2021t2/dev_data/raw/${machine_type}/target_test/* data/dcase2021t2/dev_data/raw/${machine_type}/test/
done

# download eval data
cd -
cd "${ROOT_DIR}/${parent_dir}/dcase2021t2/eval_data/raw"
for machine_type in fan gearbox pump slider ToyCar ToyTrain valve; do
wget "https://zenodo.org/record/4660992/files/eval_data_${machine_type}_train.zip"
unzip "eval_data_${machine_type}_train.zip"

wget "https://zenodo.org/record/4884786/files/eval_data_${machine_type}_test.zip"
unzip "eval_data_${machine_type}_test.zip"
mkdir -p ${ROOT_DIR}/data/dcase2021t2/eval_data/raw/${machine_type}/test
cp -r ${ROOT_DIR}/data/dcase2021t2/eval_data/raw/${machine_type}/source_test/* ${ROOT_DIR}/data/dcase2021t2/eval_data/raw/${machine_type}/test/
cp -r ${ROOT_DIR}/data/dcase2021t2/eval_data/raw/${machine_type}/target_test/* ${ROOT_DIR}/data/dcase2021t2/eval_data/raw/${machine_type}/test/
done

# Adds reference labels to test data.
python ${ROOT_DIR}/tools/rename_eval_wav.py --dataset_parent_dir=${parent_dir} --dataset_type=DCASE2021T2
