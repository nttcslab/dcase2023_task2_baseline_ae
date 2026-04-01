parent_dir=data
ROOT_DIR=$(cd $(dirname $0); pwd)/../
mkdir -p "${ROOT_DIR}/${parent_dir}/dcase2025t2/dev_data/raw"
mkdir -p "${ROOT_DIR}/${parent_dir}/dcase2025t2/eval_data/raw"

# download dev data
cd "data/dcase2025t2/dev_data/raw"
for machine_type in \
    "ToyCar" \
    "ToyTrain" \
    "bearing" \
    "fan" \
    "gearbox" \
    "slider" \
    "valve" \
; do
wget "https://zenodo.org/records/15097779/files/dev_${machine_type}.zip"
unzip "dev_${machine_type}.zip"
done

# download eval data
cd -
cd "data/dcase2025t2/eval_data/raw"
for machine_type in \
    "ToyRCCar" \
    "ToyPet" \
    "HomeCamera" \
    "AutoTrash" \
    "Polisher" \
    "ScrewFeeder" \
    "BandSealer" \
    "CoffeeGrinder" \
; do
wget "https://zenodo.org/records/15392814/files/eval_data_${machine_type}_train.zip"
unzip "eval_data_${machine_type}_train.zip"
done

for machine_type in \
    "ToyRCCar" \
    "ToyPet" \
    "HomeCamera" \
    "AutoTrash" \
    "Polisher" \
    "ScrewFeeder" \
    "BandSealer" \
    "CoffeeGrinder" \
; do
wget "https://zenodo.org/records/15519362/files/eval_data_${machine_type}_test.zip"
unzip "eval_data_${machine_type}_test.zip"
done

# Adds reference labels to test data.
python ${ROOT_DIR}/tools/rename_eval_wav.py --dataset_parent_dir=${parent_dir} --dataset_type=DCASE2025T2
