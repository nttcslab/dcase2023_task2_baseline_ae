parent_dir=data
ROOT_DIR=$(cd $(dirname $0); pwd)/../
mkdir -p "${ROOT_DIR}/${parent_dir}/dcase2024t2/dev_data/raw"
mkdir -p "${ROOT_DIR}/${parent_dir}/dcase2024t2/eval_data/raw"

# download dev data
cd "data/dcase2024t2/dev_data/raw"
for machine_type in bearing fan gearbox slider ToyCar ToyTrain valve; do
wget "https://zenodo.org/record/10902294/files/dev_${machine_type}.zip"
unzip "dev_${machine_type}.zip"
done

# download eval data
cd -
cd "data/dcase2024t2/eval_data/raw"
for machine_type in \
    3DPrinter_train_r2 \
    AirCompressor_train \
    Scanner_train \
    ToyCircuit_train \
    HoveringDrone_train \
    HairDryer_train \
    ToothBrush_train \
    RoboticArm_train_r2 \
    BrushlessMotor_train \
; do
wget "https://zenodo.org/records/11259435/files/eval_data_${machine_type}.zip"
unzip "eval_data_${machine_type}.zip"
done

for machine_type in \
    3DPrinter \
    AirCompressor \
    Scanner \
    ToyCircuit \
    HoveringDrone \
    HairDryer \
    ToothBrush \
    RoboticArm \
    BrushlessMotor \
; do
wget "https://zenodo.org/records/11363076/files/eval_data_${machine_type}_test.zip"
unzip "eval_data_${machine_type}_test.zip"
done

# Adds reference labels to test data.
python ${ROOT_DIR}/tools/rename_eval_wav.py --dataset_parent_dir=${parent_dir} --dataset_type=DCASE2024T2
