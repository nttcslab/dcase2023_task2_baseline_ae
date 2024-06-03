parent_dir=data
ROOT_DIR=$(cd $(dirname $0); pwd)/
mkdir -p "${ROOT_DIR}/${parent_dir}/dcase2024t2/eval_data/raw"

# download eval data
cd "${parent_dir}/dcase2024t2/eval_data/raw"
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

