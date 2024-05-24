mkdir -p "data/dcase2024t2/eval_data/raw"

# download eval data
cd "data/dcase2024t2/eval_data/raw"

# for machine_type in 3DPrinter AirCompressor Scanner ToyCircuit HoveringDrone HairDryer ToothBrush RoboticArm BrushlessMotor; do
# wget "https://zenodo.org/records/11183284/files/eval_data_${machine_type}_train.zip"
# unzip "eval_data_${machine_type}_train.zip"
# done

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

