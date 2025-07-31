mkdir -p "data/dcase2025t2/eval_data/raw"

# download dev data
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
wget "https://zenodo.org/records/15519362/files/eval_data_${machine_type}_test.zip"
unzip "eval_data_${machine_type}_test.zip"
done

# Adds reference labels to test data.
python tools/rename_eval_wav.py --dataset_parent_dir=data --dataset_type=DCASE2025T2
