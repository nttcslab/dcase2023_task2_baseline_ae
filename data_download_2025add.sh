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
wget "https://zenodo.org/records/15392814/files/eval_data_${machine_type}_train.zip"
unzip "eval_data_${machine_type}_train.zip"
done
