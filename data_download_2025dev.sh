mkdir -p "data/dcase2025t2/dev_data/raw"

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
