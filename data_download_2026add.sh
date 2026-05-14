mkdir -p "data/dcase2026t2/dev_data/raw"

# download dev data
cd "data/dcase2026t2/dev_data/raw"
for machine_type in \
    "ToyDrone" \
    "ToothBrush" \
    "SewingMachine" \
    "BlowerDustCollector" \
    "Sander" \
; do
wget "https://zenodo.org/records/20151556/files/eval_data_${machine_type}.zip"
unzip "eval_data_${machine_type}.zip"
done
