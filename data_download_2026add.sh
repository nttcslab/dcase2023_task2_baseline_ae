mkdir -p "data/dcase2026t2/eval_data/raw"

# download eval data
cd "data/dcase2026t2/eval_data/raw"
for machine_type in \
    "ToyDrone" \
    "ToothBrush" \
    "SewingMachine" \
    "BlowerDustCollector" \
    "Sander" \
; do
wget "https://zenodo.org/records/20151556/files/eval_data_${machine_type}_train.zip"
unzip "eval_data_${machine_type}_train.zip"
done
