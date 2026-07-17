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
wget "https://zenodo.org/records/20437238/files/eval_data_${machine_type}_test.zip"
unzip "eval_data_${machine_type}_test.zip"
done

# Adds reference labels to test data.
cd -
python3 tools/rename_eval_wav.py --dataset_parent_dir=data --dataset_type=DCASE2026T2
