mkdir -p "data/dcase2026t2/dev_data/raw"

# download dev data
cd "data/dcase2026t2/dev_data/raw"
for machine_type in \
    "ToyCar" \
    "ToyCarEmu" \
    "bearingEmu" \
    "fan" \
    "gearboxEmu" \
    "sliderEmu" \
    "valveEmu" \
; do
wget "https://zenodo.org/records/19336329/files/dev_${machine_type}.zip"
unzip "dev_${machine_type}.zip"
done
