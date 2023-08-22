mkdir -p "data/dcase2023t2/dev_data/raw"

# download dev data
cd "data/dcase2023t2/dev_data/raw"
for machine_type in bearing fan gearbox slider ToyCar ToyTrain valve; do
# wget "https://zenodo.org/record/7690148/files/dev_${machine_type}.zip" # old files with some issues
# wget "https://zenodo.org/record/7690157/files/dev_${machine_type}.zip" # fixed ones
wget "https://zenodo.org/record/7882613/files/dev_${machine_type}.zip" # fixed attributes (2023/05/01)
unzip "dev_${machine_type}.zip"
done
