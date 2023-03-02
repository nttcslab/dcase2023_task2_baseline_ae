mkdir -p "data/dcase2023t2/dev_data/raw"

# download dev data
cd "data/dcase2023t2/dev_data/raw"
for machine_type in bearing fan gearbox slider ToyCar ToyTrain valve; do
wget "https://zenodo.org/record/7690148/files/dev_${machine_type}.zip"
unzip "dev_${machine_type}.zip"
done
