mkdir -p "data/dcase2024t2/dev_data/raw"

# download dev data
cd "data/dcase2024t2/dev_data/raw"
for machine_type in ToyCar ToyTrain; do
wget "/files/dev_${machine_type}.zip"
unzip "dev_${machine_type}.zip"
done
