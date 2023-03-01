mkdir -p "data/dcase2022t2/dev_data/raw"
mkdir -p "data/dcase2022t2/eval_data/raw"

# download dev data
cd "data/dcase2022t2/dev_data/raw"
for machine_type in bearing fan gearbox slider ToyCar ToyTrain valve; do
wget "https://zenodo.org/record/6355122/files/dev_${machine_type}.zip"
unzip "dev_${machine_type}.zip"
done

# download eval data
cd -
cd "data/dcase2022t2/eval_data/raw"
for machine_type in bearing fan gearbox slider ToyCar ToyTrain valve; do
wget "https://zenodo.org/record/6462969/files/eval_data_${machine_type}_train.zip"
unzip "eval_data_${machine_type}_train.zip"

wget "https://zenodo.org/record/6586456/files/eval_data_${machine_type}_test.zip"
unzip "eval_data_${machine_type}_test.zip"
done