mkdir -p "data/dcase2023t2/eval_data/raw"

# download eval data
cd "data/dcase2023t2/eval_data/raw"
for machine_type in bandsaw grinder shaker ToyDrone ToyNscale ToyTank Vacuum; do
wget "https://zenodo.org/record/7860847/files/eval_data_${machine_type}_test.zip"
unzip "eval_data_${machine_type}_test.zip"
done
