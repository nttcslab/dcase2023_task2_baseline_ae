mkdir -p "data/dcase2023t2/eval_data/raw"

# download eval data
cd "data/dcase2023t2/eval_data/raw"
for machine_type in bandsaw grinder shaker ToyDrone ToyNscale ToyTank Vacuum; do
wget ""
unzip "eval_${machine_type}.zip"
done