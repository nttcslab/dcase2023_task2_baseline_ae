dev_eval=$1

if [ "${dev_eval}" = "" ]
then
    echo "incorrect argument"
    echo "please set option argument '-d' or '-e'"
    echo "  -d  :   dev data"
    echo "  -e  :   eval data"
    exit 1
fi

base_job="bash"
job="test_ae.sh"

for dataset in DCASE2023T2bearing DCASE2023T2fan DCASE2023T2gearbox DCASE2023T2slider DCASE2023T2ToyCar DCASE2023T2ToyTrain DCASE2023T2valve; do
    ${base_job} ${job} ${dataset} ${dev_eval} "MSE" 0
done