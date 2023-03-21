TZ=JST-9 date

ROOT_DIR=$(cd $(dirname $0); pwd)/../../
cd $ROOT_DIR

dataset=$1
dev_eval=$2

base_job="bash"
flag=0

echo "dataset : $dataset"
if [ "${dataset}" != "DCASE2020T2" ] \
    && [ "${dataset}" != "DCASE2022T2" ]
then
    echo -e "\tincorrect argument"
    echo -e "\tplease set option argument 'DCASE2022T2' or 'DCASE2020T2"
    echo 
    flag=1
fi

echo "dev_eval : $dev_eval"
if [ "${dev_eval}" != "-d" ] \
    && [ "${dev_eval}" != "-e" ] \
    && [ "${dev_eval}" != "--dev" ] \
    && [ "${dev_eval}" != "--eval" ]
then
    echo -e "\tincorrect argument"
    echo -e "\tplease set option argument '-d' or '-e'"
    echo -e "\t\t-d  :   dev data"
    echo -e "\t\t-e  :   eval data"
    echo
    flag=1
fi

if [ $flag -eq 1 ]
then
    exit
fi

for job in "train_ae.sh"; do
    if [ $dataset = "DCASE2022T2" ]; then
        if [ $dev_eval = "-d" ] || [ $dev_eval = "--dev" ]; then
            for dataset in DCASE2022T2bearing DCASE2022T2fan DCASE2022T2gearbox DCASE2022T2slider DCASE2022T2ToyCar DCASE2022T2ToyTrain DCASE2022T2valve; do
                ${base_job} $job ${dataset} ${dev_eval} 0
                ${base_job} $job ${dataset} ${dev_eval} 1
                ${base_job} $job ${dataset} ${dev_eval} 2
            done
        else # $dev_eval = "-e" || $dev_eval = "--eval"
            for dataset in DCASE2022T2bearing DCASE2022T2fan DCASE2022T2gearbox DCASE2022T2slider DCASE2022T2ToyCar DCASE2022T2ToyTrain DCASE2022T2valve; do
                ${base_job} $job ${dataset} ${dev_eval} 3
                ${base_job} $job ${dataset} ${dev_eval} 4
                ${base_job} $job ${dataset} ${dev_eval} 5
            done
        fi
    else # DCASE2020T2
        if [ $dev_eval = "-d" ] || [ $dev_eval = "--dev" ]; then
            dataset=DCASE2020T2ToyCar
            ${base_job} $job ${dataset} ${dev_eval} 1
            ${base_job} $job ${dataset} ${dev_eval} 2
            ${base_job} $job ${dataset} ${dev_eval} 3
            ${base_job} $job ${dataset} ${dev_eval} 4

            dataset=DCASE2020T2ToyConveyor
            ${base_job} $job ${dataset} ${dev_eval} 1
            ${base_job} $job ${dataset} ${dev_eval} 2
            ${base_job} $job ${dataset} ${dev_eval} 3

            for dataset in DCASE2020T2fan DCASE2020T2valve DCASE2020T2slider DCASE2020T2pump; do
                ${base_job} $job ${dataset} ${dev_eval} 0
                ${base_job} $job ${dataset} ${dev_eval} 2
                ${base_job} $job ${dataset} ${dev_eval} 4
                ${base_job} $job ${dataset} ${dev_eval} 6
            done
        else # $dev_eval = "-e" || $dev_eval = "--eval"
            dataset=DCASE2020T2ToyCar
            ${base_job} $job ${dataset} ${dev_eval} 5
            ${base_job} $job ${dataset} ${dev_eval} 6
            ${base_job} $job ${dataset} ${dev_eval} 7

            dataset=DCASE2020T2ToyConveyor
            ${base_job} $job ${dataset} ${dev_eval} 4
            ${base_job} $job ${dataset} ${dev_eval} 5
            ${base_job} $job ${dataset} ${dev_eval} 6

            for dataset in DCASE2020T2fan DCASE2020T2valve DCASE2020T2slider DCASE2020T2pump; do
                ${base_job} $job ${dataset} ${dev_eval} 1
                ${base_job} $job ${dataset} ${dev_eval} 3
                ${base_job} $job ${dataset} ${dev_eval} 5
            done
        fi
    fi
done
