TZ=JST-9 date

echo $0 $*

dev_eval=$1
echo -e "\tdev_eval = '$dev_eval'"
echo

# check args
if [ "${dev_eval}" != "-d" ] \
    && [ "${dev_eval}" != "-e" ] \
    && [ "${dev_eval}" != "--dev" ] \
    && [ "${dev_eval}" != "--eval" ]
then
    echo "$0: argument error"
    echo -e "usage\t: $0 ['-d' | '--dev' | '-e' | '--eval']"
    echo -e "\tinvalid choice '$dev_eval'"
    echo -e "\tchoice from ['-d' | '--dev' | '-e' | '--eval']."
    echo -e "\t\t-d, --dev\t: Using Development dataset. "
    echo -e "\t\t-e, --eval\t: Using Additional training dataset and Evaluation dataset. "
    echo -e "example\t: $ bash $0 -d"
    exit 1
fi

# main process
base_job="bash"
job="test_ae.sh"

if [ "${dev_eval}" = "-d" ] || [ "${dev_eval}" = "--dev" ]
then
    dataset_list="\
        DCASE2025T2ToyCar \
        DCASE2025T2ToyTrain \
        DCASE2025T2bearing \
        DCASE2025T2fan \
        DCASE2025T2gearbox \
        DCASE2025T2slider \
        DCASE2025T2valve \
    "
elif [ "${dev_eval}" = "-e" ] || [ "${dev_eval}" = "--eval" ]
then
    echo dcase2025 task2 eval data are not publish
    exit
fi

for dataset in $dataset_list; do
    ${base_job} ${job} ${dataset} ${dev_eval} "MAHALA" 0
done