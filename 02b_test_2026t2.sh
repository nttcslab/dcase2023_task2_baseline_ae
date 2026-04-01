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
mono=False

if [ "${dev_eval}" = "-d" ] || [ "${dev_eval}" = "--dev" ]
then
    dataset_list="\
        DCASE2026T2ToyCar \
        DCASE2026T2ToyCarEmu \
        DCASE2026T2bearingEmu \
        DCASE2026T2fan \
        DCASE2026T2gearboxEmu \
        DCASE2026T2sliderEmu \
        DCASE2026T2valveEmu \
    "
elif [ "${dev_eval}" = "-e" ] || [ "${dev_eval}" = "--eval" ]
then
    echo dcase2026 task2 eval data are not publish
    exit
fi

for dataset in $dataset_list; do
    ${base_job} ${job} ${dataset} ${dev_eval} ${mono} "MAHALA" 0
done