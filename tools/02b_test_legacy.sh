TZ=JST-9 date

echo $0 $*

ROOT_DIR=$(cd $(dirname $0); pwd)/../
cd $ROOT_DIR

dataset=$1
dev_eval=$2
echo -e "\tdataset = '$dataset'"
echo -e "\tdev_eval = '$dev_eval'"
echo

base_job="bash"
score="MAHALA"

# check args
args_flag=0
args_flag_dataset=0
if [ "${dataset}" != "DCASE2020T2" ] \
    && [ "${dataset}" != "DCASE2021T2" ] \
    && [ "${dataset}" != "DCASE2022T2" ] \
    && [ "${dataset}" != "DCASE2023T2" ] \
    && [ "${dataset}" != "DCASE2024T2" ]
then
    args_flag=1
    args_flag_dataset=1
fi

args_flag_dev_eval=0
if [ "${dev_eval}" != "-d" ] \
    && [ "${dev_eval}" != "-e" ] \
    && [ "${dev_eval}" != "--dev" ] \
    && [ "${dev_eval}" != "--eval" ]
then
    args_flag=1
    args_flag_dev_eval=1
fi

if [ $args_flag -eq 1 ]
then
    echo "$0: argument error"
    echo -e "usage\t: $0 ['DCASE2020T2' | 'DCASE2021T2' | 'DCASE2022T2' | 'DCASE2023T2' | 'DCASE2024T2'] ['-d' | '--dev' | '-e' | '--eval']"

    if [ $args_flag_dataset -eq 1 ]
    then
        echo -e "\tdataset: invalid choice '$dataset'"
        echo -e "\tchoice from ['DCASE2020T2' | 'DCASE2021T2' | 'DCASE2022T2' | 'DCASE2023T2']."
        echo -e "\t\tDCASE2020T2\t: Use DCASE2020 Task2 datasets. "
        echo -e "\t\tDCASE2021T2\t: Use DCASE2021 Task2 datasets. "
        echo -e "\t\tDCASE2022T2\t: Use DCASE2022 Task2 datasets. "
        echo -e "\t\tDCASE2023T2\t: Use DCASE2023 Task2 datasets. "
        echo -e "\t\tDCASE2024T2\t: Use DCASE2024 Task2 datasets. "
        echo 
    fi

    if [ $args_flag_dev_eval -eq 1 ]
    then
        echo -e "\tdev_eval: invalid choice '$dev_eval'"
        echo -e "\tchoice from ['-d' | '--dev' | '-e' | '--eval']."
        echo -e "\t\t-d, --dev\t: Using Development dataset. "
        echo -e "\t\t-e, --eval\t: Using Additional training dataset and Evaluation dataset. "
        echo
    fi

    echo -e "example\t: $ bash $0 DCASE2020T2 -d"
    exit 1
fi

# main process
for job in "test_ae.sh"; do

    if [ $dataset = "DCASE2024T2" ]; then
        if [ $dev_eval = "-d" ] || [ $dev_eval = "--dev" ]; then
            for machine_type in DCASE2024T2bearing DCASE2024T2fan DCASE2024T2gearbox DCASE2024T2slider DCASE2024T2ToyCar DCASE2024T2ToyTrain DCASE2024T2valve; do
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 0
            done
        else # $dev_eval = "-e" || $dev_eval = "--eval"
            for machine_type in \
                DCASE2024T23DPrinter \
                DCASE2024T2AirCompressor \
                DCASE2024T2Scanner \
                DCASE2024T2ToyCircuit \
                DCASE2024T2HoveringDrone \
                DCASE2024T2HairDryer \
                DCASE2024T2ToothBrush \
                DCASE2024T2RoboticArm \
                DCASE2024T2BrushlessMotor \
            ; do
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 0
            done
        fi
    elif [ $dataset = "DCASE2023T2" ]; then
        if [ $dev_eval = "-d" ] || [ $dev_eval = "--dev" ]; then
            for machine_type in DCASE2022T2bearing DCASE2022T2fan DCASE2022T2gearbox DCASE2022T2slider DCASE2022T2ToyCar DCASE2022T2ToyTrain DCASE2022T2valve; do
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 0
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 1
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 2
            done
        else # $dev_eval = "-e" || $dev_eval = "--eval"
            for machine_type in DCASE2022T2bearing DCASE2022T2fan DCASE2022T2gearbox DCASE2022T2slider DCASE2022T2ToyCar DCASE2022T2ToyTrain DCASE2022T2valve; do
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 3
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 4
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 5
            done
        fi
    elif [ $dataset = "DCASE2021T2" ]; then
        if [ $dev_eval = "-d" ] || [ $dev_eval = "--dev" ]; then
            for machine_type in DCASE2021T2fan DCASE2021T2gearbox DCASE2021T2pump DCASE2021T2slider DCASE2021T2ToyCar DCASE2021T2ToyTrain DCASE2021T2valve; do
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 0
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 1
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 2
            done
        else # $dev_eval = "-e" || $dev_eval = "--eval"
            for machine_type in DCASE2021T2fan DCASE2021T2gearbox DCASE2021T2pump DCASE2021T2slider DCASE2021T2ToyCar DCASE2021T2ToyTrain DCASE2021T2valve; do
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 3
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 4
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 5
            done
        fi
    else # DCASE2020T2
        if [ $dev_eval = "-d" ] || [ $dev_eval = "--dev" ]; then
            machine_type=DCASE2020T2ToyCar
            ${base_job} $job ${machine_type} ${dev_eval} ${score} 1
            ${base_job} $job ${machine_type} ${dev_eval} ${score} 2
            ${base_job} $job ${machine_type} ${dev_eval} ${score} 3
            ${base_job} $job ${machine_type} ${dev_eval} ${score} 4

            machine_type=DCASE2020T2ToyConveyor
            ${base_job} $job ${machine_type} ${dev_eval} ${score} 1
            ${base_job} $job ${machine_type} ${dev_eval} ${score} 2
            ${base_job} $job ${machine_type} ${dev_eval} ${score} 3

            for machine_type in DCASE2020T2fan DCASE2020T2valve DCASE2020T2slider DCASE2020T2pump; do
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 0
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 2
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 4
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 6
            done
        else # $dev_eval = "-e" || $dev_eval = "--eval"
            machine_type=DCASE2020T2ToyCar
            ${base_job} $job ${machine_type} ${dev_eval} ${score} 5
            ${base_job} $job ${machine_type} ${dev_eval} ${score} 6
            ${base_job} $job ${machine_type} ${dev_eval} ${score} 7

            machine_type=DCASE2020T2ToyConveyor
            ${base_job} $job ${machine_type} ${dev_eval} ${score} 4
            ${base_job} $job ${machine_type} ${dev_eval} ${score} 5
            ${base_job} $job ${machine_type} ${dev_eval} ${score} 6

            for machine_type in DCASE2020T2fan DCASE2020T2valve DCASE2020T2slider DCASE2020T2pump; do
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 1
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 3
                ${base_job} $job ${machine_type} ${dev_eval} ${score} 5
            done
        fi
    fi
done
