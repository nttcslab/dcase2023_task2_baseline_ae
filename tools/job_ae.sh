TZ=JST-9 date

base_job="bash"

for job in "train_ae.sh"; do

dev_eval="-d"

dataset=DCASE2022T2bearing
${base_job} $job ${dataset} ${dev_eval} 0
${base_job} $job ${dataset} ${dev_eval} 1
${base_job} $job ${dataset} ${dev_eval} 2

dataset=DCASE2022T2fan
${base_job} $job ${dataset} ${dev_eval} 0
${base_job} $job ${dataset} ${dev_eval} 1
${base_job} $job ${dataset} ${dev_eval} 2

dataset=DCASE2022T2gearbox
${base_job} $job ${dataset} ${dev_eval} 0
${base_job} $job ${dataset} ${dev_eval} 1
${base_job} $job ${dataset} ${dev_eval} 2

dataset=DCASE2022T2slider
${base_job} $job ${dataset} ${dev_eval} 0
${base_job} $job ${dataset} ${dev_eval} 1
${base_job} $job ${dataset} ${dev_eval} 2

dataset=DCASE2022T2ToyCar
${base_job} $job ${dataset} ${dev_eval} 0
${base_job} $job ${dataset} ${dev_eval} 1
${base_job} $job ${dataset} ${dev_eval} 2

dataset=DCASE2022T2ToyTrain
${base_job} $job ${dataset} ${dev_eval} 0
${base_job} $job ${dataset} ${dev_eval} 1
${base_job} $job ${dataset} ${dev_eval} 2

dataset=DCASE2022T2valve
${base_job} $job ${dataset} ${dev_eval} 0
${base_job} $job ${dataset} ${dev_eval} 1
${base_job} $job ${dataset} ${dev_eval} 2

dev_eval="-e"

dataset=DCASE2022T2bearing
${base_job} $job ${dataset} ${dev_eval} 3
${base_job} $job ${dataset} ${dev_eval} 4
${base_job} $job ${dataset} ${dev_eval} 5

dataset=DCASE2022T2fan
${base_job} $job ${dataset} ${dev_eval} 3
${base_job} $job ${dataset} ${dev_eval} 4
${base_job} $job ${dataset} ${dev_eval} 5

dataset=DCASE2022T2gearbox
${base_job} $job ${dataset} ${dev_eval} 3
${base_job} $job ${dataset} ${dev_eval} 4
${base_job} $job ${dataset} ${dev_eval} 5

dataset=DCASE2022T2slider
${base_job} $job ${dataset} ${dev_eval} 3
${base_job} $job ${dataset} ${dev_eval} 4
${base_job} $job ${dataset} ${dev_eval} 5

dataset=DCASE2022T2ToyCar
${base_job} $job ${dataset} ${dev_eval} 3
${base_job} $job ${dataset} ${dev_eval} 4
${base_job} $job ${dataset} ${dev_eval} 5

dataset=DCASE2022T2ToyTrain
${base_job} $job ${dataset} ${dev_eval} 3
${base_job} $job ${dataset} ${dev_eval} 4
${base_job} $job ${dataset} ${dev_eval} 5

dataset=DCASE2022T2valve
${base_job} $job ${dataset} ${dev_eval} 3
${base_job} $job ${dataset} ${dev_eval} 4
${base_job} $job ${dataset} ${dev_eval} 5

dev_eval="-d"
dataset=DCASE2020T2ToyCar
${base_job} $job ${dataset} ${dev_eval} 1
${base_job} $job ${dataset} ${dev_eval} 2
${base_job} $job ${dataset} ${dev_eval} 3
${base_job} $job ${dataset} ${dev_eval} 4

dataset=DCASE2020T2ToyConveyor
${base_job} $job ${dataset} ${dev_eval} 1
${base_job} $job ${dataset} ${dev_eval} 2
${base_job} $job ${dataset} ${dev_eval} 3

dataset=DCASE2020T2fan
${base_job} $job ${dataset} ${dev_eval} 0
${base_job} $job ${dataset} ${dev_eval} 2
${base_job} $job ${dataset} ${dev_eval} 4
${base_job} $job ${dataset} ${dev_eval} 6

dataset=DCASE2020T2valve
${base_job} $job ${dataset} ${dev_eval} 0
${base_job} $job ${dataset} ${dev_eval} 2
${base_job} $job ${dataset} ${dev_eval} 4
${base_job} $job ${dataset} ${dev_eval} 6

dataset=DCASE2020T2slider
${base_job} $job ${dataset} ${dev_eval} 0
${base_job} $job ${dataset} ${dev_eval} 2
${base_job} $job ${dataset} ${dev_eval} 4
${base_job} $job ${dataset} ${dev_eval} 6

dataset=DCASE2020T2pump
${base_job} $job ${dataset} ${dev_eval} 0
${base_job} $job ${dataset} ${dev_eval} 2
${base_job} $job ${dataset} ${dev_eval} 4
${base_job} $job ${dataset} ${dev_eval} 6

dev_eval="-e"

dataset=DCASE2020T2ToyCar
${base_job} $job ${dataset} ${dev_eval} 5
${base_job} $job ${dataset} ${dev_eval} 6
${base_job} $job ${dataset} ${dev_eval} 7

dataset=DCASE2020T2ToyConveyor
${base_job} $job ${dataset} ${dev_eval} 4
${base_job} $job ${dataset} ${dev_eval} 5
${base_job} $job ${dataset} ${dev_eval} 6

dataset=DCASE2020T2fan
${base_job} $job ${dataset} ${dev_eval} 1
${base_job} $job ${dataset} ${dev_eval} 3
${base_job} $job ${dataset} ${dev_eval} 5

dataset=DCASE2020T2valve
${base_job} $job ${dataset} ${dev_eval} 1
${base_job} $job ${dataset} ${dev_eval} 3
${base_job} $job ${dataset} ${dev_eval} 5

dataset=DCASE2020T2slider
${base_job} $job ${dataset} ${dev_eval} 1
${base_job} $job ${dataset} ${dev_eval} 3
${base_job} $job ${dataset} ${dev_eval} 5

dataset=DCASE2020T2pump
${base_job} $job ${dataset} ${dev_eval} 1
${base_job} $job ${dataset} ${dev_eval} 3
${base_job} $job ${dataset} ${dev_eval} 5

done
