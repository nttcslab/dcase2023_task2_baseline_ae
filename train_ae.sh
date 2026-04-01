echo $*

dataset=$1
dev_eval=$2
mono=$3
id_1=$4
id_2=$5
id_3=$6
id_4=$7

id="$id_1 $id_2 $id_3 $id_4"

echo dataset ${dataset}
echo dev_eval ${dev_eval}
echo id ${id}

tag="id("
for i in ${id}; do
tag="${tag}${i}_"
done
tag="${tag})"

python3 train.py \
    --dataset=${dataset} \
    ${dev_eval} \
    -tag=${tag} \
    --use_ids ${id} \
    --train_only \
    --mono=${mono} \
