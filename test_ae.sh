echo $*

dataset=$1
dev_eval=$2
mono=$3
score=$4
id_1=$5
id_2=$6
id_3=$7
id_4=$8

id="$id_1 $id_2 $id_3 $id_4"

echo dataset ${dataset}
echo dev_eval ${dev_eval}
echo score ${score}
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
    --score ${score} \
    --test_only \
    --mono=${mono} \
