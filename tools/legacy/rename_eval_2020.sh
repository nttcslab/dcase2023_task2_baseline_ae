ROOT_DIR=$(cd $(dirname $0); pwd)/../../
cd $ROOT_DIR

python3 tools/legacy/rename_eval_legacy_wav.py \
--dataset_parent_dir=data \
--dataset_type=DCASE2020T2 \
