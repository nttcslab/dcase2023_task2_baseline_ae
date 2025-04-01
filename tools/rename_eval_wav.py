import os
import sys
import csv
import shutil
import argparse
from pathlib import Path

ROOT_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/../"
os.chdir(ROOT_DIR)

# labeled data path
EVAL_DATA_LIST_PATH = {
    "DCASE2020T2":f"{ROOT_DIR}/datasets/eval_data_list_2020.csv",
    "DCASE2021T2":f"{ROOT_DIR}/datasets/eval_data_list_2021.csv",
    "DCASE2022T2":f"{ROOT_DIR}/datasets/eval_data_list_2022.csv",
    "DCASE2023T2":f"{ROOT_DIR}/datasets/eval_data_list_2023.csv",
    "DCASE2024T2":f"{ROOT_DIR}/datasets/eval_data_list_2024.csv",
    "DCASE2025T2":f"{ROOT_DIR}/datasets/eval_data_list_2025.csv",
}

FILENAME_COL = 0
LABELING_FILENAME_COL = 1
MACHINE_TYPE_COL = 0

CHK_MACHINE_TYPE_LINE = 2

def copy_wav(dataset_parent_dir, dataset_type):
    dataset_dir = str(Path(f"{ROOT_DIR}/{dataset_parent_dir}/raw/").relative_to(ROOT_DIR))
    eval_data_list_path = EVAL_DATA_LIST_PATH[dataset_type]
    if not eval_data_list_path:
        return None
    
    if os.path.exists(eval_data_list_path):
        with open(eval_data_list_path) as fp:
            eval_data_list = list(csv.reader(fp))
    else:
        print(f"Err:eval_data_list.csv not found : {eval_data_list_path}")
        sys.exit(1)

    count = 0
    print('copy... : test -> test_rename')
    for eval_data in eval_data_list:
        if len(eval_data) < CHK_MACHINE_TYPE_LINE:
            machine_type = eval_data[MACHINE_TYPE_COL]
            default_dir = dataset_dir.lower() + "/" + machine_type + "/test"
            save_dir = dataset_dir.lower() + "/" + machine_type + "/test_rename"
            if(not os.path.exists(save_dir)):
                Path(save_dir).mkdir(parents=True, exist_ok=True)
            count = 0
            sys.stdout.write('\n')
            sys.stdout.flush()
        else:
            if os.path.exists(default_dir + "/" + eval_data[FILENAME_COL]):
                shutil.copy2(
                    default_dir + "/" + eval_data[FILENAME_COL],
                    save_dir + "/" + eval_data[LABELING_FILENAME_COL])
                count += 1
            sys.stdout.write(f'\r\t{machine_type}: {str(count)} files\tsaved dir: {save_dir}')
            sys.stdout.flush()
    sys.stdout.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Main function to call training for different AutoEncoders')
    parser.add_argument("--dataset_parent_dir", type=str, default="data",
                        help="saving datasets directory name.")
    parser.add_argument("--dataset_type", type=str, required=True,
                        choices=[
                            "DCASE2020T2",
                            "DCASE2021T2",
                            "DCASE2022T2",
                            "DCASE2023T2",
                            "DCASE2024T2",
                            "DCASE2025T2",
                        ],
                        help="what Dataset name to renamed.")
    args = parser.parse_args()

    copy_wav(
        dataset_parent_dir=f"{args.dataset_parent_dir}/{args.dataset_type}/eval_data",
        dataset_type=args.dataset_type
    )
