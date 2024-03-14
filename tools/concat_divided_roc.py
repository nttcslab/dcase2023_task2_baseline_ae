from copy import copy
import os
import numpy as np
import re
import argparse
import glob
import statistics
from pathlib import Path
from scipy import stats
import sys
import csv

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets.loader_common import get_machine_type_dict

def csv_read(file_list):
    auc_list = []
    pauc_list = []
    pauc_s_list = []
    pauc_t_list = []
    auc_s_list = []
    auc_t_list = []
    column_list=[]
    for file in file_list:
        with open(file) as f:# list form
            reader = csv.DictReader(f)
            row_auc = []
            row_pauc = []
            row_pauc_s = []
            row_pauc_t = []
            row_auc_s = []
            row_auc_t = []
            for row in reader:
                if "AUC" in row.keys():
                    row_auc.append(float(row["AUC"]))
                if "pAUC" in row.keys():
                    row_pauc.append(float(row["pAUC"]))
                if "pAUC (source)" in row.keys():
                    row_pauc_s.append(float(row["pAUC (source)"]))
                if "pAUC (target)" in row.keys():
                    row_pauc_t.append(float(row["pAUC (target)"]))
                if "AUC (source)" in row.keys():
                    row_auc_s.append(float(row["AUC (source)"]))
                if "AUC (target)" in row.keys():
                    row_auc_t.append(float(row["AUC (target)"]))
            auc_list.append(row_auc)
            pauc_list.append(row_pauc)
            pauc_s_list.append(row_pauc_s)
            pauc_t_list.append(row_pauc_t)
            auc_s_list.append(row_auc_s)
            auc_t_list.append(row_auc_t)
    if all(auc_list):
        column_list.append("AUC")
    if all(pauc_list):
        column_list.append("pAUC")
    if all(pauc_s_list):
        column_list.append("pAUC (source)")
    if all(pauc_t_list):
        column_list.append("pAUC (target)")
    if all(auc_s_list):
        column_list.append("AUC (source)")
    if all(auc_t_list):
        column_list.append("AUC (target)")
    return [
        [list(x) for x in zip(*auc_list)],
        [list(x) for x in zip(*pauc_list)],
        [list(x) for x in zip(*pauc_s_list)],
        [list(x) for x in zip(*pauc_t_list)],
        [list(x) for x in zip(*auc_s_list)],
        [list(x) for x in zip(*auc_t_list)],
    ], column_list

def get_use_index_list(file_list,keyword):
    use_id_list = []
    for file in file_list:
        use_id = re.findall(keyword,file)
        use_id = use_id[0].split("_")
        use_id = list(filter(None, use_id))
        use_id = sorted(map(int,use_id))
        use_id_list.append(use_id)
    all_id = list(set(sum(use_id_list, [])))

    return all_id, use_id_list

def concat_auc_pauc(auc_list, all_id_list, nml_id_list):
    concat_auc_dict = {}
    if len(auc_list) == 0:
        return None
    
    for i in range(len(nml_id_list)):
        for nml_id in nml_id_list[i]:
            idx = all_id_list.index(nml_id)
            if len(auc_list) <= 2:
                concat_auc_dict[idx]=(auc_list[0][i])
            else:
                concat_auc_dict[idx]=(auc_list[idx][i])

    return [x[1] for x in sorted(concat_auc_dict.items())]

def export_csv(file_path, auc_list, column_header, machine_id_list):
    csv_lines = []
    csv_lines.append(["id"] + column_header)
    for idx, auc_row in zip(machine_id_list, zip(*auc_list)):
        csv_lines.append([idx] + list(auc_row))
    csv_lines.append(["arithmetic mean"] + np.mean(auc_list, axis=1).tolist())
    csv_lines.append(["harmonic mean"] + stats.hmean(auc_list, axis=1).tolist())
    np.savetxt(file_path, csv_lines,fmt="%s", delimiter=",")
    print(f"\tSave ROC -> {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Main function to call training for different AutoEncoders')
    parser.add_argument("--parent_dir", type=str, default='', metavar='N',
                        help='')
    parser.add_argument("--export_dir", type=str, default='', metavar='N',
                        help='')
    parser.add_argument("--keyword", type=str, default='_id\((.+)\)', metavar='N',
                        help='')
    parser.add_argument("--machine_type", type=str, default='all', metavar='N',
                        help='')
    parser.add_argument("--file_name", type=str, default='*roc.csv', metavar='N',
                        help='')
    parser.add_argument('--dataset',type=str, default="DCASE2020T2", choices=["DCASE2020T2", "DCASE2021T2", "DCASE2022T2", "DCASE2023T2", "DCASE2024T2"])
    parser.add_argument('-d', '--dev', action='store_true',
                        help='Use Development dataset')
    parser.add_argument('-e', '--eval', action='store_true',
                        help='Use Evaluation dataset')
    args = parser.parse_args()

    if args.eval:
        dev_mode = False
    elif args.dev:
        dev_mode = True
    else:
        print("incorrect argument")
        print("please set option argument '--dev' or '--eval'")
        sys.exit()

    machine_type_dict = get_machine_type_dict(dataset_name=args.dataset, mode=dev_mode)["machine_type"]
    all_types = list(machine_type_dict.keys())
    
    if args.machine_type == "all":
        types = all_types
    else:
        types = args.machine_type.split(",")

    export_dir = os.path.basename(args.parent_dir)
    if args.export_dir:
        export_dir = f"{args.export_dir}/{export_dir}"
    else:
        export_dir = f"{args.parent_dir}/{export_dir}"
    Path(export_dir).mkdir(parents=True, exist_ok=True)

    for machine_type in types:
        eval_figdata_target_list = []
        dev_figdata_target_list = []
        eval_figdata_list_tmp = []
        dev_figdata_list_tmp = []
        files = sorted(glob.glob(f"{args.parent_dir}/*{args.dataset}{machine_type}*{args.file_name}"))

        # print(files)
        # print(dev_file_list)
        # print(eval_file_list)
        file_list = {
            "dev":[file for file in files if "Eval" not in file],
            "eval":[file for file in files if "Eval" in file],
        }
        for dev_eval in ["dev", "eval"]:
            if len(file_list[dev_eval]) > 0:
                auc_list, column_list = csv_read(file_list=file_list[dev_eval])
                all_id_list, nml_id_list = get_use_index_list(file_list[dev_eval], args.keyword)
                print(f"{machine_type}\t({dev_eval}_data)\t: {len(nml_id_list)}/{len(all_id_list)}")
                if len(nml_id_list) == len(all_id_list):
                    for i in range(len(auc_list)):
                        auc_list[i] = concat_auc_pauc(
                            auc_list=auc_list[i],
                            all_id_list=all_id_list,
                            nml_id_list=nml_id_list
                        )
                    auc_list = [auc for auc in auc_list if auc is not None]
                    file_path = os.path.splitext(os.path.basename(file_list[dev_eval][0]))[0]
                    file_path = re.sub(args.keyword, "", file_path)
                    file_path = f"{export_dir}/{file_path}.csv"
                    export_csv(
                        file_path=file_path,
                        auc_list=auc_list,
                        column_header=column_list,
                        machine_id_list=machine_type_dict[machine_type][dev_eval]
                    )
                else:
                    print(f"\t{machine_type} ROC did not concat.")