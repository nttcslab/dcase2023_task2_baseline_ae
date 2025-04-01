import argparse
import csv
import numpy as np
import pandas as pd
import glob
import sys
import os
from scipy import stats

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets.loader_common import get_machine_type_dict

def load_target_dir_list(parent_dir):
    files = os.listdir(parent_dir)
    target_dirs = [dir for dir in files if os.path.isdir(os.path.join(parent_dir,dir))]
    return target_dirs

def get_column_basename(machine_type, dev_eval, index, column, tag):
    return f"{machine_type}_{dev_eval}_{index}_{column}{tag}"

def column_to_1d(columns_header, indices, machine_type_dict):
    new_columns_header = []
    tag_list = ["_ave", "_std", "_hmean"]
    for machine_type in machine_type_dict:
        for dev_eval in machine_type_dict[machine_type]:
            for column in columns_header:
                for index in machine_type_dict[machine_type][dev_eval]:
                    for tag in tag_list:
                        basename = get_column_basename(
                            machine_type=machine_type,
                            dev_eval=dev_eval,
                            column=column,
                            index=index,
                            tag=tag
                        )
                        new_columns_header.append(basename)
                for index in indices:
                    basename = get_column_basename(
                        machine_type=machine_type,
                        dev_eval=dev_eval,
                        column=column,
                        index=index,
                        tag=""
                    )
                    new_columns_header.append(basename)

    for column in columns_header:
        for tag in tag_list:
            new_columns_header.append(f"ALL_{column}_{tag}")
    return new_columns_header

def df_to_1d(df, machine_type, dev_eval, file_name):
    columns = df.columns.values
    indices = df.index.values
    summarize_df = pd.DataFrame(index=[file_name])
    for column in columns:
        for index in indices:
            base_name = get_column_basename(
                machine_type=machine_type,
                dev_eval=dev_eval,
                index=index,
                column=column,
                tag=""
            )
            summarize_df[base_name] = df.at[index, column]
    return summarize_df

def describe_df(df, df_1d, columns, file_name):
    describe_1d = df_1d.describe()
    describe_1d.loc["hmean"] =  stats.hmean(np.maximum(df_1d, sys.float_info.epsilon),axis=0)
    describe_df = pd.DataFrame(index=[file_name])
    describe = df.describe()
    describe.loc["hmean"] = stats.hmean(np.maximum(df, sys.float_info.epsilon),axis=0)
    for column_1d in columns:
        if column_1d in describe_1d.columns:
            describe_df[f"{column_1d}_ave"] = describe_1d.at["mean", column_1d]
            describe_df[f"{column_1d}_std"] = describe_1d.at["std", column_1d]
            describe_df[f"{column_1d}_hmean"] = describe_1d.at["hmean", column_1d]
        else:
            for column in describe.columns:
                if "_"+column in column_1d:
                    if "arithmetic mean" in column_1d:
                        describe_df[f"{column_1d}"] = describe.at["mean", column]
                        break
                    if "harmonic mean" in column_1d:
                        describe_df[f"{column_1d}"] = describe.at["hmean", column]
                        break
    return describe_df

def calc_all_mean(df, columns_header, machine_type_dict, df_mean):
    for column in columns_header:
        new_column_header = []
        for machine_type in machine_type_dict:
            for dev_eval in machine_type_dict[machine_type]:
                new_column_header.append(f"{machine_type}_{dev_eval}_arithmetic mean_{column}")
        new_column_header = [new_column for new_column in new_column_header if new_column in df.columns.values.tolist()]
        df_tmp = df.loc[:, new_column_header]
        df[f"ALL_{column}_ave"] = df_tmp.mean(axis=1)
        df[f"ALL_{column}_std"] = df_tmp.std(axis=1)
        df[f"ALL_{column}_hmean"] = df_mean[column]
    df["ALL_AUC_ave"] = df_mean["AUC_ave"]
    df["ALL_AUC_std"] = df_mean["AUC_std"]
    df["ALL_AUC_hmean"] = df_mean["AUC_hmean"]
    df["TOTAL_score_ave"] = df_mean["TOTAL_ave"]
    df["TOTAL_score_std"] = df_mean["TOTAL_std"]
    df["TOTAL_score_hmean"] = df_mean["TOTAL_hmean"]
    return df

def main(parent_dir, dataset, machine_type_dict, row_index=["arithmetic mean", "harmonic mean"]):
    target_dir_list = sorted(load_target_dir_list(parent_dir))
    auc_dict = {}
    summarize_df = pd.DataFrame()

    df_mean = pd.DataFrame()
    all_columns_header = set()
    for target_dir in target_dir_list:
        auc_dict[target_dir] = {}
        df_target = pd.DataFrame()
        columns_header = set()
        for machine_name in machine_type_dict.keys():
            auc_dict[target_dir][machine_name] = {}
            files = sorted(glob.glob(f"{args.parent_dir}/{target_dir}/*{dataset}{machine_name}*roc.csv"))
            file_dict = {
                "eval":[file for file in files if "Eval" in file],
                "dev":[file for file in files if "Eval" not in file]
            }
            
            for dev_eval in file_dict:
                df_machine_type_1d = pd.DataFrame()
                df_machine_type = pd.DataFrame()
                for file in file_dict[dev_eval]:
                    df = pd.read_csv(file, index_col=0)
                    columns_header = columns_header.union(df.columns.values.tolist())
                    columns = df_to_1d(
                        df=df,
                        machine_type=machine_name,
                        dev_eval=dev_eval,
                        file_name=file
                    ).columns.values.tolist()
                    df_extract = df.drop(row_index,axis=0)
                    df_1d_extract = df_to_1d(
                        df=df_extract,
                        machine_type=machine_name,
                        dev_eval=dev_eval,
                        file_name=file
                    )
                    df_machine_type_1d = pd.concat([df_machine_type_1d, df_1d_extract])
                    df_machine_type = pd.concat([df_machine_type, df_extract])
                df_target = pd.concat([df_target, df_machine_type])
                if len(df_machine_type_1d.index) > 0:
                    df_describe = describe_df(
                        df=df_machine_type,
                        df_1d=df_machine_type_1d,
                        file_name=target_dir,
                        columns=columns,
                    )
                    for column in df_describe.columns.values:
                        summarize_df.at[target_dir, column] = df_describe.at[target_dir, column]
        auc_header = [column for column in columns_header if "AUC" in column]
        if "pAUC" in auc_header:
            auc_header = [column for column in auc_header if "pAUC" not in column or column == "pAUC"]
        df_mean.loc[target_dir, list(columns_header)] = stats.hmean(np.maximum(df_target.loc[:,list(columns_header)], sys.float_info.epsilon),axis=0)
        df_mean.loc[target_dir, "TOTAL_hmean"] = stats.hmean(np.maximum(df_target.loc[:,auc_header], sys.float_info.epsilon),axis=None)
        df_mean.loc[target_dir, "TOTAL_ave"] = df_target.loc[:,auc_header].to_numpy().mean()
        df_mean.loc[target_dir, "TOTAL_std"] = df_target.loc[:,auc_header].to_numpy().std()
        auc_header = [column for column in auc_header if "pAUC" not in column]
        df_mean.loc[target_dir, "AUC_hmean"] = stats.hmean(np.maximum(df_target.loc[:,auc_header], sys.float_info.epsilon),axis=None)
        df_mean.loc[target_dir, "AUC_ave"] = df_target.loc[:,auc_header].to_numpy().mean()
        df_mean.loc[target_dir, "AUC_std"] = df_target.loc[:,auc_header].to_numpy().std()
        all_columns_header = all_columns_header.union(columns_header)
    summarize_df = calc_all_mean(
        df=summarize_df,
        columns_header=list(all_columns_header),
        machine_type_dict=machine_type_dict,
        df_mean = df_mean
    )
    export_file_path = f"{parent_dir}/{dataset}_auc_pauc.csv"
    summarize_df.to_csv(export_file_path)
    print(f"export concat results -> {export_file_path}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(
            description='Main function to call training for different AutoEncoders')
    parser.add_argument("parent_dir", type=str)
    parser.add_argument("--dataset", type=str, default="DCASE2020T2",
                        choices=[
                            "DCASE2020T2",
                            "DCASE2021T2",
                            "DCASE2022T2",
                            "DCASE2023T2",
                            "DCASE2024T2",
                            "DCASE2025T2",
                        ])
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

    machine_type_dict = get_machine_type_dict(dataset_name=args.dataset, mode=dev_mode)

    main(
        parent_dir=args.parent_dir,
        machine_type_dict=machine_type_dict["machine_type"],
        dataset=args.dataset,
    )