
import argparse
import csv
import numpy as np
import sys
import os
import pandas as pd
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets.loader_common import get_machine_type_dict

SCORE_COLUMNS = ["h-mean", "a-mean"]
SCORE_INDEXES = ["AUC (source)", "AUC (target)", "pAUC (source, target)", "AUC", "pAUC", "TOTAL score"]
MODE_LIST = ["dev", "eval"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Main function to call training for different AutoEncoders')
    parser.add_argument("parent_dir", type=str)
    parser.add_argument("--file_name", type=str, default="auc_pauc")
    parser.add_argument("--ext", type=str, default=".csv")
    parser.add_argument("--dataset", type=str, default="DCASE2020T2", choices=["DCASE2020T2", "DCASE2022T2", "DCASE2023T2"])
    parser.add_argument("--float_format", type=str, default="%.4f")
    args = parser.parse_args()

    float_format = args.float_format
    if float_format in ["None", "none", "", " "]:
        float_format = None

    file_path = f"{args.parent_dir}/{args.dataset}_{args.file_name}{args.ext}"
    machine_type_dict = get_machine_type_dict(dataset_name=args.dataset)
    machine_type_list = list(machine_type_dict["machine_type"].keys())

    all_summarize_df = pd.read_csv(file_path, index_col=0)

    index_df = pd.DataFrame({
        "System":all_summarize_df.index.values.tolist(),
        "metric":["TOTAL score"] * len(all_summarize_df.index)
    })
    multi_index = pd.MultiIndex.from_frame(index_df)
    extract_df = pd.DataFrame(
        index=multi_index,
        columns=SCORE_COLUMNS + machine_type_list
    )

    # source and target domain results
    use_source_target = False
    for domain in ["source", "target"]:
        is_pick_mean = False
        for mode in MODE_LIST:
            for machine_type in machine_type_list:
                loc_column = f"{machine_type}_{mode}_arithmetic mean_AUC ({domain})"
                if loc_column in all_summarize_df.columns.values.tolist():
                    is_pick_mean = True
                    for index in all_summarize_df.index:
                        if mode == "dev":
                            extract_df.at[(index, f"AUC ({domain})"), machine_type] = all_summarize_df.loc[index, loc_column]
                        else:
                            extract_df.at[(index, f"AUC ({domain})"), f"{machine_type} {mode}"] = all_summarize_df.loc[index, loc_column]
        if is_pick_mean:
            use_source_target = True
            for index in all_summarize_df.index:
                extract_df.at[(index, f"AUC ({domain})"), "a-mean"] = all_summarize_df.loc[index, f"ALL_AUC ({domain})_ave"]
                extract_df.at[(index, f"AUC ({domain})"), "h-mean"] = all_summarize_df.loc[index, f"ALL_AUC ({domain})_hmean"]
    
    # single domain results
    loc_column_list = []
    is_pick_mean = False
    if not use_source_target:
        for mode in MODE_LIST:
            for machine_type in machine_type_list:
                loc_column = f"{machine_type}_{mode}_arithmetic mean_AUC"
                if loc_column in all_summarize_df.columns.values.tolist():
                    is_pick_mean = True
                    for index in all_summarize_df.index:
                        if mode == "dev":
                            extract_df.at[(index, "AUC"), machine_type] = all_summarize_df.loc[index, loc_column]
                        else:
                            extract_df.at[(index, "AUC"), f"{machine_type} {mode}"] = all_summarize_df.loc[index, loc_column]
        if is_pick_mean:
            for index in all_summarize_df.index:
                extract_df.at[(index, "AUC"), "a-mean"] = all_summarize_df.loc[index, f"ALL_AUC_ave"]
                extract_df.at[(index, "AUC"), "h-mean"] = all_summarize_df.loc[index, f"ALL_AUC_hmean"]

    # pAUC
    is_pick_mean = False
    extract_column = "pAUC (source, target)" if use_source_target else "pAUC"
    for mode in MODE_LIST:
        for machine_type in machine_type_list:
            loc_column = f"{machine_type}_{mode}_arithmetic mean_pAUC"
            if loc_column in all_summarize_df.columns.values.tolist():
                is_pick_mean = True
                for index in all_summarize_df.index:
                    if mode == "dev":
                        extract_df.at[(index, extract_column), machine_type] = all_summarize_df.loc[index, loc_column]
                    else:
                        extract_df.at[(index, extract_column), f"{machine_type} {mode}"] = all_summarize_df.loc[index, loc_column]
    if is_pick_mean:
        for index in all_summarize_df.index:
            extract_df.at[(index, extract_column), "a-mean"] = all_summarize_df.loc[index, f"ALL_pAUC_ave"]
            extract_df.at[(index, extract_column), "h-mean"] = all_summarize_df.loc[index, f"ALL_pAUC_hmean"]

    # Total score
    for index in all_summarize_df.index:
        extract_df.at[(index, "TOTAL score"), "a-mean"] = all_summarize_df.loc[index, f"TOTAL_score_ave"]
        extract_df.at[(index, "TOTAL score"), "h-mean"] = all_summarize_df.loc[index, f"TOTAL_score_hmean"]

    # Sort results
    extract_df["order"] = [SCORE_INDEXES.index(metric) for _, metric in extract_df.index]
    extract_df = extract_df.sort_values("order")
    extract_df = extract_df.sort_index(level="System", sort_remaining=False)
    extract_df = extract_df.drop("order", axis=1)

    export_file_path = f"{args.parent_dir}/{args.dataset}_{args.file_name}_extract{args.ext}"
    extract_df.astype(np.float64).to_csv(export_file_path, float_format=float_format)
    print(f"export extract results -> {export_file_path}")


    
