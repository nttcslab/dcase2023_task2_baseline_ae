#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def read_csv_rows(file_path: Path) -> List[List[str]]:
    with file_path.open(newline="") as handle:
        return list(csv.reader(handle))


def load_result_metrics(result_csv: Path) -> Dict[str, float]:
    rows = read_csv_rows(result_csv)
    header = rows[0]
    data_row = None
    for row in rows[1:]:
        if row and row[0] == "arithmetic mean":
            data_row = row
            break
    if data_row is None:
        raise ValueError(f"No arithmetic mean row in: {result_csv}")

    metric_map = {name: value for name, value in zip(header, data_row)}
    auc_source = float(metric_map["AUC (source)"])
    auc_target = float(metric_map["AUC (target)"])
    pauc = float(metric_map["pAUC"])
    precision_source = float(metric_map["precision (source)"])
    precision_target = float(metric_map["precision (target)"])
    recall_source = float(metric_map["recall (source)"])
    recall_target = float(metric_map["recall (target)"])
    f1_source = float(metric_map["F1 score (source)"])
    f1_target = float(metric_map["F1 score (target)"])

    return {
        "AUC_source": auc_source,
        "AUC_target": auc_target,
        "AUC_total": (auc_source + auc_target) / 2.0,
        "pAUC": pauc,
        "Precision_source": precision_source,
        "Recall_source": recall_source,
        "F1_source": f1_source,
        "Precision_target": precision_target,
        "Recall_target": recall_target,
        "F1_target": f1_target,
    }


def load_selected_threshold_info(calibration_csv: Path) -> Dict[str, str]:
    rows = read_csv_rows(calibration_csv)
    header = rows[0]
    idx_selected = header.index("selected")
    selected_row = None
    for row in rows[1:]:
        if row and row[idx_selected] == "1":
            selected_row = row
            break
    if selected_row is None:
        raise ValueError(f"No selected threshold row found in {calibration_csv}")

    return {
        "candidate_id": selected_row[header.index("candidate_id")],
        "mode": selected_row[header.index("mode")],
        "parameter": selected_row[header.index("parameter")],
        "threshold": selected_row[header.index("threshold")],
    }


def write_comparison_csv(file_path: Path, rows: List[Tuple[str, Dict[str, float]]]) -> None:
    with file_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "system",
            "AUC_source",
            "AUC_target",
            "AUC_total",
            "pAUC",
            "Precision_source",
            "Recall_source",
            "F1_source",
            "Precision_target",
            "Recall_target",
            "F1_target",
        ])
        for label, metric in rows:
            writer.writerow([
                label,
                f"{metric['AUC_source']:.4f}",
                f"{metric['AUC_target']:.4f}",
                f"{metric['AUC_total']:.4f}",
                f"{metric['pAUC']:.4f}",
                f"{metric['Precision_source']:.4f}",
                f"{metric['Recall_source']:.4f}",
                f"{metric['F1_source']:.4f}",
                f"{metric['Precision_target']:.4f}",
                f"{metric['Recall_target']:.4f}",
                f"{metric['F1_target']:.4f}",
            ])


def write_comparison_txt(file_path: Path, rows: List[Tuple[str, Dict[str, float]]]) -> None:
    header = " System                                   | AUC_src | AUC_tgt | AUC_tot | pAUC  | F1_src | F1_tgt"
    sep = "------------------------------------------+---------+---------+---------+-------+--------+-------"
    lines = [header, sep]
    for label, metric in rows:
        lines.append(
            f" {label:<40} | {metric['AUC_source']:7.4f} | {metric['AUC_target']:7.4f} | {metric['AUC_total']:7.4f} | {metric['pAUC']:5.4f} | {metric['F1_source']:6.4f} | {metric['F1_target']:5.4f}"
        )
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_threshold_summary_txt(file_path: Path, rows: List[Tuple[str, Dict[str, float]]], selected_info: Dict[str, str]) -> None:
    metrics = {label: metric for label, metric in rows}
    temporal = metrics["GlobalNorm + Temporal (original)"]
    dynamic = metrics["GlobalNorm + Temporal + Dynamic Threshold"]
    lines = [
        "Dynamic Threshold Calibration Summary",
        "",
        f"Selected threshold: {selected_info['candidate_id']} (mode={selected_info['mode']}, parameter={selected_info['parameter']}, value={selected_info['threshold']})",
        "",
        "Temporal original metrics:",
        f"AUC_source={temporal['AUC_source']:.4f}, AUC_target={temporal['AUC_target']:.4f}, AUC_total={temporal['AUC_total']:.4f}, pAUC={temporal['pAUC']:.4f}, F1_source={temporal['F1_source']:.4f}, F1_target={temporal['F1_target']:.4f}",
        "",
        "Temporal dynamic metrics:",
        f"AUC_source={dynamic['AUC_source']:.4f}, AUC_target={dynamic['AUC_target']:.4f}, AUC_total={dynamic['AUC_total']:.4f}, pAUC={dynamic['pAUC']:.4f}, F1_source={dynamic['F1_source']:.4f}, F1_target={dynamic['F1_target']:.4f}",
        "",
        "Interpretation:",
        "Threshold calibration improved temporal source-domain F1 from collapse to a non-zero value while preserving ranking metrics. Target-domain F1 remains difficult under current score separation.",
    ]
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_threshold_analysis_md(
    file_path: Path,
    rows: List[Tuple[str, Dict[str, float]]],
    selected_info: Dict[str, str],
    calibration_csv: Path,
    stats_csv: Path,
) -> None:
    table_header = "| System | AUC_source | AUC_target | AUC_total | pAUC | F1_source | F1_target |"
    table_sep = "|---|---:|---:|---:|---:|---:|---:|"
    table_rows = [
        f"| {label} | {metric['AUC_source']:.4f} | {metric['AUC_target']:.4f} | {metric['AUC_total']:.4f} | {metric['pAUC']:.4f} | {metric['F1_source']:.4f} | {metric['F1_target']:.4f} |"
        for label, metric in rows
    ]

    metrics = {label: metric for label, metric in rows}
    temporal = metrics["GlobalNorm + Temporal (original)"]
    temporal_dyn = metrics["GlobalNorm + Temporal + Dynamic Threshold"]
    globalnorm = metrics["Baseline + Global Normalization"]

    lines = [
        "# Threshold Calibration Analysis",
        "",
        "## Methods tested",
        "- Validation percentile thresholds: P90, P95, P97, P99",
        "- Statistical thresholds: mean + k*std with k=1.5, 2.0, 2.5, 3.0",
        f"- Selected calibration: {selected_info['candidate_id']} (mode={selected_info['mode']}, parameter={selected_info['parameter']}, threshold={selected_info['threshold']})",
        "",
        "## Why temporal stacking had good AUC but poor F1",
        "Temporal stacking changed the score distribution enough that ranking quality remained useful (AUC/pAUC), but a single fixed operating threshold became miscalibrated. ROC metrics use score ordering, while F1 depends on one binary cutoff.",
        "",
        "## Why threshold calibration matters",
        "Anomaly detection pipelines often optimize ranking first, then choose an operating point. If the threshold is mismatched to the score distribution, precision/recall can collapse even when AUC is acceptable.",
        "",
        "## ROC ranking vs binary classification",
        "AUC and pAUC evaluate pairwise ranking over all thresholds. Precision/Recall/F1 evaluate exactly one threshold. Therefore, it is possible to have high AUC and low F1 simultaneously.",
        "",
        "## Results summary",
        table_header,
        table_sep,
        *table_rows,
        "",
        "## Interpretation",
        f"Dynamic calibration changed temporal-system binary behavior from F1_source={temporal['F1_source']:.4f}, F1_target={temporal['F1_target']:.4f} to F1_source={temporal_dyn['F1_source']:.4f}, F1_target={temporal_dyn['F1_target']:.4f} while preserving ranking metrics near the original temporal AUC/pAUC regime.",
        f"Compared with the global-normalization baseline, dynamic temporal AUC_total={temporal_dyn['AUC_total']:.4f} versus {globalnorm['AUC_total']:.4f}.",
        "",
        "## Calibration artifacts",
        f"- Candidate metrics: {calibration_csv}",
        f"- Validation score statistics: {stats_csv}",
        "",
    ]

    file_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dynamic-threshold comparison outputs")
    parser.add_argument("--workspace", type=Path, default=Path("/workspaces/Baseline"))
    parser.add_argument("--dataset", type=str, default="DCASE2025T2HomeCamera")
    parser.add_argument("--seed", type=int, default=13711)
    parser.add_argument("--dynamic_export", type=str, default="temporal_dynamic_threshold")
    args = parser.parse_args()

    results_root = args.workspace / "results"
    eval_root = results_root / "eval_data"

    systems = [
        ("Baseline AE (MSE)", eval_root / "baseline_clean_MSE" / f"result_{args.dataset}_test_seed{args.seed}_Eval_roc.csv"),
        ("Baseline + Global Normalization", eval_root / "globalnorm_MSE" / f"result_{args.dataset}_test_seed{args.seed}_Eval_roc.csv"),
        ("GlobalNorm + Temporal (original)", eval_root / "temporal_MSE" / f"result_{args.dataset}_test_seed{args.seed}_Eval_roc.csv"),
        ("GlobalNorm + Temporal + Dynamic Threshold", eval_root / f"{args.dynamic_export}_MSE" / f"result_{args.dataset}_test_seed{args.seed}_Eval_roc.csv"),
    ]

    rows: List[Tuple[str, Dict[str, float]]] = []
    for label, path in systems:
        rows.append((label, load_result_metrics(path)))

    comparison_csv = results_root / "threshold_comparison.csv"
    write_comparison_csv(comparison_csv, rows)
    comparison_txt = results_root / "threshold_comparison.txt"
    write_comparison_txt(comparison_txt, rows)

    calibration_csv = eval_root / f"{args.dynamic_export}_MSE" / f"threshold_calibration_{args.dataset}_seed{args.seed}_Eval.csv"
    stats_csv = eval_root / f"{args.dynamic_export}_MSE" / f"validation_score_stats_{args.dataset}_seed{args.seed}_Eval.csv"
    selected_info = load_selected_threshold_info(calibration_csv)

    analysis_md = results_root / "threshold_analysis.md"
    write_threshold_analysis_md(analysis_md, rows, selected_info, calibration_csv, stats_csv)
    summary_txt = results_root / "threshold_summary.txt"
    write_threshold_summary_txt(summary_txt, rows, selected_info)

    print(f"wrote: {comparison_csv}")
    print(f"wrote: {comparison_txt}")
    print(f"wrote: {analysis_md}")
    print(f"wrote: {summary_txt}")


if __name__ == "__main__":
    main()
