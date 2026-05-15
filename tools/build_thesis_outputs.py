#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SystemArtifacts:
    label: str
    export_dir: str
    result_csv: Path
    log_csv: Path
    result_dir: Path


def read_csv_rows(file_path: Path) -> List[List[str]]:
    with file_path.open(newline="") as handle:
        return list(csv.reader(handle))


def load_result_metrics(result_csv: Path) -> Dict[str, float]:
    rows = read_csv_rows(result_csv)
    if len(rows) < 2:
        raise ValueError(f"Result CSV has no data rows: {result_csv}")

    header = rows[0]
    data_row = None
    for row in rows[1:]:
        if row and row[0] == "arithmetic mean":
            data_row = row
            break
    if data_row is None:
        data_row = rows[1]

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


def load_convergence_stats(log_csv: Path) -> Dict[str, float]:
    rows = read_csv_rows(log_csv)
    if len(rows) < 2:
        return {
            "epochs_ran": 0,
            "best_epoch": 0,
            "best_val_loss": float("nan"),
            "final_train_loss": float("nan"),
            "final_val_loss": float("nan"),
        }

    header = rows[0]
    idx_loss = header.index("loss")
    idx_val_loss = header.index("val_loss")
    data = [[float(value) for value in row] for row in rows[1:] if row]
    val_losses = [row[idx_val_loss] for row in data]
    best_idx = min(range(len(val_losses)), key=lambda idx: val_losses[idx])

    return {
        "epochs_ran": len(data),
        "best_epoch": best_idx + 1,
        "best_val_loss": val_losses[best_idx],
        "final_train_loss": data[-1][idx_loss],
        "final_val_loss": data[-1][idx_val_loss],
    }


def copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return True


def format_metrics_text(label: str, metrics: Dict[str, float], convergence: Dict[str, float]) -> str:
    return "\n".join(
        [
            label,
            f"AUC_source: {metrics['AUC_source']:.4f}",
            f"AUC_target: {metrics['AUC_target']:.4f}",
            f"AUC_total: {metrics['AUC_total']:.4f}",
            f"pAUC: {metrics['pAUC']:.4f}",
            f"Precision_source: {metrics['Precision_source']:.4f}",
            f"Recall_source: {metrics['Recall_source']:.4f}",
            f"F1_source: {metrics['F1_source']:.4f}",
            f"Precision_target: {metrics['Precision_target']:.4f}",
            f"Recall_target: {metrics['Recall_target']:.4f}",
            f"F1_target: {metrics['F1_target']:.4f}",
            f"Epochs_ran: {int(convergence['epochs_ran'])}",
            f"Best_epoch: {int(convergence['best_epoch'])}",
            f"Best_val_loss: {convergence['best_val_loss']:.6f}",
            f"Final_train_loss: {convergence['final_train_loss']:.6f}",
            f"Final_val_loss: {convergence['final_val_loss']:.6f}",
        ]
    )


def build_comparison_table(system_rows: List[Tuple[str, Dict[str, float]]]) -> str:
    header = (
        " System                      | AUC_src | AUC_tgt | AUC_tot | pAUC  | Prec_s | Rec_s | F1_s   | Prec_t | Rec_t | F1_t"
    )
    separator = "-----------------------------+---------+---------+---------+-------+--------+-------+--------+--------+-------+-------"
    lines = [
        "================================================================",
        " HomeCamera Section 00 - Final System Comparison",
        "================================================================",
        header,
        separator,
    ]

    for label, metrics in system_rows:
        lines.append(
            f" {label:<27} | {metrics['AUC_source']:7.4f} | {metrics['AUC_target']:7.4f} | {metrics['AUC_total']:7.4f} | {metrics['pAUC']:5.4f} | {metrics['Precision_source']:6.4f} | {metrics['Recall_source']:5.4f} | {metrics['F1_source']:6.4f} | {metrics['Precision_target']:6.4f} | {metrics['Recall_target']:5.4f} | {metrics['F1_target']:6.4f}"
        )

    best_overall = max(system_rows, key=lambda item: item[1]["AUC_total"])
    best_target = max(system_rows, key=lambda item: item[1]["AUC_target"])
    best_pauc = max(system_rows, key=lambda item: item[1]["pAUC"])

    lines.extend(
        [
            "================================================================",
            f" Best overall system: {best_overall[0]}",
            f" Best target-domain system: {best_target[0]}",
            f" Best pAUC system: {best_pauc[0]}",
            "================================================================",
        ]
    )
    return "\n".join(lines)


def build_analysis_note(system_rows: List[Tuple[str, Dict[str, float]]], convergence_rows: List[Tuple[str, Dict[str, float]]]) -> str:
    metrics = {label: row for label, row in system_rows}
    baseline = metrics["Baseline AE (MSE)"]
    globalnorm = metrics["Baseline + Global Norm"]
    temporal = metrics["Global Norm + Temporal"]

    return "\n".join(
        [
            "Home Camera ASD - Thesis Analysis Note",
            "",
            "1. Global normalization",
            f"The clean baseline reached AUC_total {baseline['AUC_total']:.4f}, while global normalization improved it to {globalnorm['AUC_total']:.4f}. The gain is concentrated on the source domain, where AUC_source increased from {baseline['AUC_source']:.4f} to {globalnorm['AUC_source']:.4f}, and on thresholded detection, where both source and target F1 became non-zero.",
            "This is consistent with reduced feature-scale mismatch after normalizing the training distribution. Reconstruction error became better conditioned, and the model no longer had to spend capacity compensating for per-file amplitude shifts.",
            "",
            "2. Temporal context stacking",
            f"Adding a 5-frame context lifted AUC_target to {temporal['AUC_target']:.4f}, but AUC_source dropped to {temporal['AUC_source']:.4f} and the source-domain F1 collapsed to {temporal['F1_source']:.4f}. The model captured more local temporal structure, but the higher-dimensional input also made optimization harder and reduced the stability of the anomaly threshold.",
            "In this setting, temporal stacking improved separability on the target side in ROC space, but that benefit did not translate into reliable binary decisions under the same thresholding rule.",
            "",
            "3. Source vs. target behavior",
            f"The baseline was stronger on source than target AUC ({baseline['AUC_source']:.4f} vs {baseline['AUC_target']:.4f}), which indicates the expected domain gap. Global normalization narrowed that gap by improving both domains, whereas temporal stacking shifted the balance toward target ranking but weakened source precision and recall.",
            "",
            "4. Domain shift and stability",
            "The results suggest that the dominant issue is not model capacity alone but feature-space alignment across domains. Global normalization improves alignment directly. Temporal stacking changes the input representation more aggressively and therefore increases sensitivity to domain-specific temporal patterns and decision-threshold calibration.",
            "",
            "5. Simplicity vs. performance",
            "The simple fully connected autoencoder remains the safest thesis baseline because it is easy to reproduce and compare. Within that constraint, global normalization is the most reliable improvement. Temporal stacking is scientifically interesting, but it introduces a stronger optimization burden and a less stable operating point.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build thesis-ready tables and summaries from Home Camera ASD results")
    parser.add_argument("--workspace", type=Path, default=Path("/workspaces/Baseline"))
    parser.add_argument("--dataset", type=str, default="DCASE2025T2HomeCamera")
    parser.add_argument("--seed", type=int, default=13711)
    args = parser.parse_args()

    system_specs = [
        ("Baseline AE (MSE)", "baseline_clean"),
        ("Baseline + Global Norm", "globalnorm"),
        ("Global Norm + Temporal", "temporal"),
    ]

    system_rows: List[Tuple[str, Dict[str, float]]] = []
    convergence_rows: List[Tuple[str, Dict[str, float]]] = []

    for label, export_dir in system_specs:
        result_dir = args.workspace / "results" / "eval_data" / f"{export_dir}_MSE"
        result_csv = result_dir / f"result_{args.dataset}_test_seed{args.seed}_Eval_roc.csv"
        log_csv = args.workspace / "logs" / export_dir / f"DCASE2023T2-AE_{args.dataset}_Eval_seed{args.seed}" / "log.csv"

        metrics = load_result_metrics(result_csv)
        convergence = load_convergence_stats(log_csv)
        system_rows.append((label, metrics))
        convergence_rows.append((label, convergence))

        metrics_text = format_metrics_text(label, metrics, convergence)
        metrics_file = args.workspace / "results" / f"{export_dir}_metrics.txt"
        metrics_file.write_text(metrics_text + "\n", encoding="utf-8")

        copy_if_exists(result_csv, args.workspace / "results" / f"{export_dir}_results.csv")
        copy_if_exists(result_dir / f"{export_dir}_loss.png", args.workspace / "results" / f"{export_dir}_loss.png")
        copy_if_exists(result_dir / f"{export_dir}_roc.png", args.workspace / "results" / f"{export_dir}_roc.png")
        copy_if_exists(result_dir / f"{export_dir}_score_hist.png", args.workspace / "results" / f"{export_dir}_score_hist.png")
        copy_if_exists(result_dir / f"{export_dir}_confusion_matrix.png", args.workspace / "results" / f"{export_dir}_confusion_matrix.png")

    comparison_csv = args.workspace / "results" / "final_comparison.csv"
    with comparison_csv.open("w", newline="") as handle:
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
        for label, metrics in system_rows:
            writer.writerow([
                label,
                f"{metrics['AUC_source']:.4f}",
                f"{metrics['AUC_target']:.4f}",
                f"{metrics['AUC_total']:.4f}",
                f"{metrics['pAUC']:.4f}",
                f"{metrics['Precision_source']:.4f}",
                f"{metrics['Recall_source']:.4f}",
                f"{metrics['F1_source']:.4f}",
                f"{metrics['Precision_target']:.4f}",
                f"{metrics['Recall_target']:.4f}",
                f"{metrics['F1_target']:.4f}",
            ])

    comparison_txt = args.workspace / "results" / "final_comparison.txt"
    comparison_txt.write_text(build_comparison_table(system_rows) + "\n", encoding="utf-8")

    analysis_md = args.workspace / "results" / "analysis.md"
    analysis_md.write_text(build_analysis_note(system_rows, convergence_rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()