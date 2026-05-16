#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from sklearn import metrics

import common as com
from networks.models import Models


@dataclass
class ScoreRecord:
    basename: str
    score: float
    label: int
    domain: str


def save_csv(path: Path, rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(rows)


def ensure_dirs(workspace: Path, export_dir: str) -> Dict[str, Path]:
    result_dir = workspace / "results" / "eval_data" / export_dir
    logs_dir = workspace / "logs" / export_dir
    models_dir = workspace / "models" / export_dir
    figures_dir = result_dir / "figures"
    for d in [result_dir, logs_dir, models_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {
        "result_dir": result_dir,
        "logs_dir": logs_dir,
        "models_dir": models_dir,
        "figures_dir": figures_dir,
    }


def load_model_weights(net, args) -> Tuple[Path, Path]:
    source_export_dir = args.source_model_export_dir
    source_model_dir = Path(f"models/saved_model/{source_export_dir}")
    model_base = f"{args.model}_{args.dataset}{net.model_name_suffix}{net.eval_suffix}_seed{args.seed}"

    best_model = source_model_dir / f"{model_base}_best.pth"
    regular_model = source_model_dir / f"{model_base}.pth"
    model_path = best_model if best_model.exists() else regular_model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found in {source_model_dir}")

    score_dist_best = source_model_dir / f"score_distr_{args.model}_{args.dataset}{net.model_name_suffix}{net.eval_suffix}_seed{args.seed}_best_mse.pickle"
    score_dist = source_model_dir / f"score_distr_{args.model}_{args.dataset}{net.model_name_suffix}{net.eval_suffix}_seed{args.seed}_mse.pickle"
    score_dist_path = score_dist_best if score_dist_best.exists() else score_dist

    net.model.load_state_dict(torch.load(model_path))
    net.model.eval()

    return model_path, score_dist_path


def score_batch(net, batch_data: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        recon, _ = net.model(batch_data)
        loss = net.loss_fn(recon, batch_data)
        return loss.reshape(loss.shape[0], -1).mean(dim=1).detach().cpu().numpy()


def collect_validation_records(net) -> List[ScoreRecord]:
    records: List[ScoreRecord] = []
    with torch.no_grad():
        for _, batch in enumerate(net.valid_loader):
            data = batch[0].to(net.device).float()
            labels = batch[1].detach().cpu().numpy().astype(int).tolist()
            names = list(batch[3])
            scores = score_batch(net, data)
            for name, score, label in zip(names, scores.tolist(), labels):
                domain = "target" if "target" in name else "source"
                records.append(ScoreRecord(name, float(score), int(label), domain))
    return records


def collect_test_records(net) -> Dict[str, List[ScoreRecord]]:
    section_records: Dict[str, List[ScoreRecord]] = {}
    with torch.no_grad():
        for idx, loader in enumerate(net.test_loader):
            section = f"section_{net.data.section_id_list[idx]}"
            recs: List[ScoreRecord] = []
            for _, batch in enumerate(loader):
                data = batch[0].to(net.device).float()
                labels = batch[1].detach().cpu().numpy().astype(int).tolist()
                names = list(batch[3])
                scores = score_batch(net, data)
                for name, score, label in zip(names, scores.tolist(), labels):
                    domain = "target" if "target" in name else "source"
                    recs.append(ScoreRecord(name, float(score), int(label), domain))
            section_records[section] = recs
    return section_records


def threshold_from_percentile(scores: np.ndarray, p: float) -> float:
    return float(np.percentile(scores, p))


def threshold_from_mean_std(scores: np.ndarray, k: float) -> float:
    return float(np.mean(scores) + k * np.std(scores))


def threshold_from_fpr(normal_scores: np.ndarray, fpr_target: float) -> float:
    return float(np.quantile(normal_scores, 1.0 - fpr_target))


def build_domain_candidates(validation_records: List[ScoreRecord], percentiles: List[float], std_ks: List[float]) -> List[Dict[str, object]]:
    source_scores = np.asarray([r.score for r in validation_records if r.domain == "source"], dtype=float)
    target_scores = np.asarray([r.score for r in validation_records if r.domain == "target"], dtype=float)

    candidates: List[Dict[str, object]] = []
    for p in percentiles:
        candidates.append({
            "strategy": "domain_separate",
            "mode": "domain_percentile",
            "param": p,
            "candidate_id": f"dom_percentile_p{p:.1f}",
            "threshold_source": threshold_from_percentile(source_scores, p),
            "threshold_target": threshold_from_percentile(target_scores, p),
        })
    for k in std_ks:
        candidates.append({
            "strategy": "domain_separate",
            "mode": "domain_mean_std",
            "param": k,
            "candidate_id": f"dom_meanstd_k{k:.2f}",
            "threshold_source": threshold_from_mean_std(source_scores, k),
            "threshold_target": threshold_from_mean_std(target_scores, k),
        })

    return candidates


def build_fpr_candidates(validation_records: List[ScoreRecord], fpr_targets: List[float]) -> List[Dict[str, object]]:
    source_normal = np.asarray([r.score for r in validation_records if r.domain == "source" and r.label == 0], dtype=float)
    target_normal = np.asarray([r.score for r in validation_records if r.domain == "target" and r.label == 0], dtype=float)

    candidates: List[Dict[str, object]] = []
    for fpr in fpr_targets:
        candidates.append({
            "strategy": "fpr_calibration",
            "mode": "domain_fpr",
            "param": fpr,
            "candidate_id": f"dom_fpr_{int(fpr * 100)}",
            "threshold_source": threshold_from_fpr(source_normal, fpr),
            "threshold_target": threshold_from_fpr(target_normal, fpr),
        })
    return candidates


def predict_binary(scores: np.ndarray, domains: List[str], thr_s: float, thr_t: float) -> np.ndarray:
    out = np.zeros_like(scores, dtype=int)
    for i, (score, dom) in enumerate(zip(scores.tolist(), domains)):
        threshold = thr_t if dom == "target" else thr_s
        out[i] = 1 if score > threshold else 0
    return out


def compute_metrics_for_section(y_true: np.ndarray, y_pred: np.ndarray, domain_list: List[str], y_bin: np.ndarray, max_fpr: float) -> Dict[str, float]:
    source_mask = np.asarray([d == "source" for d in domain_list])
    target_mask = np.asarray([d == "target" for d in domain_list])

    y_true_s_auc = y_true[source_mask | (y_true == 1)]
    y_pred_s_auc = y_pred[source_mask | (y_true == 1)]
    y_true_t_auc = y_true[target_mask | (y_true == 1)]
    y_pred_t_auc = y_pred[target_mask | (y_true == 1)]

    y_true_s = y_true[source_mask]
    y_pred_s = y_pred[source_mask]
    y_true_t = y_true[target_mask]
    y_pred_t = y_pred[target_mask]

    y_bin_s = y_bin[source_mask]
    y_bin_t = y_bin[target_mask]

    auc_s = metrics.roc_auc_score(y_true_s_auc, y_pred_s_auc)
    auc_t = metrics.roc_auc_score(y_true_t_auc, y_pred_t_auc)
    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
    p_auc_s = metrics.roc_auc_score(y_true_s, y_pred_s, max_fpr=max_fpr)
    p_auc_t = metrics.roc_auc_score(y_true_t, y_pred_t, max_fpr=max_fpr)

    tn_s, fp_s, fn_s, tp_s = metrics.confusion_matrix(y_true_s, y_bin_s).ravel()
    tn_t, fp_t, fn_t, tp_t = metrics.confusion_matrix(y_true_t, y_bin_t).ravel()

    prec_s = tp_s / np.maximum(tp_s + fp_s, np.finfo(float).eps)
    rec_s = tp_s / np.maximum(tp_s + fn_s, np.finfo(float).eps)
    f1_s = 2.0 * prec_s * rec_s / np.maximum(prec_s + rec_s, np.finfo(float).eps)

    prec_t = tp_t / np.maximum(tp_t + fp_t, np.finfo(float).eps)
    rec_t = tp_t / np.maximum(tp_t + fn_t, np.finfo(float).eps)
    f1_t = 2.0 * prec_t * rec_t / np.maximum(prec_t + rec_t, np.finfo(float).eps)

    return {
        "AUC_source": float(auc_s),
        "AUC_target": float(auc_t),
        "AUC_total": float((auc_s + auc_t) / 2.0),
        "pAUC": float(p_auc),
        "pAUC_source": float(p_auc_s),
        "pAUC_target": float(p_auc_t),
        "Precision_source": float(prec_s),
        "Precision_target": float(prec_t),
        "Recall_source": float(rec_s),
        "Recall_target": float(rec_t),
        "F1_source": float(f1_s),
        "F1_target": float(f1_t),
        "F1_mean": float((f1_s + f1_t) / 2.0),
    }


def aggregate_metrics(section_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(section_metrics[0].keys())
    values = {k: float(np.mean([m[k] for m in section_metrics])) for k in keys}
    return values


def export_candidate_results(result_dir: Path, args, section_bundle: Dict[str, Dict[str, object]], candidate: Dict[str, object], result_header: List[str]) -> Dict[str, float]:
    section_metrics: List[Dict[str, float]] = []
    csv_rows: List[List[object]] = [result_header]

    for section_name, payload in section_bundle.items():
        y_true = payload["y_true"]
        y_pred = payload["y_pred"]
        domain_list = payload["domain_list"]
        names = payload["names"]

        y_bin = predict_binary(y_pred, domain_list, float(candidate["threshold_source"]), float(candidate["threshold_target"]))
        metrics_sec = compute_metrics_for_section(y_true, y_pred, domain_list, y_bin, max_fpr=args.max_fpr)
        section_metrics.append(metrics_sec)

        section_id = section_name.split("_", 1)[1]
        csv_rows.append([
            section_id,
            metrics_sec["AUC_source"],
            metrics_sec["AUC_target"],
            metrics_sec["pAUC"],
            metrics_sec["pAUC_source"],
            metrics_sec["pAUC_target"],
            metrics_sec["Precision_source"],
            metrics_sec["Precision_target"],
            metrics_sec["Recall_source"],
            metrics_sec["Recall_target"],
            metrics_sec["F1_source"],
            metrics_sec["F1_target"],
        ])

        decision_rows = [[n, int(v)] for n, v in zip(names, y_bin.tolist())]
        suffix = str(candidate["candidate_id"]).replace(".", "p")
        decision_path = result_dir / f"decision_result_{args.dataset}_{section_name}_test_seed{args.seed}_Eval_{suffix}.csv"
        save_csv(decision_path, decision_rows)

    arr = np.asarray([[m[k] for k in section_metrics[0].keys()] for m in section_metrics], dtype=float)
    amean = np.mean(arr, axis=0)
    hmean = scipy.stats.hmean(np.maximum(arr, np.finfo(float).eps), axis=0)
    metric_keys = list(section_metrics[0].keys())

    csv_rows.append(["arithmetic mean"] + [float(v) for v in [
        amean[metric_keys.index("AUC_source")],
        amean[metric_keys.index("AUC_target")],
        amean[metric_keys.index("pAUC")],
        amean[metric_keys.index("pAUC_source")],
        amean[metric_keys.index("pAUC_target")],
        amean[metric_keys.index("Precision_source")],
        amean[metric_keys.index("Precision_target")],
        amean[metric_keys.index("Recall_source")],
        amean[metric_keys.index("Recall_target")],
        amean[metric_keys.index("F1_source")],
        amean[metric_keys.index("F1_target")],
    ]])
    csv_rows.append(["harmonic mean"] + [float(v) for v in [
        hmean[metric_keys.index("AUC_source")],
        hmean[metric_keys.index("AUC_target")],
        hmean[metric_keys.index("pAUC")],
        hmean[metric_keys.index("pAUC_source")],
        hmean[metric_keys.index("pAUC_target")],
        hmean[metric_keys.index("Precision_source")],
        hmean[metric_keys.index("Precision_target")],
        hmean[metric_keys.index("Recall_source")],
        hmean[metric_keys.index("Recall_target")],
        hmean[metric_keys.index("F1_source")],
        hmean[metric_keys.index("F1_target")],
    ]])
    csv_rows.append([])

    result_path = result_dir / f"result_{args.dataset}_test_seed{args.seed}_Eval_roc_{str(candidate['candidate_id']).replace('.', 'p')}.csv"
    save_csv(result_path, csv_rows)

    aggregate = aggregate_metrics(section_metrics)
    candidate["aggregate"] = aggregate
    candidate["result_path"] = result_path
    return aggregate


def select_candidate(candidates: List[Dict[str, object]], metric_key: str = "F1_mean") -> Dict[str, object]:
    return max(candidates, key=lambda c: float(c["aggregate"][metric_key]))


def export_strategy_summary(result_dir: Path, args, strategy_name: str, candidates: List[Dict[str, object]], selected: Dict[str, object]) -> Path:
    path = result_dir / f"threshold_strategy_{strategy_name}_{args.dataset}_seed{args.seed}_Eval.csv"
    rows = [[
        "candidate_id", "mode", "parameter", "threshold_source", "threshold_target",
        "AUC_source", "AUC_target", "AUC_total", "pAUC",
        "Precision_source", "Recall_source", "F1_source", "Precision_target", "Recall_target", "F1_target", "F1_mean", "selected"
    ]]
    for c in candidates:
        m = c["aggregate"]
        rows.append([
            c["candidate_id"], c["mode"], c["param"], c["threshold_source"], c["threshold_target"],
            m["AUC_source"], m["AUC_target"], m["AUC_total"], m["pAUC"],
            m["Precision_source"], m["Recall_source"], m["F1_source"], m["Precision_target"], m["Recall_target"], m["F1_target"], m["F1_mean"],
            int(c["candidate_id"] == selected["candidate_id"]),
        ])
    save_csv(path, rows)
    return path


def plot_and_save(fig: plt.Figure, base_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".png"), dpi=220)
    try:
        fig.savefig(base_path.with_suffix(".pdf"))
    except Exception:
        pass
    plt.close(fig)


def export_selected_figures(figures_dir: Path, strategy_name: str, selected: Dict[str, object], section_bundle: Dict[str, Dict[str, object]], temporal_gamma_threshold: float) -> None:
    all_scores = []
    all_labels = []
    all_domains = []
    all_bins = []
    all_bins_temporal = []
    for payload in section_bundle.values():
        y_pred = payload["y_pred"]
        y_true = payload["y_true"]
        domain_list = payload["domain_list"]
        y_bin = predict_binary(y_pred, domain_list, float(selected["threshold_source"]), float(selected["threshold_target"]))
        y_bin_temporal = np.asarray([1 if s > temporal_gamma_threshold else 0 for s in y_pred], dtype=int)
        all_scores.extend(y_pred.tolist())
        all_labels.extend(y_true.tolist())
        all_domains.extend(domain_list)
        all_bins.extend(y_bin.tolist())
        all_bins_temporal.extend(y_bin_temporal.tolist())

    y_scores = np.asarray(all_scores, dtype=float)
    y_true = np.asarray(all_labels, dtype=int)
    y_bin = np.asarray(all_bins, dtype=int)
    y_bin_temporal = np.asarray(all_bins_temporal, dtype=int)

    source_mask = np.asarray([d == "source" for d in all_domains])
    target_mask = ~source_mask

    fig, ax = plt.subplots(figsize=(6.5, 5))
    fpr, tpr, _ = metrics.roc_curve(y_true, y_scores)
    auc = metrics.roc_auc_score(y_true, y_scores)
    ax.plot(fpr, tpr, label=f"overall AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{strategy_name} ROC")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    plot_and_save(fig, figures_dir / f"{strategy_name}_roc")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    prec, rec, _ = metrics.precision_recall_curve(y_true, y_scores)
    ap = metrics.average_precision_score(y_true, y_scores)
    ax.plot(rec, prec, label=f"overall AP={ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{strategy_name} Precision-Recall")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    plot_and_save(fig, figures_dir / f"{strategy_name}_pr")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(y_scores[(y_true == 0) & source_mask], bins=40, alpha=0.5, density=True, label="source normal", color="tab:blue")
    ax.hist(y_scores[(y_true == 1) & source_mask], bins=40, alpha=0.5, density=True, label="source anomaly", color="tab:cyan")
    ax.hist(y_scores[(y_true == 0) & target_mask], bins=40, alpha=0.5, density=True, label="target normal", color="tab:orange")
    ax.hist(y_scores[(y_true == 1) & target_mask], bins=40, alpha=0.5, density=True, label="target anomaly", color="tab:red")
    ax.axvline(float(selected["threshold_source"]), color="tab:blue", linestyle="--", linewidth=1.8, label="threshold source")
    ax.axvline(float(selected["threshold_target"]), color="tab:orange", linestyle="--", linewidth=1.8, label="threshold target")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Density")
    ax.set_title(f"{strategy_name} Source vs Target Score Distribution")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    plot_and_save(fig, figures_dir / f"{strategy_name}_score_distribution")

    cm = metrics.confusion_matrix(y_true, y_bin)
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, str(int(cm[r, c])), ha="center", va="center")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{strategy_name} Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plot_and_save(fig, figures_dir / f"{strategy_name}_confusion")

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    labels = ["Temporal (single)", strategy_name]
    f1_temporal = metrics.f1_score(y_true, y_bin_temporal)
    f1_cal = metrics.f1_score(y_true, y_bin)
    ax.bar(labels, [f1_temporal, f1_cal], color=["gray", "tab:green"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Overall F1")
    ax.set_title("Temporal vs Calibrated Threshold Visualization")
    ax.grid(alpha=0.3, axis="y")
    plot_and_save(fig, figures_dir / f"{strategy_name}_temporal_vs_calibrated")


def export_sweep_plot(figures_dir: Path, strategy_name: str, candidates: List[Dict[str, object]]) -> None:
    x = np.arange(len(candidates))
    labels = [str(c["candidate_id"]) for c in candidates]
    f1s = [float(c["aggregate"]["F1_source"]) for c in candidates]
    f1t = [float(c["aggregate"]["F1_target"]) for c in candidates]
    f1m = [float(c["aggregate"]["F1_mean"]) for c in candidates]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, f1s, marker="o", label="F1 source")
    ax.plot(x, f1t, marker="o", label="F1 target")
    ax.plot(x, f1m, marker="o", linewidth=2.5, label="F1 mean")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("F1")
    ax.set_title(f"{strategy_name} Threshold Sweep")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    plot_and_save(fig, figures_dir / f"{strategy_name}_threshold_sweep")


def build_section_bundle(section_records: Dict[str, List[ScoreRecord]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for section, recs in section_records.items():
        out[section] = {
            "names": [r.basename for r in recs],
            "y_true": np.asarray([r.label for r in recs], dtype=int),
            "y_pred": np.asarray([r.score for r in recs], dtype=float),
            "domain_list": [r.domain for r in recs],
        }
    return out


def write_run_notes(logs_dir: Path, args, model_path: Path, score_dist_path: Path) -> None:
    txt = "\n".join([
        "Temporal Domain Threshold Calibration Run",
        f"dataset={args.dataset}",
        f"seed={args.seed}",
        f"source_model_export_dir={args.source_model_export_dir}",
        f"loaded_model={model_path}",
        f"score_distribution={score_dist_path}",
        "no_training=true",
    ])
    (logs_dir / "run_notes.txt").write_text(txt + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Domain-aware threshold calibration using an existing temporal model")
    parser.add_argument("--workspace", type=Path, default=Path("/workspaces/Baseline"))
    parser.add_argument("--dataset", type=str, default="DCASE2025T2HomeCamera")
    parser.add_argument("--seed", type=int, default=13711)
    parser.add_argument("--export_dir", type=str, default="temporal_domain_threshold")
    parser.add_argument("--source_model_export_dir", type=str, default="temporal")
    parser.add_argument("--percentiles", type=float, nargs="*", default=[90.0, 95.0, 97.0, 99.0])
    parser.add_argument("--std_ks", type=float, nargs="*", default=[1.5, 2.0, 2.5, 3.0])
    parser.add_argument("--fpr_targets", type=float, nargs="*", default=[0.05, 0.10, 0.15])
    parser.add_argument("--selection_metric", type=str, default="F1_mean", choices=["F1_mean", "F1_source", "F1_target"])
    args = parser.parse_args()

    dirs = ensure_dirs(args.workspace, args.export_dir)

    yaml_params = com.yaml_load()
    arg_parser = com.get_argparse()
    flat = com.param_to_args_list(yaml_params)
    parsed = arg_parser.parse_args(args=flat)
    parsed = arg_parser.parse_args(args=[], namespace=parsed)

    parsed.dataset = args.dataset
    parsed.seed = args.seed
    parsed.eval = True
    parsed.dev = False
    parsed.train_only = False
    parsed.test_only = True
    parsed.export_dir = args.export_dir
    parsed.use_global_norm = True
    parsed.use_temporal_stack = True
    parsed.source_model_export_dir = args.source_model_export_dir
    parsed.cuda = parsed.use_cuda and torch.cuda.is_available()

    random.seed(parsed.seed)
    np.random.seed(parsed.seed)
    torch.manual_seed(parsed.seed)
    torch.cuda.manual_seed(parsed.seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    net = Models(parsed.model).net(args=parsed, train=False, test=True)
    model_path, score_dist_path = load_model_weights(net, parsed)
    write_run_notes(dirs["logs_dir"], parsed, model_path, score_dist_path)

    gamma_threshold = float(net.calc_decision_threshold(score_distr_file_path=score_dist_path))

    val_records = collect_validation_records(net)
    test_records = collect_test_records(net)
    section_bundle = build_section_bundle(test_records)

    result_header = [
        "section", "AUC (source)", "AUC (target)", "pAUC", "pAUC (source)", "pAUC (target)",
        "precision (source)", "precision (target)", "recall (source)", "recall (target)", "F1 score (source)", "F1 score (target)"
    ]

    domain_candidates = build_domain_candidates(val_records, args.percentiles, args.std_ks)
    for c in domain_candidates:
        export_candidate_results(dirs["result_dir"], parsed, section_bundle, c, result_header)
    selected_domain = select_candidate(domain_candidates, metric_key=args.selection_metric)
    summary_domain_path = export_strategy_summary(dirs["result_dir"], parsed, "domain_separate", domain_candidates, selected_domain)

    fpr_candidates = build_fpr_candidates(val_records, args.fpr_targets)
    for c in fpr_candidates:
        export_candidate_results(dirs["result_dir"], parsed, section_bundle, c, result_header)
    selected_fpr = select_candidate(fpr_candidates, metric_key=args.selection_metric)
    summary_fpr_path = export_strategy_summary(dirs["result_dir"], parsed, "fpr_calibration", fpr_candidates, selected_fpr)

    # Canonical selected result files per strategy.
    save_csv(
        dirs["result_dir"] / f"selected_strategy_summary_{parsed.dataset}_seed{parsed.seed}_Eval.csv",
        [
            ["strategy", "candidate_id", "mode", "parameter", "threshold_source", "threshold_target", "selection_metric", "selection_value"],
            ["domain_separate", selected_domain["candidate_id"], selected_domain["mode"], selected_domain["param"], selected_domain["threshold_source"], selected_domain["threshold_target"], args.selection_metric, selected_domain["aggregate"][args.selection_metric]],
            ["fpr_calibration", selected_fpr["candidate_id"], selected_fpr["mode"], selected_fpr["param"], selected_fpr["threshold_source"], selected_fpr["threshold_target"], args.selection_metric, selected_fpr["aggregate"][args.selection_metric]],
        ],
    )

    export_sweep_plot(dirs["figures_dir"], "domain_separate", domain_candidates)
    export_sweep_plot(dirs["figures_dir"], "fpr_calibration", fpr_candidates)
    export_selected_figures(dirs["figures_dir"], "domain_separate", selected_domain, section_bundle, gamma_threshold)
    export_selected_figures(dirs["figures_dir"], "fpr_calibration", selected_fpr, section_bundle, gamma_threshold)

    # Store exact strategy-best result tables for downstream summary scripts.
    best_domain_result = Path(selected_domain["result_path"])
    best_fpr_result = Path(selected_fpr["result_path"])
    (dirs["result_dir"] / f"result_{parsed.dataset}_test_seed{parsed.seed}_Eval_roc_domain_separate.csv").write_text(best_domain_result.read_text(encoding="utf-8"), encoding="utf-8")
    (dirs["result_dir"] / f"result_{parsed.dataset}_test_seed{parsed.seed}_Eval_roc_fpr_calibration.csv").write_text(best_fpr_result.read_text(encoding="utf-8"), encoding="utf-8")

    # Human-readable notes.
    notes = [
        "Domain-Aware Threshold Calibration",
        f"summary_domain_csv={summary_domain_path}",
        f"summary_fpr_csv={summary_fpr_path}",
        f"selected_domain={selected_domain['candidate_id']}",
        f"selected_fpr={selected_fpr['candidate_id']}",
        f"temporal_gamma_threshold={gamma_threshold}",
    ]
    (dirs["models_dir"] / "README.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")

    print(f"Saved domain-aware calibration outputs in: {dirs['result_dir']}")
    print(f"Saved strategy figures in: {dirs['figures_dir']}")


if __name__ == "__main__":
    main()
