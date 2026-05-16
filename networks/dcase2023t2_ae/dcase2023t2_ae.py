import os
import sys
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import scipy
from sklearn import metrics
import csv
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from networks.base_model import BaseModel
from networks.dcase2023t2_ae.network import AENet
from tools.plot_anm_score import AnmScoreFigData
from tools.plot_loss_curve import csv_to_figdata


class DCASE2023T2AE(BaseModel):
    def __init__(self, args, train, test):
        super().__init__(args=args, train=train, test=test)
        parameter_list = [{"params": self.model.parameters()}]
        self.optimizer = optim.Adam(parameter_list, lr=self.args.learning_rate)
        self.mse_score_distr_file_path = self.model_dir / f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_mse.pickle"
        self.best_mse_score_distr_file_path = self.model_dir / f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_best_mse.pickle"
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def init_model(self):
        return AENet(input_dim=self.data.input_dim)

    def get_log_header(self):
        self.column_heading_list = [
            ["loss"],
            ["val_loss"],
            ["recon_loss"],
            ["recon_loss_source", "recon_loss_target"],
        ]
        return "loss,val_loss,recon_loss,recon_loss_source,recon_loss_target"

    def train(self, epoch):
        if epoch <= self.epoch:
            return {"epoch": epoch, "should_stop": False, "is_best": False, "val_loss": None}

        train_loss = 0
        train_recon_loss = 0
        train_recon_loss_source = 0
        train_recon_loss_target = 0
        y_pred = []

        self.model.train()
        train_loader = self.train_loader

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            data = batch[0].to(self.device).float()
            if data.shape[0] <= 1:
                continue

            data_name_list = batch[3]
            is_target_list = ["target" in data_name for data_name in data_name_list]
            is_source_list = np.logical_not(is_target_list).tolist()
            n_source = is_source_list.count(True)
            n_target = is_target_list.count(True)

            self.optimizer.zero_grad()
            recon_batch, _ = self.model(data)

            score_2d = self.loss_fn(recon_batch, data)
            n_loss = len(score_2d)
            score = self.loss_reduction_1d(score=score_2d)

            recon_loss = self.loss_reduction(score=score, n_loss=n_loss)
            recon_loss_source = self.loss_reduction(score=score[is_source_list], n_loss=n_source)
            recon_loss_target = self.loss_reduction(score=score[is_target_list], n_loss=n_target) if n_target > 0 else 0

            self.loss = recon_loss
            self.loss.backward()
            self.optimizer.step()

            train_loss += float(self.loss)
            train_recon_loss += float(recon_loss)
            train_recon_loss_source += float(recon_loss_source)
            train_recon_loss_target += float(recon_loss_target)
            y_pred.append(self.loss.item())

            if batch_idx % self.args.log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        self.loss.item(),
                    )
                )

        val_loss = 0
        with torch.no_grad():
            self.model.eval()
            for _, batch in enumerate(self.valid_loader):
                data = batch[0].to(self.device).float()
                recon_batch, _ = self.model(data)
                score = self.loss_fn(recon_batch, data)
                loss = score.mean()
                val_loss += float(loss)
                y_pred.append(loss.item())

        print(
            "====> Epoch: {} Average loss: {:.4f} Validation loss: {:.4f}".format(
                epoch,
                train_loss / len(train_loader),
                val_loss / len(self.valid_loader),
            )
        )

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(self.valid_loader)
        is_best = avg_val_loss < (self.best_val_loss - self.args.min_delta)
        if is_best:
            self.best_val_loss = avg_val_loss
            self.best_epoch = epoch

        with open(self.log_path, "a") as log:
            np.savetxt(
                log,
                [
                    "{0},{1},{2},{3},{4}".format(
                        avg_train_loss,
                        avg_val_loss,
                        train_recon_loss / len(train_loader),
                        train_recon_loss_source / len(train_loader),
                        train_recon_loss_target / len(train_loader),
                    )
                ],
                fmt="%s",
            )

        csv_to_figdata(
            file_path=self.log_path,
            column_heading_list=self.column_heading_list,
            ylabel="loss",
            fig_count=len(self.column_heading_list),
            cut_first_epoch=True,
        )

        if self.loss_curve_path.exists():
            shutil.copyfile(self.loss_curve_path, self.result_dir / f"{self.export_dir}_loss.png")
            shutil.copyfile(self.loss_curve_path, self.eval_data_result_dir / f"{self.export_dir}_loss.png")

        self.fit_anomaly_score_distribution(
            y_pred=y_pred,
            score_distr_file_path=self.mse_score_distr_file_path,
        )
        if is_best:
            shutil.copyfile(self.mse_score_distr_file_path, self.best_mse_score_distr_file_path)

        torch.save(self.model.state_dict(), self.model_path)
        if is_best:
            torch.save(self.model.state_dict(), self.best_model_path)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
            },
            self.checkpoint_path,
        )
        if is_best:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": self.loss,
                },
                self.model_dir / f"checkpoint_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_best.tar",
            )

        return {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "is_best": is_best,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "should_stop": False,
        }

    def loss_reduction_1d(self, score):
        return torch.mean(score, dim=1)

    def loss_reduction(self, score, n_loss):
        return torch.sum(score) / n_loss

    def loss_fn(self, recon_x, x):
        return F.mse_loss(recon_x, x.view(recon_x.shape), reduction="none")

    def _resolve_model_paths(self):
        source_export_dir = self.args.source_model_export_dir
        if source_export_dir is None:
            source_export_dir = self.export_dir

        source_model_dir = Path(f"models/saved_model/{source_export_dir}/")
        source_model_path = source_model_dir / f"{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}.pth"
        source_best_model_path = source_model_dir / f"{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_best.pth"
        source_score_distr = source_model_dir / f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_mse.pickle"
        source_best_score_distr = source_model_dir / f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_best_mse.pickle"
        return source_model_path, source_best_model_path, source_score_distr, source_best_score_distr

    def _collect_validation_scores(self):
        scores = []
        with torch.no_grad():
            self.model.eval()
            for _, batch in enumerate(self.valid_loader):
                data = batch[0].to(self.device).float()
                recon_batch, _ = self.model(data)
                score = self.loss_fn(recon_batch, data)
                per_sample = score.reshape(score.shape[0], -1).mean(dim=1)
                scores.extend(per_sample.detach().cpu().numpy().tolist())
        return np.asarray(scores, dtype=float)

    def _build_threshold_candidates(self, validation_scores, score_distr_file_path):
        candidates = []
        mode = self.args.threshold_mode
        if mode == "gamma":
            candidates.append({
                "candidate_id": "gamma_q{:.3f}".format(self.args.decision_threshold),
                "mode": "gamma",
                "param": self.args.decision_threshold,
                "threshold": float(self.calc_decision_threshold(score_distr_file_path=score_distr_file_path)),
            })
        elif mode == "validation_percentile":
            candidates.append({
                "candidate_id": "percentile_p{:.1f}".format(self.args.percentile_value),
                "mode": "validation_percentile",
                "param": self.args.percentile_value,
                "threshold": float(np.percentile(validation_scores, self.args.percentile_value)),
            })
        elif mode == "mean_std":
            mean_v = float(np.mean(validation_scores))
            std_v = float(np.std(validation_scores))
            k = self.args.std_multiplier
            candidates.append({
                "candidate_id": "meanstd_k{:.2f}".format(k),
                "mode": "mean_std",
                "param": k,
                "threshold": float(mean_v + k * std_v),
            })
        else:
            for p in self.args.threshold_percentile_candidates:
                candidates.append({
                    "candidate_id": "percentile_p{:.1f}".format(p),
                    "mode": "validation_percentile",
                    "param": float(p),
                    "threshold": float(np.percentile(validation_scores, p)),
                })

            mean_v = float(np.mean(validation_scores))
            std_v = float(np.std(validation_scores))
            for k in self.args.threshold_std_candidates:
                candidates.append({
                    "candidate_id": "meanstd_k{:.2f}".format(k),
                    "mode": "mean_std",
                    "param": float(k),
                    "threshold": float(mean_v + k * std_v),
                })

        return candidates

    def _compute_section_metrics(self, y_true, y_pred, domain_list, decision_threshold):
        y_true_s_auc = [y_true[i] for i in range(len(y_true)) if domain_list[i] == "source" or y_true[i] == 1]
        y_pred_s_auc = [y_pred[i] for i in range(len(y_true)) if domain_list[i] == "source" or y_true[i] == 1]
        y_true_t_auc = [y_true[i] for i in range(len(y_true)) if domain_list[i] == "target" or y_true[i] == 1]
        y_pred_t_auc = [y_pred[i] for i in range(len(y_true)) if domain_list[i] == "target" or y_true[i] == 1]

        y_true_s = [y_true[i] for i in range(len(y_true)) if domain_list[i] == "source"]
        y_pred_s = [y_pred[i] for i in range(len(y_true)) if domain_list[i] == "source"]
        y_true_t = [y_true[i] for i in range(len(y_true)) if domain_list[i] == "target"]
        y_pred_t = [y_pred[i] for i in range(len(y_true)) if domain_list[i] == "target"]

        auc_s = metrics.roc_auc_score(y_true_s_auc, y_pred_s_auc)
        p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=self.args.max_fpr)
        p_auc_s = metrics.roc_auc_score(y_true_s, y_pred_s, max_fpr=self.args.max_fpr)
        tn_s, fp_s, fn_s, tp_s = metrics.confusion_matrix(
            y_true_s, [1 if x > decision_threshold else 0 for x in y_pred_s]
        ).ravel()
        prec_s = tp_s / np.maximum(tp_s + fp_s, sys.float_info.epsilon)
        recall_s = tp_s / np.maximum(tp_s + fn_s, sys.float_info.epsilon)
        f1_s = 2.0 * prec_s * recall_s / np.maximum(prec_s + recall_s, sys.float_info.epsilon)

        auc_t = metrics.roc_auc_score(y_true_t_auc, y_pred_t_auc)
        p_auc_t = metrics.roc_auc_score(y_true_t, y_pred_t, max_fpr=self.args.max_fpr)
        tn_t, fp_t, fn_t, tp_t = metrics.confusion_matrix(
            y_true_t, [1 if x > decision_threshold else 0 for x in y_pred_t]
        ).ravel()
        prec_t = tp_t / np.maximum(tp_t + fp_t, sys.float_info.epsilon)
        recall_t = tp_t / np.maximum(tp_t + fn_t, sys.float_info.epsilon)
        f1_t = 2.0 * prec_t * recall_t / np.maximum(prec_t + recall_t, sys.float_info.epsilon)

        return {
            "auc_s": float(auc_s),
            "auc_t": float(auc_t),
            "p_auc": float(p_auc),
            "p_auc_s": float(p_auc_s),
            "p_auc_t": float(p_auc_t),
            "prec_s": float(prec_s),
            "prec_t": float(prec_t),
            "recall_s": float(recall_s),
            "recall_t": float(recall_t),
            "f1_s": float(f1_s),
            "f1_t": float(f1_t),
            "auc_total": float((auc_s + auc_t) / 2.0),
            "f1_mean": float((f1_s + f1_t) / 2.0),
        }

    def _save_threshold_diagnostics(self, result_dir, validation_scores, candidates, selected_candidate):
        summary_csv = result_dir / f"threshold_calibration_{self.args.dataset}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv"
        summary_rows = [[
            "candidate_id", "mode", "parameter", "threshold", "AUC_source", "AUC_target", "AUC_total", "pAUC",
            "Precision_source", "Recall_source", "F1_source", "Precision_target", "Recall_target", "F1_target", "f1_mean", "selected"
        ]]
        for candidate in candidates:
            m = candidate["aggregate"]
            summary_rows.append([
                candidate["candidate_id"],
                candidate["mode"],
                candidate["param"],
                candidate["threshold"],
                m["AUC_source"],
                m["AUC_target"],
                m["AUC_total"],
                m["pAUC"],
                m["Precision_source"],
                m["Recall_source"],
                m["F1_source"],
                m["Precision_target"],
                m["Recall_target"],
                m["F1_target"],
                m["f1_mean"],
                int(candidate["candidate_id"] == selected_candidate["candidate_id"]),
            ])
        save_csv(summary_csv, summary_rows)

        stats_csv = result_dir / f"validation_score_stats_{self.args.dataset}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv"
        stats_rows = [
            ["count", "mean", "std", "min", "max", "p90", "p95", "p97", "p99", "selected_mode", "selected_parameter", "selected_threshold"],
            [
                int(validation_scores.size),
                float(np.mean(validation_scores)),
                float(np.std(validation_scores)),
                float(np.min(validation_scores)),
                float(np.max(validation_scores)),
                float(np.percentile(validation_scores, 90)),
                float(np.percentile(validation_scores, 95)),
                float(np.percentile(validation_scores, 97)),
                float(np.percentile(validation_scores, 99)),
                selected_candidate["mode"],
                selected_candidate["param"],
                selected_candidate["threshold"],
            ],
        ]
        save_csv(stats_csv, stats_rows)

        fig_path = result_dir / f"{self.export_dir}_validation_score_distribution.png"
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(validation_scores, bins=60, alpha=0.75, color="tab:blue", density=True)
        for candidate in candidates:
            linewidth = 2.5 if candidate["candidate_id"] == selected_candidate["candidate_id"] else 1.2
            linestyle = "-" if candidate["candidate_id"] == selected_candidate["candidate_id"] else "--"
            ax.axvline(candidate["threshold"], linestyle=linestyle, linewidth=linewidth, label=candidate["candidate_id"])
        ax.set_title("Validation Score Distribution with Candidate Thresholds")
        ax.set_xlabel("Reconstruction score")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)

        sweep_fig_path = result_dir / f"{self.export_dir}_threshold_sweep.png"
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(candidates))
        f1_source = [c["aggregate"]["F1_source"] for c in candidates]
        f1_target = [c["aggregate"]["F1_target"] for c in candidates]
        f1_mean = [c["aggregate"]["f1_mean"] for c in candidates]
        ax.plot(x, f1_source, marker="o", label="F1 source")
        ax.plot(x, f1_target, marker="o", label="F1 target")
        ax.plot(x, f1_mean, marker="o", linewidth=2, label="F1 mean")
        selected_index = [i for i, c in enumerate(candidates) if c["candidate_id"] == selected_candidate["candidate_id"]][0]
        ax.scatter([selected_index], [f1_mean[selected_index]], color="red", s=80, label="Selected")
        ax.set_xticks(x)
        ax.set_xticklabels([c["candidate_id"] for c in candidates], rotation=30, ha="right")
        ax.set_ylabel("F1")
        ax.set_title("Threshold Calibration Sweep")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(sweep_fig_path, dpi=200)
        plt.close(fig)

    def test(self):
        anm_score_figdata = AnmScoreFigData()
        mode = self.data.mode
        if not mode:
            print("Dynamic threshold calibration is intended for development mode with labels.")
            return

        print("============== MODEL LOAD ==============")
        source_model_path, source_best_model_path, source_score_distr, source_best_score_distr = self._resolve_model_paths()
        model_path = source_best_model_path if os.path.exists(source_best_model_path) else source_model_path
        if not os.path.exists(model_path):
            print(f"model not found -> {model_path} ")
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        score_distr_file_path = source_best_score_distr if os.path.exists(source_best_score_distr) else source_score_distr
        validation_scores = self._collect_validation_scores()
        candidates = self._build_threshold_candidates(validation_scores=validation_scores, score_distr_file_path=score_distr_file_path)

        dir_name = "test"
        section_records = []
        for idx, test_loader_tmp in enumerate(self.test_loader):
            section_name = f"section_{self.data.section_id_list[idx]}"
            result_dir = self.result_dir if self.args.dev else self.eval_data_result_dir

            anomaly_score_csv = result_dir / f"anomaly_score_{self.args.dataset}_{section_name}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv"
            anomaly_score_list = []
            domain_list = []

            print("\n============== BEGIN TEST FOR A SECTION ==============")
            y_pred = []
            y_true = []
            basename_list = []

            with torch.no_grad():
                y_pred, anomaly_score_list, _, domain_list = self.eval(
                    test_loader=test_loader_tmp,
                    y_pred=y_pred,
                    anomaly_score_list=anomaly_score_list,
                    decision_result_list=[],
                    domain_list=domain_list,
                    y_true=y_true,
                    decision_threshold=np.inf,
                    mode=mode,
                )
                basename_list = [row[0] for row in anomaly_score_list]

            save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
            print(f"anomaly score result ->  {anomaly_score_csv}")

            section_records.append({
                "section_name": section_name,
                "y_true": np.array(y_true, dtype=int),
                "y_pred": np.array(y_pred, dtype=float),
                "domain_list": list(domain_list),
                "basename_list": basename_list,
            })

            print("\n============ END OF TEST FOR A SECTION ============")

        selection_key = {
            "f1_mean": "f1_mean",
            "f1_source": "F1_source",
            "f1_target": "F1_target",
        }[self.args.threshold_selection_metric]

        for candidate in candidates:
            performance = []
            csv_lines = [self.result_column_dict["source_target"]]

            for record in section_records:
                metrics_sec = self._compute_section_metrics(
                    y_true=record["y_true"],
                    y_pred=record["y_pred"],
                    domain_list=record["domain_list"],
                    decision_threshold=candidate["threshold"],
                )

                section_id = record["section_name"].split("_", 1)[1]
                csv_lines.append([
                    section_id,
                    metrics_sec["auc_s"],
                    metrics_sec["auc_t"],
                    metrics_sec["p_auc"],
                    metrics_sec["p_auc_s"],
                    metrics_sec["p_auc_t"],
                    metrics_sec["prec_s"],
                    metrics_sec["prec_t"],
                    metrics_sec["recall_s"],
                    metrics_sec["recall_t"],
                    metrics_sec["f1_s"],
                    metrics_sec["f1_t"],
                ])
                performance.append([
                    metrics_sec["auc_s"],
                    metrics_sec["auc_t"],
                    metrics_sec["p_auc"],
                    metrics_sec["p_auc_s"],
                    metrics_sec["p_auc_t"],
                    metrics_sec["prec_s"],
                    metrics_sec["prec_t"],
                    metrics_sec["recall_s"],
                    metrics_sec["recall_t"],
                    metrics_sec["f1_s"],
                    metrics_sec["f1_t"],
                ])

                decision_rows = [[name, 1 if score > candidate["threshold"] else 0] for name, score in zip(record["basename_list"], record["y_pred"])]
                decision_suffix = candidate["candidate_id"].replace(".", "p")
                decision_result_csv = result_dir / f"decision_result_{self.args.dataset}_{record['section_name']}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}_{decision_suffix}.csv"
                save_csv(save_file_path=decision_result_csv, save_data=decision_rows)

            perf_array = np.array(performance, dtype=float)
            amean_performance = np.mean(perf_array, axis=0)
            hmean_performance = scipy.stats.hmean(np.maximum(perf_array, sys.float_info.epsilon), axis=0)
            csv_lines.append(["arithmetic mean"] + list(amean_performance))
            csv_lines.append(["harmonic mean"] + list(hmean_performance))
            csv_lines.append([])

            candidate["aggregate"] = {
                "AUC_source": float(amean_performance[0]),
                "AUC_target": float(amean_performance[1]),
                "AUC_total": float((amean_performance[0] + amean_performance[1]) / 2.0),
                "pAUC": float(amean_performance[2]),
                "Precision_source": float(amean_performance[5]),
                "Precision_target": float(amean_performance[6]),
                "Recall_source": float(amean_performance[7]),
                "Recall_target": float(amean_performance[8]),
                "F1_source": float(amean_performance[9]),
                "F1_target": float(amean_performance[10]),
                "f1_mean": float((amean_performance[9] + amean_performance[10]) / 2.0),
            }

            candidate_result_path = result_dir / f"result_{self.args.dataset}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}_roc_{candidate['candidate_id'].replace('.', 'p')}.csv"
            save_csv(save_file_path=candidate_result_path, save_data=csv_lines)
            print(f"calibration result -> {candidate_result_path}")

        selected_candidate = max(candidates, key=lambda c: c["aggregate"][selection_key])

        print("Selected threshold calibration:")
        print(f"  candidate={selected_candidate['candidate_id']}")
        print(f"  mode={selected_candidate['mode']}")
        print(f"  parameter={selected_candidate['param']}")
        print(f"  threshold={selected_candidate['threshold']:.6f}")

        best_result_path = result_dir / f"result_{self.args.dataset}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}_roc_{selected_candidate['candidate_id'].replace('.', 'p')}.csv"
        result_path = result_dir / f"result_{self.args.dataset}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}_roc.csv"
        shutil.copyfile(best_result_path, result_path)
        print(f"results -> {result_path}")

        # Export default decision files for the selected calibration so existing tools can read canonical file names.
        for record in section_records:
            decision_rows = [[name, 1 if score > selected_candidate["threshold"] else 0] for name, score in zip(record["basename_list"], record["y_pred"])]
            decision_result_csv = result_dir / f"decision_result_{self.args.dataset}_{record['section_name']}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv"
            save_csv(save_file_path=decision_result_csv, save_data=decision_rows)

            self.export_evaluation_figures(
                result_dir=result_dir,
                section_name=record["section_name"],
                y_true=np.array(record["y_true"], dtype=int),
                y_pred=np.array(record["y_pred"], dtype=float),
                domain_list=record["domain_list"],
                decision_threshold=selected_candidate["threshold"],
            )

            y_true_s = [record["y_true"][i] for i in range(len(record["y_true"])) if record["domain_list"][i] == "source"]
            y_pred_s = [record["y_pred"][i] for i in range(len(record["y_true"])) if record["domain_list"][i] == "source"]
            y_true_t = [record["y_true"][i] for i in range(len(record["y_true"])) if record["domain_list"][i] == "target"]
            y_pred_t = [record["y_pred"][i] for i in range(len(record["y_true"])) if record["domain_list"][i] == "target"]
            if len(y_true_s) > 0:
                auc_s = metrics.roc_auc_score([record["y_true"][i] for i in range(len(record["y_true"])) if record["domain_list"][i] == "source" or record["y_true"][i] == 1],
                                              [record["y_pred"][i] for i in range(len(record["y_true"])) if record["domain_list"][i] == "source" or record["y_true"][i] == 1])
                anm_score_figdata.append_figdata(
                    anm_score_figdata.anm_score_to_figdata(
                        scores=[[t, p] for t, p in zip(y_true_s, y_pred_s)],
                        title=f"{record['section_name']}_source_AUC{auc_s}",
                    )
                )
            if len(y_true_t) > 0:
                auc_t = metrics.roc_auc_score([record["y_true"][i] for i in range(len(record["y_true"])) if record["domain_list"][i] == "target" or record["y_true"][i] == 1],
                                              [record["y_pred"][i] for i in range(len(record["y_true"])) if record["domain_list"][i] == "target" or record["y_true"][i] == 1])
                anm_score_figdata.append_figdata(
                    anm_score_figdata.anm_score_to_figdata(
                        scores=[[t, p] for t, p in zip(y_true_t, y_pred_t)],
                        title=f"{record['section_name']}_target_AUC{auc_t}",
                    )
                )

        anm_score_figdata.show_fig(
            title=self.args.model + "_" + self.args.dataset + self.model_name_suffix + self.eval_suffix + "_anm_score",
            export_dir=result_dir,
        )

        self._save_threshold_diagnostics(
            result_dir=result_dir,
            validation_scores=validation_scores,
            candidates=candidates,
            selected_candidate=selected_candidate,
        )

    def eval(
        self,
        test_loader,
        y_pred,
        anomaly_score_list,
        decision_result_list,
        domain_list,
        y_true,
        decision_threshold,
        mode,
    ):
        for _, batch in enumerate(test_loader):
            data = batch[0].to(self.device).float()
            y_true.append(batch[1][0].item())
            basename = batch[3][0]

            recon_data, _ = self.model(data)
            y_pred.append(self.loss_fn(recon_x=recon_data, x=data).mean().item())

            anomaly_score_list.append([basename, y_pred[-1]])
            decision_result_list.append([basename, 1 if y_pred[-1] > decision_threshold else 0])

            if mode:
                domain_list.append("target" if "target" in basename else "source")

        return y_pred, anomaly_score_list, decision_result_list, domain_list

    def export_evaluation_figures(self, result_dir, section_name, y_true, y_pred, domain_list, decision_threshold):
        if domain_list is None or len(domain_list) == 0:
            return

        source_mask = np.array([domain == "source" for domain in domain_list])
        target_mask = np.array([domain == "target" for domain in domain_list])

        roc_fig, roc_ax = plt.subplots(figsize=(6, 5))
        for mask, label, color in [
            (source_mask, "source", "tab:blue"),
            (target_mask, "target", "tab:orange"),
        ]:
            if mask.any() and len(np.unique(y_true[mask])) > 1:
                fpr, tpr, _ = metrics.roc_curve(y_true[mask], y_pred[mask])
                auc_value = metrics.roc_auc_score(y_true[mask], y_pred[mask])
                roc_ax.plot(fpr, tpr, label=f"{label} AUC={auc_value:.4f}", color=color)
        roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        roc_ax.set_xlabel("False Positive Rate")
        roc_ax.set_ylabel("True Positive Rate")
        roc_ax.set_title(f"{section_name} ROC")
        roc_ax.legend(loc="lower right")
        roc_ax.grid(alpha=0.3)
        roc_path = result_dir / f"{self.export_dir}_roc.png"
        roc_fig.tight_layout()
        roc_fig.savefig(roc_path, dpi=200)
        plt.close(roc_fig)

        hist_fig, hist_ax = plt.subplots(figsize=(6, 5))
        normal_scores = y_pred[y_true == 0]
        anomaly_scores = y_pred[y_true == 1]
        hist_ax.hist(normal_scores, bins=40, alpha=0.6, label="normal", color="tab:green", density=True)
        hist_ax.hist(anomaly_scores, bins=40, alpha=0.6, label="anomaly", color="tab:red", density=True)
        hist_ax.axvline(decision_threshold, color="white", linestyle="--", linewidth=1.5, label="threshold")
        hist_ax.set_xlabel("Anomaly score")
        hist_ax.set_ylabel("Density")
        hist_ax.set_title(f"{section_name} Score Distribution")
        hist_ax.legend(loc="upper right")
        hist_ax.grid(alpha=0.3)
        hist_path = result_dir / f"{self.export_dir}_score_hist.png"
        hist_fig.tight_layout()
        hist_fig.savefig(hist_path, dpi=200)
        plt.close(hist_fig)

        cm_fig, cm_ax = plt.subplots(figsize=(5, 5))
        cm = metrics.confusion_matrix(y_true, [1 if x > decision_threshold else 0 for x in y_pred])
        im = cm_ax.imshow(cm, cmap="Blues")
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                cm_ax.text(col, row, int(cm[row, col]), ha="center", va="center", color="black")
        cm_ax.set_xticks([0, 1])
        cm_ax.set_yticks([0, 1])
        cm_ax.set_xticklabels(["Normal", "Anomaly"])
        cm_ax.set_yticklabels(["Normal", "Anomaly"])
        cm_ax.set_xlabel("Predicted")
        cm_ax.set_ylabel("True")
        cm_ax.set_title(f"{section_name} Confusion Matrix")
        cm_fig.colorbar(im, ax=cm_ax, fraction=0.046, pad=0.04)
        cm_path = result_dir / f"{self.export_dir}_confusion_matrix.png"
        cm_fig.tight_layout()
        cm_fig.savefig(cm_path, dpi=200)
        plt.close(cm_fig)


def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(save_data)
