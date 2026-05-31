import os
import sys
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import scipy
from sklearn import metrics
import csv
from tqdm import tqdm
import re
import pickle
from collections import defaultdict

from networks.base_model import BaseModel
from networks.dcase2023t2_ae.network import AENet
from networks.criterion.mahala import cov_v, loss_function_mahala, calc_inv_cov
from tools.plot_anm_score import AnmScoreFigData
from tools.plot_loss_curve import csv_to_figdata


def torch_smote(x, target_count, k=5):
    num_samples, num_features = x.size()
    if num_samples >= target_count: return x
    dist_matrix = torch.cdist(x, x)
    _, knn_indices = torch.topk(dist_matrix, k=k + 1, largest=False)
    knn_indices = knn_indices[:, 1:]
    num_to_generate = target_count - num_samples
    synthetic_samples = []
    for _ in range(num_to_generate):
        idx = np.random.randint(0, num_samples)
        neighbor_idx = knn_indices[idx, np.random.randint(0, k)].item()
        lam = np.random.rand()
        syn = x[idx] + lam * (x[neighbor_idx] - x[idx])
        synthetic_samples.append(syn.unsqueeze(0))
    return torch.cat([x, torch.cat(synthetic_samples, dim=0)], dim=0)


class DCASE2023T2AE(BaseModel):
    def __init__(self, args, train, test):
        super().__init__(args=args, train=train, test=test)
        parameter_list = [{"params": self.model.parameters()}]
        self.optimizer = optim.Adam(parameter_list, lr=self.args.learning_rate)
        self.mse_score_distr_file_path = self.model_dir / f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_mse.pickle"
        self.mahala_score_distr_file_path = self.model_dir / f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_mahala.pickle"

    def init_model(self):
        self.block_size = self.data.height
        return AENet(input_dim=self.data.input_dim, block_size=self.block_size)

    def get_log_header(self):
        self.column_heading_list = [["loss"], ["val_loss"], ["recon_loss"], ["recon_loss_source", "recon_loss_target"]]
        return "loss,val_loss,recon_loss,recon_loss_source,recon_loss_target"

    def train(self, epoch):
        if epoch <= self.epoch: return
        torch.autograd.set_detect_anomaly(True)
        train_loss, train_recon_loss, train_recon_loss_source, train_recon_loss_target = 0, 0, 0, 0
        y_pred = []
        train_loader = self.train_loader

        if epoch == self.args.epochs + 1:
            print("\n============== CALCULATE COVARIANCE (SUB-CLUSTERS + SMOTE) ==============")
            is_calc_cov = True
            self.model.eval()
            torch.set_grad_enabled(False)
            residual_pool = defaultdict(list)
            epoch = self.args.epochs
        else:
            self.model.train()
            is_calc_cov = False

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            data = batch[0].float().to(self.device)
            if data.shape[0] <= 1: continue
            data_name_list = batch[3]
            is_target_list = ["target" in data_name for data_name in data_name_list]
            is_source_list = np.logical_not(is_target_list).tolist()
            n_source, n_target = is_source_list.count(True), is_target_list.count(True)

            if not is_calc_cov: self.optimizer.zero_grad()
            recon_batch, z = self.model(data)

            if is_calc_cov:
                score_2d = self.loss_fn(recon_batch, data)
                diff = (data - recon_batch).view(-1, self.block_size)

                # Assign residuals to specific speed/mic clusters
                for i, name in enumerate(data_name_list):
                    match = re.search(r'spd_(\d+)_mic_(\d+)', name)
                    attr = f"spd_{match.group(1)}_mic_{match.group(2)}" if match else "global"
                    domain = "target" if "target" in name else "source"
                    residual_pool[(domain, attr)].append(diff[i].detach())
            else:
                score_2d = self.loss_fn(recon_batch, data)

            n_loss = len(score_2d)
            score = self.loss_reduction_1d(score=score_2d)
            recon_loss = self.loss_reduction(score=score, n_loss=n_loss)
            recon_loss_source = self.loss_reduction(score=score[is_source_list], n_loss=n_source)
            recon_loss_target = self.loss_reduction(score=score[is_target_list], n_loss=n_target) if n_target > 0 else 0

            self.loss = recon_loss
            if not is_calc_cov:
                self.loss.backward()
                self.optimizer.step()
            train_loss += float(self.loss)
            train_recon_loss += float(recon_loss)
            train_recon_loss_source += float(recon_loss_source)
            train_recon_loss_target += float(recon_loss_target)
            y_pred.append(self.loss.item())

            if batch_idx % self.args.log_interval == 0 and not is_calc_cov:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    self.loss.item()))

        if is_calc_cov:
            inv_cov_source_dict = {}
            inv_cov_target_dict = {}

            # Process each cluster independently
            for (domain, attr), vector_list in residual_pool.items():
                group_tensor = torch.stack(vector_list, dim=0)

                # Apply SMOTE if sample size is too small for stable 640x640 inversion
                if group_tensor.size(0) < 700:
                    group_tensor = torch_smote(group_tensor, target_count=700, k=5)

                mu = torch.mean(group_tensor.t(), dim=1)
                c_diff = torch.sub(group_tensor, mu)
                cov = torch.matmul(c_diff.t(), c_diff) / (group_tensor.size(0) - 1)
                cov += torch.eye(self.block_size, device=self.device) * 1e-2

                if domain == "source":
                    inv_cov_source_dict[attr] = torch.inverse(cov)
                else:
                    inv_cov_target_dict[attr] = torch.inverse(cov)

            if len(inv_cov_target_dict) == 0:
                inv_cov_target_dict = {k: v.clone() for k, v in inv_cov_source_dict.items()}

            self.inv_cov_source, self.inv_cov_target = inv_cov_source_dict, inv_cov_target_dict
            with open(self.model_dir / f"cov_dict_{self.args.dataset}_seed{self.args.seed}.pickle", 'wb') as f:
                pickle.dump({'source': inv_cov_source_dict, 'target': inv_cov_target_dict}, f)

            y_pred_mahala = []
            for batch_idx, batch in enumerate(tqdm(train_loader)):
                y_pred_mahala = self.calc_valid_mahala_score(batch[0], y_pred_mahala, self.inv_cov_source,
                                                             self.inv_cov_target)
            for batch_idx, batch in enumerate(self.valid_loader):
                y_pred_mahala = self.calc_valid_mahala_score(batch[0], y_pred_mahala, self.inv_cov_source,
                                                             self.inv_cov_target)
            self.fit_anomaly_score_distribution(y_pred=y_pred_mahala,
                                                score_distr_file_path=self.mahala_score_distr_file_path)

        val_loss = 0
        with torch.no_grad():
            self.model.eval()
            for batch_idx, batch in enumerate(self.valid_loader):
                data = batch[0].float().to(self.device)
                recon_batch, _ = self.model(data)
                loss = self.loss_fn(recon_batch, data).mean()
                val_loss += float(loss)
                y_pred.append(loss.item())

        if not is_calc_cov:
            print('====> Epoch: {} Average loss: {:.4f} Validation loss: {:.4f}'.format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        val_loss / len(
                                                                                            self.valid_loader)))
            with open(self.log_path, 'a') as log:
                np.savetxt(log, [
                    "{0},{1},{2},{3},{4}".format(train_loss / len(train_loader), val_loss / len(self.valid_loader),
                                                 train_recon_loss / len(train_loader),
                                                 train_recon_loss_source / len(train_loader),
                                                 train_recon_loss_target / len(train_loader))], fmt="%s")

            # FIXED KWARGS CALL HERE
            csv_to_figdata(
                file_path=self.log_path,
                column_heading_list=self.column_heading_list,
                ylabel="loss",
                fig_count=len(self.column_heading_list),
                cut_first_epoch=True
            )
            self.fit_anomaly_score_distribution(y_pred=y_pred, score_distr_file_path=self.mse_score_distr_file_path)

        torch.save(self.model.state_dict(), self.model_path)
        torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(), 'loss': self.loss}, self.checkpoint_path)

    def calc_valid_mahala_score(self, data, y_pred, inv_cov_source, inv_cov_target):
        data = data.float().to(self.device)
        recon_data, _ = self.model(data)
        loss_source, num = loss_function_mahala(recon_x=recon_data, x=data, block_size=self.block_size,
                                                cov=inv_cov_source, use_precision=True, reduction=False)
        loss_source = self.loss_reduction(score=self.loss_reduction_1d(loss_source), n_loss=num)
        loss_target, num = loss_function_mahala(recon_x=recon_data, x=data, block_size=self.block_size,
                                                cov=inv_cov_target, use_precision=True, reduction=False)
        loss_target = self.loss_reduction(score=self.loss_reduction_1d(loss_target), n_loss=num)
        y_pred.append(min(loss_target.item(), loss_source.item()))
        return y_pred

    def loss_reduction_1d(self, score):
        return torch.mean(score, dim=1)

    def loss_reduction(self, score, n_loss):
        return torch.sum(score) / n_loss

    def loss_fn(self, recon_x, x):
        return F.mse_loss(recon_x, x.view(recon_x.shape), reduction="none")

    def test(self):
        anm_score_figdata = AnmScoreFigData()
        mode = True
        csv_lines = []
        block_size = self.data.height
        if mode:
            performance_over_all, performance = [], []

        print("============== MODEL LOAD ==============")
        if not os.path.exists(self.model_path): print(f"model not found -> {self.model_path} ")
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        if self.args.score == "MAHALA":
            decision_threshold = self.calc_decision_threshold(score_distr_file_path=self.mahala_score_distr_file_path)
        else:
            decision_threshold = self.calc_decision_threshold(score_distr_file_path=self.mse_score_distr_file_path)

        dir_name = "test"

        cov_dict_path = self.model_dir / f"cov_dict_{self.args.dataset}_seed{self.args.seed}.pickle"
        if os.path.exists(cov_dict_path):
            with open(cov_dict_path, 'rb') as f:
                cov_dicts = pickle.load(f)
            inv_cov_source, inv_cov_target = cov_dicts['source'], cov_dicts['target']
        else:
            inv_cov_source, inv_cov_target = calc_inv_cov(model=self.model, device=self.device)

        for idx, test_loader_tmp in enumerate(self.test_loader):
            section_name = f"section_{self.data.section_id_list[idx]}"
            result_dir = self.result_dir if self.args.dev else self.eval_data_result_dir
            anomaly_score_csv = result_dir / f"anomaly_score_{self.args.dataset}_{section_name}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv"
            anomaly_score_list, decision_result_list = [], []
            decision_result_csv = result_dir / f"decision_result_{self.args.dataset}_{section_name}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv"
            domain_list = [] if mode else None

            print("\n============== BEGIN TEST FOR A SECTION ==============")
            y_pred, y_true = [], []
            with torch.no_grad():
                y_pred, anomaly_score_list, decision_result_list, domain_list = self.eval(
                    test_loader=test_loader_tmp, y_pred=y_pred, anomaly_score_list=anomaly_score_list,
                    decision_result_list=decision_result_list, domain_list=domain_list, y_true=y_true,
                    decision_threshold=decision_threshold, mode=mode, inv_cov_source=inv_cov_source,
                    inv_cov_target=inv_cov_target,
                )

            save_csv(anomaly_score_csv, anomaly_score_list)
            save_csv(decision_result_csv, decision_result_list)

            if mode:
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
                tn_s, fp_s, fn_s, tp_s = metrics.confusion_matrix(y_true_s, [1 if x > decision_threshold else 0 for x in
                                                                             y_pred_s]).ravel()
                prec_s = tp_s / np.maximum(tp_s + fp_s, sys.float_info.epsilon)
                recall_s = tp_s / np.maximum(tp_s + fn_s, sys.float_info.epsilon)
                f1_s = 2.0 * prec_s * recall_s / np.maximum(prec_s + recall_s, sys.float_info.epsilon)

                print("AUC (source) : {}".format(auc_s))
                print("pAUC : {}".format(p_auc))
                print("pAUC (source) : {}".format(p_auc_s))

                if len(y_true_t) > 0:
                    auc_t = metrics.roc_auc_score(y_true_t_auc, y_pred_t_auc)
                    p_auc_t = metrics.roc_auc_score(y_true_t, y_pred_t, max_fpr=self.args.max_fpr)
                    tn_t, fp_t, fn_t, tp_t = metrics.confusion_matrix(y_true_t,
                                                                      [1 if x > decision_threshold else 0 for x in
                                                                       y_pred_t]).ravel()
                    prec_t = tp_t / np.maximum(tp_t + fp_t, sys.float_info.epsilon)
                    recall_t = tp_t / np.maximum(tp_t + fn_t, sys.float_info.epsilon)
                    f1_t = 2.0 * prec_t * recall_t / np.maximum(prec_t + recall_t, sys.float_info.epsilon)

                    if len(csv_lines) == 0: csv_lines.append(self.result_column_dict["source_target"])
                    csv_lines.append(
                        [section_name.split("_", 1)[1], auc_s, auc_t, p_auc, p_auc_s, p_auc_t, prec_s, prec_t, recall_s,
                         recall_t, f1_s, f1_t])
                    performance.append(
                        [auc_s, auc_t, p_auc, p_auc_s, p_auc_t, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])
                    print("AUC (target) : {}".format(auc_t))
                    print("pAUC (target) : {}".format(p_auc_t))

        if mode:
            amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
            csv_lines.append(["arithmetic mean"] + list(amean_performance))
        result_path = result_dir / f"result_{self.args.dataset}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}_roc.csv"
        save_csv(save_file_path=result_path, save_data=csv_lines)

    def eval(self, test_loader, y_pred, anomaly_score_list, decision_result_list, domain_list, y_true,
             decision_threshold, mode, inv_cov_source, inv_cov_target):
        for j, batch in enumerate(test_loader):
            data = batch[0].float().to(self.device)
            y_true.append(batch[1][0].item())
            basename = batch[3][0]
            recon_data, _ = self.model(data)

            if self.args.score == "MAHALA":
                loss_source, num = loss_function_mahala(recon_x=recon_data, x=data, block_size=self.block_size,
                                                        cov=inv_cov_source, use_precision=True, reduction=False)
                loss_source = self.loss_reduction(score=self.loss_reduction_1d(loss_source), n_loss=num)
                loss_target, num = loss_function_mahala(recon_x=recon_data, x=data, block_size=self.block_size,
                                                        cov=inv_cov_target, use_precision=True, reduction=False)
                loss_target = self.loss_reduction(score=self.loss_reduction_1d(loss_target), n_loss=num)
                y_pred.append(min(loss_target.item(), loss_source.item()))
            else:
                y_pred.append(self.loss_fn(recon_x=recon_data, x=data).mean().item())

            anomaly_score_list.append([basename, y_pred[-1]])
            if y_pred[-1] > decision_threshold:
                decision_result_list.append([basename, 1])
            else:
                decision_result_list.append([basename, 0])
            if mode: domain_list.append("target" if "target" in basename else "source")
        return y_pred, anomaly_score_list, decision_result_list, domain_list


def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)