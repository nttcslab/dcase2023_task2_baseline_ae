import pickle
import os
from pathlib import Path
import scipy
import torch
import sys
import numpy as np
import json
import pandas as pd
import shutil

from tools.plot_time_frequency import TimeFrequencyFigData
from datasets.datasets import Datasets

class BaseModel(object):
    def __init__(self, args, train, test):
        self.args = args
        print("selected gpu id:{}".format(args.gpu_id))
        self.device = torch.device("cuda" if args.cuda else "cpu",args.gpu_id[0])
        print(self.device)
        try: 
            self.data = Datasets(self.args.dataset).data(self.args)
        except Exception as e: 
            print('dataset "{}" is not supported'.format(self.args.dataset))
            print("please set another name \n{}".format(Datasets.show_list()))
            raise e
        self.train_loader = self.data.train_loader
        self.valid_loader = self.data.valid_loader
        self.test_loader = self.data.test_loader

        self.epoch = 0
        self.model = self.init_model()

        self.export_dir = f"{self.args.export_dir}" if self.args.export_dir else ""
        self.result_dir = Path(f"{args.result_directory}/dev_data/{self.export_dir}_{args.score}/")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.eval_data_result_dir = Path(f"{args.result_directory}/eval_data/{self.export_dir}_{args.score}/")
        self.eval_data_result_dir.mkdir(parents=True, exist_ok=True)
        self.model_name_suffix = "_"+self.args.model_name_suffix if self.args.model_name_suffix else ""
        self.eval_suffix = "_Eval" if self.args.eval else ""
        base_name = f"{self.export_dir}/{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}"
        self.checkpoint_dir = f"models/checkpoint/{base_name}"
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = f"{self.checkpoint_dir}/checkpoint.tar"
        Path(f"models/checkpoint/{self.export_dir}").mkdir(parents=True, exist_ok=True)
        self.logs_dir = Path(f"logs/{base_name}")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.logs_dir / "log.csv"

        log_data = self.get_log_header()
        if (train and not self.args.restart):
            print(f"Generate blank log -> {self.log_path}")
            np.savetxt(self.log_path,[log_data],fmt="%s")

        # select using checkpoint file
        if self.args.checkpoint_path:
            checkpoint_path = self.args.checkpoint_path
            saved_log_path = Path(f"logs")
            for dir in Path(os.path.dirname(checkpoint_path)).parts[2:]:
                saved_log_path /= dir
            saved_log_path /= "log.csv"
            if os.path.exists(saved_log_path):
                print(f"copy log: {saved_log_path}\n\t->{self.log_path}")
                shutil.copyfile(saved_log_path, self.log_path)
            else:
                print(f"Generate blank log: {self.log_path}")
                np.savetxt(self.log_path,[log_data],fmt="%s")
        else:
            checkpoint_path = self.checkpoint_path
        
        # load checkpoint
        self.optim_state_dict = None
        if self.args.restart:
            if os.path.exists(checkpoint_path):
                print(f"load checkpoint -> {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path)
                self.load_state_dict(checkpoint=checkpoint)
                self.optim_state_dict = self.load_optim_state_dict(checkpoint=checkpoint)
                if os.path.exists(self.log_path):
                    print(f"log reindex: {self.log_path}")
                    log_data = pd.read_csv(self.log_path).reindex(columns=log_data.split(',')).fillna(0)
                    log_data.to_csv(self.log_path, index=False)
            else :
                print(f"not found -> {checkpoint_path}")
                np.savetxt(self.log_path,[log_data],fmt="%s")
        
        self.model.to(self.device)
        self.model_dir = Path(f"models/saved_model/{self.export_dir}/")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir/f"{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}.pth"
        self.score_distr_file_path = self.model_dir/f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}.pickle"
        self.history_img = self.model_dir/f"history_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}.png"

        self.tf_figdata = TimeFrequencyFigData(
            max_imgs=4,
            max_extract=1,
            frames=args.frames,
            frame_hop_length=args.frame_hop_length,
            shape=(self.data.channel, self.data.width, self.data.height)
        )

        self.result_column_dict = {
            "single_domain":["section", "AUC", "pAUC", "precision", "recall", "F1 score"],
            "source_target":["section", "AUC (source)", "AUC (target)", "pAUC", "pAUC (source)", "pAUC (target)",
                            "precision (source)", "precision (target)", "recall (source)", "recall (target)",
                            "F1 score (source)", "F1 score (target)"]
        }

        # output parameter to json
        self.args_path = f"{self.checkpoint_dir}/args.json"
        tf = open(self.args_path, "w")
        json.dump(vars(self.args), tf, indent=2, ensure_ascii=False)
        print(f"save args -> {self.args_path}")
        tf.close()

    def init_model(self):
        pass
    
    def get_log_header(self):
        return "loss,loss_var,time"

    def load_state_dict(self, checkpoint):
        pretrain_net_dict = checkpoint['model_state_dict']
        net_dict = self.model.state_dict()
        net_dict.update(pretrain_net_dict)
        self.model.load_state_dict(net_dict)
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']

    def load_optim_state_dict(self, checkpoint, key='optimizer_state_dict'):
        return checkpoint[key]
    
    def fit_anomaly_score_distribution(self, y_pred, score_distr_file_path=None):
        if not score_distr_file_path:
            score_distr_file_path = self.score_distr_file_path
        shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(y_pred)
        gamma_params = [shape_hat, loc_hat, scale_hat]
        with open(score_distr_file_path, "wb") as f:
            pickle.dump(gamma_params, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def calc_decision_threshold(self, score_distr_file_path=None):
        if not score_distr_file_path:
            score_distr_file_path = self.score_distr_file_path
        # load anomaly score distribution for determining threshold
        with open(score_distr_file_path, "rb") as f:
            shape_hat, loc_hat, scale_hat = pickle.load(f)

        # determine threshold for decision
        return scipy.stats.gamma.ppf(q=self.args.decision_threshold, a=shape_hat, loc=loc_hat, scale=scale_hat)

    def train(self, epoch):
        pass

    def test(self):
        pass

    def copy_eval_data_score(self, decision_result_csv_path, anomaly_score_csv_path):
        eval_data_decision_result_csv_path = self.eval_data_result_dir / os.path.basename(decision_result_csv_path).replace(self.model_name_suffix, "")
        print(f"copy decision result: {decision_result_csv_path}\n\t->{eval_data_decision_result_csv_path}")
        shutil.copyfile(decision_result_csv_path, eval_data_decision_result_csv_path)

        eval_data_anomaly_score_csv_path = self.eval_data_result_dir / os.path.basename(anomaly_score_csv_path).replace(self.model_name_suffix, "")
        print(f"copy anomaly score: {anomaly_score_csv_path}\n\t->{eval_data_anomaly_score_csv_path}")
        shutil.copyfile(anomaly_score_csv_path, eval_data_anomaly_score_csv_path)
