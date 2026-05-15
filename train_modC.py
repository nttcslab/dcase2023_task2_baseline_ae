raise SystemExit('REMOVED: train_modC.py — experimental ModC code disabled to restore baseline')
import common as com
from networks.criterion.mahala import cov_v, loss_function_mahala, calc_inv_cov
from networks.base_model import BaseModel
from networks.dcase2023t2_ae.network_modC import AENet
from networks.models import Models
from tools.plot_loss_curve import csv_to_figdata


def _split_smoke_test_args(argv):
    config_path = "baseline.yaml"
    machine_type = None
    dataset_prefix = None
    cli_args = [argv[0]]

    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            i += 1
            continue
        if arg == "--machine_type" and i + 1 < len(argv):
            machine_type = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--machine_type="):
            machine_type = arg.split("=", 1)[1]
            i += 1
            continue
        if arg == "--is_DCASE2025T2":
            dataset_prefix = "DCASE2025T2"
            i += 1
            continue
        if arg == "--is_DCASE2026T2":
            dataset_prefix = "DCASE2026T2"
            i += 1
            continue

        cli_args.append(arg)
        i += 1

    return config_path, machine_type, dataset_prefix, cli_args


EVAL_MACHINE_TYPES_2025 = {
    "ToyRCCar",
    "ToyPet",
    "HomeCamera",
    "AutoTrash",
    "Polisher",
    "ScrewFeeder",
    "BandSealer",
    "CoffeeGrinder",
}


class DCASE2023T2AE_ModC(BaseModel):
    def __init__(self, args, train, test):
        super().__init__(
            args=args,
            train=train,
            test=test
        )
        parameter_list = [{"params": self.model.parameters()}]
        self.optimizer = torch.optim.Adam(parameter_list, lr=self.args.learning_rate)
        self.mse_score_distr_file_path = self.model_dir / f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_mse.pickle"
        self.mahala_score_distr_file_path = self.model_dir / f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_mahala.pickle"

    def init_model(self):
        self.block_size = self.data.height
        return AENet(
            input_dim=self.data.input_dim,
            block_size=self.block_size,
            bottleneck_dim=self.args.bottleneck_dim,
            n_layers=self.args.n_layers
        )

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
            return
        torch.autograd.set_detect_anomaly(True)
        train_loss = 0
        train_recon_loss = 0
        train_recon_loss_source = 0
        train_recon_loss_target = 0
        y_pred = []

        train_loader = self.train_loader

        if epoch == self.args.epochs + 1:
            print("\n============== CALCULATE COVARIANCE ==============")
            is_calc_cov = True
            self.model.eval()
            torch.set_grad_enabled(False)
            cov_x_source = np.zeros((self.block_size, self.block_size))
            cov_x_source = torch.from_numpy(cov_x_source)
            cov_x_source = cov_x_source.to(self.device).float()
            cov_x_target = cov_x_source.clone().detach()
            num_source = 0
            num_target = 0
            epoch = self.args.epochs
        else:
            self.model.train()
            is_calc_cov = False

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            data = batch[0]
            data = data.to(self.device).float()
            if data.shape[0] <= 1:
                continue
            data_name_list = batch[3]
            machine_id = torch.argmax(batch[2], dim=1).long()
            machine_id = machine_id.to(self.device)

            is_target_list = ["target" in data_name for data_name in data_name_list]
            is_source_list = np.logical_not(is_target_list).tolist()
            n_source = is_source_list.count(True)
            n_target = is_target_list.count(True)

            if not is_calc_cov:
                self.optimizer.zero_grad()
            recon_batch, z = self.model(data)

            if is_calc_cov:
                score_2d, cov_diff_source, cov_diff_target = loss_function_mahala(
                    recon_x=recon_batch,
                    x=data,
                    block_size=self.block_size,
                    update_cov=True,
                    reduction=False,
                    is_source_list=is_source_list,
                    is_target_list=is_target_list
                )
                cov_x_source_batch = cov_v(
                    diff=cov_diff_source,
                    num=1
                )
                cov_x_source += cov_x_source_batch.clone().detach()
                num_source += n_source
                if n_target > 0:
                    cov_x_target_batch = cov_v(
                        diff=cov_diff_target,
                        num=1
                    )
                    cov_x_target += cov_x_target_batch.clone().detach()
                    num_target += n_target
            else:
                score_2d = self.loss_fn(
                    recon_batch,
                    data
                )

            n_loss = len(score_2d)
            score = self.loss_reduction_1d(score=score_2d)

            recon_loss = self.loss_reduction(score=score, n_loss=n_loss)
            recon_loss_source = self.loss_reduction(score=score[is_source_list], n_loss=n_source)
            if n_target > 0:
                recon_loss_target = self.loss_reduction(score=score[is_target_list], n_loss=n_target)
            else:
                recon_loss_target = 0

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
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    self.loss.item()))

        if is_calc_cov:
            cov_x_source /= num_source - 1
            if num_target == 0:
                cov_x_target = cov_x_source.clone().detach()
            else:
                cov_x_target /= num_target - 1
            self.model.cov_source.data = cov_x_source
            self.model.cov_target.data = cov_x_target

            inv_cov_source, inv_cov_target = calc_inv_cov(
                model=self.model,
                device=self.device
            )
            y_pred_mahala = []
            for batch_idx, batch in enumerate(tqdm(train_loader)):
                y_pred_mahala = self.calc_valid_mahala_score(
                    data=batch[0],
                    y_pred=y_pred_mahala,
                    inv_cov_source=inv_cov_source,
                    inv_cov_target=inv_cov_target,
                )
            for batch_idx, batch in enumerate(self.valid_loader):
                y_pred_mahala = self.calc_valid_mahala_score(
                    data=batch[0],
                    y_pred=y_pred_mahala,
                    inv_cov_source=inv_cov_source,
                    inv_cov_target=inv_cov_target,
                )
            self.fit_anomaly_score_distribution(
                y_pred=y_pred_mahala,
                score_distr_file_path=self.mahala_score_distr_file_path
            )

        # validation test
        val_loss = 0
        with torch.no_grad():
            self.model.eval()
            for batch_idx, batch in enumerate(self.valid_loader):
                data = batch[0]
                data = data.to(self.device).float()

                recon_batch, _ = self.model(data)
                score = self.loss_fn(
                    recon_batch,
                    data
                )
                loss = score.mean()

                val_loss += float(loss)
                y_pred.append(loss.item())

        if not is_calc_cov:
            print('====> Epoch: {} Average loss: {:.4f} Validation loss: {:.4f}'.format(
                epoch,
                train_loss / len(train_loader),
                val_loss / len(self.valid_loader)))
            with open(self.log_path, 'a') as log:
                np.savetxt(log, ["{0},{1},{2},{3},{4}".format(
                    train_loss / len(train_loader),
                    val_loss / len(self.valid_loader),
                    train_recon_loss / len(train_loader),
                    train_recon_loss_source / len(train_loader),
                    train_recon_loss_target / len(train_loader),
                )], fmt="%s")
            csv_to_figdata(
                file_path=self.log_path,
                column_heading_list=self.column_heading_list,
                ylabel="loss",
                fig_count=len(self.column_heading_list),
                cut_first_epoch=True
            )
            self.fit_anomaly_score_distribution(
                y_pred=y_pred,
                score_distr_file_path=self.mse_score_distr_file_path
            )

        # save model
        torch.save(self.model.state_dict(), self.model_path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss
        }, self.checkpoint_path)

    def calc_valid_mahala_score(self, data, y_pred, inv_cov_source, inv_cov_target):
        data = data.to(self.device).float()
        recon_data, _ = self.model(data)
        loss_source, num = loss_function_mahala(
            recon_x=recon_data,
            x=data,
            block_size=self.block_size,
            cov=inv_cov_source,
            use_precision=True,
            reduction=False
        )
        loss_source = self.loss_reduction(score=self.loss_reduction_1d(loss_source), n_loss=num)

        loss_target, num = loss_function_mahala(
            recon_x=recon_data,
            x=data,
            block_size=self.block_size,
            cov=inv_cov_target,
            use_precision=True,
            reduction=False
        )
        loss_target = self.loss_reduction(score=self.loss_reduction_1d(loss_target), n_loss=num)
        y_pred.append(min(loss_target.item(), loss_source.item()))
        return y_pred

    def loss_reduction_1d(self, score):
        return torch.mean(score, dim=1)

    def loss_reduction(self, score, n_loss):
        return torch.sum(score) / n_loss

    def loss_fn(self, recon_x, x):
        loss = torch.nn.functional.mse_loss(recon_x, x.view(recon_x.shape), reduction="none")
        return loss

    def test(self):
        from sklearn import metrics
        from tools.plot_anm_score import AnmScoreFigData
        anm_score_figdata = AnmScoreFigData()
        mode = self.data.mode

        csv_lines = []

        block_size = self.data.height
        if mode:
            performance_over_all = []
            performance = []

        print("============== MODEL LOAD ==============")
        if not self.model_path.exists():
            print(f"model not found -> {self.model_path} ")
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        if self.args.score == "MAHALA":
            decision_threshold = self.calc_decision_threshold(score_distr_file_path=self.mahala_score_distr_file_path)
        else:
            decision_threshold = self.calc_decision_threshold(score_distr_file_path=self.mse_score_distr_file_path)

        dir_name = "test"
        inv_cov_source = None
        inv_cov_target = None
        if self.args.score == "MAHALA":
            inv_cov_source, inv_cov_target = calc_inv_cov(
                model=self.model,
                device=self.device
            )

        print("============== BEGIN TEST ==============")
        y_pred = []
        y_true = []

        for idx, test_loader_tmp in enumerate(self.test_loader):
            section_name = f"section_{self.data.section_id_list[idx]}"
            result_dir = self.result_dir if self.args.dev else self.eval_data_result_dir

            # setup anomaly score file path
            anomaly_score_csv = result_dir/f"anomaly_score_{self.args.dataset}_{section_name}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv"
            anomaly_score_list = []

            # setup decision result file path
            decision_result_csv = result_dir/f"decision_result_{self.args.dataset}_{section_name}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv"
            decision_result_list = []

            domain_list = None
            if mode:
                domain_list = []

            print("\n============== BEGIN TEST FOR A SECTION ==============")
            y_pred_section = []
            y_true_section = []
            test_loader = test_loader_tmp

            with torch.no_grad():
                for j, batch in enumerate(test_loader):
                    data = batch[0]
                    data = data.to(self.device).float()
                    y_true_section.append(batch[1][0].item())
                    basename = batch[3][0]

                    recon_data, _ = self.model(data)

                    if self.args.score == "MAHALA":
                        loss_source, num = loss_function_mahala(
                            recon_x=recon_data,
                            x=data,
                            block_size=self.block_size,
                            cov=inv_cov_source,
                            use_precision=True,
                            reduction=False
                        )
                        loss_source = self.loss_reduction(score=self.loss_reduction_1d(loss_source), n_loss=num)

                        loss_target, num = loss_function_mahala(
                            recon_x=recon_data,
                            x=data,
                            block_size=self.block_size,
                            cov=inv_cov_target,
                            use_precision=True,
                            reduction=False
                        )
                        loss_target = self.loss_reduction(score=self.loss_reduction_1d(loss_target), n_loss=num)
                        y_pred_section.append(min(loss_target.item(), loss_source.item()))
                    else:
                        y_pred_section.append(self.loss_fn(recon_x=recon_data, x=data).mean().item())

                    # store anomaly scores
                    anomaly_score_list.append([basename, y_pred_section[-1]])

                    # store decision results
                    if y_pred_section[-1] > decision_threshold:
                        decision_result_list.append([basename, 1])
                    else:
                        decision_result_list.append([basename, 0])

                    if mode:
                        domain_list.append("target" if "target" in basename else "source")

            # output anomaly scores
            from networks.dcase2023t2_ae.dcase2023t2_ae import save_csv
            save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
            print("anomaly score result ->  {}".format(anomaly_score_csv))

            # output decision results
            save_csv(save_file_path=decision_result_csv, save_data=decision_result_list)
            print("decision result ->  {}".format(decision_result_csv))

            y_pred.extend(y_pred_section)
            y_true.extend(y_true_section)

        # calculate max AUC
        if len(y_true) > 0 and len(y_pred) > 0:
            max_auc = metrics.roc_auc_score(y_true, y_pred)
            
            # Calculate pAUC
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=self.args.max_fpr)
            
            # Calculate confusion matrix metrics
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, [1 if x > decision_threshold else 0 for x in y_pred]).ravel()
            prec = tp / np.maximum(tp + fp, np.finfo(float).eps)
            recall = tp / np.maximum(tp + fn, np.finfo(float).eps)
            f1 = 2.0 * prec * recall / np.maximum(prec + recall, np.finfo(float).eps)

            print("============== END OF TEST ==============")

            # output results
            print("AUC: {0:.4f}".format(max_auc))
            print("pAUC: {0:.4f}".format(p_auc))
            print("Precision: {0:.4f}".format(prec))
            print("Recall: {0:.4f}".format(recall))
            print("F1 Score: {0:.4f}".format(f1))
            print("\n" + "=" * 100)

            return max_auc, p_auc
        else:
            print("No test data available")
            return 0, 0


def main():
    # Parse smoke test specific args
    config_path, machine_type, dataset_prefix, cli_args = _split_smoke_test_args(sys.argv)

    # Load yaml config
    with open(config_path) as stream:
        param = yaml.safe_load(stream)

    # Create argument parser with ModC specific arguments
    parser = com.get_argparse()
    parser.add_argument('--bottleneck_dim', type=int, default=32,
                        help='bottleneck dimension (default: 32)')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='number of hidden layers (default: 4)')

    # Convert yaml params to args list
    flat_param = com.param_to_args_list(params=param)

    # Parse yaml params first
    args = parser.parse_args(args=flat_param)

    # Parse command line args (override yaml)
    args = parser.parse_args(args=cli_args[1:], namespace=args)

    # Handle dataset override
    if dataset_prefix and machine_type:
        args.dataset = f"{dataset_prefix}{machine_type}"
        if not any(flag in cli_args[1:] for flag in ["--dev", "-d", "--eval", "-e"]):
            if dataset_prefix == "DCASE2025T2" and machine_type in EVAL_MACHINE_TYPES_2025:
                args.dev = False
                args.eval = True
            else:
                args.dev = True
                args.eval = False

    print(args)

    if args.train_only and args.test_only:
        raise ValueError("--train_only and --test_only cannot be used together.")
    elif args.train_only:
        train = True
        test = False
    elif args.test_only:
        train = False
        test = True
    else:
        train = True
        test = True

    # Set export directory with bottleneck and layer info
    args.export_dir = f"modC_b{args.bottleneck_dim}_l{args.n_layers}"
    args.cuda = args.use_cuda and torch.cuda.is_available()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # Register custom model class with the Models registry
    Models.ModelsDic["DCASE2023T2-AE"] = DCASE2023T2AE_ModC

    # Create model
    net = Models(args.model).net(
        args=args,
        train=train,
        test=test
    )

    print(args.model)

    print("============== BEGIN TRAIN ==============")
    if train:
        for epoch in range(1, args.epochs + 2):
            net.train(epoch)
    print("============ END OF TRAIN ============")

    if test:
        net.test()


if __name__ == "__main__":
    main()
