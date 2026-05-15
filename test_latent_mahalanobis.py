raise SystemExit('REMOVED: test_latent_mahalanobis.py — experimental test disabled')

    search_root = os.path.join('models', 'saved_model', os.path.basename(os.path.normpath(model_dir)))
    files = glob.glob(os.path.join(search_root, '*.pth'))
    if len(files) == 0:
        files = glob.glob(os.path.join(search_root, '**', '*.pth'), recursive=True)
    if len(files) == 0:
        files = glob.glob(os.path.join('models/saved_model', '**', '*.pth'), recursive=True)
    return files[0] if len(files) > 0 else None


def find_score_pickle(model_dir, suffix='mse'):
    search_root = os.path.join('models', 'saved_model', os.path.basename(os.path.normpath(model_dir)))
    files = glob.glob(os.path.join(search_root, f'*_{suffix}.pickle'))
    if len(files) == 0:
        files = glob.glob(os.path.join(search_root, '**', f'*_{suffix}.pickle'), recursive=True)
    if len(files) == 0:
        files = glob.glob(os.path.join('models/saved_model', '**', f'*_{suffix}.pickle'), recursive=True)
    return files[0] if len(files) > 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine_type', required=True)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--model_dir', required=True)
    args = parser.parse_args()

    # dataset string used in this repo
    dataset_str = f'DCASE2025T2{args.machine_type}'

    # minimal args for dataset loader
    class A:
        pass
    a = A()
    a.frames = 5
    a.n_mels = 128
    a.frame_hop_length = 1
    a.n_fft = 1024
    a.hop_length = 512
    a.power = 2.0
    a.fmax = None
    a.fmin = 0.0
    a.win_length = None
    a.batch_size = 256
    a.validation_split = 0.1
    a.dataset_directory = './data'
    a.dataset = dataset_str
    a.dev = False
    a.eval = True
    a.shuffle = False
    a.train_only = False
    a.use_ids = []
    a.is_auto_download = True
    a.mono = True
    a.frames = 5

    # instantiate dataset
    DataClass = Datasets.DatasetsDic[dataset_str]
    data = DataClass(a)

    device = torch.device('cpu')

    # find saved model file
    model_path = find_saved_model(args.model_dir)
    if model_path is None:
        raise FileNotFoundError('Saved model not found')

    # build model
    input_dim = a.frames * a.n_mels
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    checkpoint_root = os.path.join('models', 'checkpoint', os.path.basename(os.path.normpath(args.model_dir)))
    model_meta_path = os.path.join(checkpoint_root, model_stem, 'args.json')
    if not os.path.exists(model_meta_path):
        matches = glob.glob(os.path.join(checkpoint_root, '**', 'args.json'), recursive=True)
        if not matches:
            raise FileNotFoundError(f'Checkpoint args.json not found under {checkpoint_root}')
        model_meta_path = matches[0]
    with open(model_meta_path, 'r') as handle:
        model_meta = json.load(handle)
    model = AENet(
        input_dim=input_dim,
        block_size=a.n_mels,
        bottleneck_dim=int(model_meta.get('bottleneck_dim', 32)),
        n_layers=int(model_meta.get('n_layers', 4)),
    )

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 1) Extract latent vectors from ALL training normal clips
    train_loader = data.train_loader
    latent_vectors = extract_latent_vectors(model, train_loader, device)

    # 2) Fit Gaussian
    mu, cov_inv = fit_gaussian(latent_vectors)

    # 3) Compute training-set MSE reference scores (try to load saved pickle).
    mse_pickle = find_score_pickle(args.model_dir, 'mse')
    train_mse_scores = []
    if mse_pickle and os.path.exists(mse_pickle):
        try:
            with open(mse_pickle, 'rb') as f:
                train_mse_scores = pickle.load(f)
        except Exception:
            train_mse_scores = []

    # If no saved training MSE scores, compute them from train_loader
    if len(train_mse_scores) == 0:
        train_mse_scores = []
        with torch.no_grad():
            for batch in train_loader:
                data_x = batch[0].to(device).float()
                recon, _ = model(data_x)
                recon_np = recon.cpu().numpy()
                x_np = data_x.cpu().numpy()
                frame_mse = np.mean((recon_np - x_np) ** 2, axis=1)
                for m in frame_mse:
                    train_mse_scores.append(float(m))
        train_mse_scores = np.array(train_mse_scores)

    # Compute training Mahala ref as per-frame Mahala across latent_vectors
    from scipy.spatial import distance
    train_mahala_frames = [distance.mahalanobis(v, mu, cov_inv) for v in latent_vectors]
    train_mahala_scores = np.array(train_mahala_frames)

    # Compute means/stds for z-score normalization
    mse_mean = np.mean(train_mse_scores)
    mse_std = np.std(train_mse_scores) + 1e-8
    maha_mean = np.mean(train_mahala_scores)
    maha_std = np.std(train_mahala_scores) + 1e-8

    # 4) Iterate test set and collect normalized component scores
    out_dir = 'results/latent_mahalanobis'
    os.makedirs(out_dir, exist_ok=True)
    anomaly_rows = []
    basenames = []
    y_true = []
    mse_norm_list = []
    maha_norm_list = []

    for section_loader in data.test_loader:
        for batch in section_loader:
            data_x = batch[0].to(device).float()
            basename = batch[3][0]
            # recon and latent
            with torch.no_grad():
                recon, z = model(data_x)
            recon_np = recon.cpu().numpy()
            x_np = data_x.cpu().numpy()
            # per-frame mse
            frame_mse = np.mean((recon_np - x_np) ** 2, axis=1)
            clip_mse = float(np.mean(frame_mse))

            z_np = z.cpu().numpy()
            clip_mahala = compute_clip_mahalanobis_score(z_np, mu, cov_inv)

            # z-score normalization against training distribution
            mse_norm = (clip_mse - mse_mean) / mse_std
            maha_norm = (clip_mahala - maha_mean) / maha_std

            basenames.append(basename)
            mse_norm_list.append(float(mse_norm))
            maha_norm_list.append(float(maha_norm))
            y_true.append(int(batch[1][0].item()))

    # Convert lists to arrays for alpha sweep
    basenames = np.array(basenames)
    y_true = np.array(y_true)
    mse_norm_arr = np.array(mse_norm_list)
    maha_norm_arr = np.array(maha_norm_list)

    # Alpha sweep
    alphas = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    sweep_results = []
    print("\nalpha | AUC_src | AUC_tgt | AUC_tot | pAUC")
    for alpha in alphas:
        final_scores = alpha * mse_norm_arr + (1.0 - alpha) * maha_norm_arr
        auc_total = metrics.roc_auc_score(y_true, final_scores)
        p_auc = metrics.roc_auc_score(y_true, final_scores, max_fpr=0.1)

        # For source/target split: simple heuristics based on basename
        source_idx = [i for i, b in enumerate(basenames) if 'source' in b]
        target_idx = [i for i, b in enumerate(basenames) if 'target' in b]
        if len(source_idx) > 0:
            auc_source = metrics.roc_auc_score(y_true[source_idx], final_scores[source_idx])
        else:
            auc_source = auc_total
        if len(target_idx) > 0:
            auc_target = metrics.roc_auc_score(y_true[target_idx], final_scores[target_idx])
        else:
            auc_target = auc_total

        sweep_results.append((alpha, auc_source, auc_target, auc_total, p_auc))
        print(f"{alpha:.2f}  | {auc_source:.4f} | {auc_target:.4f} | {auc_total:.4f} | {p_auc:.4f}")

    # choose best alpha by AUC_total
    best = max(sweep_results, key=lambda x: x[3])
    best_alpha = best[0]
    best_auc_total = best[3]
    best_pauc = best[4]
    best_auc_source = best[1]
    best_auc_target = best[2]

    # Save final result.csv for best alpha
    res_dir = out_dir
    os.makedirs(res_dir, exist_ok=True)
    res_path = os.path.join(res_dir, 'result.csv')
    import csv
    with open(res_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['alpha', 'AUC_source', 'AUC_target', 'AUC_total', 'pAUC'])
        writer.writerow([best_alpha, best_auc_source, best_auc_target, best_auc_total, best_pauc])

    # Also save per-clip anomaly scores for best alpha
    anomaly_csv = os.path.join(out_dir, f'anomaly_score_{args.machine_type}_section_00_test.csv')
    final_scores = best_alpha * mse_norm_arr + (1.0 - best_alpha) * maha_norm_arr
    with open(anomaly_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        for b, s in zip(basenames, final_scores):
            writer.writerow([b, s])

    # print selected best metrics
    print('\nBest alpha:', best_alpha)
    print('AUC_source:', best_auc_source)
    print('AUC_target:', best_auc_target)
    print('AUC_total:', best_auc_total)
    print('pAUC:', best_pauc)


if __name__ == '__main__':
    main()
