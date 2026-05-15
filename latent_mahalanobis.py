raise SystemExit('REMOVED: latent_mahalanobis.py — experimental latent Mahalanobis disabled')

    """
    Run all training samples through encoder only (no decoder).
    Return np.ndarray of shape (N_frames_total, bottleneck_dim).
    Set model.eval() and use torch.no_grad().
    The encoder is all layers up to and including the bottleneck.
    """
    model.to(device)
    model.eval()
    latents = []
    with torch.no_grad():
        for batch in dataloader:
            data = batch[0]
            x = data.to(device).float()
            # encoder is exposed as model.encoder
            if hasattr(model, "encoder"):
                z = model.encoder(x.view(-1, x.shape[1]))
            else:
                # fallback: call model and take second output if available
                out = model(x)
                if isinstance(out, tuple) and len(out) >= 2:
                    z = out[1]
                else:
                    raise RuntimeError("Cannot find encoder in model")
            # Convert to numpy safely - handle NumPy 2.x compatibility
            z_np = z.detach().cpu().numpy() if hasattr(z, 'detach') else np.array(z)
            latents.append(z_np)
    if len(latents) == 0:
        return np.zeros((0, 0))
    return np.vstack(latents)


def fit_gaussian(latent_vectors):
    """
    Compute mean mu and inverse covariance cov_inv.
    Add 1e-6 * identity to covariance before inverting (regularization).
    Return: mu (array, bottleneck_dim), cov_inv (array, bottleneck_dim x bottleneck_dim)
    """
    mu = np.mean(latent_vectors, axis=0)
    cov = np.cov(latent_vectors, rowvar=False)
    # regularize
    eps = 1e-6
    cov += np.eye(cov.shape[0]) * eps
    cov_inv = np.linalg.inv(cov)
    return mu, cov_inv


def compute_clip_mahalanobis_score(latent_frames, mu, cov_inv):
    """
    latent_frames: np.ndarray (T_frames, bottleneck_dim) for one clip
    Compute Mahalanobis distance for each frame, return mean over frames.
    Use scipy.spatial.distance.mahalanobis per frame.
    """
    scores = []
    for i in range(latent_frames.shape[0]):
        scores.append(distance.mahalanobis(latent_frames[i], mu, cov_inv))
    if len(scores) == 0:
        return 0.0
    return float(np.mean(scores))


def normalize_scores(scores, ref_scores):
    """
    Normalize scores by dividing by mean of ref_scores (training set scores).
    Prevents scale mismatch between MSE and Mahalanobis before ensembling.
    scores: array of test scores
    ref_scores: array of training scores (used to compute normalization factor)
    Return normalized scores.
    """
    ref_mean = float(np.mean(ref_scores)) if len(ref_scores) > 0 else 1.0
    eps = 1e-12
    return np.array(scores) / (ref_mean + eps)
