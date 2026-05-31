import torch


def cov_v_diff(in_v):
    in_v_tmp = in_v.clone()
    mu = torch.mean(in_v_tmp.t(), 1)
    diff = torch.sub(in_v, mu)
    return diff, mu


def cov_v(diff, num):
    var = torch.matmul(diff.t(), diff) / num
    return var


def mahalanobis(u, v, cov_x, use_precision=False, reduction=True):
    num, dim = v.size()
    if use_precision:
        inv_cov = cov_x
    else:
        inv_cov = torch.inverse(cov_x)

    delta = torch.sub(u, v)
    m_loss = torch.matmul(torch.matmul(delta, inv_cov), delta.t())

    if reduction:
        return torch.sum(m_loss) / num
    else:
        return m_loss, num


def loss_function_mahala(recon_x, x, block_size, cov=None, is_source_list=None, is_target_list=None, update_cov=False,
                         use_precision=False, reduction=True):
    ### Modified mahalanobis loss to support Attribute-Conditioned Dictionaries ###
    if not update_cov:
        # If we pass a dictionary of covariance matrices (Attribute-Conditioned)
        if isinstance(cov, dict):
            min_loss = None
            # Calculate distance to each attribute cluster and take the minimum
            for attr_key, inv_cov in cov.items():
                loss, num = mahalanobis(recon_x.view(-1, block_size), x.view(-1, block_size), inv_cov,
                                        use_precision=use_precision, reduction=False)
                if min_loss is None:
                    min_loss = loss
                else:
                    min_loss = torch.min(min_loss, loss)

            if reduction:
                return torch.sum(min_loss) / recon_x.view(-1, block_size).size(0)
            else:
                return min_loss, recon_x.view(-1, block_size).size(0)
        else:
            # Standard single matrix fallback
            loss = mahalanobis(recon_x.view(-1, block_size), x.view(-1, block_size), cov, use_precision,
                               reduction=reduction)
            return loss
    else:
        # Variance diff calculations for the training phase
        diff = x - recon_x
        cov_diff_source, _ = cov_v_diff(in_v=diff[is_source_list].view(-1, block_size))

        cov_diff_target = None
        is_calc_cov_target = any(is_target_list)
        if is_calc_cov_target:
            cov_diff_target, _ = cov_v_diff(in_v=diff[is_target_list].view(-1, block_size))

        loss = diff ** 2
        if reduction:
            loss = torch.mean(loss, dim=1)

        return loss, cov_diff_source, cov_diff_target


def loss_reduction_mahala(loss):
    return torch.mean(loss)


def calc_inv_cov(model, device="cpu"):
    # This will be overridden in the main loop to handle dictionaries
    inv_cov_source = torch.inverse(model.cov_source.data.float().to(device))
    inv_cov_target = torch.inverse(model.cov_target.data.float().to(device))

    return inv_cov_source, inv_cov_target