import torch
from torch import nn
import numpy as np
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal

def nll_with_covariances(gt, predictions, confidences, avails, covariance_matrices):
    #代码作者：loss: You can read about the loss function in the technical report, actually it is nothing but NLL that was used in many papers
    precision_matrices = torch.inverse(covariance_matrices) #方阵的逆矩阵，(B, 6, 80, 2, 2)
    gt = torch.unsqueeze(gt, 1) #([B, 1, 80, 2])
    avails = avails[:, None, :, None] #([B, 1, 80, 1])
    coordinates_delta = (gt - predictions).unsqueeze(-1) #(B,6,80,2,1)
    # 通过协方差矩阵得到概率 https://zhuanlan.zhihu.com/p/480918380
    errors = coordinates_delta.permute(0, 1, 2, 4, 3) @ precision_matrices @ coordinates_delta #([B, 6, 80, 1, 1])，
    # logdet:行列式值再取对数，协方差矩阵行列式值是非负的， 所以前一部分是概率，后一部分是数据云的体积（数据协方差的大小，体积越大，协方差也越大）
    errors = avails * (-0.5 * errors.squeeze(-1) - 0.5 * torch.logdet(covariance_matrices).unsqueeze(-1))# (B,6,80,1) 有个回复用这个torch.log(torch.det(covariance_matrices).unsqueeze(-1)+1e-6)
    assert torch.isfinite(errors).all()
    with np.errstate(divide="ignore"):
        errors = nn.functional.log_softmax(confidences, dim=1) + \
            torch.sum(errors, dim=[2, 3]) #(B,6)
    errors = -torch.logsumexp(errors, dim=-1, keepdim=True) #相当于log_sum_exp(x) (B,1)
    return torch.mean(errors)

def pytorch_neg_multi_log_likelihood_batch(gt, predictions, confidences, avails):
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        predictions (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords
    error = torch.sum(
        ((gt - predictions) * avails) ** 2, dim=-1
    )  # reduce coords and use availability
    with np.errstate(
        divide="ignore"
    ):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = nn.functional.log_softmax(confidences, dim=1) - 0.5 * torch.sum(
            error, dim=-1
        )  # reduce time
    # error (batch_size, num_modes)
    error = -torch.logsumexp(error, dim=-1, keepdim=True)
    return torch.mean(error)