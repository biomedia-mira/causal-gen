import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def log_prob_from_logits(x):
    """numerically stable log_softmax implementation that prevents overflow"""
    axis = len(x.shape) - 1
    m = x.max(dim=axis, keepdim=True)[0]
    return x - m - torch.log(torch.exp(x - m).sum(dim=axis, keepdim=True))


def const_max(t, constant):
    other = torch.ones_like(t) * constant
    return torch.max(t, other)


def const_min(t, constant):
    other = torch.ones_like(t) * constant
    return torch.min(t, other)


def discretized_mix_logistic_loss(x, l, low_bit=False):
    """log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval"""
    # Adapted from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
    xs = [
        s for s in x.shape
    ]  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = [s for s in l.shape]  # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(
        ls[-1] / 10
    )  # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = const_max(l[:, :, :, :, nr_mix : 2 * nr_mix], -7.0)
    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix : 3 * nr_mix])
    x = torch.reshape(x, xs + [1]) + torch.zeros(xs + [nr_mix]).to(
        x.device
    )  # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = torch.reshape(
        means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :],
        [xs[0], xs[1], xs[2], 1, nr_mix],
    )
    m3 = torch.reshape(
        means[:, :, :, 2, :]
        + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :]
        + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :],
        [xs[0], xs[1], xs[2], 1, nr_mix],
    )
    means = torch.cat(
        [torch.reshape(means[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix]), m2, m3],
        dim=3,
    )
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    if low_bit:
        plus_in = inv_stdv * (centered_x + 1.0 / 31.0)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 31.0)
    else:
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = torch.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(
        plus_in
    )  # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -F.softplus(
        min_in
    )  # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = (
        mid_in - log_scales - 2.0 * F.softplus(mid_in)
    )  # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    if low_bit:
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(
                x > 0.999,
                log_one_minus_cdf_min,
                torch.where(
                    cdf_delta > 1e-5,
                    torch.log(const_max(cdf_delta, 1e-12)),
                    log_pdf_mid - np.log(15.5),
                ),
            ),
        )
    else:
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(
                x > 0.999,
                log_one_minus_cdf_min,
                torch.where(
                    cdf_delta > 1e-5,
                    torch.log(const_max(cdf_delta, 1e-12)),
                    log_pdf_mid - np.log(127.5),
                ),
            ),
        )
    log_probs = log_probs.sum(dim=3) + log_prob_from_logits(logit_probs)
    mixture_probs = torch.logsumexp(log_probs, -1)
    return -1.0 * mixture_probs.sum(dim=[1, 2]) / np.prod(xs[1:])


def sample_from_discretized_mix_logistic(l, nr_mix, return_scale=False, t=None):
    ls = [s for s in l.shape]
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    eps = torch.empty(logit_probs.shape, device=l.device).uniform_(1e-5, 1.0 - 1e-5)
    amax = torch.argmax(logit_probs - torch.log(-torch.log(eps)), dim=3)
    sel = F.one_hot(amax, num_classes=nr_mix).float()
    sel = torch.reshape(sel, xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = (l[:, :, :, :, :nr_mix] * sel).sum(dim=4)
    log_scales = const_max((l[:, :, :, :, nr_mix : nr_mix * 2] * sel).sum(dim=4), -7.0)
    coeffs = (torch.tanh(l[:, :, :, :, nr_mix * 2 : nr_mix * 3]) * sel).sum(dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.empty(means.shape, device=means.device).uniform_(1e-5, 1.0 - 1e-5)
    if t is not None:
        log_scales = log_scales + torch.tensor(t).to(log_scales.device).log()
    x = means + log_scales.exp() * (torch.log(u) - torch.log(1.0 - u))
    x0 = const_min(const_max(x[:, :, :, 0], -1.0), 1.0)
    x1 = const_min(const_max(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.0), 1.0)
    x2 = const_min(
        const_max(
            x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.0
        ),
        1.0,
    )
    sample = torch.cat(
        [
            torch.reshape(x0, xs[:-1] + [1]),
            torch.reshape(x1, xs[:-1] + [1]),
            torch.reshape(x2, xs[:-1] + [1]),
        ],
        dim=3,
    )
    if return_scale:
        return sample, log_scales.exp()
    else:
        return sample


def mean_discretized_mix_logistic(l, nr_mix, mask="soft", return_scale=False):
    # no sampling in observation space
    ls = [s for s in l.shape]
    xs = ls[:-1] + [3]
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    if mask == "soft":
        # soft mask of mixture components according to their probs
        sel = log_prob_from_logits(logit_probs).exp().unsqueeze(-2)
    elif mask == "hard":
        # hard mask select most likely mixture component
        amax = torch.argmax(logit_probs, dim=3)  # no sampling
        sel = F.one_hot(amax, num_classes=nr_mix).float()
        sel = torch.reshape(sel, xs[:-1] + [1, nr_mix])
    elif "top" in mask:
        top_k = int(mask[-1])
        assert top_k < nr_mix, "invalid top_k"
        # sel = log_prob_from_logits(logit_probs).exp().unsqueeze(-2)
        v, _ = torch.sort(logit_probs, descending=True, dim=-1)
        logit_probs[
            (logit_probs < v[..., top_k - 1][..., None])
        ] -= np.inf  # turn off not top k
        sel = (
            log_prob_from_logits(logit_probs).exp().unsqueeze(-2)
        )  # renormalise top k mixture components
        # print(sel[0,16,16,:])
    else:
        NotImplementedError
    means = (l[:, :, :, :, :nr_mix] * sel).sum(dim=4)
    log_scales = const_max((l[:, :, :, :, nr_mix : nr_mix * 2] * sel).sum(dim=4), -7.0)
    coeffs = (torch.tanh(l[:, :, :, :, nr_mix * 2 : nr_mix * 3]) * sel).sum(dim=4)
    x = means  # no sampling
    x0 = const_min(const_max(x[:, :, :, 0], -1.0), 1.0)
    x1 = const_min(const_max(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.0), 1.0)
    x2 = const_min(
        const_max(
            x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.0
        ),
        1.0,
    )
    sample = torch.cat(
        [
            torch.reshape(x0, xs[:-1] + [1]),
            torch.reshape(x1, xs[:-1] + [1]),
            torch.reshape(x2, xs[:-1] + [1]),
        ],
        dim=3,
    )
    if return_scale:
        return sample, log_scales.exp()
    else:
        return sample


class DmolNet(nn.Module):
    def __init__(self, args):
        super(DmolNet, self).__init__()
        self.width = args.widths[0]
        self.num_mixtures = 10
        self.conv = nn.Conv2d(
            self.width, self.num_mixtures * 10, kernel_size=1, stride=1, padding=0
        )
        self.mask = "soft"

    def forward(self, h):
        return self.conv(h).permute(0, 2, 3, 1)  # channels last

    def nll(self, h, x):
        return discretized_mix_logistic_loss(x=x.permute(0, 2, 3, 1), l=self.forward(h))

    def sample(self, h, return_loc=True, t=None):
        xhat = self.forward(h)
        if return_loc:
            x, scale = mean_discretized_mix_logistic(
                xhat, self.num_mixtures, return_scale=True, mask=self.mask
            )
        else:
            x, scale = sample_from_discretized_mix_logistic(
                xhat, self.num_mixtures, return_scale=True, t=t
            )
        x = x.clamp(min=-1, max=1).permute(0, 3, 1, 2)  # channels first
        return x, scale.permute(0, 3, 1, 2)
