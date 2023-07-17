from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as dist
import torch.distributions.transforms as T
from torch import Tensor, nn

from hps import Hparams

EPS = -9
EPS_z = -9
enc_act = nn.LeakyReLU()
dec_act = nn.ReLU()


@torch.jit.script
def gaussian_kl(q_loc, q_logscale, p_loc, p_logscale):
    return (
        -0.5
        + p_logscale
        - q_logscale
        + 0.5
        * (q_logscale.exp().pow(2) + (q_loc - p_loc).pow(2))
        / p_logscale.exp().pow(2)
    )


@torch.jit.script
def sample_gaussian(loc, logscale):
    return loc + logscale.exp() * torch.randn_like(loc)


class Encoder(nn.Module):
    def __init__(self, args: Hparams):
        super().__init__()
        n_channels = args.hidden_dim // 4
        self.conv = nn.Sequential(
            nn.Conv2d(
                args.input_channels, n_channels, kernel_size=5, stride=2, padding=1
            ),  # 16x16
            enc_act,
            nn.Conv2d(
                n_channels, n_channels, kernel_size=3, stride=2, padding=1
            ),  # 8x8
            enc_act,
            nn.Conv2d(
                n_channels, n_channels, kernel_size=3, stride=2, padding=1
            ),  # 4x4
            enc_act,
        )
        self.fc = nn.Sequential(nn.Linear(n_channels * 4 * 4, args.hidden_dim), enc_act)
        self.embed = nn.Sequential(
            nn.Linear(args.hidden_dim + args.context_dim, args.hidden_dim), enc_act
        )
        self.z_loc = nn.Linear(args.hidden_dim, args.z_dim)
        self.z_logscale = nn.Linear(args.hidden_dim, args.z_dim)

    def forward(
        self, x: Tensor, y: Tensor, t: Optional[float] = None
    ) -> Tuple[Tensor, Tensor]:
        x = self.conv(x).reshape(x.size(0), -1)
        x = self.fc(x)
        if len(y.shape) > 2:
            y = y[:, :, 0, 0]
        x = self.embed(torch.cat((x, y), dim=-1))
        loc, logscale = self.z_loc(x), self.z_logscale(x).clamp(min=EPS_z)
        if t is not None:
            logscale = logscale + torch.tensor(t).to(x.device).log()
        return loc, logscale


class CondPrior(nn.Module):
    def __init__(self, args: Hparams):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(args.context_dim, args.hidden_dim),
            enc_act,
            nn.Linear(args.hidden_dim, args.hidden_dim),
            enc_act,
        )
        self.z_loc = nn.Linear(args.hidden_dim, args.z_dim)
        self.z_logscale = nn.Linear(args.hidden_dim, args.z_dim)
        self.p_feat = nn.Linear(args.hidden_dim, args.z_dim)

        nn.init.zeros_(self.z_loc.weight)
        nn.init.zeros_(self.z_loc.bias)
        nn.init.zeros_(self.z_logscale.weight)
        nn.init.zeros_(self.z_logscale.bias)

    def forward(
        self, y: Tensor, t: Optional[float] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if len(y.shape) > 2:
            y = y[:, :, 0, 0]
        y = self.fc(y)
        loc, logscale = self.z_loc(y), self.z_logscale(y).clamp(min=EPS_z)
        if t is not None:
            logscale = logscale + torch.tensor(t).to(y.device).log()
        return loc, logscale, self.p_feat(y)


class DGaussNet(nn.Module):
    def __init__(self, args: Hparams):
        super(DGaussNet, self).__init__()
        self.x_loc = nn.Conv2d(
            args.widths[0], args.input_channels, kernel_size=1, stride=1
        )
        self.x_logscale = nn.Conv2d(
            args.widths[0], args.input_channels, kernel_size=1, stride=1
        )

        if args.std_init > 0:  # if std_init=0, we random init weights for diag cov
            nn.init.zeros_(self.x_logscale.weight)
            nn.init.constant_(self.x_logscale.bias, np.log(args.std_init))

            covariance = args.x_like.split("_")[0]
            if covariance == "fixed":
                self.x_logscale.weight.requires_grad = False
                self.x_logscale.bias.requires_grad = False
            elif covariance == "shared":
                self.x_logscale.weight.requires_grad = False
                self.x_logscale.bias.requires_grad = True
            elif covariance == "diag":
                self.x_logscale.weight.requires_grad = True
                self.x_logscale.bias.requires_grad = True
            else:
                NotImplementedError(f"{args.x_like} not implemented.")

    def forward(self, h: Tensor, t: Optional[float] = None) -> Tensor:
        loc, logscale = self.x_loc(h), self.x_logscale(h).clamp(min=EPS)
        if t is not None:
            logscale = logscale + torch.tensor(t).to(h.device).log()
        return loc, logscale

    def approx_cdf(self, x: Tensor) -> Tensor:
        return 0.5 * (
            1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
        )

    def nll(self, h: Tensor, x: Tensor) -> Tensor:
        loc, logscale = self.forward(h)
        centered_x = x - loc
        inv_stdv = torch.exp(-logscale)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = self.approx_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = self.approx_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(
                x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))
            ),
        )
        return -1.0 * log_probs.mean(dim=(1, 2, 3))

    def sample(
        self, h: Tensor, return_loc: bool = True, t: Optional[float] = None
    ) -> Tuple[Tensor, Tensor]:
        if return_loc:
            x, logscale = self.forward(h)
        else:
            loc, logscale = self.forward(h, t)
            x = loc + torch.exp(logscale) * torch.randn_like(loc)
        x = torch.clamp(x, min=-1.0, max=1.0)
        return x, logscale.exp()


class GaussNet(nn.Module):
    def __init__(self, args: Hparams):
        super(GaussNet, self).__init__()
        self.x_loc = nn.Conv2d(
            args.widths[0], args.input_channels, kernel_size=1, stride=1
        )
        self.x_logscale = nn.Conv2d(
            args.widths[0], args.input_channels, kernel_size=1, stride=1
        )

        if args.std_init > 0:  # if std_init=0, we random init weights for diag cov
            nn.init.zeros_(self.x_logscale.weight)
            nn.init.constant_(self.x_logscale.bias, np.log(args.std_init))

            covariance = args.x_like.split("_")[0]
            if covariance == "fixed":
                self.x_logscale.weight.requires_grad = False
                self.x_logscale.bias.requires_grad = False
            elif covariance == "shared":
                self.x_logscale.weight.requires_grad = False
                self.x_logscale.bias.requires_grad = True
            elif covariance == "diag":
                self.x_logscale.weight.requires_grad = True
                self.x_logscale.bias.requires_grad = True
            else:
                NotImplementedError(f"{args.x_like} not implemented.")

    def forward(
        self, h: Tensor, t: Optional[float] = None
    ) -> Union[Tensor, torch.distributions.Distribution]:
        loc, logscale = self.x_loc(h), self.x_logscale(h).clamp(min=EPS)
        if t is not None:
            logscale = logscale + torch.tensor(t).to(h.device).log()
        scale = torch.exp(logscale)
        if torch.isnan(loc).any() or torch.isnan(scale).any():
            # hacky way to return nans for skipping grad update during training
            return loc * float("nan")
        else:
            return dist.Independent(dist.Normal(loc, scale), 3)

    def nll(
        self, h: Tensor, x: Tensor
    ) -> Union[Tensor, torch.distributions.Distribution]:
        x_dist = self.forward(h)
        if isinstance(x_dist, torch.Tensor):  # when x_dist is just nans
            print("nan")
            return x_dist
        else:
            x = (x + 1.0) * 127.5  # [-1,1] back to [0,255]
            x = x + torch.rand_like(x)  # dequantize x target, [0,256]^D
            x = self.x_preprocess()(x)  # logit(alpha + (1 - alpha) * x / 256)
            # x = torch.logit(x / 256, eps=1e-5)
            # per pixel nll
            return -1.0 * x_dist.log_prob(x) / np.prod(x.shape[1:])

    def sample(
        self, h: Tensor, return_loc: bool = True, t: Optional[float] = None
    ) -> Tuple[Tensor, Tensor]:
        x_dist = self.forward(h, t)
        x = x_dist.base_dist.loc if return_loc else x_dist.sample()
        x = self.x_preprocess().inv(x)  # (sigmoid(x) - alpha) / (1 - alpha) * 256
        x = torch.clamp((x - 128) / 128, min=-1.0, max=1.0)  # return in [-1,1]
        return x, x_dist.base_dist.scale

    def x_preprocess(self) -> torch.distributions.transforms.Transform:
        """(x + uniform_noise) pixel values are [0, 256]^D
        realnvp: model density of: logit(alpha + (1 - alpha) * x / 256)."""
        alpha, num_bits = 0.0, 8
        return T.ComposeTransform(
            [
                T.AffineTransform(0.0, (1.0 / 2**num_bits)),
                T.AffineTransform(alpha, (1 - alpha)),
                T.SigmoidTransform().inv,
            ]
        )


class Decoder(nn.Module):
    def __init__(self, args: Hparams):
        super().__init__()
        self.cond_prior = args.cond_prior
        in_width = args.z_dim + args.context_dim
        if self.cond_prior:
            self.prior = CondPrior(args)
            in_width += args.z_dim
        else:
            self.register_buffer("p_loc", torch.zeros(1, args.z_dim))
            self.register_buffer("p_scale", torch.ones(1, args.z_dim))

        n_channels = args.hidden_dim // 4
        self.fc = nn.Sequential(
            nn.Linear(in_width, args.hidden_dim),
            dec_act,
            nn.Linear(args.hidden_dim, n_channels * 4 * 4),
            dec_act,
        )

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 8x8
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1),
            dec_act,
            nn.Upsample(scale_factor=2, mode="nearest"),  # 16x16
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1),
            dec_act,
            nn.Upsample(scale_factor=2, mode="nearest"),  # 32x32
            nn.Conv2d(n_channels, 16, kernel_size=5, stride=1, padding=2),
            dec_act,
        )

    def forward(self, y: Tensor, z: Optional[Tensor] = None, t: Optional[float] = None):
        if len(y.shape) > 2:
            y = y[:, :, 0, 0]

        if self.training and self.cond_prior:
            p1, p2 = self.drop_cond()
        else:
            p1, p2 = (1, 1)
        y_drop1 = y.clone()
        y_drop1[:, 2:] = y_drop1[:, 2:] * p1
        y_drop2 = y.clone()
        y_drop2[:, 2:] = y_drop2[:, 2:] * p2

        if self.cond_prior:
            p_loc, p_logscale, p_feat = self.prior(y_drop1, t)
        else:
            p_loc = self.p_loc.repeat(y.shape[0], 1)
            p_logscale = self.p_scale.log().repeat(y.shape[0], 1)
            if t is not None:
                p_logscale = p_logscale + torch.tensor(t).to(y.device).log()

        if z is None:  # random sampling
            z = sample_gaussian(p_loc, p_logscale)

        if self.cond_prior:
            z = torch.cat((p_feat, z), dim=-1)

        x = torch.cat((z, y_drop2), dim=-1)
        x = self.fc(x).reshape(x.size(0), -1, 4, 4)
        return self.conv(x), (p_loc, p_logscale)

    def drop_cond(self) -> Tuple[int, int]:
        opt = dist.Categorical(1 / 3 * torch.ones(3)).sample()
        if opt == 0:
            p1, p2 = 0, 1
        elif opt == 1:
            p1, p2 = 1, 0
        elif opt == 2:
            p1, p2 = 1, 1
        return p1, p2


class VAE(nn.Module):
    def __init__(self, args: Hparams):
        super().__init__()
        args.hidden_dim = 128
        self.cond_prior = args.cond_prior
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        x_dist = args.x_like.split("_")[1]
        if x_dist == "gauss":
            self.likelihood = GaussNet(args)
        elif x_dist == "dgauss":
            self.likelihood = DGaussNet(args)
        elif x_dist == "dmol":
            from dmol import DmolNet

            self.likelihood = DmolNet(args)
        else:
            NotImplementedError(f"{args.x_like} not implemented.")

    def forward(self, x: Tensor, parents: Tensor, beta: int = 1) -> Dict[str, Tensor]:
        q_loc, q_logscale = self.encoder(x, y=parents)
        z = sample_gaussian(q_loc, q_logscale)
        h, prior = self.decoder(y=parents, z=z)
        p_loc, p_logscale = prior
        nll_pp = self.likelihood.nll(h, x)  # per pixel
        kl_pp = gaussian_kl(q_loc, q_logscale, p_loc, p_logscale)
        kl_pp = kl_pp.sum(dim=-1) / np.prod(x.shape[1:])  # per pixel
        elbo = nll_pp.mean() + beta * kl_pp.mean()  # negative elbo (free energy)
        return dict(elbo=elbo, nll=nll_pp.mean(), kl=kl_pp.mean())

    def sample(
        self, parents: Tensor, return_loc: bool = True, t: Optional[float] = None
    ):
        h, _ = self.decoder(y=parents, t=t)
        return self.likelihood.sample(h, return_loc, t=t)

    def abduct(
        self,
        x: Tensor,
        parents: Tensor,
        cf_parents: Optional[Tensor] = None,
        alpha: float = 0.5,
        t: Optional[float] = None,
    ) -> List[Tensor]:
        q_loc, q_logscale = self.encoder(x, y=parents)  # q(z|x,pa)
        z = sample_gaussian(q_loc, q_logscale)

        if self.cond_prior:
            q_stats = {"z": z, "q_loc": q_loc, "q_logscale": q_logscale}
            if cf_parents is None:
                # part of abduction for z* if conditional prior
                return [q_stats]
            else:
                p_loc, p_logscale, p_feat = self.decoder.prior(
                    cf_parents, t
                )  # p(z|pa*)

                q_scale = q_logscale.exp()
                u = (z - q_loc) / q_scale  # abduct exogenouse noise u ~ N(0,I)
                p_var = p_logscale.exp().pow(2)  # p(z|pa*)

                # Option1: mixture distribution
                # r(z|x,pa,pa*) = a*q(z|x,pa) + (1-a)*p(z|pa*)
                r_loc = alpha * q_loc + (1 - alpha) * p_loc
                # assumes independence
                r_var = alpha * q_scale.pow(2) + (1 - alpha) * p_var
                # r_var = a*(q_loc.pow(2) + q_var) + (1-a)*(p_loc.pow(2) + p_var) - r_loc.pow(2)

                # # Option 2: precision weighted distribution
                # q_prec = 1 / q_scale.pow(2)
                # p_prec = 1 / p_var
                # joint_prec = q_prec + p_prec
                # r_loc = (q_loc * q_prec + p_loc * p_prec) / joint_prec
                # r_var = 1 / joint_prec

                # sample: z* ~ r(z|x,pa,pa*)
                r_scale = r_var.sqrt()
                if t is not None:
                    r_scale = r_scale * torch.tensor(t).to(x.device)
                return [r_loc + r_scale * u]  # inferred z*
        else:  # z if exogenous prior
            return [z.detach()]

    def forward_latents(
        self,
        latents: List[Tensor],
        parents: Tensor,
        return_loc: bool = True,
        t: Optional[float] = None,
    ):
        h, _ = self.decoder(y=parents, z=latents[0], t=t)
        return self.likelihood.sample(h, return_loc, t=t)
