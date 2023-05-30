import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from matplotlib import colors


def seed_all(seed, deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def linear_warmup(warmup_iters):
    def f(iter):
        return 1.0 if iter > warmup_iters else iter / warmup_iters

    return f


def beta_anneal(beta, step, anneal_steps):
    return min(beta, (max(1e-11, step) / anneal_steps) ** 2)


def normalize(x, zero_one=False):
    x_min = x.min()
    x_max = x.max()
    print(f"max: {x_max}, min: {x_min}")
    x = (x - x_min) / (x_max - x_min)  # [0,1]
    return x if zero_one else 2 * x - 1  # else [-1,1]


def log_standardize(x):
    log_x = torch.log(x.clamp(min=1e-12))
    return (log_x - log_x.mean()) / log_x.std().clamp(min=1e-12)  # mean=0, std=1


def exists(val):
    return val is not None


def is_float_dtype(dtype):
    return any(
        [
            dtype == float_dtype
            for float_dtype in (
                torch.float64,
                torch.float32,
                torch.float16,
                torch.bfloat16,
            )
        ]
    )


def clamp(value, min_value=None, max_value=None):
    assert exists(min_value) or exists(max_value)
    if exists(min_value):
        value = max(value, min_value)

    if exists(max_value):
        value = min(value, max_value)

    return value


class EMA(nn.Module):
    """
    Adapted from: https://github.com/lucidrains/ema-pytorch/blob/main/ema_pytorch/ema_pytorch.py
    Implements exponential moving average shadowing for your model.

    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your specified beta.

    @crowsonkb's notes on EMA Warmup:

    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """

    def __init__(
        self,
        model,
        beta=0.999,
        update_after_step=100,
        update_every=1,
        inv_gamma=1.0,
        power=1.0,
        min_value=0.0,
        param_or_buffer_names_no_ema=set(),
    ):
        super().__init__()
        self.beta = beta
        self.online_model = model

        try:
            self.ema_model = deepcopy(model)
        except:
            print(
                "Your model was not copyable. Please make sure you are not using any LazyLinear"
            )
            exit()

        self.ema_model.requires_grad_(False)
        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = (
            param_or_buffer_names_no_ema  # parameter or buffer
        )

        self.register_buffer("initted", torch.Tensor([False]))
        self.register_buffer("step", torch.tensor([0]))

    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def copy_params_from_model_to_ema(self):
        for ma_params, current_params in zip(
            list(self.ema_model.parameters()), list(self.online_model.parameters())
        ):
            if not is_float_dtype(current_params.dtype):
                continue

            ma_params.data.copy_(current_params.data)

        for ma_buffers, current_buffers in zip(
            list(self.ema_model.buffers()), list(self.online_model.buffers())
        ):
            if not is_float_dtype(current_buffers.dtype):
                continue

            ma_buffers.data.copy_(current_buffers.data)

    def get_current_decay(self):
        epoch = clamp(self.step.item() - self.update_after_step - 1, min_value=0.0)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power

        if epoch <= 0:
            return 0.0

        return clamp(value, min_value=self.min_value, max_value=self.beta)

    def update(self):
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.Tensor([True]))

        self.update_moving_average(self.ema_model, self.online_model)

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        current_decay = self.get_current_decay()

        for (name, current_params), (_, ma_params) in zip(
            list(current_model.named_parameters()), list(ma_model.named_parameters())
        ):
            if not is_float_dtype(current_params.dtype):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_params.data.copy_(current_params.data)
                continue

            difference = ma_params.data - current_params.data
            difference.mul_(1.0 - current_decay)
            ma_params.sub_(difference)

        for (name, current_buffer), (_, ma_buffer) in zip(
            list(current_model.named_buffers()), list(ma_model.named_buffers())
        ):
            if not is_float_dtype(current_buffer.dtype):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_buffer.data.copy_(current_buffer.data)
                continue

            difference = ma_buffer - current_buffer
            difference.mul_(1.0 - current_decay)
            ma_buffer.sub_(difference)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)


def vae_preprocess(args, pa):
    pa = torch.cat([pa[k] for k in args.parents_x], dim=1)
    pa = pa[..., None, None].repeat(1, 1, *(args.input_res,) * 2).cuda().float()
    return pa


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max([np.abs(self.vmin), np.abs(self.vmax)])
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def preprocess(batch):
    for k, v in batch.items():
        if k == "x":
            batch["x"] = batch["x"].float().cuda()  # [0,1]
        elif k in ["age"]:
            batch[k] = batch[k].float().cuda()
        elif k in ["race"]:
            batch[k] = batch[k].float().cuda()
        #             print("batch[race]: ", batch[k])
        elif k in ["finding"]:
            batch[k] = batch[k].float().cuda()
            # batch[k] = F.one_hot(batch[k], num_classes=2).squeeze().float().cuda()
        else:
            batch[k] = batch[k].float().cuda()
    return batch


def plot(x, fig=None, ax=None, nrows=1, cmap="Greys_r", norm=None, cbar=False):
    m, n = nrows, x.shape[0] // nrows
    if ax is None:
        fig, ax = plt.subplots(m, n, figsize=(n * 4, 8))
    im = []
    for i in range(m):
        for j in range(n):
            idx = (i, j) if m > 1 else j
            ax = [ax] if n == 1 else ax
            _x = x[i * n + j].squeeze()
            if norm is not None:
                norm = MidpointNormalize(vmin=_x.min(), midpoint=0, vmax=_x.max())
                # norm = colors.TwoSlopeNorm(vmin=_x.min(), vcenter=0., vmax=_x.max())
            _im = ax[idx].imshow(_x, cmap=cmap, norm=norm)
            im.append(_im)
            ax[idx].axes.xaxis.set_ticks([])
            ax[idx].axes.yaxis.set_ticks([])

    plt.tight_layout()

    if cbar:
        if fig:
            # fig.subplots_adjust(wspace=-0.525, hspace=0.3)
            fig.subplots_adjust(wspace=-0.3, hspace=0.3)
        for i in range(m):
            for j in range(n):
                idx = [i, j] if m > 1 else j
                cbar_ax = fig.add_axes(
                    [
                        ax[idx].get_position().x0,
                        ax[idx].get_position().y0 - 0.02,
                        ax[idx].get_position().width,
                        0.01,
                    ]
                )
                # , ticks=mticker.MultipleLocator(25)) #, ticks=mticker.AutoLocator())
                cbar = plt.colorbar(
                    im[i * n + j], cax=cbar_ax, orientation="horizontal"
                )
                _x = x[i * n + j].squeeze()

                d = 20
                _vmin, _vmax = _x.min().abs().item(), _x.max().item()
                _vmin = -(_vmin - (_vmin % d))
                _vmax = _vmax - (_vmax % d)

                lt = [_vmin, 0, _vmax]

                if (np.abs(_vmin) - 0) > d:
                    lt.insert(1, _vmin // 2)
                if (_vmax - 0) > d:
                    lt.insert(-2, _vmax // 2)

                cbar.set_ticks(lt)
                cbar.outline.set_visible(False)
    return fig, ax


@torch.no_grad()
def plot_cf_rec(args, x, cf_x, pa, cf_pa, do, rec_loc):
    def undo_norm(pa):
        # reverse [-1,1] parent preprocessing back to original range
        for k, v in pa.items():
            if k == "age":
                pa[k] = (v + 1) / 2 * 100  # [-1,1] -> [0,100]
        return pa

    do = undo_norm(do)
    pa = undo_norm(pa)
    cf_pa = undo_norm(cf_pa)

    fs = 15
    m, s = 6, 3
    n = 8
    fig, ax = plt.subplots(m, n, figsize=(n * s - 2, m * s))
    x = (x[:n].detach().cpu() + 1) * 127.5
    _, _ = plot(x, ax=ax[0])

    cf_x = (cf_x[:n].detach().cpu() + 1) * 127.5
    rec_loc = (rec_loc[:n].detach().cpu() + 1) * 127.5
    _, _ = plot(rec_loc, ax=ax[1])
    _, _ = plot(cf_x, ax=ax[2])
    _, _ = plot(
        rec_loc - x,
        ax=ax[3],
        fig=fig,
        cmap="RdBu_r",
        cbar=True,
        norm=MidpointNormalize(midpoint=0),
    )
    _, _ = plot(
        cf_x - x,
        ax=ax[4],
        fig=fig,
        cmap="RdBu_r",
        cbar=True,
        norm=MidpointNormalize(midpoint=0),
    )
    _, _ = plot(
        cf_x - rec_loc,
        ax=ax[5],
        fig=fig,
        cmap="RdBu_r",
        cbar=True,
        norm=MidpointNormalize(midpoint=0),
    )
    sex_categories = ["female", "male"]  # 0,1
    race_categories = ["White", "Asian", "Black"]  # 0,1,2

    for j in range(n):
        msg = ""
        for i, (k, v) in enumerate(do.items()):
            if k == "sex":
                vv = sex_categories[int(v[j].item())]
                kk = "s"
            elif k == "age":
                vv = str(v[j].item())
                kk = "a"
            elif k == "race":
                vv = race_categories[int(torch.argmax(v[j], dim=-1))]
                kk = "r"
            msg += kk + "{{=}}" + vv
            msg += ", " if (i + 1) < len(list(do.keys())) else ""

        s = str(sex_categories[int(pa["sex"][j].item())])
        r = str(race_categories[int(torch.argmax(pa["race"][j], dim=-1))])
        a = str(int(pa["age"][j].item()))

        ax[0, j].set_title(
            rf"$a{{=}}{a}, \ s{{=}}{s}, \ r{{=}}{r}$",
            pad=8,
            fontsize=fs - 5,
            multialignment="center",
            linespacing=1.5,
        )
        ax[1, j].set_title("rec_loc")
        ax[2, j].set_title(rf"do(${msg}$)", fontsize=fs - 2, pad=8)
        ax[3, j].set_title("rec_loc - x")
        ax[4, j].set_title(
            "cf_loc - x",
            pad=8,
            fontsize=fs - 5,
            multialignment="center",
            linespacing=1.5,
        )
        ax[5, j].set_title("cf_loc - rec_loc")

    # plt.show()
    fig.savefig(
        os.path.join(args.save_dir, f"viz-{args.iter}.png"), bbox_inches="tight"
    )


def write_images(args, model, batch):
    bs, c, h, w = batch["x"].shape
    batch = preprocess(batch)
    # reconstructions
    zs = model.abduct(x=batch["x"], parents=batch["pa"])
    pa = {k: v for k, v in batch.items() if k != "x"}
    _pa = vae_preprocess(args, {k: v.clone() for k, v in pa.items()})

    rec_loc, _ = model.forward_latents(zs, parents=_pa)
    # counterfactuals (focus on changing sex)
    cf_pa = copy.deepcopy(pa)
    cf_pa["sex"] = 1 - cf_pa["sex"]
    do = {"sex": cf_pa["sex"]}
    _cf_pa = vae_preprocess(args, {k: v.clone() for k, v in cf_pa.items()})
    cf_loc, _ = model.forward_latents(zs, parents=_cf_pa)
    # plot this figure
    plot_cf_rec(args, batch["x"], cf_loc, pa, cf_pa, do, rec_loc)
