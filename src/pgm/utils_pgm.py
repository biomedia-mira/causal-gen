import os
import sys
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from layers import TraceStorage_ELBO
from matplotlib import colors
from torch import Tensor, nn

sys.path.append("..")
from hps import Hparams

# plt.rcParams['figure.facecolor'] = 'white'


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max([np.abs(self.vmin), np.abs(self.vmax)])
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def check_nan(input_dict: Dict[str, Tensor]):
    nans = 0
    for k, v in input_dict.items():
        k_nans = torch.isnan(v).sum()
        nans += k_nans
        if k_nans > 0:
            print(f"\nFound {k_nans} nan(s) in {k}, skipping step.")
    return nans


def update_stats(stats: Dict[str, Any], elbo_fn: TraceStorage_ELBO):
    """Accumulate tracked summary statistics."""

    def _update(trace, dist="p"):
        for name, node in trace.nodes.items():
            if node["type"] == "sample":
                k = "log" + dist + "(" + name + ")"
                if k not in stats:
                    stats[k] = 0
                stats[k] += node["log_prob"].sum().item()
        return stats

    _update(elbo_fn.trace_storage["model"], dist="p")
    _update(elbo_fn.trace_storage["guide"], dist="q")
    return stats


def plot_joint(
    args: Hparams, model: nn.Module, dataset: torch.utils.data.Dataset, step: int
):
    plt.close("all")
    if "ukbb" in args.dataset:
        NotImplementedError
    elif args.dataset == "morphomnist":
        file = os.path.join(args.save_dir, "joint_data.pdf")
        if not os.path.exists(file):
            df = pd.DataFrame(
                {
                    "thickness": dataset.metrics["thickness"],
                    "intensity": dataset.metrics["intensity"],
                    # 'digit': np.argmax(dataset.labels, axis=-1)
                }
            )
            sns.jointplot(data=df, x="thickness", y="intensity")  # , kind='kde')
            plt.suptitle("Data Joint")
            plt.tight_layout()
            plt.savefig(file)

        with torch.no_grad():
            samples = model.sample(n_samples=10000)
        df = pd.DataFrame(
            {
                "thickness": samples["thickness"].squeeze().cpu(),
                "intensity": samples["intensity"].squeeze().cpu(),
            }
        )
        sns.jointplot(data=df, x="thickness", y="intensity")  # , kind='kde')
        plt.suptitle(f"Model Joint (step {step})")
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, f"joint_model_{step}.pdf"))


def plot(
    x,
    fig=None,
    ax=None,
    nrows=1,
    cmap="Greys_r",
    norm=None,
    cbar=False,
    set_cbar_ticks=True,
):
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

    # plt.tight_layout()

    if cbar:
        if fig:
            fig.subplots_adjust(wspace=-0.275, hspace=0.25)
        for i in range(m):
            for j in range(n):
                idx = [i, j] if m > 1 else j
                # cbar_ax = fig.add_axes([
                #     ax[idx].get_position().x0 + 0.0025, # left
                #     ax[idx].get_position().y1, # bottom
                #     0.003, # width
                #     ax[idx].get_position().height # height
                # ])
                cbar_ax = fig.add_axes(
                    [
                        ax[idx].get_position().x0,
                        ax[idx].get_position().y0 - 0.015,
                        ax[idx].get_position().width,
                        0.0075,
                    ]
                )
                cbar = plt.colorbar(
                    im[i * n + j], cax=cbar_ax, orientation="horizontal"
                )  # , ticks=mticker.MultipleLocator(25)) #, ticks=mticker.AutoLocator())
                # cbar.ax.tick_params(rotation=0)
                # cbar.ax.locator_params(nbins=5)
                _x = x[i * n + j].squeeze()

                if set_cbar_ticks:
                    d = 20
                    _vmin, _vmax = _x.min().abs().item(), _x.max().item()
                    _vmin = -(_vmin - (_vmin % d))
                    _vmax = _vmax - (_vmax % d)
                    lt = [_vmin, 0, _vmax]

                    if (np.abs(_vmin) - 0) > d or (_vmax - 0) > d:
                        lt.insert(1, _vmin // 2)
                        lt.insert(-2, _vmax // 2)
                    cbar.set_ticks(lt)
                else:
                    cbar.ax.locator_params(nbins=5)
                    cbar.formatter.set_powerlimits((0, 0))

                cbar.outline.set_visible(False)
    return fig, ax


@torch.no_grad()
def plot_cf(
    x: Tensor,
    cf_x: Tensor,
    pa: Dict[str, Tensor],
    cf_pa: Dict[str, Tensor],
    do: Dict[str, Tensor],
    var_cf_x: Optional[Tensor],
    num_images: int = 8,
):
    n = num_images  # 8 columns
    x = (x[:n].detach().cpu() + 1) * 127.5
    cf_x = (cf_x[:n].detach().cpu() + 1) * 127.5

    fs = 24  # font size
    pad = 8
    m = 3 if var_cf_x is None else 4  # nrows
    s = 5
    fig, ax = plt.subplots(m, n, figsize=(n * s - 6, m * s), facecolor="white")
    # fig, ax = plt.subplots(m, n, figsize=(n*s, m*s+2))
    _, _ = plot(x, ax=ax[0])
    _, _ = plot(cf_x, ax=ax[1])
    _, _ = plot(
        cf_x - x,
        ax=ax[2],
        fig=fig,
        cmap="RdBu_r",
        cbar=True,
        norm=MidpointNormalize(midpoint=0),
    )
    if var_cf_x is not None:
        _, _ = plot(
            var_cf_x[:n].clamp(min=0).detach().sqrt().cpu(),
            fig=fig,
            cmap="jet",
            ax=ax[3],
            cbar=True,
            set_cbar_ticks=False,
        )

    sex_categories = ["female", "male"]  # 0,1
    mriseq_categories = ["T1", "T2"]  # 0,1

    for j in range(n):
        msg = r"$do($"
        for i, (k, v) in enumerate(do.items()):
            if k == "sex":
                vv = sex_categories[int(v[j].item())]
                kk = "s"
            elif k == "mri_seq":
                vv = mriseq_categories[int(v[j].item())]
                kk = "m"
            elif k == "age":
                vv = str(v[j].item())
                kk = "a"
            else:
                vv = str(np.round(v[j].item() / 1000, 1))
                kk = k[0]
            msg += rf"${kk}{{=}}$" + f"{vv}"
            msg += ", " if (i + 1) < len(list(do.keys())) else ""

        ax[1, j].set_title(msg + r"$)$", fontsize=fs - 2, pad=pad + 4)

        s = str(sex_categories[int(pa["sex"][j].item())])
        m = str(mriseq_categories[int(pa["mri_seq"][j].item())])
        a = str(int(pa["age"][j].item()))
        b = str(np.round(pa["brain_volume"][j].item() / 1000, 1))  # ml
        v = str(np.round(pa["ventricle_volume"][j].item() / 1000, 1))  # ml

        # ax[0,j].set_title(rf'$a{{=}}{a}, \ s{{=}}{s}, \ m{{=}}{m}$' +'\n'+ rf'$b{{=}}{b}, \ v{{=}}{v}$',
        #                 pad=pad, fontsize=fs-4, multialignment='center', linespacing=1.5)

        ax[0, j].set_title(
            rf"$m{{=}}$"
            + f"{m}"
            + "$, \ a{{=}}$"
            + f"{a}"
            + rf"$, \ s{{=}}$"
            + f"{s}"
            + "\n"
            + rf"$b{{=}}{b}, \ v{{=}}{v}$",
            pad=pad + 2,
            fontsize=fs - 8,
            multialignment="center",
            linespacing=1.5,
        )

        # plot counterfactual
        cf_s = str(sex_categories[int(cf_pa["sex"][j].item())])
        cf_m = str(mriseq_categories[int(cf_pa["mri_seq"][j].item())])
        cf_a = str(np.round(cf_pa["age"][j].item(), 1))
        cf_b = str(np.round(cf_pa["brain_volume"][j].item() / 1000, 1))  # ml
        cf_v = str(np.round(cf_pa["ventricle_volume"][j].item() / 1000, 1))  # ml

        # ax[1, j].set_xlabel(rf'$\widetilde{{a}}{{=}}{cf_a}, \ \widetilde{{s}}{{=}}{cf_s}, \ \widetilde{{m}}{{=}}{cf_m}$' +'\n'+
        #     rf'$\widetilde{{b}}{{=}}{cf_b}, \ \widetilde{{v}}{{=}}{cf_v}$',
        #                 labelpad=pad+1, fontsize=fs-4, multialignment='center', linespacing=1.25)

        ax[1, j].set_xlabel(
            rf"$\widetilde{{m}}{{=}}$"
            + f"{cf_m}"
            + "$, \ \widetilde{{a}}{{=}}$"
            + f"{cf_a}"
            rf"$, \ \widetilde{{s}}{{=}}$"
            + f"{cf_s}"
            + "\n"
            + rf"$\widetilde{{b}}{{=}}{cf_b}, \ \widetilde{{v}}{{=}}{cf_v}$",
            labelpad=pad + 4,
            fontsize=fs - 8,
            multialignment="center",
            linespacing=1.25,
        )

    ax[0, 0].set_ylabel("Observation", fontsize=fs + 4, labelpad=pad)
    ax[1, 0].set_ylabel("Counterfactual", fontsize=fs + 4, labelpad=pad)
    ax[2, 0].set_ylabel("Direct Effect", fontsize=fs + 4, labelpad=pad)
    if var_cf_x is not None:
        ax[3, 0].set_ylabel("Uncertainty", fontsize=fs + 4, labelpad=pad)
    return fig
