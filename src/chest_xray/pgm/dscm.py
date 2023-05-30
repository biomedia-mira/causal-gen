import torch
import torch.nn as nn

from utils_pgm import check_nan, calculate_loss


class DSCM(nn.Module):
    def __init__(self, args, pgm, predictor, vae):
        super().__init__()
        self.args = args
        self.pgm = pgm
        self.pgm.eval()
        self.pgm.requires_grad = False
        self.predictor = predictor
        self.predictor.eval()
        self.predictor.requires_grad = False
        self.vae = vae
        # lagrange multiplier
        self.lmbda = nn.Parameter(args.lmbda_init * torch.ones(1))
        self.register_buffer("eps", args.elbo_constraint * torch.ones(1))

    def forward(self, obs, do, elbo_fn=None, cf_particles=1):
        pa = {k: v for k, v in obs.items() if k != "x"}
        # forward vae with factual parents
        _pa = vae_preprocess(self.args, {k: v.clone() for k, v in pa.items()})
        vae_out = self.vae(obs["x"], _pa, beta=self.args.beta)

        if cf_particles > 1:  # for calculating counterfactual variance
            cfs = {"x": torch.zeros_like(obs["x"])}
            cfs.update({"x2": torch.zeros_like(obs["x"])})

        for _ in range(cf_particles):
            # forward pgm, get counterfactual parents
            cf_pa = self.pgm.counterfactual(obs=pa, intervention=do, num_particles=1)
            _cf_pa = vae_preprocess(self.args, {k: v.clone() for k, v in cf_pa.items()})
            # forward vae with counterfactual parents
            zs = self.vae.abduct(obs["x"], parents=_pa)  # z ~ q(z|x,pa)
            cf_loc, cf_scale = self.vae.forward_latents(zs, parents=_cf_pa)
            rec_loc, rec_scale = self.vae.forward_latents(zs, parents=_pa)
            u = (obs["x"] - rec_loc) / rec_scale.clamp(min=1e-12)
            cf_x = torch.clamp(cf_loc + cf_scale * u, min=-1, max=1)

            if cf_particles > 1:
                cfs["x"] += cf_x
                with torch.no_grad():
                    cfs["x2"] += cf_x**2
            else:
                cfs = {"x": cf_x}

        # Var[X] = E[X^2] - E[X]^2
        if cf_particles > 1:
            with torch.no_grad():
                var_cf_x = (cfs["x2"] - cfs["x"] ** 2 / cf_particles) / cf_particles
                cfs.pop("x2", None)
            cfs["x"] = cfs["x"] / cf_particles
        else:
            var_cf_x = None

        cfs.update(cf_pa)
        if check_nan(vae_out) > 0 or check_nan(cfs) > 0:
            return {"loss": torch.tensor(float("nan"))}

        # calculate sup loss outside pyro
        pred_batch = self.predictor.predict_unnorm(**cfs)
        aux_loss = calculate_loss(
            pred_batch=pred_batch, target_batch=cfs, loss_norm="l2"
        )

        # aux_loss = elbo_fn.differentiable_loss(
        #     self.predictor.model_anticausal,
        #     self.predictor.guide_pass, **cfs
        # ) / cfs['x'].shape[0]

        with torch.no_grad():
            sg = self.eps - vae_out["elbo"]
        damp = self.args.damping * sg
        loss = aux_loss - (self.lmbda - damp) * (self.eps - vae_out["elbo"])

        out = {}
        out.update(vae_out)
        out.update(
            {
                "loss": loss,
                "aux_loss": aux_loss,
                "cfs": cfs,
                "var_cf_x": var_cf_x,
                "cf_pa": cf_pa,
            }
        )
        return out


def vae_preprocess(args, pa):
    pa = torch.cat([pa[k] for k in args.parents_x], dim=1)
    pa = pa[..., None, None].repeat(1, 1, *(args.input_res,) * 2).cuda().float()
    return pa
