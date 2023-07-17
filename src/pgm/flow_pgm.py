from typing import Dict

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import torch
import torch.nn.functional as F
from layers import (  # fmt: skip
    CNN,
    MLP,
    ConditionalAffineTransform,
    ConditionalGumbelMax,
    ConditionalTransformedDistributionGumbelMax,
)
from pyro.distributions.conditional import ConditionalTransformedDistribution
from pyro.infer.reparam.transform import TransformReparam
from pyro.nn import DenseNN
from torch import Tensor, nn

from hps import Hparams


class BasePGM(nn.Module):
    def __init__(self):
        super().__init__()

    def scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg["fn"], dist.TransformedDistribution):
                return TransformReparam()
            else:
                return None

        return pyro.poutine.reparam(self.model, config=config)(*args, **kwargs)

    def sample_scm(self, n_samples: int = 1):
        with pyro.plate("obs", n_samples):
            samples = self.scm()
        return samples

    def sample(self, n_samples: int = 1):
        with pyro.plate("obs", n_samples):
            samples = self.model()  # NOTE: not ideal as model is defined in child class
        return samples

    def infer_exogeneous(self, obs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        batch_size = list(obs.values())[0].shape[0]
        # assuming that we use transformed distributions for everything:
        cond_model = pyro.condition(self.sample, data=obs)
        cond_trace = pyro.poutine.trace(cond_model).get_trace(batch_size)

        output = {}
        for name, node in cond_trace.nodes.items():
            if "z" in name or "fn" not in node.keys():
                continue
            fn = node["fn"]
            if isinstance(fn, dist.Independent):
                fn = fn.base_dist
            if isinstance(fn, dist.TransformedDistribution):
                # compute exogenous base dist (created with TransformReparam) at all sites
                output[name + "_base"] = T.ComposeTransform(fn.transforms).inv(
                    node["value"]
                )
        return output

    def counterfactual(
        self,
        obs: Dict[str, Tensor],
        intervention: Dict[str, Tensor],
        num_particles: int = 1,
        detach: bool = True,
    ) -> Dict[str, Tensor]:
        # NOTE: not ideal as "variables" is defined in child class
        dag_variables = self.variables.keys()
        assert set(obs.keys()) == set(dag_variables)
        avg_cfs = {k: torch.zeros_like(obs[k]) for k in obs.keys()}
        batch_size = list(obs.values())[0].shape[0]

        for _ in range(num_particles):
            # Abduction
            exo_noise = self.infer_exogeneous(obs)
            exo_noise = {k: v.detach() if detach else v for k, v in exo_noise.items()}
            # condition on root node variables (no exogeneous noise available)
            for k in dag_variables:
                if k not in intervention.keys():
                    if k not in [i.split("_base")[0] for i in exo_noise.keys()]:
                        exo_noise[k] = obs[k]
            # Abducted SCM
            abducted_scm = pyro.poutine.condition(self.sample_scm, data=exo_noise)
            # Action
            counterfactual_scm = pyro.poutine.do(abducted_scm, data=intervention)
            # Prediction
            counterfactuals = counterfactual_scm(batch_size)

            if hasattr(self, "discrete_variables"):  # hack for MIMIC
                # Check if we should change "finding", i.e. if its parents and/or
                # itself are not intervened on, then we use its observed value.
                # This is used due to stochastic abduction of discrete variables
                if (
                    "age" not in intervention.keys()
                    and "finding" not in intervention.keys()
                ):
                    counterfactuals["finding"] = obs["finding"]

            for k, v in counterfactuals.items():
                avg_cfs[k] += v / num_particles
        return avg_cfs


class FlowPGM(BasePGM):
    def __init__(self, args: Hparams):
        super().__init__()
        self.variables = {
            "sex": "binary",
            "mri_seq": "binary",
            "age": "continuous",
            "brain_volume": "continuous",
            "ventricle_volume": "continuous",
        }
        # priors: s, m, a, b and v
        self.s_logit = nn.Parameter(torch.zeros(1))
        self.m_logit = nn.Parameter(torch.zeros(1))
        for k in ["a", "b", "v"]:
            self.register_buffer(f"{k}_base_loc", torch.zeros(1))
            self.register_buffer(f"{k}_base_scale", torch.ones(1))

        # constraint, assumes data is [-1,1] normalized
        # normalize_transform = T.ComposeTransform([
        #     T.AffineTransform(loc=0, scale=2), T.SigmoidTransform(), T.AffineTransform(loc=-1, scale=2)])
        # normalize_transform = T.ComposeTransform([T.TanhTransform(cache_size=1)])
        # normalize_transform = T.ComposeTransform([T.AffineTransform(loc=0, scale=1)])

        # age flow
        self.age_module = T.ComposeTransformModule(
            [T.Spline(1, count_bins=4, order="linear")]
        )
        self.age_flow = T.ComposeTransform([self.age_module])
        # self.age_module, normalize_transform])

        # brain volume (conditional) flow: (sex, age) -> brain_vol
        bvol_net = DenseNN(2, args.widths, [1, 1], nonlinearity=nn.LeakyReLU(0.1))
        self.bvol_flow = ConditionalAffineTransform(context_nn=bvol_net, event_dim=0)
        # self.bvol_flow = [self.bvol_flow, normalize_transform]

        # ventricle volume (conditional) flow: (brain_vol, age) -> ventricle_vol
        vvol_net = DenseNN(2, args.widths, [1, 1], nonlinearity=nn.LeakyReLU(0.1))
        self.vvol_flow = ConditionalAffineTransform(context_nn=vvol_net, event_dim=0)
        # self.vvol_flow = [self.vvol_transf, normalize_transform]

        # if args.setup != 'sup_pgm':
        # anticausal predictors
        input_shape = (args.input_channels, args.input_res, args.input_res)
        # q(s | x, b) = Bernoulli(f(x,b))
        self.encoder_s = CNN(input_shape, num_outputs=1, context_dim=1)
        # q(m | x) = Bernoulli(f(x))
        self.encoder_m = CNN(input_shape, num_outputs=1)
        # q(a | b, v) = Normal(mu(b, v), sigma(b, v))
        self.encoder_a = MLP(num_inputs=2, num_outputs=2)
        # q(b | x, v) = Normal(mu(x, v), sigma(x, v))
        self.encoder_b = CNN(input_shape, num_outputs=2, context_dim=1)
        # q(v | x) = Normal(mu(x), sigma(x))
        self.encoder_v = CNN(input_shape, num_outputs=2)
        self.f = (
            lambda x: args.std_fixed * torch.ones_like(x)
            if args.std_fixed > 0
            else F.softplus(x)
        )

    def model(self) -> Dict[str, Tensor]:
        # p(s), sex dist
        ps = dist.Bernoulli(logits=self.s_logit).to_event(1)
        sex = pyro.sample("sex", ps)

        # p(m), mri_seq dist
        pm = dist.Bernoulli(logits=self.m_logit).to_event(1)
        mri_seq = pyro.sample("mri_seq", pm)

        # p(a), age flow
        pa_base = dist.Normal(self.a_base_loc, self.a_base_scale).to_event(1)
        pa = dist.TransformedDistribution(pa_base, self.age_flow)
        age = pyro.sample("age", pa)

        # p(b | s, a), brain volume flow
        pb_sa_base = dist.Normal(self.b_base_loc, self.b_base_scale).to_event(1)
        pb_sa = ConditionalTransformedDistribution(
            pb_sa_base, [self.bvol_flow]
        ).condition(torch.cat([sex, age], dim=1))
        bvol = pyro.sample("brain_volume", pb_sa)
        # _ = self.bvol_transf  # register with pyro

        # p(v | b, a), ventricle volume flow
        pv_ba_base = dist.Normal(self.v_base_loc, self.v_base_scale).to_event(1)
        pv_ba = ConditionalTransformedDistribution(
            pv_ba_base, [self.vvol_flow]
        ).condition(torch.cat([bvol, age], dim=1))
        vvol = pyro.sample("ventricle_volume", pv_ba)
        # _ = self.vvol_transf  # register with pyro

        return {
            "sex": sex,
            "mri_seq": mri_seq,
            "age": age,
            "brain_volume": bvol,
            "ventricle_volume": vvol,
        }

    def guide(self, **obs) -> None:
        # guide for (optional) semi-supervised learning
        pyro.module("FlowPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(m | x)
            if obs["mri_seq"] is None:
                m_prob = torch.sigmoid(self.encoder_m(obs["x"]))
                m = pyro.sample("mri_seq", dist.Bernoulli(probs=m_prob).to_event(1))

            # q(v | x)
            if obs["ventricle_volume"] is None:
                v_loc, v_logscale = self.encoder_v(obs["x"]).chunk(2, dim=-1)
                qv_x = dist.Normal(v_loc, self.f(v_logscale)).to_event(1)
                obs["ventricle_volume"] = pyro.sample("ventricle_volume", qv_x)

            # q(b | x, v)
            if obs["brain_volume"] is None:
                b_loc, b_logscale = self.encoder_b(
                    obs["x"], y=obs["ventricle_volume"]
                ).chunk(2, dim=-1)
                qb_xv = dist.Normal(b_loc, self.f(b_logscale)).to_event(1)
                obs["brain_volume"] = pyro.sample("brain_volume", qb_xv)

            # q(s | x, b)
            if obs["sex"] is None:
                s_prob = torch.sigmoid(
                    self.encoder_s(obs["x"], y=obs["brain_volume"])
                )  # .squeeze()
                pyro.sample("sex", dist.Bernoulli(probs=s_prob).to_event(1))

            # q(a | b, v)
            if obs["age"] is None:
                ctx = torch.cat([obs["brain_volume"], obs["ventricle_volume"]], dim=-1)
                a_loc, a_logscale = self.encoder_a(ctx).chunk(2, dim=-1)
                pyro.sample("age", dist.Normal(a_loc, self.f(a_logscale)).to_event(1))

    def model_anticausal(self, **obs) -> None:
        # assumes all variables are observed
        pyro.module("FlowPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(v | x)
            v_loc, v_logscale = self.encoder_v(obs["x"]).chunk(2, dim=-1)
            qv_x = dist.Normal(v_loc, self.f(v_logscale)).to_event(1)
            pyro.sample("ventricle_volume_aux", qv_x, obs=obs["ventricle_volume"])

            # q(b | x, v)
            b_loc, b_logscale = self.encoder_b(
                obs["x"], y=obs["ventricle_volume"]
            ).chunk(2, dim=-1)
            qb_xv = dist.Normal(b_loc, self.f(b_logscale)).to_event(1)
            pyro.sample("brain_volume_aux", qb_xv, obs=obs["brain_volume"])

            # q(a | b, v)
            ctx = torch.cat([obs["brain_volume"], obs["ventricle_volume"]], dim=-1)
            a_loc, a_logscale = self.encoder_a(ctx).chunk(2, dim=-1)
            pyro.sample(
                "age_aux",
                dist.Normal(a_loc, self.f(a_logscale)).to_event(1),
                obs=obs["age"],
            )

            # q(s | x, b)
            s_prob = torch.sigmoid(self.encoder_s(obs["x"], y=obs["brain_volume"]))
            qs_xb = dist.Bernoulli(probs=s_prob).to_event(1)
            pyro.sample("sex_aux", qs_xb, obs=obs["sex"])

            # q(m | x)
            m_prob = torch.sigmoid(self.encoder_m(obs["x"]))
            qm_x = dist.Bernoulli(probs=m_prob).to_event(1)
            pyro.sample("mri_seq_aux", qm_x, obs=obs["mri_seq"])

    def predict(self, **obs) -> Dict[str, Tensor]:
        # q(v | x)
        v_loc, v_logscale = self.encoder_v(obs["x"]).chunk(2, dim=-1)
        # v_loc = torch.tanh(v_loc)
        # q(b | x, v)
        b_loc, b_logscale = self.encoder_b(obs["x"], y=obs["ventricle_volume"]).chunk(
            2, dim=-1
        )
        # b_loc = torch.tanh(b_loc)
        # q(a | b, v)
        ctx = torch.cat([obs["brain_volume"], obs["ventricle_volume"]], dim=-1)
        a_loc, a_logscale = self.encoder_a(ctx).chunk(2, dim=-1)
        # a_loc = torch.tanh(b_loc)
        # q(s | x, b)
        s_prob = torch.sigmoid(self.encoder_s(obs["x"], y=obs["brain_volume"]))
        # q(m | x)
        m_prob = torch.sigmoid(self.encoder_m(obs["x"]))

        return {
            "sex": s_prob,
            "mri_seq": m_prob,
            "age": a_loc,
            "brain_volume": b_loc,
            "ventricle_volume": v_loc,
        }

    def svi_model(self, **obs) -> None:
        with pyro.plate("observations", obs["x"].shape[0]):
            pyro.condition(self.model, data=obs)()

    def guide_pass(self, **obs) -> None:
        pass


class MorphoMNISTPGM(BasePGM):
    def __init__(self, args):
        super().__init__()
        self.variables = {
            "thickness": "continuous",
            "intensity": "continuous",
            "digit": "categorical",
        }
        # priors
        self.digit_logits = nn.Parameter(torch.zeros(1, 10))  # uniform prior
        for k in ["t", "i"]:  # thickness, intensity, standard Gaussian
            self.register_buffer(f"{k}_base_loc", torch.zeros(1))
            self.register_buffer(f"{k}_base_scale", torch.ones(1))

        # constraint, assumes data is [-1,1] normalized
        normalize_transform = T.ComposeTransform(
            [T.SigmoidTransform(), T.AffineTransform(loc=-1, scale=2)]
        )

        # thickness flow
        self.thickness_module = T.ComposeTransformModule(
            [T.Spline(1, count_bins=4, order="linear")]
        )
        self.thickness_flow = T.ComposeTransform(
            [self.thickness_module, normalize_transform]
        )

        # intensity (conditional) flow: thickness -> intensity
        intensity_net = DenseNN(1, args.widths, [1, 1], nonlinearity=nn.GELU())
        self.context_nn = ConditionalAffineTransform(
            context_nn=intensity_net, event_dim=0
        )
        self.intensity_flow = [self.context_nn, normalize_transform]

        if args.setup != "sup_pgm":
            # anticausal predictors
            input_shape = (args.input_channels, args.input_res, args.input_res)
            # q(t | x, i) = Normal(mu(x, i), sigma(x, i)), 2 outputs: loc & scale
            self.encoder_t = CNN(input_shape, num_outputs=2, context_dim=1, width=8)
            # q(i | x) = Normal(mu(x), sigma(x))
            self.encoder_i = CNN(input_shape, num_outputs=2, width=8)
            # q(y | x) = Categorical(pi(x))
            self.encoder_y = CNN(input_shape, num_outputs=10, width=8)
            self.f = (
                lambda x: args.std_fixed * torch.ones_like(x)
                if args.std_fixed > 0
                else F.softplus(x)
            )

    def model(self) -> Dict[str, Tensor]:
        pyro.module("MorphoMNISTPGM", self)
        # p(y), digit label prior dist
        py = dist.OneHotCategorical(
            probs=F.softmax(self.digit_logits, dim=-1)
        )  # .to_event(1)
        # with pyro.poutine.scale(scale=0.05):
        digit = pyro.sample("digit", py)

        # p(t), thickness flow
        pt_base = dist.Normal(self.t_base_loc, self.t_base_scale).to_event(1)
        pt = dist.TransformedDistribution(pt_base, self.thickness_flow)
        thickness = pyro.sample("thickness", pt)

        # p(i | t), intensity conditional flow
        pi_t_base = dist.Normal(self.i_base_loc, self.i_base_scale).to_event(1)
        pi_t = ConditionalTransformedDistribution(
            pi_t_base, self.intensity_flow
        ).condition(thickness)
        intensity = pyro.sample("intensity", pi_t)
        _ = self.context_nn

        return {"thickness": thickness, "intensity": intensity, "digit": digit}

    def guide(self, **obs) -> None:
        # guide for (optional) semi-supervised learning
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(i | x)
            if obs["intensity"] is None:
                i_loc, i_logscale = self.encoder_i(obs["x"]).chunk(2, dim=-1)
                qi_t = dist.Normal(torch.tanh(i_loc), self.f(i_logscale)).to_event(1)
                obs["intensity"] = pyro.sample("intensity", qi_t)

            # q(t | x, i)
            if obs["thickness"] is None:
                t_loc, t_logscale = self.encoder_t(obs["x"], y=obs["intensity"]).chunk(
                    2, dim=-1
                )
                qt_x = dist.Normal(torch.tanh(t_loc), self.f(t_logscale)).to_event(1)
                obs["thickness"] = pyro.sample("thickness", qt_x)

            # q(y | x)
            if obs["digit"] is None:
                y_prob = F.softmax(self.encoder_y(obs["x"]), dim=-1)
                qy_x = dist.OneHotCategorical(probs=y_prob)  # .to_event(1)
                pyro.sample("digit", qy_x)

    def model_anticausal(self, **obs) -> None:
        # assumes all variables are observed & continuous ones are in [-1,1]
        pyro.module("MorphoMNISTPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(t | x, i)
            t_loc, t_logscale = self.encoder_t(obs["x"], y=obs["intensity"]).chunk(
                2, dim=-1
            )
            qt_x = dist.Normal(torch.tanh(t_loc), self.f(t_logscale)).to_event(1)
            pyro.sample("thickness_aux", qt_x, obs=obs["thickness"])

            # q(i | x)
            i_loc, i_logscale = self.encoder_i(obs["x"]).chunk(2, dim=-1)
            qi_t = dist.Normal(torch.tanh(i_loc), self.f(i_logscale)).to_event(1)
            pyro.sample("intensity_aux", qi_t, obs=obs["intensity"])

            # q(y | x)
            y_prob = F.softmax(self.encoder_y(obs["x"]), dim=-1)
            qy_x = dist.OneHotCategorical(probs=y_prob)  # .to_event(1)
            pyro.sample("digit_aux", qy_x, obs=obs["digit"])

    def predict(self, **obs) -> Dict[str, Tensor]:
        # q(t | x, i)
        t_loc, t_logscale = self.encoder_t(obs["x"], y=obs["intensity"]).chunk(
            2, dim=-1
        )
        t_loc = torch.tanh(t_loc)
        # q(i | x)
        i_loc, i_logscale = self.encoder_i(obs["x"]).chunk(2, dim=-1)
        i_loc = torch.tanh(i_loc)
        # q(y | x)
        y_prob = F.softmax(self.encoder_y(obs["x"]), dim=-1)
        return {"thickness": t_loc, "intensity": i_loc, "digit": y_prob}

    def svi_model(self, **obs) -> None:
        with pyro.plate("observations", obs["x"].shape[0]):
            pyro.condition(self.model, data=obs)()

    def guide_pass(self, **obs) -> None:
        pass


class ColourMNISTPGM(BasePGM):
    def __init__(self, args):
        super().__init__()
        self.variables = {
            "digit": "categorical",
            "colour": "categorical",
        }
        self.digit_logits = nn.Parameter(torch.zeros(1, 10))  # uniform prior
        self.colour_logits = nn.Parameter(torch.zeros(1, 10))  # uniform prior

        if args.setup != "sup_pgm":
            # anticausal predictors
            input_shape = (args.input_channels, args.input_res, args.input_res)
            # q(y | x) = Categorical(pi(x))
            self.encoder_y = CNN(input_shape, num_outputs=10, width=8)
            # q(c | x) = Categorical(pi(x))
            self.encoder_c = CNN(input_shape, num_outputs=10, width=8)
            self.f = (
                lambda x: args.std_fixed * torch.ones_like(x)
                if args.std_fixed > 0
                else F.softplus(x)
            )

    def model(self) -> Dict[str, Tensor]:
        pyro.module("ColourMNISTPGM", self)
        # p(y), digit label prior dist
        py = dist.OneHotCategorical(
            probs=F.softmax(self.digit_logits, dim=-1)
        )  # .to_event(1)
        digit = pyro.sample("digit", py)

        # p(c), colour label prior dist
        pc = dist.OneHotCategorical(
            probs=F.softmax(self.colour_logits, dim=-1)
        )  # .to_event(1)
        colour = pyro.sample("colour", pc)
        return {"digit": digit, "colour": colour}

    def guide(self, **obs) -> None:
        # guide for (optional) semi-supervised learning
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(y | x)
            if obs["digit"] is None:
                y_prob = F.softmax(self.encoder_y(obs["x"]), dim=-1)
                qy_x = dist.OneHotCategorical(probs=y_prob)  # .to_event(1)
                pyro.sample("digit", qy_x)

            # q(y | x)
            if obs["colour"] is None:
                c_prob = F.softmax(self.encoder_c(obs["x"]), dim=-1)
                qc_x = dist.OneHotCategorical(probs=c_prob)  # .to_event(1)
                pyro.sample("colour", qc_x)

    def model_anticausal(self, **obs) -> None:
        # assumes all variables are observed
        pyro.module("ColourMNISTPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(y | x)
            y_prob = F.softmax(self.encoder_y(obs["x"]), dim=-1)
            qy_x = dist.OneHotCategorical(probs=y_prob)  # .to_event(1)
            pyro.sample("digit_aux", qy_x, obs=obs["digit"])

            # q(c | x)
            c_prob = F.softmax(self.encoder_c(obs["x"]), dim=-1)
            qc_x = dist.OneHotCategorical(probs=c_prob)  # .to_event(1)
            pyro.sample("colour_aux", qc_x, obs=obs["colour"])

    def predict(self, **obs) -> Dict[str, Tensor]:
        # q(y | x)
        y_prob = F.softmax(self.encoder_y(obs["x"]), dim=-1)
        # q(c | x)
        c_prob = F.softmax(self.encoder_c(obs["x"]), dim=-1)
        return {"digit": y_prob, "colour": c_prob}

    def svi_model(self, **obs) -> None:
        with pyro.plate("observations", obs["x"].shape[0]):
            pyro.condition(self.model, data=obs)()

    def guide_pass(self, **obs) -> None:
        pass


class ChestPGM(BasePGM):
    def __init__(self, args: Hparams):
        super().__init__()
        self.variables = {
            "race": "categorical",
            "sex": "binary",
            "finding": "binary",
            "age": "continuous",
        }
        # Discrete variables that are not root nodes
        self.discrete_variables = {"finding": "binary"}
        # define base distributions
        for k in ["a", "f"]:
            self.register_buffer(f"{k}_base_loc", torch.zeros(1))
            self.register_buffer(f"{k}_base_scale", torch.ones(1))
        # age spline flow
        self.age_flow_components = T.ComposeTransformModule([T.Spline(1)])
        # self.age_constraints = T.ComposeTransform([
        #     T.AffineTransform(loc=4.09541458484, scale=0.32548387126),
        #     T.ExpTransform()])
        self.age_flow = T.ComposeTransform(
            [
                self.age_flow_components,
                # self.age_constraints,
            ]
        )
        # Finding (conditional) via MLP, a -> f
        finding_net = DenseNN(1, [8, 16], param_dims=[2], nonlinearity=nn.Sigmoid())
        self.finding_transform_GumbelMax = ConditionalGumbelMax(
            context_nn=finding_net, event_dim=0
        )
        # log space for sex and race
        self.sex_logit = nn.Parameter(np.log(1 / 2) * torch.ones(1))
        self.race_logits = nn.Parameter(np.log(1 / 3) * torch.ones(1, 3))

        if args.setup != "sup_pgm":
            from resnet import CustomBlock, ResNet, ResNet18

            shared_model = ResNet(
                CustomBlock,
                layers=[2, 2, 2, 2],
                widths=[64, 128, 256, 512],
                norm_layer=lambda c: nn.GroupNorm(min(32, c // 4), c),
            )
            # shared_model = torchvision.models.resnet18(weights=None)
            shared_model.conv1 = nn.Conv2d(
                args.input_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            kwargs = {
                "in_shape": (args.input_channels, *(args.input_res,) * 2),
                "base_model": shared_model,
            }
            # q(s | x) ~ Bernoulli(f(x))
            self.encoder_s = ResNet18(num_outputs=1, **kwargs)
            # q(r | x) ~ OneHotCategorical(f(x))
            self.encoder_r = ResNet18(num_outputs=3, **kwargs)
            # q(f | x) ~ Bernoulli(f(x))
            self.encoder_f = ResNet18(num_outputs=1, **kwargs)
            # q(a | x, f) ~ Normal(mu(x), sigma(x))
            self.encoder_a = ResNet18(num_outputs=2, context_dim=1, **kwargs)
            self.f = (
                lambda x: args.std_fixed * torch.ones_like(x)
                if args.std_fixed > 0
                else F.softplus(x)
            )

    def model(self) -> Dict[str, Tensor]:
        pyro.module("ChestPGM", self)
        # p(s), sex dist
        ps = dist.Bernoulli(logits=self.sex_logit).to_event(1)
        sex = pyro.sample("sex", ps)

        # p(a), age flow
        pa_base = dist.Normal(self.a_base_loc, self.a_base_scale).to_event(1)
        pa = dist.TransformedDistribution(pa_base, self.age_flow)
        age = pyro.sample("age", pa)
        # age_ = self.age_constraints.inv(age)
        _ = self.age_flow_components  # register with pyro

        # p(r), race dist
        pr = dist.OneHotCategorical(logits=self.race_logits)  # .to_event(1)
        race = pyro.sample("race", pr)

        # p(f | a), finding as OneHotCategorical conditioned on age
        finding_dist_base = dist.Gumbel(self.f_base_loc, self.f_base_scale).to_event(1)
        finding_dist = ConditionalTransformedDistributionGumbelMax(
            finding_dist_base, [self.finding_transform_GumbelMax]
        ).condition(age)
        finding = pyro.sample("finding", finding_dist)

        return {
            "sex": sex,
            "race": race,
            "age": age,
            "finding": finding,
        }

    def guide(self, **obs) -> None:
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(s | x)
            if obs["sex"] is None:
                s_prob = torch.sigmoid(self.encoder_s(obs["x"]))
                pyro.sample("sex", dist.Bernoulli(probs=s_prob).to_event(1))
            # q(r | x)
            if obs["race"] is None:
                r_probs = F.softmax(self.encoder_r(obs["x"]), dim=-1)
                qr_x = dist.OneHotCategorical(probs=r_probs)  # .to_event(1)
                pyro.sample("race", qr_x)
            # q(f | x)
            if obs["finding"] is None:
                f_prob = torch.sigmoid(self.encoder_f(obs["x"]))
                qf_x = dist.Bernoulli(probs=f_prob).to_event(1)
                obs["finding"] = pyro.sample("finding", qf_x)
            # q(a | x, f)
            if obs["age"] is None:
                a_loc, a_logscale = self.encoder_a(obs["x"], y=obs["finding"]).chunk(
                    2, dim=-1
                )
                qa_xf = dist.Normal(a_loc, self.f(a_logscale)).to_event(1)
                pyro.sample("age_aux", qa_xf)

    def model_anticausal(self, **obs) -> None:
        # assumes all variables are observed, train classfiers
        pyro.module("ChestPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(s | x)
            s_prob = torch.sigmoid(self.encoder_s(obs["x"]))
            qs_x = dist.Bernoulli(probs=s_prob).to_event(1)
            # with pyro.poutine.scale(scale=0.8):
            pyro.sample("sex_aux", qs_x, obs=obs["sex"])

            # q(r | x)
            r_probs = F.softmax(self.encoder_r(obs["x"]), dim=-1)
            qr_x = dist.OneHotCategorical(probs=r_probs)  # .to_event(1)
            # with pyro.poutine.scale(scale=0.5):
            pyro.sample("race_aux", qr_x, obs=obs["race"])

            # q(f | x)
            f_prob = torch.sigmoid(self.encoder_f(obs["x"]))
            qf_x = dist.Bernoulli(probs=f_prob).to_event(1)
            pyro.sample("finding_aux", qf_x, obs=obs["finding"])

            # q(a | x, f)
            a_loc, a_logscale = self.encoder_a(obs["x"], y=obs["finding"]).chunk(
                2, dim=-1
            )
            qa_xf = dist.Normal(a_loc, self.f(a_logscale)).to_event(1)
            # with pyro.poutine.scale(scale=2):
            pyro.sample("age_aux", qa_xf, obs=obs["age"])

    def predict(self, **obs) -> Dict[str, Tensor]:
        # q(s | x)
        s_prob = torch.sigmoid(self.encoder_s(obs["x"]))
        # q(r | x)
        r_probs = F.softmax(self.encoder_r(obs["x"]), dim=-1)
        # q(f | x)
        f_prob = torch.sigmoid(self.encoder_f(obs["x"]))
        # q(a | x, f)
        a_loc, _ = self.encoder_a(obs["x"], y=obs["finding"]).chunk(2, dim=-1)

        return {
            "sex": s_prob,
            "race": r_probs,
            "finding": f_prob,
            "age": a_loc,
        }

    def svi_model(self, **obs) -> None:
        with pyro.plate("observations", obs["x"].shape[0]):
            pyro.condition(self.model, data=obs)()

    def guide_pass(self, **obs) -> None:
        pass
