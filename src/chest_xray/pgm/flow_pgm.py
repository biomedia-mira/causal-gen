import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import numpy as np
from pyro.nn import DenseNN
from pyro.infer.reparam.transform import TransformReparam
from layers import (
    ConditionalTransformedDistributionGumbelMax,
    ConditionalGumbelMax,
    CNN,
)
from resnet import ResNets_custom


class FlowPGM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.variables = {
            "race": "categorical",
            "sex": "binary",
            "finding": "binary",
            "age": "continuous",
        }
        # Discrete variables that are not root nodes
        self.discrete_variables = {
            "finding": "binary",
        }

        # prior age
        for k in ["a"]:
            self.register_buffer(f"{k}_base_loc", torch.zeros(1))
            self.register_buffer(f"{k}_base_scale", torch.ones(1))

        # age flow
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
        finding_net = DenseNN(
            1, [8, 16], param_dims=[2], nonlinearity=nn.Sigmoid()
        ).cuda()
        self.finding_transform_GumbelMax = ConditionalGumbelMax(
            context_nn=finding_net, event_dim=0
        )
        # log space for sex and race
        self.sex_logit = nn.Parameter(torch.zeros(1))
        # self.sex_logit = pyro.param(torch.zeros(1))
        self.race_logits = nn.Parameter(np.log(1 / 3) * torch.ones(1, 3))

        input_shape = (args.input_channels, args.input_res, args.input_res)

        if args.enc_net == "cnn":
            # q(s | x) ~ Bernoulli(f(x))
            self.encoder_s = CNN(input_shape, num_outputs=1)
            # q(r | x) ~ OneHotCategorical(logits=f(x))
            self.encoder_r = CNN(input_shape, num_outputs=3)
            # q(f | x) ~ Bernoulli(f(x))
            self.encoder_f = CNN(input_shape, num_outputs=1)
            # q(a | x, f) ~ Normal(mu(x), sigma(x))
            self.encoder_a = CNN(input_shape, num_outputs=1, context_dim=1)
        elif "resnet" in args.enc_net:
            # q(s | x) ~ Bernoulli(f(x))
            self.encoder_s = ResNets_custom(
                in_channels=1, out_channels=1, name=args.enc_net
            )
            # q(r | x) ~ OneHotCategorical(logits=f(x))
            self.encoder_r = ResNets_custom(
                in_channels=1, out_channels=3, name=args.enc_net
            )
            # q(f | x) ~ Bernoulli(f(x))
            self.encoder_f = ResNets_custom(
                in_channels=1, out_channels=1, name=args.enc_net
            )
            # q(a | x, f) ~ Normal(mu(x), sigma(x))
            self.encoder_a = ResNets_custom(
                in_channels=1, out_channels=1, name=args.enc_net, context_dim=1
            )

    def model(self, t=None):
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
        race_dist = dist.OneHotCategorical(logits=self.race_logits).to_event(0)
        race = pyro.sample("race", race_dist)

        # p(f | a), finding as OneHotCategorical conditioned on age
        finding_dist_base = dist.Gumbel(torch.zeros(1), torch.ones(1)).to_event(1)
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

    def guide(self, **obs):
        # print([k for k, v in obs.items() if v is not None])
        pyro.module("FlowPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(s | x)
            if obs["sex"] is None:
                s_prob = torch.sigmoid(self.encoder_s(obs["x"]))
                s = pyro.sample("sex", dist.Bernoulli(probs=s_prob).to_event(1))
            # q(r | x)
            if obs["race"] is None:
                r_logits = F.softmax(self.encoder_r(obs["x"]), dim=-1)  # .squeeze()
                r = pyro.sample(
                    "race", dist.OneHotCategorical(logits=r_logits).to_event(1)
                )
            # q(f | x)
            if obs["finding"] is None:
                f_prob = torch.sigmoid(self.encoder_ff(obs["x"]))
                f = pyro.sample("finding", dist.Bernoulli(probs=f_prob).to_event(1))
            # q(a | x, f)
            if obs["age"] is None:
                a_loc = self.encoder_a(obs["x"], y=obs["finding"])
                pyro.sample(
                    "age", dist.Normal(a_loc, torch.ones_like(a_loc)).to_event(1)
                )

    def model_anticausal(self, **obs):
        # assumes all variables are observed, train classfiers
        pyro.module("FlowPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(s | x)
            s_prob = torch.sigmoid(self.encoder_s(obs["x"]))
            s = pyro.sample("sex", dist.Bernoulli(probs=s_prob).to_event(1))

            # q(r | x)
            r_logits = F.softmax(self.encoder_r(obs["x"]), dim=-1)  # .squeeze()
            r = pyro.sample("race", dist.OneHotCategorical(logits=r_logits).to_event(1))

            # q(f | x)
            f_prob = torch.sigmoid(self.encoder_f(obs["x"]))
            qf_x = dist.Bernoulli(probs=f_prob).to_event(1)
            obs["finding"] = pyro.sample("finding", qf_x)

            # q(a | x, f)
            a_loc = self.encoder_a(obs["x"], y=obs["finding"])
            pyro.sample("age", dist.Normal(a_loc, torch.ones_like(a_loc)).to_event(1))

    def predict(self, **obs):
        # q(s | x)
        s_prob = torch.sigmoid(self.encoder_s(obs["x"]))
        # q(r | x)
        r_logits = F.softmax(self.encoder_r(obs["x"]), dim=-1)  # .squeeze()
        # q(f | x)
        f_prob = torch.sigmoid(self.encoder_f(obs["x"]))
        # q(a | x, f)
        a_loc = self.encoder_a(obs["x"], y=obs["finding"])

        return {
            "sex": s_prob,
            "race": r_logits,
            "age": a_loc,
            "finding": f_prob,
        }

    def predict_unnorm(self, **obs):
        # q(s | x)
        s_prob = self.encoder_s(obs["x"])
        # q(r | x)
        r_logits = self.encoder_r(obs["x"])
        # q(f | x)
        f_prob = self.encoder_f(obs["x"])
        qf_x = dist.Bernoulli(probs=torch.sigmoid(f_prob)).to_event(1)
        obs_finding = pyro.sample("finding", qf_x)
        # q(a | x, f)
        a_loc = self.encoder_a(
            obs["x"],
            # y=obs['finding'],
            y=obs_finding,
        )

        return {
            "sex": s_prob,
            "race": r_logits,
            "age": a_loc,
            "finding": f_prob,
        }

    def svi_model(self, **obs):
        with pyro.plate("observations", obs["x"].shape[0]):
            pyro.condition(self.model, data=obs)()

    def guide_pass(self, **obs):
        pass

    def infer_exogeneous(self, obs):
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
                # print(f"name: {name}; fn: {fn}; fn transforms: {fn.transforms}; node value {node['value'].size()}")
                # compute exogenous base dist (created with TransformReparam) at all sites
                output[name + "_base"] = T.ComposeTransform(fn.transforms).inv(
                    node["value"]
                )
        return output

    def infer_discrete_exogeneous(self, obs):
        r"""Infer exogeneous (i.e. using Gumbel max trick) for discrete factors"""
        raise NotImplementedError

    def scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg["fn"], dist.TransformedDistribution):
                return TransformReparam()
            # elif isinstance(msg['fn'], dist.Independent):
            #     return TransformReparam()
            else:
                return None

        return pyro.poutine.reparam(self.model, config=config)(*args, **kwargs)

    def sample_scm(self, n_samples=1, t=None):
        with pyro.plate("obs", n_samples):
            samples = self.scm(t)
        return samples

    def sample(self, n_samples=1, t=None):
        with pyro.plate("obs", n_samples):
            samples = self.model(t)
        return samples

    def counterfactual(self, obs, intervention, num_particles=1, detach=True, t=None):
        dag_variables = self.variables.keys()

        obs_ = {k: v for k, v in obs.items() if k in dag_variables}

        assert set(obs_.keys()) == set(dag_variables)
        # For continuos variables
        avg_cfs = {k: torch.zeros_like(obs_[k]) for k in obs_.keys()}
        batch_size = list(obs_.values())[0].shape[0]

        for _ in range(num_particles):
            # Abduction
            exo_noise = self.infer_exogeneous(obs_)
            exo_noise = {k: v.detach() if detach else v for k, v in exo_noise.items()}
            # condition on root node variables (no exogeneous noise available)
            for k in dag_variables:
                if k not in intervention.keys():
                    if k not in [i.split("_base")[0] for i in exo_noise.keys()]:
                        exo_noise[k] = obs_[k]
            # Abducted SCM
            abducted_scm = pyro.poutine.condition(self.sample_scm, data=exo_noise)
            # Action
            counterfactual_scm = pyro.poutine.do(abducted_scm, data=intervention)
            # Prediction
            counterfactuals = counterfactual_scm(batch_size, t)
            # Check if we should change finding or not, i.e. if its parents and it are not touched,
            # then we do not change it
            if (
                "age" not in intervention.keys()
                and "finding" not in intervention.keys()
            ):
                counterfactuals["finding"] = obs_["finding"]

            for k, v in counterfactuals.items():
                # print(f"k: {k}; v: {v.size()}")
                avg_cfs[k] += v / num_particles

        return avg_cfs


class FlowPGM_full(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.variables = {
            "race": "categorical",
            "sex": "binary",
            "finding": "binary",
            "age": "continuous",
        }
        # Discrete variables that are not root nodes
        self.discrete_variables = {
            "finding": "binary",
        }

        # priors: a
        for k in ["a"]:
            self.register_buffer(f"{k}_base_loc", torch.zeros(1))
            self.register_buffer(f"{k}_base_scale", torch.ones(1))

        # age flow
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

        # Finding (conditional) via MLP, a (1), s (1), r (3) -> f
        finding_net = DenseNN(
            5, [8, 16], param_dims=[2], nonlinearity=nn.Sigmoid()
        ).cuda()
        self.finding_transform_GumbelMax = ConditionalGumbelMax(
            context_nn=finding_net, event_dim=0
        )
        # log space for sex and race
        self.sex_logit = nn.Parameter(torch.zeros(1))
        # self.sex_logit = pyro.param(torch.zeros(1))
        self.race_logits = nn.Parameter(np.log(1 / 3) * torch.ones(1, 3))

        input_shape = (args.input_channels, args.input_res, args.input_res)

        if args.enc_net == "cnn":
            # q(s | x) ~ Bernoulli(f(x))
            self.encoder_s = CNN(input_shape, num_outputs=1, context_dim=1)
            # q(r | x) ~ OneHotCategorical(logits=f(x))
            self.encoder_r = CNN(input_shape, num_outputs=3, context_dim=1)
            # q(f | x) ~ Bernoulli(f(x))
            self.encoder_f = CNN(input_shape, num_outputs=1)
            # q(a | x, f) ~ Normal(mu(x), sigma(x))
            self.encoder_a = CNN(input_shape, num_outputs=1, context_dim=1)
        elif "resnet" in args.enc_net:
            # q(s | x) ~ Bernoulli(f(x))
            self.encoder_s = ResNets_custom(
                in_channels=1, out_channels=1, name=args.enc_net, context_dim=1
            )
            # q(r | x) ~ OneHotCategorical(logits=f(x))
            self.encoder_r = ResNets_custom(
                in_channels=1, out_channels=3, name=args.enc_net, context_dim=1
            )
            # q(f | x) ~ Bernoulli(f(x))
            self.encoder_f = ResNets_custom(
                in_channels=1, out_channels=1, name=args.enc_net
            )
            # q(a | x, f) ~ Normal(mu(x), sigma(x))
            self.encoder_a = ResNets_custom(
                in_channels=1, out_channels=1, name=args.enc_net, context_dim=1
            )

    def model(self, t=None):
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
        race_dist = dist.OneHotCategorical(logits=self.race_logits).to_event(0)
        race = pyro.sample("race", race_dist)

        # p(f | a), finding as OneHotCategorical conditioned on age
        finding_dist_base = dist.Gumbel(torch.zeros(1), torch.ones(1)).to_event(1)
        finding_dist = ConditionalTransformedDistributionGumbelMax(
            finding_dist_base, [self.finding_transform_GumbelMax]
        ).condition(torch.cat([age, sex, race], dim=-1))
        finding = pyro.sample("finding", finding_dist)

        return {
            "sex": sex,
            "race": race,
            "age": age,
            "finding": finding,
        }

    def guide(self, **obs):
        # print([k for k, v in obs.items() if v is not None])
        pyro.module("FlowPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(s | x)
            if obs["sex"] is None:
                s_prob = torch.sigmoid(self.encoder_s(obs["x"], y=obs["finding"]))
                s = pyro.sample("sex", dist.Bernoulli(probs=s_prob).to_event(1))
            # q(r | x)
            if obs["race"] is None:
                r_logits = F.softmax(
                    self.encoder_r(obs["x"], y=obs["finding"]), dim=-1
                )  # .squeeze()
                r = pyro.sample(
                    "race", dist.OneHotCategorical(logits=r_logits).to_event(1)
                )
            # q(f | x)
            if obs["finding"] is None:
                f_prob = torch.sigmoid(self.encoder_ff(obs["x"]))
                f = pyro.sample("finding", dist.Bernoulli(probs=f_prob).to_event(1))
            # q(a | x, f)
            if obs["age"] is None:
                a_loc = self.encoder_a(obs["x"], y=obs["finding"])
                pyro.sample(
                    "age", dist.Normal(a_loc, torch.ones_like(a_loc)).to_event(1)
                )

    def model_anticausal(self, **obs):
        # assumes all variables are observed, train classfiers
        pyro.module("FlowPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(s | x, f)
            s_prob = torch.sigmoid(self.encoder_s(obs["x"], y=obs["finding"]))
            s = pyro.sample("sex", dist.Bernoulli(probs=s_prob).to_event(1))

            # q(r | x, f)
            r_logits = F.softmax(self.encoder_r(obs["x"], y=obs["finding"]), dim=-1)
            r = pyro.sample("race", dist.OneHotCategorical(logits=r_logits).to_event(1))

            # q(f | x)
            f_prob = torch.sigmoid(self.encoder_f(obs["x"]))
            qf_x = dist.Bernoulli(probs=f_prob).to_event(1)
            obs["finding"] = pyro.sample("finding", qf_x)

            # q(a | x, f)
            a_loc = self.encoder_a(obs["x"], y=obs["finding"])
            pyro.sample("age", dist.Normal(a_loc, torch.ones_like(a_loc)).to_event(1))

    def predict(self, **obs):
        # q(s | x)
        s_prob = torch.sigmoid(self.encoder_s(obs["x"], y=obs["finding"]))
        # q(r | x)
        r_logits = F.softmax(
            self.encoder_r(obs["x"], y=obs["finding"]), dim=-1
        )  # .squeeze()
        # q(f | x)
        f_prob = torch.sigmoid(self.encoder_f(obs["x"]))
        # q(a | x, f)
        a_loc = self.encoder_a(obs["x"], y=obs["finding"])

        return {
            "sex": s_prob,
            "race": r_logits,
            "age": a_loc,
            "finding": f_prob,
        }

    def predict_unnorm(self, **obs):
        # q(s | x)
        s_prob = self.encoder_s(obs["x"], y=obs["finding"])
        # q(r | x)
        r_logits = self.encoder_r(obs["x"], y=obs["finding"])
        # q(f | x)
        f_prob = self.encoder_f(obs["x"])
        # q(a | x, f)
        a_loc = self.encoder_a(obs["x"], y=obs["finding"])

        return {
            "sex": s_prob,
            "race": r_logits,
            "age": a_loc,
            "finding": f_prob,
        }

    def svi_model(self, **obs):
        with pyro.plate("observations", obs["x"].shape[0]):
            pyro.condition(self.model, data=obs)()

    def guide_pass(self, **obs):
        pass

    def infer_exogeneous(self, obs):
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
                # print(f"name: {name}; fn: {fn}; fn transforms: {fn.transforms}; node value {node['value'].size()}")
                # compute exogenous base dist (created with TransformReparam) at all sites
                output[name + "_base"] = T.ComposeTransform(fn.transforms).inv(
                    node["value"]
                )
        return output

    def infer_discrete_exogeneous(self, obs):
        r"""Infer exogeneous (i.e. using Gumbel max trick) for discrete factors"""
        raise NotImplementedError

    def scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg["fn"], dist.TransformedDistribution):
                return TransformReparam()
            # elif isinstance(msg['fn'], dist.Independent):
            #     return TransformReparam()
            else:
                return None

        return pyro.poutine.reparam(self.model, config=config)(*args, **kwargs)

    def sample_scm(self, n_samples=1, t=None):
        with pyro.plate("obs", n_samples):
            samples = self.scm(t)
        return samples

    def sample(self, n_samples=1, t=None):
        with pyro.plate("obs", n_samples):
            samples = self.model(t)
        return samples

    def counterfactual(self, obs, intervention, num_particles=1, detach=True, t=None):
        dag_variables = self.variables.keys()

        obs_ = {k: v for k, v in obs.items() if k in dag_variables}
        assert set(obs_.keys()) == set(dag_variables)
        # For continuos variables
        avg_cfs = {k: torch.zeros_like(obs_[k]) for k in obs_.keys()}
        # dis_cfs =
        batch_size = list(obs_.values())[0].shape[0]

        for _ in range(num_particles):
            # Abduction
            exo_noise = self.infer_exogeneous(obs_)
            # for k, v in exo_noise.items():
            #     print(f'exo_noise, k: {k}, v: {v.size()}')
            exo_noise = {k: v.detach() if detach else v for k, v in exo_noise.items()}
            # condition on root node variables (no exogeneous noise available)
            for k in dag_variables:
                if k not in intervention.keys():
                    if k not in [i.split("_base")[0] for i in exo_noise.keys()]:
                        exo_noise[k] = obs_[k]

            # Abducted SCM
            abducted_scm = pyro.poutine.condition(self.sample_scm, data=exo_noise)
            # Action
            counterfactual_scm = pyro.poutine.do(abducted_scm, data=intervention)
            # Prediction
            counterfactuals = counterfactual_scm(batch_size, t)

            for k, v in counterfactuals.items():
                # print(f"k: {k}; v: {v.size()}")
                avg_cfs[k] += v / num_particles

        return avg_cfs


class FlowPGM_without_finding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.variables = {
            "race": "categorical",
            "sex": "binary",
            # 'finding': 'binary',
            "age": "continuous",
        }

        # priors: r, s, f, a
        for k in ["a"]:
            self.register_buffer(f"{k}_base_loc", torch.zeros(1))
            self.register_buffer(f"{k}_base_scale", torch.ones(1))

        # age flow
        self.age_flow_components = T.ComposeTransformModule([T.Spline(1)])
        self.age_flow = T.ComposeTransform(
            [
                self.age_flow_components,
            ]
        )

        # log space for sex and race
        self.sex_logit = nn.Parameter(torch.zeros(1))
        self.race_logits = nn.Parameter(np.log(1 / 3) * torch.ones(1, 3))

        input_shape = (args.input_channels, args.input_res, args.input_res)
        if args.enc_net == "cnn":
            # q(s | x) ~ Bernoulli(f(x))
            self.encoder_s = CNN(input_shape, num_outputs=1)
            # q(r | x) ~ OneHotCategorical(logits=f(x))
            self.encoder_r = CNN(input_shape, num_outputs=3)
            # q(a | x) ~ Normal(mu(x), sigma(x))
            self.encoder_a = CNN(input_shape, num_outputs=1)
        elif "resnet" in args.enc_net:
            # q(s | x) ~ Bernoulli(f(x))
            self.encoder_s = ResNets_custom(
                in_channels=1, out_channels=1, name=args.enc_net
            )
            # q(r | x) ~ OneHotCategorical(logits=f(x))
            self.encoder_r = ResNets_custom(
                in_channels=1, out_channels=3, name=args.enc_net
            )
            # q(a | x) ~ Normal(mu(x), sigma(x))
            self.encoder_a = ResNets_custom(
                in_channels=1, out_channels=1, name=args.enc_net
            )

    def model(self, t=None):
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
        race_dist = dist.OneHotCategorical(logits=self.race_logits).to_event(0)
        race = pyro.sample("race", race_dist)

        return {
            "sex": sex,
            "race": race,
            "age": age,
        }

    def guide(self, **obs):
        # print([k for k, v in obs.items() if v is not None])
        pyro.module("FlowPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(s | x)
            if obs["sex"] is None:
                s_prob = torch.sigmoid(self.encoder_s(obs["x"]))
                s = pyro.sample("sex", dist.Bernoulli(probs=s_prob).to_event(1))
            # q(r | x)
            if obs["race"] is None:
                r_logits = F.softmax(self.encoder_r(obs["x"]))  # .squeeze()
                r = pyro.sample(
                    "race", dist.OneHotCategorical(logits=r_logits).to_event(1)
                )
            # q(a | x)
            if obs["age"] is None:
                a_loc = self.encoder_a(obs["x"], y=obs["find"])
                a = pyro.sample(
                    "age", dist.Normal(a_loc, torch.ones_like(a_loc)).to_event(1)
                )

    def model_anticausal(self, **obs):
        # assumes all variables are observed, train classfiers
        pyro.module("FlowPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(s | x)
            s_prob = torch.sigmoid(self.encoder_s(obs["x"]))
            s = pyro.sample("sex", dist.Bernoulli(probs=s_prob).to_event(1))

            # q(r | x)
            r_logits = F.softmax(self.encoder_r(obs["x"]), dim=-1)  # .squeeze()
            r = pyro.sample("race", dist.OneHotCategorical(logits=r_logits).to_event(1))

            # q(a | x)
            a_loc = self.encoder_a(obs["x"])
            pyro.sample("age", dist.Normal(a_loc, torch.ones_like(a_loc)).to_event(1))

    def predict(self, **obs):
        # q(s | x)
        s_prob = torch.sigmoid(self.encoder_s(obs["x"]))
        # q(r | x)
        r_logits = F.softmax(self.encoder_r(obs["x"]), dim=-1)  # .squeeze()
        # q(a | x)
        a_loc = self.encoder_a(obs["x"])

        return {
            "sex": s_prob,
            "race": r_logits,
            "age": a_loc,
            # 'finding': f_logits,
        }

    def predict_unnorm(self, **obs):
        s = self.encoder_s(obs["x"])
        r = self.encoder_r(obs["x"])
        a = self.encoder_a(obs["x"])

        return {
            "sex": s,
            "race": r,
            "age": a,
            # 'finding': f_logits,
        }

    def svi_model(self, **obs):
        with pyro.plate("observations", obs["x"].shape[0]):
            pyro.condition(self.model, data=obs)()

    def guide_pass(self, **obs):
        pass

    def infer_exogeneous(self, obs):
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

    def infer_discrete_exogeneous(self, obs):
        r"""Infer exogeneous (i.e. using Gumbel max trick) for discrete factors"""
        raise NotImplementedError

    def scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg["fn"], dist.TransformedDistribution):
                return TransformReparam()
            # elif isinstance(msg['fn'], dist.Independent):
            #     return TransformReparam()
            else:
                return None

        return pyro.poutine.reparam(self.model, config=config)(*args, **kwargs)

    def sample_scm(self, n_samples=1, t=None):
        with pyro.plate("obs", n_samples):
            samples = self.scm(t)
        return samples

    def sample(self, n_samples=1, t=None):
        with pyro.plate("obs", n_samples):
            samples = self.model(t)
        return samples

    def counterfactual(self, obs, intervention, num_particles=1, detach=True, t=None):
        dag_variables = self.variables.keys()

        obs_ = {k: v for k, v in obs.items() if k in dag_variables}
        # print(f"dag_variables keys: {dag_variables}; obs_ keys: {obs_.keys()}")

        assert set(obs_.keys()) == set(dag_variables)
        # For continuos variables
        avg_cfs = {k: torch.zeros_like(obs_[k]) for k in obs_.keys()}
        # dis_cfs =
        batch_size = list(obs_.values())[0].shape[0]

        for _ in range(num_particles):
            # Abduction
            exo_noise = self.infer_exogeneous(obs_)
            exo_noise = {k: v.detach() if detach else v for k, v in exo_noise.items()}

            # condition on root node variables (no exogeneous noise available)
            for k in dag_variables:
                if k not in intervention.keys():
                    if k not in [i.split("_base")[0] for i in exo_noise.keys()]:
                        exo_noise[k] = obs_[k]

            # Abducted SCM
            abducted_scm = pyro.poutine.condition(self.sample_scm, data=exo_noise)
            # Action
            counterfactual_scm = pyro.poutine.do(abducted_scm, data=intervention)
            # Prediction
            counterfactuals = counterfactual_scm(batch_size, t)

            for k, v in counterfactuals.items():
                avg_cfs[k] += v / num_particles

        return avg_cfs
