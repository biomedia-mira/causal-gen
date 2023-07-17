from typing import Dict

import pyro
import torch
import torch.nn.functional as F
from pyro.distributions.conditional import (
    ConditionalTransformedDistribution,
    ConditionalTransformModule,
    TransformedDistribution,
)
from pyro.distributions.torch_distribution import TorchDistributionMixin
from torch import nn
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch.distributions.utils import _sum_rightmost


class TraceStorage_ELBO(pyro.infer.Trace_ELBO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trace_storage = {"model": None, "guide": None}

    def _get_trace(self, model, guide, args, kwargs):
        model_trace, guide_trace = super()._get_trace(model, guide, args, kwargs)

        self.trace_storage["model"] = model_trace
        self.trace_storage["guide"] = guide_trace

        return model_trace, guide_trace


class ConditionalAffineTransform(ConditionalTransformModule):
    def __init__(self, context_nn, event_dim=0, **kwargs):
        super().__init__(**kwargs)
        self.event_dim = event_dim
        self.context_nn = context_nn

    def condition(self, context):
        loc, log_scale = self.context_nn(context)
        return torch.distributions.transforms.AffineTransform(
            loc, log_scale.exp(), event_dim=self.event_dim
        )


class MLP(nn.Module):
    def __init__(self, num_inputs=1, width=32, num_outputs=1):
        super().__init__()
        activation = nn.LeakyReLU()
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, width, bias=False),
            nn.BatchNorm1d(width),
            activation,
            nn.Linear(width, width, bias=False),
            nn.BatchNorm1d(width),
            activation,
            nn.Linear(width, num_outputs),
        )

    def forward(self, x):
        return self.mlp(x)


class CNN(nn.Module):
    def __init__(self, in_shape=(1, 192, 192), width=16, num_outputs=1, context_dim=0):
        super().__init__()
        in_channels = in_shape[0]
        res = in_shape[1]
        s = 2 if res > 64 else 1
        activation = nn.LeakyReLU()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, width, 7, s, 3, bias=False),
            nn.BatchNorm2d(width),
            activation,
            (nn.MaxPool2d(2, 2) if res > 32 else nn.Identity()),
            nn.Conv2d(width, 2 * width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(2 * width),
            activation,
            nn.Conv2d(2 * width, 2 * width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2 * width),
            activation,
            nn.Conv2d(2 * width, 4 * width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(4 * width),
            activation,
            nn.Conv2d(4 * width, 4 * width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(4 * width),
            activation,
            nn.Conv2d(4 * width, 8 * width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(8 * width),
            activation,
        )
        self.fc = nn.Sequential(
            nn.Linear(8 * width + context_dim, 8 * width, bias=False),
            nn.BatchNorm1d(8 * width),
            activation,
            nn.Linear(8 * width, num_outputs),
        )

    def forward(self, x, y=None):
        x = self.cnn(x)
        x = x.mean(dim=(-2, -1))  # avg pool
        if y is not None:
            x = torch.cat([x, y], dim=-1)
        return self.fc(x)


class ArgMaxGumbelMax(Transform):
    def __init__(self, logits, event_dim=0, cache_size=0):
        super(ArgMaxGumbelMax, self).__init__(cache_size=cache_size)
        self.logits = logits
        self._event_dim = event_dim
        self._categorical = pyro.distributions.torch.Categorical(
            logits=self.logits
        ).to_event(0)

    @property
    def event_dim(self):
        return self._event_dim

    def __call__(self, gumbels):
        assert self.logits != None, "Logits not defined."
        if self._cache_size == 0:
            return self._call(gumbels)
        y = self._call(gumbels)
        return y

    def _call(self, gumbels):
        assert self.logits != None, "Logits not defined."
        y = gumbels + self.logits
        return y.argmax(-1, keepdim=True)

    @property
    def domain(self):
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)

    @property
    def codomain(self):
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)

    def inv(self, k):
        """Infer the Gumbel posteriors"""
        assert self.logits != None, "Logits not defined."

        uniforms = torch.rand(
            self.logits.shape,
            dtype=self.logits.dtype,
            device=self.logits.device,
        )
        gumbels = -((-(uniforms.log())).log())
        # (batch_size, num_classes) mask to select kth class
        mask = F.one_hot(
            k.squeeze(-1).to(torch.int64), num_classes=self.logits.shape[-1]
        )
        # (batch_size, 1) select topgumbel for truncation of other classes
        topgumbel = (mask * gumbels).sum(dim=-1, keepdim=True) - (
            mask * self.logits
        ).sum(dim=-1, keepdim=True)
        mask = 1 - mask  # invert mask to select other != k classes
        g = gumbels + self.logits
        # (batch_size, num_classes)
        epsilons = -torch.log(mask * torch.exp(-g) + torch.exp(-topgumbel)) - (
            mask * self.logits
        )
        return epsilons

    def log_abs_det_jacobian(self, y):
        return -self._categorical.log_prob(y.squeeze(-1)).unsqueeze(-1)


class ConditionalGumbelMax(ConditionalTransformModule):
    def __init__(self, context_nn, event_dim=0, **kwargs):
        super().__init__(**kwargs)
        self.context_nn = context_nn
        self.event_dim = event_dim

    def condition(self, context):
        logits = self.context_nn(context)
        return ArgMaxGumbelMax(logits)

    def _logits(self, context):
        return self.context_nn(context)

    @property
    def domain(self):
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)

    @property
    def codomain(self):
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)


class TransformedDistributionGumbelMax(TransformedDistribution, TorchDistributionMixin):
    arg_constraints: Dict[str, constraints.Constraint] = {}

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        event_dim = len(self.event_shape)
        log_prob = 0.0
        y = value
        for transform in reversed(self.transforms):
            x = transform.inv(y)
            event_dim += transform.domain.event_dim - transform.codomain.event_dim
            log_prob = log_prob - _sum_rightmost(
                transform.log_abs_det_jacobian(x, y),
                event_dim - transform.domain.event_dim,
            )
            y = x
        return log_prob


class ConditionalTransformedDistributionGumbelMax(ConditionalTransformedDistribution):
    def condition(self, context):
        base_dist = self.base_dist.condition(context)
        transforms = [t.condition(context) for t in self.transforms]
        return TransformedDistributionGumbelMax(base_dist, transforms)

    def clear_cache(self):
        pass
