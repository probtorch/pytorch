import math

import torch
from torch._six import inf, nan
from torch.distributions import Chi2, constraints
from torch.distributions.distribution import Distribution
from torch.distributions.distribution.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all

def _map_unbind(t, f):
    return torch.stack([f(m) for m in t.unbind(dim=0)], dim=0)

class Wishart(ExponentialFamily):
    # rsample/sample, log_prob, mean, variance, cdf, icdf, entropy
    r"""
    Creates a Wishart distribution parameterized by degree of freedom
    :attr:`df` and scale matrix :attr:`scale_tril`.

    Example::

        >>> m = StudentT(torch.tensor([2.0]))
        >>> m.sample()  # Student's t-distributed with degrees of freedom=2
        tensor([ 0.1046])

    Args:
        df (float or Tensor): degrees of freedom
        scale_tril (Tensor): scale of the distribution
    """
    arg_constraints = {'df': constraints.positive,
                       'scale_tril': constraints.lower_cholesky}
    support = constraints.real
    has_rsample = True

    def __init__(self, df, scale_tril, validate_args=None):
        self.df, self.scale_tril = broadcast_all(df, scale_tril)
        self._chi2 = Chi2(self.df)
        batch_shape = self.df.size()
        event_shape = self.scale_tril.shape[-2:]
        self._dim = event_shape[0]
        super(Wishart, self).__init__(batch_shape, event_shape,
                                      validate_args=validate_args)

    @property
    def mean(self):
        return self.df * self._scale_matrix

    @property
    def variance(self):
        addend = [[self.scale_tril[:, i, i] * self.scale_tril[:, j, j]
                   for j in range(self._dim)] for i in range(self._dim)]
        addend = [torch.stack(addend[i], dim=-1) for i in range(self._dim)]
        addend = torch.stack(addend, dim=-2)
        return self.df * (self.scale_tril ** 2 + addend)

    @property
    def _scale_matrix(self):
        scale_trilt = _map_unbind(self.scale_tril, lambda st: st.t())
        return self._scale_tril @ scale_trilt

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Wishart, _instance)
        batch_shape = torch.Size(batch_shape)
        new.df = self.df.expand(batch_shape)
        new.scale_tril = self.scale_tril.expand(batch_shape)
        new._chi2 = self._chi2.expand(batch_shape)
        super(Wishart, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        Z = _standard_normal(shape, dtype=self.df.dtype, device=self.df.device)
        cs = _map_unbind(self._chi2.rsample(sample_shape), torch.diagflat)
        eye_mask = torch.eye(self._dim).expand(*shape)
        A = torch.where(eye_mask, cs.sqrt(), torch.tril(Z))

        At = _map_unbind(A, lambda a: a.t())
        scale_trilt = _map_unbind(self.scale_tril, lambda st: st.t())
        return self.scale_tril, A, At, scale_trilt

    def rsample(self, sample_shape=torch.Size()):
        scale_tril, A, At, scale_trilt = self._rsample(sample_shape)
        return scale_tril @ A @ At @ scale_trilt

    @property
    def _natural_params(self):
        return (-1./2. * self._scale_matrix, 1./2. * (self.df - self._dim - 1))

    def _log_normalizer(self, natural_scale, natural_df):
        naturals_factor = -(natural_df * (self._dim + 1) / 2) * _map_unbind(
            -natural_scale, torch.logdet
        )
        log_gamma_factor = torch.mvlgamma(natural_df + (self._dim + 1) / 2,
                                          self._dim)
        return naturals_factor + log_gamma_factor

    @property
    def _mean_carrier_measure(self):
        return torch.zeros_like(self.df)

    def log_prob(self, value):
        det_factor = (self.df - self._dim - 1) / 2 * _map_unbind(value,
                                                                 torch.logdet)
        exp_factor = -1./2 * _map_unbind(
            torch.inverse(self._scale_matrix) @ value, torch.trace
        )
        log_num = det_factor + exp_factor

        return log_num - self._log_normalizer(*self._natural_params)
