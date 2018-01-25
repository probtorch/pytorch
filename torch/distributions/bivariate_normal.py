import math
from numbers import Number

import torch
from torch.autograd import Variable
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, lazy_property

class BivariateNormal(Distribution):
    r"""
    Creates a two-dimenaional bivariate normal distribution
    parameterized by a mean vector and a covariance matrix.
    
    The multivariate normal distribution can be parameterized either 
    in terms of a positive definite covariance matrix :math:`\mathbf{\Sigma}` 
    or a lower-triangular matrix :math:`\mathbf{L}` such that 
    :math:`\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top` as obtained via e.g. 
    Cholesky decomposition of the covariance.

    Example:

        >>> m = BivariateNormal(torch.zeros(2), torch.eye(2))
        >>> m.sample()  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
        -0.2102
        -0.5429
        [torch.FloatTensor of size 2]

    Args:
        loc (Tensor or Variable): mean of the distribution
        covariance_matrix (Tensor or Variable): covariance matrix (sigma positive-definite).
        scale_tril (Tensor or Variable): lower-triangular factor of covariance.
        
    Note:
        Only one of `covariance_matrix` or `scale_tril` can be specified.
        
    """
    params = {'loc': constraints.real,
              'covariance_matrix': constraints.real, # TODO: obviously not...
              'scale_tril': constraints.lower_triangular }
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, covariance_matrix=None, scale_tril=None):
        batch_shape, event_shape = loc.shape[:-1], loc.shape[-1:]
        if event_shape != (2,):
            raise ValueError("A bivariate normal distribution is a distribution over 2d vectors")
        if (covariance_matrix is None) == (scale_tril is None):
            raise ValueError("Exactly one of covariance_matrix or scale_tril may be specified (but not both).")
        if scale_tril is None:
            if covariance_matrix.dim() < 2:
                raise ValueError("covariance_matrix must be two-dimensional")
            self.covariance_matrix = covariance_matrix
            # TODO: why doesn't the following do what I would expect?
            #self.loc, self.covariance_matrix = broadcast_all(loc, covariance_matrix)
        else:
            if scale_tril.dim() < 2:
                raise ValueError("scale_tril matrix must be two-dimensional")
            self.scale_tril = scale_tril
            #self.loc, self.scale_tril = broadcast_all(loc, scale_tril)
        self.loc = loc
        super(BivariateNormal, self).__init__(batch_shape, event_shape)

    @lazy_property
    def scale_tril(self):
        L = self.covariance_matrix.clone()
        L[...,0,1] = 0.0
        temp_expand = L.dim() == 2
        if temp_expand:
            L.unsqueeze_(0) # workaround for tensors with no batch shape?
        L[...,0,0].sqrt_()
        L[...,1,0] /= L[...,0,0]
        L[...,1,1] -= L[...,1,0]
        L[...,1,1].sqrt_()
        if temp_expand:
            L.squeeze_(0)
        return L

    @lazy_property
    def covariance_matrix(self):
        # Note: this is never needed internally.
        return torch.bmm(self.scale_tril, self.scale_tril.transpose(-1,-2))

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        print shape
        eps = self.loc.new(*shape).normal_().unsqueeze_(-1)
        # TODO using `torch.bmm` is surprisingly clunky... only works when `.dim() == 3`
        if eps.dim() == 2:
            eps.unsqueeze_(0)
        # Now we need to "flatten" everything
        eps = eps.view((-1, self._event_shape[0], 1))
        flattened_scale = self.scale_tril.view((-1, self._event_shape[0], self.event_shape[0])).expand((eps.shape[0], -1, -1))
        return self.loc + torch.bmm(flattened_scale, eps).squeeze(-1).view(shape)

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        delta = value - self.loc
        z1 = delta[...,1] / self.scale_tril[...,1,1]
        z0 = (delta[...,0] - self.scale_tril[...,1,0] * z1) / self.scale_tril[...,0,0]
        M = (torch.stack([z0, z1], -1)**2).sum(-1)
        det = self.scale_tril[...,0,0]*self.scale_tril[...,1,1]
        log = math.log if isinstance(det, Number) else torch.log
        return -0.5*(M + self.loc.size(-1)*math.log(2*math.pi)) - log(det)

    def entropy(self):
        # TODO failing the monte carlo test at the moment
        det = self.scale_tril[...,0,0]*self.scale_tril[...,1,1]
        log = math.log if isinstance(det, Number) else torch.log
        H = (1.0 + (math.log(2*math.pi) + log(det)))*0.5*self.loc.shape[-1]
        if len(self._batch_shape) == 0:
            return H
        else:
            # TODO fails on empty batch shape
            return H.expand(self._batch_shape)

