import math
from numbers import Number

import numpy as np

import torch
from torch.autograd import Variable
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property

def _expand_batch_shape(bvec, bmat):
    """
    Given a batch of vectors and a batch of matrices, expand both to have the same 
    `batch_shape`.
    """
    try:
        vec_shape = torch._C._infer_size(bvec.shape, bmat.shape[:-1])
    except RuntimeError:
        raise ValueError("Incompatible batch shapes: vector {}, matrix {}".format(bvec.shape, bmat.shape))
    event_shape = bmat.shape[-1:]
    return bvec.expand(vec_shape), bmat.expand(vec_shape + event_shape)


def _batch_mv(bmat, bvec):
    """
    Performs a batched matrix-vector product, with an arbitrary batch shape.
    """
    batch_shape = bvec.shape[:-1]
    event_dim = bvec.shape[-1]
    bmat = bmat.expand(batch_shape + (event_dim, event_dim))
    if batch_shape != bmat.shape[:-2]:
        print "SHAPE MISMATCH:", batch_shape, bmat.shape[:-2]
        assert False
    bvec = bvec.unsqueeze(-1)
    
    # using `torch.bmm` is surprisingly clunky... only works when `.dim() == 3`
    if bvec.dim() == 2:
        bvec.unsqueeze(0) #_
     # flatten batch dimensions
    bvec = bvec.contiguous().view((-1, event_dim, 1))
    bmat = bmat.contiguous().view((-1, event_dim, event_dim)).expand((bvec.shape[0], -1, -1))
    return torch.bmm(bmat, bvec).squeeze(-1).view(batch_shape+(event_dim,)) 


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
        covariance_matrix (Tensor or Variable): positive-definite covariance matrix
        scale_tril (Tensor or Variable): lower-triangular factor of covariance.
        
    Note:
        Only one of `covariance_matrix` or `scale_tril` can be specified.
        
    """
    params = {'loc': constraints.real_vector,
              'covariance_matrix': constraints.real, # TODO: positive_definite,
              'scale_tril': constraints.real} # TODO: lower_cholesky }
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, loc, covariance_matrix=None, scale_tril=None):
        event_shape = torch.Size(loc.shape[-1:])
        if event_shape[0] != 2:
            raise ValueError("A bivariate normal distribution is a distribution over 2d vectors")
        if (covariance_matrix is None) == (scale_tril is None):
            raise ValueError("Exactly one of covariance_matrix or scale_tril may be specified (but not both).")
        if scale_tril is None:
            if covariance_matrix.dim() < 2:
                raise ValueError("covariance_matrix must be two-dimensional")
            self.loc, self.covariance_matrix = _expand_batch_shape(loc, covariance_matrix)
        else:
            if scale_tril.dim() < 2:
                raise ValueError("scale_tril matrix must be two-dimensional")
            self.loc, self.scale_tril = _expand_batch_shape(loc, scale_tril)
        super(BivariateNormal, self).__init__(torch.Size(self.loc.shape[:-1]), event_shape)

    @lazy_property
    def scale_tril(self):
        L = self.covariance_matrix #.clone()
        # TODO: explicit computation (previous version used in-place operations and was not differentiable...)
        L = torch.stack([torch.potrf(Li, upper=False) for Li in L.contiguous().view((-1,2,2))])
        return L.view(self._batch_shape + self._event_shape*2)

    @lazy_property
    def covariance_matrix(self):
        # Note: this is never needed internally. Possibly useful anyway?
        return torch.bmm(self.scale_tril, self.scale_tril.transpose(-1,-2))

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new(*shape).normal_()
        return self.loc + _batch_mv(self.scale_tril, eps)

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        delta = value - self.loc
        z0 = delta[...,0] / self.scale_tril[...,0,0]
        z1 = (delta[...,1] - self.scale_tril[...,1,0] * z0) / self.scale_tril[...,1,1]
        M = (torch.stack([z0, z1], -1)**2).sum(-1)
        diag = torch.arange(2).long()
        log_det = self.scale_tril[...,diag,diag].abs().log().sum(-1)
        return -0.5*M - math.log(2*math.pi) - log_det

    def entropy(self):
        diag = torch.arange(2).long()
        log_det = self.scale_tril[...,diag,diag].abs().log().sum(-1)
        H = 1.0 + math.log(2*math.pi) + log_det
        if len(self._batch_shape) == 0:
            return H
        else:
            # TODO probably fails on empty batch shape?
            return H.expand(self._batch_shape)

