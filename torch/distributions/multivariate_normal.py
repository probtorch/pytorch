import math
from numbers import Number

import torch
from torch.autograd import Variable
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, lazy_property

class MultivariateNormal(Distribution):
    r"""
    Creates a multivariate normal (also called Gaussian) distribution
    parameterized by a mean vector and a covariance matrix.
    
    The multivariate normal distribution can be parameterized either 
    in terms of a positive definite covariance matrix :math:`\mathbf{\Sigma}` 
    or a lower-triangular matrix :math:`\mathbf{L}` such that 
    :math:`\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top` as obtained via e.g. 
    Cholesky decomposition of the covariance.

    Example:

        >>> m = MultivariateNormal(torch.zeros(2), torch.eye(2))
        >>> m.sample()  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
        -0.2102
        -0.5429
        [torch.FloatTensor of size 2]

    Args:
        loc (Tensor or Variable): mean of the distribution
        covariance_matrix (Tensor or Variable): positive-definite covariance matrix
        scale_tril (Tensor or Variable): lower-triangular factor of covariance
        
    Note:
        Only one of `covariance_matrix` or `scale_tril` can be specified.
        
    """
    params = {'loc': constraints.real_vector,
              'covariance_matrix': constraints.positive_definite,
              'scale_tril': constraints.lower_cholesky }
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, covariance_matrix=None, scale_tril=None):
        batch_shape, event_shape = torch.Size(loc.shape[:-1]), torch.Size(loc.shape[-1:])
        if (covariance_matrix is None) == (scale_tril is None):
            raise ValueError("Exactly one of covariance_matrix or scale_tril may be specified (but not both).")
        if scale_tril is None:
            if covariance_matrix.dim() > 2:
                # TODO support batch_shape for covariance
                raise NotImplementedError("batch_shape for covariance_matrix is not yet supported")
            elif covariance_matrix.dim() < 2:
                raise ValueError("covariance_matrix must be two-dimensional")
            self.covariance_matrix = covariance_matrix
        else:
            if scale_tril.dim() > 2:
                # TODO support batch_shape for scale_tril
                raise NotImplementedError("batch_shape for scale_tril is not yet supported")
            elif scale_tril.dim() < 2:
                raise ValueError("scale_tril matrix must be two-dimensional")
            self.scale_tril = scale_tril
        self.loc = loc
        super(MultivariateNormal, self).__init__(batch_shape, event_shape)

    @lazy_property
    def scale_tril(self):
        return torch.potrf(self.covariance_matrix, upper=False)

    @lazy_property
    def covariance_matrix(self):
        return torch.mm(self.scale_tril, self.scale_tril.t())

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new(*shape).normal_()
        return self.loc + torch.matmul(eps, self.scale_tril.t())

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        delta = value - self.loc
        # TODO replace torch.gesv with appropriate solver (e.g. potrs)
        M = (delta * torch.gesv(delta.view(-1,delta.shape[-1]).t(), self.covariance_matrix)[0].t().view(delta.shape)).sum(-1)
        log_det = torch.log(self.scale_tril.diag()).sum()
        return -0.5*(M + self.loc.size(-1)*math.log(2*math.pi)) - log_det

    def entropy(self):
        # TODO this will need modified to support batched covariance
        log_det = self.scale_tril.diag().log().sum(-1, keepdim=True)
        H = (1.0 + (math.log(2*math.pi) + log_det))*0.5*self.loc.shape[-1]
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)

