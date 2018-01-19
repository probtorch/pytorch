import math
from numbers import Number

import torch
from torch.autograd import Variable
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all

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
        mean (Tensor or Variable): mean of the distribution
        covariance_matrix (Tensor or Variable): covariance matrix (sigma positive-definite).
        scale_tril (Tensor or Variable): lower-triangular factor of covariance.
        
    Note:
        Only one of `cov` or `scale_tril` can be specified.
        
    """

    has_rsample = True

    def __init__(self, mean, covariance_matrix=None, scale_tril=None):
        batch_shape, event_shape = mean.shape[:-1], mean.shape[-1:]
        if covariance_matrix is not None and scale_tril is not None:
            raise ValueError("Either covariance matrix or scale_tril may be specified, not both.")
        if covariance_matrix is None and scale_tril is None:
            raise ValueError("One of either covariance matrix or scale_tril must be specified")
        if scale_tril is None:
            assert covariance_matrix.dim() >= 2
            if covariance_matrix.dim() > 2:
                # TODO support batch_shape for covariance
                raise NotImplementedError("batch_shape for covariance matrix is not yet supported")
            else:
                scale_tril = torch.potrf(covariance_matrix, upper=False)
        else:
            assert scale_tril.dim() >= 2
            if scale_tril.dim() > 2:
                # TODO support batch_shape for scale_tril
                raise NotImplementedError("batch_shape for scale_tril is not yet supported")
            else:
                covariance_matrix = torch.mm(scale_tril, scale_tril.t())
        self.mean = mean
        self.covariance_matrix = covariance_matrix
        self.scale_tril = scale_tril
        super(MultivariateNormal, self).__init__(batch_shape, event_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.mean.new(*shape).normal_()
        return self.mean + torch.matmul(eps, self.scale_tril.t())

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        delta = value - self.mean
        # TODO replace torch.gesv with appropriate solver (e.g. potrs)
        M = (delta * torch.gesv(delta.view(-1,delta.shape[-1]).t(), self.covariance_matrix)[0].t().view(delta.shape)).sum(-1)
        #M = (delta * torch.matmul(delta, torch.inverse(self.cov))).sum(-1)
        log_det = torch.log(self.scale_tril.diag()).sum()
        return -0.5*(M + self.mean.size(-1)*math.log(2*math.pi)) - log_det

    def entropy(self):
        # TODO this will need modified to support batched covariance
        log_det = torch.log(self.scale_tril.diag()).sum()
        H = 0.5 + 0.5*(math.log(2*math.pi) + log_det)
        return H if len(self._batch_shape) == 0 else self.scale_tril.new(*self._batch_shape).fill_(H)


