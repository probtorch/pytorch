import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from torch.nn.functional import sigmoid

__all__ = [
    'AbsTransform',
    'AffineTransform',
    'BoltzmannTransform',
    'CachedTransform',
    'ExpTransform',
    'InverseTransform',
    'SigmoidTransform',
    'StickBreakingTransform',
    'Transform',
]


class Transform(object):
    """
    Abstract class for transformations with computable inverse log
    det jacobians. They are primarily used in
    :class:`torch.distributions.TransformedDistribution`.

    Derived classes should implement one or both of :meth:`_forward` or
    :meth:`_inverse` and should implement :meth:`log_abs_det_jacobian`.
    Derived classes may store intermediate results in the `._cache` dict.
    """
    bijective = False

    def __eq__(self, other):
        return type(other) is type(self)

    def __ne__(self, other):
        # Necessary for Python2
        return not self.__eq__(other)

    def forward(self, x):
        """
        Abstract method to compute forward transformation.
        """
        raise NotImplementedError

    def inverse(self, y):
        """
        Abstract method to compute inverse transformation.
        """
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y):
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        """
        raise NotImplementedError


class CachedTransform(Transform):
    """
    Abstract base class for transforms that implement one of :meth:`forward`
    or :meth:`backward` via by caching the latest value (i.e. an LRU(1) cache).

    This class is useful for tranforms whose inverses are either expensive or
    numerically unstable. Note that care must be taken with memoized values
    since the autograd graph may be reversed. For example while the following
    works::

        y = t.forward(x)
        t.log_abs_det_jacobian(x, y).backward()  # x will receive gradients.

    However the following will error due to dependency reversal::

        y = t.forward(x)
        z = t.inverse(y)
        grad(z.sum(), [y])  # error because z is x

    Derived classes should implement one or both of :meth:`_forward` and
    :meth:`_backward`.
    """
    def __init__(self):
        self._cached_x_y = None, None

    def forward(self, x):
        """
        Invokes the memoized transform `x => y`.
        """
        x_old, y_old = self._cached_x_y
        if x is x_old:
            return y_old
        y = self._forward(x)
        self._cached_x_y = x, y
        return y

    def inverse(self, y):
        """
        Inverts the memoized transform `y => x`.
        """
        x_old, y_old = self._cached_x_y
        if y is y_old:
            return x_old
        x = self._inverse(y)
        self._cached_x_y = x, y
        return x

    def _forward(self, x):
        """
        Abstract method to compute forward transformation.
        """
        raise NotImplementedError

    def _inverse(self, y):
        """
        Abstract method to compute inverse transformation.
        """
        raise NotImplementedError


class InverseTransform(Transform):
    """
    Inverts a single :class:`Transform`.
    """
    def __init__(self, transform):
        self.transform = transform

    @constraints.dependent_property
    def domain(self):
        return self.transform.codomain

    @constraints.dependent_property
    def codomain(self):
        return self.transform.domain

    @property
    def bijective(self):
        return self.transform.bijective

    def forward(self, x):
        return self.transform.inverse(x)

    def inverse(self, y):
        return self.transform.forward(y)

    def log_abs_det_jacobian(self, x, y):
        return -self.transform.log_abs_det_jacobian(y, x)


class ExpTransform(Transform):
    """
    Transform via the mapping `y = exp(x)`.
    """
    domain = constraints.real
    codomain = constraints.positive
    bijective = True

    def forward(self, x):
        return x.exp()

    def inverse(self, y):
        return y.log()

    def log_abs_det_jacobian(self, x, y):
        return x


class SigmoidTransform(Transform):
    """
    Transform via the mapping `y = sigmoid(x)` and `x = logit(y)`.
    """
    domain = constraints.real
    codomain = constraints.unit_interval
    bijective = True

    def forward(self, x):
        return sigmoid(x)

    def inverse(self, y):
        return y.log() - (-y).log1p()

    def log_abs_det_jacobian(self, x, y):
        return -(y.reciprocal() + (1 - y).reciprocal()).log()


class AbsTransform(Transform):
    """
    Transform via the mapping `y = abs(x)`.
    """
    domain = constraints.real
    codomain = constraints.positive

    def forward(self, x):
        return x.abs()

    def inverse(self, y):
        return y


class AffineTransform(Transform):
    """
    Transform via the pointwise affine mapping `y = loc + scale * x`.

    Args:
        loc (Tensor or Variable): Location parameter.
        scale (Tensor or Variable): Scale parameter.
        event_dim (int): Optional size of `event_shape`. This should be zero
            for univariate random variables, 1 for distributions over vectors,
            2 for distributions over matrices, etc.
    """
    domain = constraints.real
    codomain = constraints.real
    bijective = True

    def __init__(self, loc, scale, event_dim=0):
        super(AffineTransform, self).__init__()
        self.loc, self.scale = broadcast_all(loc, scale)
        self.event_dim = event_dim

    def __eq__(self, other):
        return (type(other) is AffineTransform) and self.loc.eq(other.loc).all() and self.scale.eq(other.scale).all()

    def forward(self, x):
        return self.loc + self.scale * x

    def inverse(self, y):
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, x, y):
        result = torch.abs(self.scale).log()
        shape = x.shape
        if self.event_dim:
            result_size = result.size()[:-self.event_dim] + (-1,)
            result = result.view(result_size).sum(-1)
            shape = shape[:-self.event_dim]
        return result.expand(shape)


class BoltzmannTransform(CachedTransform):
    """
    Transform from unconstrained space to the simplex via `y = exp(x)` then
    normalizing.

    This is not bijective and cannot be used for HMC. However this acts mostly
    coordinate-wise (except for the final normalization), and this is
    appropriate for coordinate-wise optimization algorithms.
    """
    domain = constraints.real
    codomain = constraints.simplex

    def _forward(self, x):
        logprobs = x
        probs = (logprobs - logprobs.max(-1, True)[0]).exp()
        probs /= probs.sum(-1, True)
        return probs

    def _inverse(self, y):
        probs = y
        return probs.log()


class StickBreakingTransform(CachedTransform):
    """
    Transform from unconstrained space to the simplex of one additional
    dimension via a stick-breaking process.

    This transform arises as an iterated sigmoid transform in a stick-breaking
    construction of the `Dirichlet` distribution: the first logit is
    transformed via sigmoid to the first probability and the probability of
    everything else, and then the process recurses.

    This is bijective and appropriate for use in HMC; however it mixes
    coordinates together and is less appropriate for optimization.
    """
    domain = constraints.real
    codomain = constraints.simplex
    bijective = True

    def _forward(self, x):
        shape = x.shape[:-1] + (1 + x.shape[-1],)
        one = x.new([1]).expand(x.shape[:-1] + (1,))
        numer = sigmoid(x)
        denom = (1 - numer).cumprod(-1)
        probs = torch.cat([numer, one], -1) * torch.cat([one, denom], -1)
        return probs

    def _inverse(self, y):
        pmf = y
        cmf = pmf.cumsum(-1)
        sf = 1 - cmf
        units = y[..., :-1] / sf[..., :-1]
        return units.log()

    # TODO implement .log_abs_det_jacobian()
