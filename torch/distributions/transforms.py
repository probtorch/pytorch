from __future__ import division

import torch
from torch.autograd import Variable
from torch.distributions import constraints
from torch.nn.functional import sigmoid, softmax

__all__ = ['transform', 'register_transform', 'Transform']

_TRANSFORMS = {}


def transform(constraint):
    """
    Looks up a pair of transforms to- and from- unconstrained space, given
    a constraint object. Usage::

        constraint = Normal.params['scale']
        scale = transform(constraint).to_constrained(torch.zeros(1))
        u = transform(constraint).to_unconstrained(scale)
    """
    # Look up by singleton instance.
    try:
        return _TRANSFORMS[constraint]
    except KeyError:
        pass
    # Look up by Constraint subclass.
    try:
        Trans = _TRANSFORMS[type(constraint)]
    except KeyError:
        raise NotImplementedError(
            'Cannot transform {} constraints'.format(type(constraint).__name__))
    return Trans(constraint)


def register_transform(constraint):
    """
    Decorator to register a `Constraint` subclass or singleton object with the
    `torch.distributions.transforms.transform()` function. Usage::

        @register_transform(MyConstraintClass)
        class MyTransform(Transform):
            def to_unconstrained(self, x):
                ...
            def to_constrained(self, u):
                ...

    Args:
        constraint (Constraint subclass or instance): Either a specific
            constraint or a class of constraints.
    """

    def decorator(transform_class):
        if isinstance(constraint, constraints.Constraint):
            # Register singleton instances.
            _TRANSFORMS[constraint] = transform_class(constraint)
        elif issubclass(constraint, constraints.Constraint):
            # Register Constraint subclass.
            _TRANSFORMS[constraint] = transform_class
        else:
            raise TypeError('Expected constraint to be either a Constraint subclass or instance, '
                            'but got {}'.format(constraint))
        return transform_class

    return decorator


class Transform(object):
    """
    Each constraint class registers a pseudoinverse pair of transforms
    `to_unconstrained` and `from_unconstrained`. These allow standard
    parameters to be transformed to an unconstrained space for optimization and
    transformed back after optimization. Note that these are not necessarily
    inverse pairs since the unconstrained space may have extra dimensions that
    are projected out; only the one-sided inverse equation is guaranteed::

        x == t.to_constrained(t.to_unconstrained(x))

    """
    def __init__(self, constraint):
        self.constraint = constraint

    def to_unconstrained(self, x):
        """
        Transform from constrained coordinates to unconstrained coordinates.
        """
        raise NotImplementedError

    def to_constrained(self, u):
        """
        Transform from unconstrained coordinates to constrained coordinates.
        """
        raise NotImplementedError


@register_transform(constraints.real)
class IdentityTransform(Transform):
    """
    Identity transform for arbitrary real-valued data.
    """
    def to_unconstrained(self, x):
        return x

    def to_constrained(self, u):
        return u


@register_transform(constraints.positive)
class LogExpTransform(Transform):
    """
    Transform from the positive reals and back via `log()` and `exp()`.
    """
    def to_unconstrained(self, x):
        return torch.log(x)

    def to_constrained(self, u):
        return torch.exp(u)


@register_transform(constraints.greater_than)
class ShiftLogExpTransform(Transform):
    """
    Transform from lower-bounded reals and back via `+`, `log()`, and `exp()`.
    """
    def to_unconstrained(self, x):
        return torch.log(x - self.constraint.lower_bound)

    def to_constrained(self, u):
        return torch.exp(u) + self.constraint.lower_bound


@register_transform(constraints.interval)
class SigmoidLogitTransform(Transform):
    """
    Transform from an arbitrary interval and back via an affine transform and
    the `logit()` and `sigmoid()` functions.
    """
    def to_unconstrained(self, x):
        c = self.constraint
        unit = (x - c.lower_bound) / (c.upper_bound - c.lower_bound)
        return torch.log(unit / (1 - unit))

    def to_constrained(self, u):
        c = self.constraint
        unit = torch.sigmoid(u)
        return c.lower_bound + unit * (c.upper_bound - c.lower_bound)


@register_transform(constraints.simplex)
class LogSoftmaxTransform(Transform):
    """
    Transform from the unit simplex and back via `log()` and `softmax()`.
    """
    def to_unconstrained(self, x):
        return torch.log(x)

    def to_constrained(self, u):
        if isinstance(u, Variable):
            return softmax(u, dim=-1)
        return softmax(Variable(u), dim=-1).data


@register_transform(constraints.lower_triangular)
class LowerTriangularTransform(Transform):
    """
    Transform from lower-triangular square matrices of size `(n,n)` to
    contiguous vectors of size `m = n*(n+1)/2`. Dimensions left of the
    rightmost shape `(n,n)` or `(m,)` are preserved.
    """
    def _mask(self, x):
        mask = torch.tril(x.new([1]).byte().expand(x.shape[-2:]))
        if x.dim() > 2:
            mask = mask.view((1,) * (x.dim() - 2) + mask.shape).expand_as(x)
        return mask

    def to_unconstrained(self, x):
        n = x.size(-1)
        m = n * (n + 1) // 2
        return x[self._mask(x)].view(x.shape[:-2] + (m,))

    def to_constrained(self, u):
        n = int(round(((8 * u.size(-1) + 1)**0.5 - 1) / 2))
        x = u.new(u.shape[:-1] + (n, n)).zero_()
        x[self._mask(x)] = u
        return x
