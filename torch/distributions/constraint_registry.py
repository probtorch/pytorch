r"""
PyTorch provides two :class:`ConstraintRegistry` objects that link
:class:`~torch.distributions.constraints.Constraint` objects to
:class:`~torch.distributions.transforms.Transform` objects:

1. ``biject_to(constraint)`` looks up a bijective
   :class:`~torch.distributions.transforms.Transform` from
   ``constraints.real`` to the given ``constraint``.
2. ``transform_to(constraint)`` looks up a not-necessarily bijective
   :class:`~torch.distributions.transforms.Transform` from
   ``constraints.real`` to the given ``constraint``.

The ``transform_to`` object is useful for performing unconstrained optimization
on constrained parameters of probability distributions, which are indicated by
each distribution's ``.params`` dict::

    loc = Variable(torch.zeros(100), requires_grad=True)
    unconstrained = Variable(torch.zeros(100), requires_grad=True)
    scale = transform_to(Normal.params['scale'])(unconstrained)
    loss = -Normal(loc, scale).log_prob(data).sum()

The ``biject_to`` object is useful for Hamiltonian Monte Carlo, where samples
from a probability distribution with constrained ``.support`` are propagated in
an unconstrained space::

    dist = Exponential(rate)
    unconstrained = Variable(torch.zeros(100), requires_grad=True)
    sample = biject_to(dist.support)(unconstrained)
    potential_energy = -dist.log_prob(sample).sum()

The ``biject_to`` and ``transform_to`` objects can be extended by user-defined
constraints and transforms using their ``.register()`` method. you can create
your own registry by creating a new :class:`ConstraintRegistry` object.
"""

from torch.distributions import constraints, transforms

__all__ = [
    'ConstraintRegistry',
    'biject_to',
    'transform_to',
]


class ConstraintRegistry(object):
    """
    Registry to link constraints to transforms.
    """
    def __init__(self):
        self._registry = {}

    def register(self, constraint, transform=None):
        """
        Registers a :class:`~torch.distributions.constraints.Constraint`
        subclass or singleton object in this registry. Usage as decorator::

            @my_registry.register(MyConstraintClass)
            def construct_transform(constraint):
                assert isinstance(constraint, MyConstraint)
                return MyTransform(constraint.params)

        Usage on singleton instances::

            my_registry.register(my_constraint_singleton, MyTransform())

        Args:
            constraint (:class:`~torch.distributions.constraints.Constraint`):
                Either a specific constraint instance or a subclass of
                constraints.
            transform (:class:`~torch.distributions.transforms.Transform`):
                Either a transform object or a callable that inputs a
                constraint object and returns a transform object.
        """
        # Support use as decorator.
        if transform is None:
            return lambda transform: self.register(constraint, transform)

        if isinstance(constraint, constraints.Constraint):
            # Register singleton instances.
            self._registry[constraint] = transform
        elif issubclass(constraint, constraints.Constraint):
            # Register Constraint subclass.
            self._registry[constraint] = transform
        else:
            raise TypeError('Expected constraint to be either a Constraint subclass or instance, '
                            'but got {}'.format(constraint))
        return transform

    def __call__(self, constraint):
        """
        Looks up a transform to constrained space, given a constraint object.
        Usage::

            constraint = Normal.params['scale']
            scale = transform_to(constraint)(torch.zeros(1))  # constrained
            u = transform_to(constraint).inv(scale)           # unconstrained

        Args:
            constraint (:class:`~torch.distributions.constraints.Constraint`):
                A constraint object.

        Returns:
            A :class:`~torch.distributions.transforms.Transform` object.

        Raises:
            `NotImplementedError` if no transform has been registered.
        """
        # Look up by singleton instance.
        try:
            return self._registry[constraint]
        except KeyError:
            pass
        # Look up by Constraint subclass.
        try:
            factory = self._registry[type(constraint)]
        except KeyError:
            raise NotImplementedError(
                'Cannot transform {} constraints'.format(type(constraint).__name__))
        return factory(constraint)


biject_to = ConstraintRegistry()
transform_to = ConstraintRegistry()

################################################################################
# Registration Table
################################################################################

biject_to.register(constraints.real, transforms.identity_transform)
transform_to.register(constraints.real, transforms.identity_transform)

biject_to.register(constraints.positive, transforms.ExpTransform())
transform_to.register(constraints.positive, transforms.ExpTransform())


@biject_to.register(constraints.greater_than)
@transform_to.register(constraints.greater_than)
def _transform_to_greater_than(constraint):
    loc = constraint.lower_bound
    scale = loc.new([1]).expand_as(loc)
    return transforms.ComposeTransform([transforms.ExpTransform(),
                                        transforms.AffineTransform(loc, scale)])


@biject_to.register(constraints.less_than)
@transform_to.register(constraints.less_than)
def _transform_to_less_than(constraint):
    loc = constraint.upper_bound
    scale = loc.new([-1]).expand_as(loc)
    return transforms.ComposeTransform([transforms.ExpTransform(),
                                        transforms.AffineTransform(loc, scale)])


biject_to.register(constraints.unit_interval, transforms.SigmoidTransform())
transform_to.register(constraints.unit_interval, transforms.SigmoidTransform())


@biject_to.register(constraints.interval)
@transform_to.register(constraints.interval)
def _transform_to_interval(constraint):
    loc = constraint.lower_bound
    scale = constraint.upper_bound - constraint.lower_bound
    return transforms.ComposeTransform([transforms.SigmoidTransform(),
                                        transforms.AffineTransform(loc, scale)])


biject_to.register(constraints.simplex, transforms.StickBreakingTransform())
transform_to.register(constraints.simplex, transforms.BoltzmannTransform())

# TODO define a bijection for LowerCholeskyTransform
transform_to.register(constraints.lower_cholesky, transforms.LowerCholeskyTransform())
