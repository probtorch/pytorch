import torch
from common import TestCase


class TestEinsum(TestCase):
    def test_jit(self):

        def fn(x, y):
            return torch.einsum('i,j->ij', x, y)

        jit_fn = torch.jit.trace(fn, (torch.ones(2), torch.ones(3)), check_trace=False)

        x = torch.randn(2)
        y = torch.randn(3)
        expected = fn(x, y)
        actual = jit_fn(x, y)
        self.assertLess(torch.abs(actual - expected) < 1e-6)
