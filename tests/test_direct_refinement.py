import unittest
import numpy as np
from scipy import sparse
from solver import _direct_refined


class RefinedDirectTests(unittest.TestCase):
    def test_manufactured_scaled_complex_system(self):
        n = 48
        base = sparse.diags([-np.ones(n-1), (3+0.1j)*np.ones(n),
                             -np.ones(n-1)], [-1, 0, 1], format='csr')
        rows = np.geomspace(1e-8, 1e8, n)
        cols = np.geomspace(1e-5, 1e5, n)[::-1]
        A = (sparse.diags(rows) @ base @ sparse.diags(cols)).tocsr()
        expected = np.exp(1j*np.linspace(0, 1, n)) / cols
        b = A @ expected
        x = _direct_refined(A, b, 1e-12)
        self.assertLess(np.linalg.norm(A @ x - b)/np.linalg.norm(b), 1e-12)
        np.testing.assert_allclose(x, expected, rtol=1e-9, atol=1e-12)

    def test_zero_source_produces_zero_field(self):
        A = sparse.eye(16, format='csr', dtype=complex)
        x = _direct_refined(A, np.zeros(16, dtype=complex), 1e-9)
        np.testing.assert_array_equal(x, np.zeros(16))


if __name__ == '__main__':
    unittest.main()
