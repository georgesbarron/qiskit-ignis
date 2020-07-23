import unittest

import scipy as sp
from qiskit.ignis.mitigation.measurement.ctmp_method.markov_compiled import *
from qiskit.ignis.mitigation.measurement.ctmp_method import *


# pylint disable=bare-except


def statistical_test(num_tests: int, fraction_passes: float):
    def stat_test(func):
        def wrapper_func(*args, **kwargs):
            num_failures = 0
            num_passes = 0
            for _ in range(num_tests):
                try:
                    func(*args, **kwargs)
                    num_passes += 1
                except:
                    num_failures += 1
            if num_passes / num_tests < fraction_passes:
                raise ValueError('Passed {} out of {} trials, needed {}%'.format(
                    num_passes,
                    num_tests,
                    100 * fraction_passes
                ))

        return wrapper_func

    return stat_test


class TestMarkov(unittest.TestCase):

    def setUp(self):
        self.lo = 0.1
        self.hi = 1.0 - self.lo

        self.r_dict = {
            # (final, start, qubits)
            ('0', '1', (0,)): 1e-3,
            ('1', '0', (0,)): 1e-1,
            ('0', '1', (1,)): 1e-3,
            ('1', '0', (1,)): 1e-1,

            ('00', '11', (0, 1)): 1e-3,
            ('11', '00', (0, 1)): 1e-1,
            ('01', '10', (0, 1)): 1e-3,
            ('10', '01', (0, 1)): 1e-3
        }
        gen_set = StandardGeneratorSet.from_num_qubits(2)

        self.G = MeasurementCalibrator(None, gen_set).total_g_matrix(self.r_dict).toarray()
        self.B = sp.linalg.expm(self.G)

        self.num_steps = 100

    @statistical_test(50, 0.7)
    def test_markov_chain_int_0(self):
        y = markov_chain_int(self.B, 0, self.num_steps)
        self.assertEqual(y, 3)

    @statistical_test(50, 0.7)
    def test_markov_chain_int_1(self):
        y = markov_chain_int(self.B, 1, self.num_steps)
        self.assertEqual(y, 3)

    @statistical_test(50, 0.7)
    def test_markov_chain_int_2(self):
        y = markov_chain_int(self.B, 2, self.num_steps)
        self.assertEqual(y, 3)

    @statistical_test(50, 0.7)
    def test_markov_chain_int_3(self):
        y = markov_chain_int(self.B, 3, self.num_steps)
        self.assertEqual(y, 3)


if __name__ == '__main__':
    unittest.main()
