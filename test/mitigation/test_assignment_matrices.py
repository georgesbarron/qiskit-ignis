# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Test assignment matrices
"""

import unittest
from typing import List, Dict, Tuple
from itertools import combinations, product, chain, permutations

import numpy as np

from qiskit import execute, QuantumCircuit
from qiskit.result import Result
from qiskit.providers.aer import QasmSimulator, noise
from qiskit.ignis.mitigation.measurement import (
    MeasMitigatorGenerator,
    MeasMitigatorFitter,
    counts_expectation_value
)
from ddt import ddt, unpack, data

from test_mitigators import NoisySimulationTest


_QUBIT_SUBSETS = [(1, 0, 2), (3, 0), (2,), (0, 3, 1, 2)]
_PAIRINGS = [
    *product(['complete', 'tensored'], _QUBIT_SUBSETS),
    ('ctmp', None)
]


@ddt
class TestMatrices(NoisySimulationTest):

    def get_mitigator(self, method: str, noise_model):
        circs, meta, _ = MeasMitigatorGenerator(self.num_qubits, method=method).run()
        cal_res = self.execute_circs(circs, noise_model=noise_model)
        mitigator = MeasMitigatorFitter(cal_res, meta).fit(method=method)
        return mitigator

    @data(*_PAIRINGS)
    @unpack
    def test_no_noise(self, method: str, qubits: List[int]):
        if qubits is not None:
            num_qubits = len(qubits)
        else:
            num_qubits = self.num_qubits
        mitigator = self.get_mitigator(method, noise_model=None)
        assignment_matrix = mitigator.assignment_matrix(qubits)
        mitigation_matrix = mitigator.mitigation_matrix(qubits)

        np.testing.assert_array_almost_equal(
            assignment_matrix,
            np.eye(2**num_qubits, 2**num_qubits),
            decimal=6
        )

        np.testing.assert_array_almost_equal(
            mitigation_matrix,
            np.eye(2**num_qubits, 2**num_qubits),
            decimal=6
        )
    
    @data(*_PAIRINGS)
    @unpack
    def test_with_noise(self, method: str, qubits: List[int]):
        if qubits is not None:
            num_qubits = len(qubits)
        else:
            num_qubits = self.num_qubits
        mitigator = self.get_mitigator(method, noise_model=self.noise_model)
        assignment_matrix = mitigator.assignment_matrix(qubits)
        mitigation_matrix = mitigator.mitigation_matrix(qubits)

        np.testing.assert_array_almost_equal(
            assignment_matrix,
            np.eye(2**num_qubits, 2**num_qubits),
            decimal=1 # This is pretty loose, but the diagonal elements are only 1 decimal away
        )

        np.testing.assert_array_almost_equal(
            mitigation_matrix,
            np.eye(2**num_qubits, 2**num_qubits),
            decimal=1 # This is pretty loose, but the diagonal elements are only 1 decimal away
        )


if __name__ == '__main__':
    unittest.main()