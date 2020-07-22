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
Full-matrix measurement error mitigation generator.
"""
from typing import Optional, List, Dict
import numpy as np

from qiskit.exceptions import QiskitError
from ..meas_mit_utils import counts_probability_vector
from ..base_meas_mitigator import BaseMeasureErrorMitigator


class FullMeasureErrorMitigator(BaseMeasureErrorMitigator):
    """Measurement error mitigator via full N-qubit mitigation."""

    def __init__(self, amat: np.ndarray):
        """Initialize a TensorMeasurementMitigator

        Args:
            amat (np.array): readout error A matrix.
        """
        self._num_qubits = int(np.log2(amat.shape[0]))
        self._amat = amat
        self._mitigation_matrices = {}

    def expectation_value(self,
                          counts: Dict,
                          clbits: Optional[List[int]] = None,
                          qubits: Optional[List[int]] = None) -> float:
        """Compute mitigated expectation value.

        The ``qubits`` kwarg is used so that count bitstrings correpond to
        measurements of the form ``circuit.measure(qubits, range(num_qubits))``.

        Args:
            counts: counts object
            clbits: Optional, marginalize counts to just these bits.
            qubits: qubits the count bitstrings correspond to.

        Raises:
            QiskitError: if qubit and clbit kwargs are not valid.

        Returns:
            float: expval.
        """
        # Get probability vector
        probs = counts_probability_vector(counts, clbits=clbits, qubits=qubits)
        num_qubits = int(np.log2(probs.shape[0]))

        # Get qubit mitigation matrix and mitigate probs
        if qubits is None:
            qubits = tuple(range(num_qubits))
        if len(qubits) != num_qubits:
            raise QiskitError("Num qubits does not match number of clbits.")
        mit_probs = self.mitigation_matrix(qubits).dot(probs)

        # Compute mitigated expval
        parity = self._parity_vector(2 ** num_qubits)
        return mit_probs.dot(parity)

    def mitigation_matrix(self, qubits=None):
        """Return the mitigation matrix for specified qubits."""
        if qubits is None:
            qubits = tuple(range(self._num_qubits))
        else:
            qubits = tuple(sorted(qubits))

        # Check for cached mitigation matrix
        # if not present compute
        if qubits not in self._mitigation_matrices:
            marginal_matrix = self._compute_marginal_matrix(qubits)
            try:
                mit_mat = np.linalg.inv(marginal_matrix)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if matrix is singular
                mit_mat = np.linalg.pinv(marginal_matrix)
            self._mitigation_matrices[qubits] = mit_mat

        return self._mitigation_matrices[qubits]

    @staticmethod
    def _keep_indexes(qubits):
        indexes = [0]
        for i in sorted(qubits):
            indexes += [idx + (1 << i) for idx in indexes]
        return indexes

    def _compute_marginal_matrix(self, qubits):
        # Compute marginal matrix
        axis = tuple([self._num_qubits - 1 - i for i in set(
            range(self._num_qubits)).difference(qubits)])
        num_qubits = len(qubits)
        new_amat = np.zeros(2 * [2 ** num_qubits], dtype=float)
        for i, col in enumerate(self._amat.T[self._keep_indexes(qubits)]):
            new_amat[i] = np.reshape(col, self._num_qubits * [2]).sum(axis=axis).reshape(
                [2 ** num_qubits])
        new_amat = new_amat.T
        return new_amat
