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
Single-qubit tensor-product measurement error mitigation generator.
"""
from typing import Optional, List, Dict
import numpy as np

from qiskit.exceptions import QiskitError
from ..meas_mit_utils import counts_probability_vector
from ..base_meas_mitigator import BaseMeasureErrorMitigator


class TensorMeasureErrorMitigator(BaseMeasureErrorMitigator):
    """Measurement error mitigator via 1-qubit tensor product mitigation."""

    def __init__(self, mats: List[np.ndarray]):
        """Initialize a TensorMeasurementMitigator

        Args:
            mats: list of single-qubit readout error matrices.
        """
        self._num_qubits = len(mats)
        self._amats = mats
        self._mitigation_matrices = np.zeros([self._num_qubits, 2, 2], dtype=float)

        for i in range(self._num_qubits):
            mat = self._amats[i]
            try:
                ainv = np.linalg.inv(mat)
            except np.linalg.LinAlgError:
                ainv = np.linalg.pinv(mat)
            self._mitigation_matrices[i] = ainv

    def mitigation_matrix(self, qubit):
        """Return the mitigation matrix for specified qubits."""
        return self._mitigation_matrices[qubit]

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

        Returns:
            float: expval.
        """
        # Get expectation value on specified qubits
        probs = counts_probability_vector(counts, clbits=clbits)
        num_qubits = int(np.log2(probs.shape[0]))

        if qubits is not None:
            ainvs = self._mitigation_matrices[list(qubits)]
        else:
            ainvs = self._mitigation_matrices

        probs = np.reshape(probs, num_qubits * [2])
        einsum_args = [probs, list(range(num_qubits))]
        for i, ainv in enumerate(reversed(ainvs)):
            einsum_args += [ainv, [num_qubits + i, i]]
        einsum_args += [list(range(num_qubits, 2 * num_qubits))]

        probs_mit = np.einsum(*einsum_args).ravel()
        parity = self._parity_vector(2 ** num_qubits)
        return probs_mit.dot(parity)
