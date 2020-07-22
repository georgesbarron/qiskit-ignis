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
Measurement error mitigation utility functions.
"""

from typing import Optional, List, Dict
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.ignis.verification.tomography import marginal_counts


def counts_probability_vector(
        counts: Dict,
        clbits: Optional[List[int]] = None,
        qubits: Optional[List[int]] = None) -> np.ndarray:
    """Compute mitigated expectation value.

    Args:
        counts: counts object
        clbits: Optional, marginalize counts to just these bits.
        qubits: qubits the count bitstrings correspond to.

    Raises:
        QiskitError: if qubit and clbit kwargs are not valid.

    Returns:
        np.ndarray: a probability vector for all count outcomes.
    """
    # Marginalize counts
    if clbits is not None:
        counts = marginal_counts(counts, meas_qubits=clbits)

    # Get total number of qubits
    num_qubits = len(next(iter(counts)))

    # Get vector
    vec = np.zeros(2**num_qubits, dtype=float)
    shots = 0
    for key, val in counts.items():
        shots += val
        vec[int(key, 2)] = val
    vec /= shots

    # Remap qubits
    if qubits is not None:
        if len(qubits) != num_qubits:
            raise QiskitError("Num qubits does not match vector length.")
        axes = [num_qubits - 1 - i for i in reversed(np.argsort(qubits))]
        vec = np.reshape(vec,
                         num_qubits * [2]).transpose(axes).reshape(vec.shape)

    return vec


def counts_expectation_value(counts: Dict,
                             clbits: Optional[List[int]] = None) -> float:
    """Convert counts to a probability vector.

    Args:
        counts: counts dictionary.
        clbits: Optional, marginalize counts to just these bits.

    Returns:
        float: expectation value.
    """
    # Marginalize counts
    if clbits is not None:
        counts = marginal_counts(counts, meas_qubits=clbits)

    # Compute expval via reduction
    expval = 0
    shots = 0
    for key, val in counts.items():
        shots += val
        parity = key.count('1') // 2
        expval += (-1)**parity * val
    return expval / shots
