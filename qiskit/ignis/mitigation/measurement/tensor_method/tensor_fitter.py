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
Tensor-product matrix measurement error mitigation generator.
"""
from typing import List, Dict
import numpy as np

from qiskit.result import Result
from qiskit.exceptions import QiskitError
from qiskit.ignis.verification.tomography import marginal_counts
from ..meas_mit_utils import counts_probability_vector
from .tensor_mitigator import TensorMeasureErrorMitigator


def fit_tensor_measure_error_mitigator(
        results: Result,
        metadata: List[Dict[str, any]]) -> TensorMeasureErrorMitigator:
    """Return TensorMeasureErrorMitigator from result data.

    Args:
        results: Qiskit result object.
        metadata: mitigation generator metadata.

    Raises:
        QiskitError: if input Result object and metadata are not valid.

    Returns:
        Measurement error mitigator object.
    """
    # TODO: Put num qubits in results
    # Get mitigation counts and labels
    num_qubits = None
    cal_counts0 = None
    cal_counts1 = None
    for i, meta in enumerate(metadata):
        if cal_counts0 is not None and cal_counts1 is not None:
            break
        if meta.get('experiment') == 'meas_mit':
            cal = ''.join(set(meta['cal']))
            if cal == '0':
                cal_counts0 = results.get_counts(i)
            elif cal == '1':
                cal_counts1 = results.get_counts(i)
                num_qubits = len(meta['cal'])
    if cal_counts0 is None or cal_counts1 is None:
        raise QiskitError("Invalid result data for tensor product error mitigation.")

    amats = []
    for qubit in range(num_qubits):
        amat = np.zeros([2, 2], dtype=float)
        counts0 = marginal_counts(cal_counts0, meas_qubits=[qubit])
        counts1 = marginal_counts(cal_counts1, meas_qubits=[qubit])
        amat = np.array([counts_probability_vector(counts0),
                         counts_probability_vector(counts1)]).T
        amats.append(amat)
    return TensorMeasureErrorMitigator(amats)
