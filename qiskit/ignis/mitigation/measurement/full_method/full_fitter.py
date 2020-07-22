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
from typing import List, Dict
import numpy as np

from qiskit.result import Result
from ..meas_mit_utils import counts_probability_vector
from .full_mitigator import FullMeasureErrorMitigator


def fit_full_measure_error_mitigator(
        results: Result,
        metadata: List[Dict[str, any]]) -> FullMeasureErrorMitigator:
    """Return FullMeasureErrorMitigator from result data.

    Args:
        results: Qiskit result object.
        metadata: mitigation generator metadata.

    Returns:
        Measurement error mitigator object.
    """
    # TODO: Put num qubits in results
    # Get mitigation counts and labels
    cal_indices = []
    cal_counts = []
    for i, meta in enumerate(metadata):
        if meta.get('experiment') == 'meas_mit':
            cal_indices.append(int(meta['cal'], 2))
            cal_counts.append(results.get_counts(i))

    num_qubits = len(next(iter(cal_counts[0])))
    amat = np.zeros(2 * [2**num_qubits], dtype=float)
    for index, counts in zip(cal_indices, cal_counts):
        vec = counts_probability_vector(counts)
        amat[:, index] = vec

    return FullMeasureErrorMitigator(amat)
