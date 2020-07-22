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
Full A-matrix measurement migitation generator.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict
import numpy as np


class BaseMeasureErrorMitigator(ABC):
    """Base measurement error mitigator class."""

    @abstractmethod
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

    @staticmethod
    def _parity_vector(dim):
        parity = np.zeros(dim, dtype=np.int)
        for i in range(dim):
            parity[i] = bin(i)[2:].count('1')
        return (-1)**(parity // 2)
