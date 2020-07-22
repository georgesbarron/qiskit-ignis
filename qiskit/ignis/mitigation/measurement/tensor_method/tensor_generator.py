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
Tensor-product measurement error mitigation generator.
"""

from qiskit.circuit import QuantumCircuit
from ..base_meas_mit_generator import BaseMeasureErrorMitigationGenerator


class TensorMeasureErrorMitigationGenerator(BaseMeasureErrorMitigationGenerator):
    """Tensor product measurement mitigation generator."""
    # pylint: disable=arguments-differ

    def __init__(self, num_qubits: int):
        """Initialize measurement mitigation calibration generator."""
        label0 = num_qubits * '0'
        circ0 = QuantumCircuit(num_qubits, name='cal_' + label0)
        circ0.measure_all()

        label1 = num_qubits * '1'
        circ1 = QuantumCircuit(num_qubits, name='cal_' + label1)
        circ1.x(list(range(num_qubits)))
        circ1.measure_all()

        super().__init__(num_qubits, [circ0, circ1], [label0, label1])
