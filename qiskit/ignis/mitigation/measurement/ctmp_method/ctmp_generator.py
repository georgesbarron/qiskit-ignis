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
CTMP measurement error mitigation generator.
"""

from qiskit.circuit import QuantumCircuit

from ..base_meas_mit_generator import BaseMeasureErrorMitigationGenerator


class CTMPMeasureErrorMitigationGenerator(BaseMeasureErrorMitigationGenerator):
    """CTMP measurement mitigation generator."""

    # pylint: disable=arguments-differ

    def __init__(self, num_qubits: int):
        """Initialize measurement mitigation calibration generator."""
        circuits = []
        labels = []

        label0 = num_qubits * '0'
        circ0 = QuantumCircuit(num_qubits, name='cal_' + label0)
        circ0.measure_all()

        label1 = num_qubits * '1'
        circ1 = QuantumCircuit(num_qubits, name='cal_' + label1)
        circ1.x(list(range(num_qubits)))
        circ1.measure_all()

        circuits += [circ0, circ1]
        labels += [label0, label1]

        for i in range(num_qubits):
            label = ((num_qubits - i - 1) * '0') + '1' + (i * '0')
            circ = QuantumCircuit(num_qubits, name='cal_' + label)
            circ.x(i)
            circ.measure_all()
            circuits.append(circ)
            labels.append(label)
        super().__init__(num_qubits, circuits, labels)
