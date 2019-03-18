# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Measurement calibration circuits. To apply the measurement mitigation
use the fitters to produce a filter.
"""

from qiskit import QuantumRegister, ClassicalRegister, \
    QuantumCircuit, QiskitError
from ...verification.tomography import count_keys


def complete_meas_cal(qubit_list=None, qr=None, cr=None, circlabel=''):
    """
    Return a list of measurement calibration circuits for the full
    Hilbert space.

    Each of the 2**n circuits creates a basis state

    Args:
        qubit_list: A list of qubits to perform the measurement correction on,
        if None and qr is given then assumed to be performed over the entire
        qr. The calibration states will be labelled according to this ordering

        qr (QuantumRegister): A quantum register. If none one is created

        cr (ClassicalRegister): A classical register. If none one is created

        circlabel: A string to add to the front of circuit names for
        unique identification

    Returns:
        A list of QuantumCircuit objects containing the calibration circuits

        A list of calibration state labels

    Additional Information:
        The returned circuits are named circlabel+cal_XXX
        where XXX is the basis state,
        e.g., cal_1001

        Pass the results of these circuits to "MeasurementFitter" constructor
    """

    if qubit_list is None and qr is None:
        raise QiskitError("Must give one of a qubit_list or a qr")

    # Create the registers if not already done
    if qr is None:
        qr = QuantumRegister(max(qubit_list)+1)

    if qubit_list is None:
        qubit_list = range(len(qr))

    cal_circuits = []
    nqubits = len(qubit_list)

    # create classical bit registers
    if cr is None:
        cr = ClassicalRegister(nqubits)

    # labels for 2**n qubit states
    state_labels = count_keys(nqubits)

    for basis_state in state_labels:
        qc_circuit = QuantumCircuit(qr, cr,
                                    name='%scal_%s' % (circlabel, basis_state))
        for qind, _ in enumerate(basis_state):
            if int(basis_state[nqubits-qind-1]):
                # the index labeling of the label is backwards with
                # the list
                qc_circuit.x(qr[qubit_list[qind]])

            # add measurements
            qc_circuit.measure(qr[qubit_list[qind]], cr[qind])

        cal_circuits.append(qc_circuit)

    return cal_circuits, state_labels


def tensored_meas_cal(qubit_list=None, qr=None, cr=None, circlabel=''):
    """
    Return a list of calibration circuits for the all zeros and all ones basis states.

    Args:
        qubit_list: A list of qubits to perform the measurement correction on,
        if None and qr is given then assumed to be performed over the entire
        qr.

        qr (QuantumRegister): A quantum register. If none one is created

        cr (ClassicalRegister): A classical register. If none one is created

        circlabel: A string to add to the front of circuit names for
        unique identification

    Returns:
        A list of two QuantumCircuit objects containing the calibration circuits

        A list of calibration state labels

    Additional Information:
        The returned circuits are named circlabel+cal_XXX
        where XXX is the basis state,
        i.e., cal_000 and cal_111

        Pass the results of these circuits to "MeasurementFitter" constructor
    """

    if qubit_list is None and qr is None:
        raise QiskitError("Must give one of a qubit_list or a qr")

    # Create the registers if not already done
    if qr is None:
        qr = QuantumRegister(max(qubit_list)+1)

    if qubit_list is None:
        qubit_list = range(len(qr))

    nqubits = len(qubit_list)

    # create classical bit registers
    if cr is None:
        cr = ClassicalRegister(nqubits)

    zero_label = ''.zfill(nqubits)
    zero_circ = QuantumCircuit(qr, cr,
                               name=circlabel+'cal_'+zero_label)

    for qind, qubit in enumertae(qubit_list):
        zero_circ.measure(qr[qubit], cr[qind])

    one_label = zero_label
    for qubit in qubit_list:
        one_label[nqubits-qubit-1] = 1
    one_circ = QuantumCircuit(qr, cr, name=circlabel+'cal_'+one_label)

    for qind, qubit in enumerate(qubit_list):
        one_circ.x(qr[qubit])
        one_circ.measure(qr[qubit], cr[qind])

    return [zero_circ, one_circ], [zero_label, one_label]
