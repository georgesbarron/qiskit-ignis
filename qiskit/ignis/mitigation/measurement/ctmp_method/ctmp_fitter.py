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
import logging
from typing import List, Dict, Set
import itertools as it
import numpy as np
import scipy.linalg as la

from qiskit.exceptions import QiskitError
from qiskit.result import Result
from .ctmp_mitigator import CTMPMeasMitigator
from .ctmp_generator_set import Generator, standard_generator_set
from ..meas_mit_utils import filter_calibration_data

logger = logging.getLogger(__name__)


def fit_ctmp_meas_mitigator(result: Result,
                            metadata: List[Dict[str, any]],
                            generators: List[Generator] = None) -> CTMPMeasMitigator:
    """Return FullMeasureErrorMitigator from result data.

    Args:
        result: Qiskit result object.
        metadata: mitigation generator metadata.

    Returns:
        Measurement error mitigator object.
    """
    # Filter mitigation calibration data
    cal_data, num_qubits = filter_calibration_data(result, metadata)
    if generators is None:
        generators = standard_generator_set(num_qubits)

    gen_mat_dict = {}
    for gen in generators + _supplementary_generators(generators):
        if len(gen[2]) > 1:
            mat = _local_g_matrix(gen, cal_data)
            gen_mat_dict[gen] = mat

    # Compute rates for generators
    rates = [_get_ctmp_error_rate(gen, gen_mat_dict, num_qubits) for gen in generators]
    return CTMPMeasMitigator(generators, rates)


# Utility functions used for fitting (Should be moved to Fitter class)

def _ctmp_err_rate_1_q(a: str, b: str, j: int,
                       g_mat_dict: Dict[Generator, np.array],
                       num_qubits: int) -> float:
    """Compute the 1q error rate for a given generator.
    """
    rate_list = []
    if a == '0' and b == '1':
        g1 = ('00', '10')
        g2 = ('01', '11')
    elif a == '1' and b == '0':
        g1 = ('10', '00')
        g2 = ('11', '01')
    else:
        raise ValueError('Invalid a,b encountered...')
    for k in range(num_qubits):
        if k == j:
            continue
        c = (j, k)
        for g_strs in [g1, g2]:
            gen = g_strs + (c,)
            r = _ctmp_err_rate_2_q(gen, g_mat_dict)
            rate_list.append(r)
    if len(rate_list) != 2 * (num_qubits - 1):
        raise ValueError('Rate list has wrong number of elements')
    rate = np.mean(rate_list)
    return rate


def _ctmp_err_rate_2_q(gen, g_mat_dict) -> float:
    """Compute the 2 qubit error rate for a given generator.
    """
    g_mat = g_mat_dict[gen]
    b, a, _ = gen
    r = g_mat[int(b, 2), int(a, 2)]
    return r


def _get_ctmp_error_rate(gen: Generator,
                         g_mat_dict: Dict[Generator, np.array],
                         num_qubits: int) -> float:
    """Compute the error rate r_i for generator G_i.

    Args:
        gen (Generator): Generator to calibrate.
        g_mat_dict (Dict[Generator, np.array]): Dictionary of local G(j,k) matrices.
        num_qubits (int): number of qubits.

    Returns:
        float: The coefficient r_i for generator G_i.

    Raises:
        ValueError: The provided generator is not already in the set of generators.
    """
    b, a, c = gen
    if len(b) == 1:
        rate = _ctmp_err_rate_1_q(a, b, c[0], g_mat_dict, num_qubits)
    elif len(b) == 2:
        rate = _ctmp_err_rate_2_q(gen, g_mat_dict)
    return rate


def _match_on_set(str_1: str, str_2: str, qubits: Set[int]) -> bool:
    """Ask whether or not two bitstrings are equal on a set of bits.

    Args:
        str_1 (str): First string.
        str_2 (str): Second string.
        qubits (Set[int]): Qubits to check.

    Returns:
        bool: Whether or not the strings match on the given indices.

    Raises:
        ValueError: When the strings do not have equal length.
    """
    num_qubits = len(str_1)
    if len(str_1) != len(str_2):
        raise ValueError('Strings must have same length')
    q_inds = [num_qubits - i - 1 for i in qubits]
    for i in q_inds:
        if str_1[i] != str_2[i]:
            return False
    return True


def _no_error_out_set(
        in_set: Set[int],
        counts: Dict[str, int],
        input_state: str
) -> Dict[str, int]:
    """Given a counts dictionary, a desired bitstring, and an "input set",
    return the dictionary of counts where there are no errors on qubits
    not in `in_set`, as determined by `input_state`.
    """
    output_dict = {}
    num_qubits = len(input_state)
    out_set = set(range(num_qubits)) - in_set
    for output_state, freq in counts.items():
        if _match_on_set(output_state, input_state, out_set):
            output_dict[output_state] = freq
    return output_dict


def _local_a_matrix(j: int, k: int, cal_data: Dict[str, Dict[str, int]]) -> np.array:
    """Computes the A(j,k) matrix in the basis [00, 01, 10, 11]."""
    if j == k:
        raise QiskitError('Encountered j=k={}'.format(j))
    a_out = np.zeros((4, 4))
    indices = ['00', '01', '10', '11']
    index_dict = {b: int(b, 2) for b in indices}
    for w, v in it.product(indices, repeat=2):
        v_to_w_err_cts = 0
        tot_cts = 0
        for input_str, c_dict in cal_data.items():
            if input_str[::-1][j] == v[0] and input_str[::-1][k] == v[1]:
                no_err_out_dict = _no_error_out_set({j, k}, c_dict, input_str)
                tot_cts += np.sum(list(no_err_out_dict.values()))
                for output_str, counts in no_err_out_dict.items():
                    if output_str[::-1][j] == w[0] and output_str[::-1][k] == w[1]:
                        v_to_w_err_cts += counts
        a_out[index_dict[w], index_dict[v]] = v_to_w_err_cts / tot_cts
    return a_out


def _local_g_matrix(gen: Generator, cal_data: Dict[str, Dict[str, int]]) -> np.array:
    """Computes the G(j,k) matrix in the basis [00, 01, 10, 11]."""
    _, _, c = gen
    j, k = c
    a_mat = _local_a_matrix(j, k, cal_data)
    g = la.logm(a_mat)
    if np.linalg.norm(np.imag(g)) > 1e-3:
        raise QiskitError('Encountered complex entries in G_i={}'.format(g))
    g = np.real(g)
    for i in range(4):
        for j in range(4):
            if i != j:
                if g[i, j] < 0:
                    g[i, j] = 0
    return g


def _supplementary_generators(gen_list: List[Generator]) -> List[Generator]:
    """Supplementary generators needed to run 1q calibrations.

    Args:
        gen_list (List[Generator]): List of generators.

    Returns:
        List[Generator]: List of additional generators needed.
    """
    pairs = {tuple(gen[2]) for gen in gen_list}
    supp_gens = []
    for tup in pairs:
        supp_gens.append(('10', '00', tup))
        supp_gens.append(('00', '10', tup))
        supp_gens.append(('11', '01', tup))
        supp_gens.append(('01', '11', tup))
    return supp_gens