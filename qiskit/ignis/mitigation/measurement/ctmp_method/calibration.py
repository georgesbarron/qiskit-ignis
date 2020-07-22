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
"""Perform CTMP calibration for error mitigation.
"""

# pylint: disable=missing-function-docstring

from abc import abstractmethod
from copy import deepcopy
from itertools import combinations, product
import logging
from typing import List, Union, Tuple, Dict, Set

import numpy as np
from scipy.linalg import logm
from scipy import sparse

from qiskit import QuantumCircuit
from qiskit.result import Result

logger = logging.getLogger(__name__)

"""Generators are uniquely determined by two bitstrings,
and a list of qubits on which the bitstrings act. For instance,
the generator `|b^i><a^i| - |a^i><a^i|` acting on the (ordered)
set `C_i` is represented by `g = ('1', '0', [5])`
"""
Generator = Tuple[str, str, Tuple[int]]


_KET_BRA_DICT = {
    '00': np.array([[1, 0], [0, 0]]),
    '01': np.array([[0, 1], [0, 0]]),
    '10': np.array([[0, 0], [1, 0]]),
    '11': np.array([[0, 0], [0, 1]])
}


def tensor_list(l: List[np.array]) -> np.array: # pylint: disable=invalid-name
    """Given a list [a, b, c, ...], return
    the array a otimes b otimes c otimes ...
    """
    res = l[0]
    for m in l[1:]:
        res = sparse.kron(res, m)
    return res


def generator_to_sparse_matrix(gen: Generator, num_qubits: int) -> sparse.coo_matrix:
    b, a, c = gen
    shape = (2**num_qubits,) * 2
    res = sparse.coo_matrix(shape)
    ba_strings = list(map(lambda x: ''.join(x), list(zip(*[b, a]))))
    aa_strings = list(map(lambda x: ''.join(x), list(zip(*[a, a]))))
    ba_mats = [sparse.eye(2, 2).tocoo()] * num_qubits
    aa_mats = [sparse.eye(2, 2).tocoo()] * num_qubits
    for _c, _ba, _aa in zip(c, ba_strings, aa_strings):
        ba_mats[_c] = _KET_BRA_DICT[_ba]
        aa_mats[_c] = _KET_BRA_DICT[_aa]
    res += tensor_list(ba_mats[::-1])
    res -= tensor_list(aa_mats[::-1])
    return res


def str_del_inds(in_str: str, indices: List[int]) -> str:
    l = list(in_str[::-1])
    for i in sorted(indices, reverse=True):
        del l[i]
    return ''.join(l[::-1])


def delete_qubits(input_str: str, q_list: Set[int]) -> str:
    num_qubits = len(input_str)
    q_inds = [num_qubits-i-1 for i in q_list]
    l = list(input_str)
    for i in sorted(q_inds, reverse=True):
        del l[i]
    return ''.join(l)


def match_on_set(str_1: str, str_2: str, qubits: Set[int]) -> bool:
    num_qubits = len(str_1)
    if len(str_1) != len(str_2):
        raise ValueError('Strings must have same length')
    q_inds = [num_qubits - i - 1 for i in qubits]
    for i in q_inds:
        if str_1[i] != str_2[i]:
            return False
    return True


def no_error_out_set(in_set: List[int], counts_dict: Dict[str, int], input_state: str) -> Dict[str, int]:
    """Given a counts dictionary, a desired bitstring, and an "input set", return the dictionary of counts
    where there are no errors on qubits not in `in_set`, as determined by `input_state`.
    """
    output_dict = {}
    num_qubits = len(input_state)
    out_set = set(range(num_qubits)) - in_set
    for output_state, counts in counts_dict.items():
        if match_on_set(output_state, input_state, out_set):
            output_dict[output_state] = counts
    return output_dict


def compute_gamma(g_matrix: sparse.csr_matrix) -> float:
    if isinstance(g_matrix, sparse.csr_matrix):
        _g_matrix = g_matrix
    else:
        _g_matrix = sparse.csr_matrix(g_matrix)
    cg = -_g_matrix.tocoo()
    current_max = -np.inf
    logger.info('Computing gamma...')
    for i, j, v in zip(cg.row, cg.col, cg.data):
        if i == j:
            if v > current_max:
                current_max = v
    logger.info('Computed gamma={}'.format(current_max))
    if current_max < 0:
        raise ValueError('gamma should be non-negative, found gamma={}'.format(current_max))
    return current_max


def local_a_matrix(j: int, k: int, counts_dicts: Dict[str, Dict[str, int]]) -> np.array:
    """Computes the A(j,k) matrix in the basis:
    00, 01, 10, 11
    """
    if j == k:
        raise ValueError('Encountered j=k={}'.format(j))
    a_out = np.zeros((4, 4))
    indices = ['00', '01', '10', '11']
    index_dict = {b: int(b, 2) for b in indices}
    for w, v in product(indices, repeat=2):
        v_to_w_err_cts = 0
        tot_cts = 0
        for input_str, c_dict in counts_dicts.items():
            if input_str[::-1][j] == v[0] and input_str[::-1][k] == v[1]:
                no_err_out_dict = no_error_out_set({j, k}, c_dict, input_str)
                tot_cts += np.sum(list(no_err_out_dict.values()))
                for output_str, counts in no_err_out_dict.items():
                    if output_str[::-1][j] == w[0] and output_str[::-1][k] == w[1]:
                        v_to_w_err_cts += counts
        a_out[index_dict[w], index_dict[v]] = v_to_w_err_cts / tot_cts
    return a_out


class BaseGeneratorSet:

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self._generators = []

    def __len__(self) -> int:
        return len(self._generators)

    def __getitem__(self, i: int) -> Generator:
        return self._generators[i]

    def __list__(self) -> List[Generator]:
        return self._generators

    def add_generators(self, gen_list: List[Generator]):
        self._generators.extend(gen_list)

    def __add__(self, other):
        if self.num_qubits != other.num_qubits:
            raise ValueError('Generator sets must act on same number of qubits')
        res = deepcopy(self)
        res.add_generators(list(other))
        return res

    @abstractmethod
    def get_ctmp_error_rate(self, gen: Generator, g_mat_dict: Dict[Generator, np.array]) -> float:
        pass

    @abstractmethod
    def local_g_matrix(self, gen: Generator, counts_dicts: Dict[str, Dict[str, int]]) -> np.array:
        pass
    
    @abstractmethod
    def supplementary_generators(self, gen_list: List[Generator]) -> List[Generator]:
        """These generators do not have rates directly associated with them, but are used to compute
        rates for other generators.
        """
        pass
    
    @classmethod
    def from_generator_list(cls, gen_list: List[Generator], num_qubits: int):
        res = cls(num_qubits=num_qubits)
        res.add_generators(gen_list=gen_list)
        return res


class StandardGeneratorSet(BaseGeneratorSet):

    @staticmethod
    def standard_single_qubit_bitstrings(num_qubits: int) -> List[Generator]:
        """Returns a list of tuples `[(C_1, b_1, a_1), (C_2, b_2, a_2), ...]` that represent
        the generators .
        """
        res = [('1', '0', (i,)) for i in range(num_qubits)]
        res += [('0', '1', (i,)) for i in range(num_qubits)]
        if len(res) != 2*num_qubits:
            raise ValueError('Should have gotten 2n qubits, got {}'.format(len(res)))
        return res

    @staticmethod
    def standard_two_qubit_bitstrings_symmetric(num_qubits: int, pairs=None) -> List[Generator]:
        """
        """
        if pairs is None:
            pairs = list(combinations(range(num_qubits), r=2))
        res = [('11', '00', (i, j)) for i, j in pairs if i < j]
        res += [('00', '11', (i, j)) for i, j in pairs if i < j]
        if len(res) != num_qubits*(num_qubits-1):
            raise ValueError('Should have gotten n(n-1) qubits, got {}'.format(len(res)))
        return res

    @staticmethod
    def standard_two_qubit_bitstrings_asymmetric(num_qubits: int, pairs=None) -> List[Generator]:
        """
        """
        if pairs is None:
            pairs = list(combinations(range(num_qubits), r=2))
        res = [('10', '01', (i, j)) for i, j in pairs]
        res += [('10', '01', (j, i)) for i, j in pairs]
        if len(res) != num_qubits*(num_qubits-1):
            raise ValueError('Should have gotten n(n-1) qubits, got {}'.format(len(res)))
        return res

    @classmethod
    def from_num_qubits(cls, num_qubits: int, pairs=None):
        if not isinstance(num_qubits, int):
            raise ValueError('num_qubits needs to be an int')
        if num_qubits <= 0:
            raise ValueError('Need num_qubits at least 1')
        res = cls(num_qubits=num_qubits)
        res.add_generators(res.standard_single_qubit_bitstrings(num_qubits))
        if num_qubits > 1:
            res.add_generators(res.standard_two_qubit_bitstrings_symmetric(num_qubits, pairs=pairs))
            res.add_generators(res.standard_two_qubit_bitstrings_asymmetric(num_qubits, pairs=pairs))
            if len(res) != 2*num_qubits**2:
                raise ValueError('Should have gotten 2n^2 generators, got {}...'.format(len(res)))
        return res
    
    def supplementary_generators(self, gen_list: List[Generator]) -> List[Generator]:
        pairs = {tuple(gen[2]) for gen in gen_list}
        supp_gens = []
        for tup in pairs:
            supp_gens.append(('10', '00', tup))
            supp_gens.append(('00', '10', tup))
            supp_gens.append(('11', '01', tup))
            supp_gens.append(('01', '11', tup))
        return supp_gens

    def get_ctmp_error_rate(self, gen: Generator, g_mat_dict: Dict[Generator, np.array]) -> float:
        b, a, c = gen
        if gen not in self._generators:
            raise ValueError('Invalid generator encountered: {}'.format(gen))
        if len(b) == 1:
            rate = self._ctmp_err_rate_1_q(a=a, b=b, j=c[0], g_mat_dict=g_mat_dict)
        elif len(b) == 2:
            rate = self._ctmp_err_rate_2_q(gen, g_mat_dict)
        logger.info('Generator {} calibrated with error rate {}'.format(
            gen, rate
        ))
        return rate
    
    def _ctmp_err_rate_1_q(self, a: str, b: str, j: int, g_mat_dict: Dict[Generator, np.array]) -> float:
        rate_list = []
        if a == '0' and b == '1':
            g1 = ('00', '10')
            g2 = ('01', '11')
        elif a == '1' and b == '0':
            g1 = ('10', '00')
            g2 = ('11', '01')
        else:
            raise ValueError('Invalid a,b encountered...')
        for k in range(self.num_qubits):
            if k == j:
                continue
            c = (j, k)
            for g_strs in [g1, g2]:
                gen = g_strs + (c,)
                r = self._ctmp_err_rate_2_q(gen, g_mat_dict)
                rate_list.append(r)
        if len(rate_list) != 2*(self.num_qubits-1):
            raise ValueError('Rate list has wrong number of elements')
        rate = np.mean(rate_list)
        return rate

    def _ctmp_err_rate_2_q(self, gen, g_mat_dict) -> float:
        g_mat = g_mat_dict[gen]
        b, a, _ = gen
        r = g_mat[int(b, 2), int(a, 2)]
        return r

    def local_g_matrix(self, gen: Generator, counts_dicts: Dict[str, Dict[str, int]]) -> np.array:
        """Computes the G(j,k) matrix in the basis:
        00, 01, 10, 11
        """
        _, _, c = gen
        j, k = c
        a = local_a_matrix(j, k, counts_dicts)
        g = self.amat_to_gmat(a)
        if np.linalg.norm(np.imag(g)) > 1e-3:
            raise ValueError('Encountered complex entries in G_i={}'.format(g))
        g = np.real(g)
        for i in range(4):
            for j in range(4):
                if i != j:
                    if g[i, j] < 0:
                        logger.debug('Found negative element of size: {}'.format(g[i, j]))
                        g[i, j] = 0
        return g
    
    @staticmethod
    def amat_to_gmat(a_mat: np.array) -> np.array:
        return logm(a_mat)


class BaseCalibrationCircuitSet:

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.cal_circ_dict = {} # type: Dict[str, QuantumCircuit]

    @property
    def circs(self):
        return list(self.cal_circ_dict.values())
    
    def __dict__(self):
        return self.cal_circ_dict
    
    def bitstring_to_circ(self, bits: Union[str, int]) -> QuantumCircuit:
        if isinstance(bits, int):
            bitstring = np.binary_repr(bits, width=self.num_qubits) # type: str
        elif isinstance(bits, str):
            bitstring = bits # type: str
        else:
            raise ValueError('Input bits must be either str or int')
        circ = QuantumCircuit(self.num_qubits, name='cal-{}'.format(bitstring))
        for i, b in enumerate(bitstring[::-1]):
            if b == '1':
                circ.x(i)
        circ.measure_all()
        return circ
    
    def get_weight_1_str(self, index: int) -> str:
        out = ['0']*self.num_qubits
        out[index] = '1'
        return ''.join(out)[::-1]
    
    @classmethod
    def from_dict(cls, num_qubits: int, cal_circ_dict: Dict[str, QuantumCircuit]):
        res = cls(num_qubits)
        res.cal_circ_dict = cal_circ_dict
        return res


class StandardCalibrationCircuitSet(BaseCalibrationCircuitSet):

    @classmethod
    def from_num_qubits(cls, num_qubits: int):
        res = cls(num_qubits=num_qubits)
        cal_strings = ['0'*num_qubits, '1'*num_qubits]
        cal_strings.extend([res.get_weight_1_str(i) for i in range(num_qubits)])
        res.cal_circ_dict = {cal_str: res.bitstring_to_circ(cal_str) for cal_str in cal_strings}
        return res


class WeightTwoCalibrationCircuitSet(BaseCalibrationCircuitSet):

    @classmethod
    def from_num_qubits(cls, num_qubits: int):
        res = cls(num_qubits=num_qubits)
        cal_strings = []
        cal_strings.extend(['0'*num_qubits])
        cal_strings.extend([res.get_weight_1_str(i) for i in range(num_qubits)])
        for i, j in product(range(num_qubits), repeat=2):
            if i != j:
                two_hot_str = ['0']*num_qubits
                two_hot_str[i] = '1'
                two_hot_str[j] = '1'
                cal_strings.append(''.join(two_hot_str))
        res.cal_circ_dict = {cal_str: res.bitstring_to_circ(cal_str) for cal_str in cal_strings}
        return res


class MeasurementCalibrator:
    
    def __init__(
            self, 
            cal_circ_set: BaseCalibrationCircuitSet,
            gen_set: BaseGeneratorSet
        ):
        self.cal_circ_set = cal_circ_set
        self.gen_set = gen_set
        self._num_qubits = self.gen_set.num_qubits
        self._gamma = None
        self._r_dict = {} # type: Dict[Generator, float]
        self._tot_g_mat = None
        self._b_mat = None
        self.calibrated = False

    @classmethod
    def standard_construction(cls, num_qubits: int):
        res = cls(
            cal_circ_set=WeightTwoCalibrationCircuitSet.from_num_qubits(num_qubits),
            gen_set=StandardGeneratorSet.from_num_qubits(num_qubits)
        )
        return res

    @property
    def gamma(self) -> float:
        if not self.calibrated:
            raise ValueError('Calibration has not been run yet')
        return self._gamma

    @property
    def r_dict(self) -> Dict[Generator, float]:
        if not self.calibrated:
            raise ValueError('Calibration has not been run yet')
        return self._r_dict

    @property
    def G(self) -> float:
        if not self.calibrated:
            raise ValueError('Calibration has not been run yet')
        return self._tot_g_mat

    @property
    def B(self):
        if not self.calibrated:
            raise ValueError('Calibration has not been run yet')
        return self._b_mat

    def __repr__(self):
        res = "Operator error mitigation calibrator\n"
        res += "Num generators: {}, Num qubits: {}\n".format(len(self.gen_set), self._num_qubits)
        if self.calibrated:
            r_min = np.min(list(self.r_dict.values()))
            r_max = np.max(list(self.r_dict.values()))
            r_mean = np.mean(list(self.r_dict.values()))
            res += "gamma={}, r_mean={}\nr_min={}, r_max={}".format(self.gamma, r_mean, r_min, r_max)
        else:
            res += "Not yet calibrated"
        return res

    def circ_dicts(self, result: Result) -> Dict[str, Dict[str, int]]:
        circ_dicts = {}
        for bits, circ in self.cal_circ_set.cal_circ_dict.items():
            circ_dicts[bits] = result.get_counts(circ)
        return circ_dicts

    def calibrate(self, result: Result) -> Tuple[float, Dict[Generator, float]]:
        logger.info('Beginning calibration with {} generators on {} qubits'.format(
            len(self.gen_set), self._num_qubits
        ))
        gen_mat_dict = {}
        # Compute G(j,k) matrices
        logger.info('Computing local G matrices...')
        circ_dicts = self.circ_dicts(result)
        for gen in list(self.gen_set)+self.gen_set.supplementary_generators(list(self.gen_set)):
            if len(gen[2]) > 1:
                mat = self.gen_set.local_g_matrix(gen, circ_dicts)
                gen_mat_dict[gen] = mat
        logger.info('Computed local G matrices')
        # Compute r-parameters
        logger.info('Computing generator coefficients...')
        for gen in self.gen_set:
            r = self.gen_set.get_ctmp_error_rate(gen, gen_mat_dict)
            self._r_dict[gen] = r
            logger.info('Generator G={} has error rate r={}'.format(gen, r))
        logger.info('Computed generator coefficients')
        # Compute gamma
        self._tot_g_mat = self.total_g_matrix(self._r_dict)
        self._gamma = compute_gamma(self._tot_g_mat)
        self._b_mat = sparse.eye(2**self._num_qubits) + self._tot_g_mat / self._gamma
        self._b_mat = self._b_mat.tocsc()
        logger.info('Finished calibration...')
        logger.info('num_qubits = {}, gamma = {}'.format(
            self._num_qubits, self._gamma
        ))
        r_vec = np.array(list(self._r_dict.values()))
        logger.info('r_min={}, r_max={}, r_mean={}'.format(
            np.min(r_vec), np.max(r_vec), np.mean(r_vec)
        ))
        logger.info('Finished calibration')
        self.calibrated = True

        return self.gamma, self.r_dict

    def total_g_matrix(self, r_dict: Dict[Generator, float]) -> sparse.csr_matrix:
        res = sparse.csr_matrix((2**self._num_qubits, 2**self._num_qubits), dtype=np.float)
        logger.info('Computing sparse G matrix')
        for gen, r in r_dict.items():
            res += r*generator_to_sparse_matrix(gen, self._num_qubits)
        num_elts = res.shape[0]**2
        try:
            nnz = res.nnz
            if nnz == num_elts:
                sparsity = '+inf'
            else:
                sparsity = nnz / (num_elts - nnz)
            logger.info('Computed sparse G matrix with sparsity {}'.format(sparsity))
        except AttributeError:
            pass
        return res
