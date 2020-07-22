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

from typing import Optional, Tuple, List

from qiskit import QuantumCircuit, transpile
from qiskit.providers import BaseBackend


class BaseMeasureErrorMitigationGenerator:
    """Base measurement error mitigation generator."""
    # pylint: disable=arguments-differ

    def __init__(self, num_qubits, circuits, labels):
        """Initialize measurement mitigation calibration generator."""
        self._num_qubits = num_qubits
        self._circuits = circuits
        self._metadata = []
        for label in labels:
            self._metadata.append({
                'experiment': 'meas_mit',
                'cal': label,
            })

    def run(self, initial_layout: Optional[List[int]] = None,
            backend: Optional[BaseBackend] = None) -> Tuple[List[QuantumCircuit], List[dict], dict]:
        """Return experiment payload data.
​
        Args:
            initial_lay
​
        Returns:
            tuple: (circuits, metadata, run_config)
​
        Circuits is a list of circuits for the experiments, metadata is a list of metadata
        for the experiment that is required by the fitter to interpreting results, run_config
        is a dictionary of parameters for configuring a backend (or running the experiment)
        """
        circuits = self.circuits(initial_layout=initial_layout,
                                 backend=backend)
        metadata = self.metadata(initial_layout=initial_layout,
                                 backend=backend)
        run_config = self.run_config(initial_layout=initial_layout,
                                     backend=backend)
        return circuits, metadata, run_config

    def circuits(self,
                 initial_layout: Optional[List[int]] = None,
                 backend: Optional[BaseBackend] = None) -> List[QuantumCircuit]:
        """Return only the circuits generated by the run command.
​
        Args:
            initial_layout: Optional, Initial layout
            backend: Optional, backend to transpile to.
​
        Returns:
            Generated circuits.
        """
        circuits = self._circuits
        if initial_layout or backend:
            if backend is None:
                # We need to provide a dummy coupling map for initial_layout
                # to work. So we apply a 1D layout here
                coupling_map = [[i - 1, i] for i in range(1, max(initial_layout))]
            else:
                coupling_map = None
            circuits = transpile(circuits,
                                 backend=backend,
                                 coupling_map=coupling_map,
                                 initial_layout=initial_layout)
        return circuits

    def metadata(self,
                 initial_layout: Optional[List[int]] = None,
                 backend: Optional[BaseBackend] = None) -> List[dict]:
        """Generate a list of metadata.
​
        Args:
            initial_layout: Optional, Initial layout.
            backend: Optional, backend to transpile to.
​
        Returns:
            Metdata dictionaries.
        """
        if initial_layout is None and backend is None:
            return self._metadata
        new_meta = []
        for meta in self._metadata:
            tmp = meta.copy()
            if initial_layout:
                tmp['initial_layout'] = initial_layout
            if backend:
                tmp['backend'] = backend.name
            new_meta.append(tmp)
        return new_meta

    def run_config(self,
                   initial_layout: Optional[List[int]] = None,
                   backend: Optional[BaseBackend] = None) -> dict:
        """Generate any backend config needed for execution.
        Args:
            initial_layout: Optional, Initial layout
            backend: Optional, backend to transpile to.
​
        Returns:
            Runconfig
        """
        # pylint: disable=unused-argument
        return {}
