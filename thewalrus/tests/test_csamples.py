# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Tests for the classical sampling functions"""

import numpy as np
from thewalrus.csamples import (
    rescale_adjacency_matrix_thermal,
    generate_thermal_samples,
)

rel_tol = 10

def generate_positive_definite_matrix(n):
    r"""Generates a positive definite matrix of size n
	Args:
		n (int) : Size of the matrix
	Returns:
		array: Positive definite matrix
	"""
    A = np.random.rand(n, n)
    return A.T @ A

def test_rescaling_thermal():
    r"""Test that the rescaled eigenvalues of the matrix give the correct mean photon number for thermal rescaling"""

    n = 10
    A = generate_positive_definite_matrix(n)
    n_mean = 1.0
    ls, _ = rescale_adjacency_matrix_thermal(A, n_mean)
    assert np.allclose(n_mean, np.sum(ls / (1 - ls)))

def test_mean_thermal():
    r"""Test that the thermal samples have the correct mean photon number"""
    n = 10
    n_samples = 100000
    A = generate_positive_definite_matrix(n)
    n_mean = 10.0
    ls, O = rescale_adjacency_matrix_thermal(A, n_mean)
    samples = np.array(generate_thermal_samples(ls, O, num_samples=n_samples))
    tot_photon_sample = np.sum(samples, axis=1)
    n_mean_calc = tot_photon_sample.mean()
    assert np.allclose(n_mean_calc, n_mean, rtol=10 / np.sqrt(n_samples))

def test_dist_thermal():
    r"""Test that the thermal sampling for a single mode produces the correct photon number distribution"""
    n_samples = 100000
    n_mean = 1.0
    A = generate_positive_definite_matrix(1)
    ls, O = rescale_adjacency_matrix_thermal(A, n_mean)

    samples = np.array(generate_thermal_samples(ls, O, num_samples=n_samples))
    bins = np.arange(0, max(samples), 1)
    (freq, _) = np.histogram(samples, bins=bins)
    rel_freq = freq / n_samples
    expected = (1 / (1 + n_mean)) * (n_mean / (1 + n_mean)) ** (np.arange(len(rel_freq)))
    assert np.allclose(rel_freq, expected, atol=10 / np.sqrt(n_samples))

def test_multimode_thermal()