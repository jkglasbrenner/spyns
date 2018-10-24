# -*- coding: utf-8 -*-

import numpy as np
import pytest

from isingmodel.lattice import BinaryLattice


@pytest.mark.parametrize(
    "dimensions,expected", [
        ((10, 10), 10 * 10),
        ((12, 15), 12 * 15),
        ((5, 7), 5 * 7),
    ]
)
def test_2d_initialization(dimensions, expected) -> None:
    lattice: BinaryLattice = BinaryLattice(dimensions, "Neumann", [1])
    state: np.ndarray = lattice.sample_random_state()
    lattice_shape: np.ndarray = np.array(state.shape)
    assert lattice_shape.prod() == expected


@pytest.mark.parametrize(
    "dimensions,site,neighborhood,neighbors,interaction", [
        ((10, 10), (0, 0), "Neumann", [[0, 1], [0, 9], [1, 0], [9, 0]], [1]),
        ((10, 10), (0, 0), "Moore", [[0, 1], [0, 9], [1, 0], [9, 0], [1, 1], [1, 9],
                                     [9, 1], [9, 9]], [1, 1]),
        ((10, 10), (4, 4), "Neumann", [[4, 5], [4, 3], [5, 4], [3, 4]], [1]),
        ((10, 10), (4, 4), "Moore", [[4, 5], [4, 3], [5, 4], [3, 4], [5, 5], [3, 5],
                                     [5, 3], [3, 3]], [1, 1]),
    ]
)
def test_finding_neighbor_indices(
    dimensions,
    site,
    neighborhood,
    neighbors,
    interaction,
) -> None:
    lattice: BinaryLattice = BinaryLattice(dimensions, neighborhood, interaction)
    neighbor_check: np.ndarray = lattice._get_neighbor_indices(
        site_index=site,
        neighborhood=lattice.neighborhood,
    )
    neighbors_verify = np.array(neighbors)
    neighbors_verify = neighbors_verify[:, 1] + neighbors_verify[:, 0] * dimensions[1]
    assert np.all(np.sort(neighbors_verify) == np.sort(neighbor_check))
