# -*- coding: utf-8 -*-

from typing import List, Tuple

import numpy as np

from isingmodel.data import SimulationData
from isingmodel.distributions import BinaryLattice


def ising_test_flip(
    site_index: int,
    lattice: BinaryLattice,
    data: SimulationData,
) -> float:
    """Compute the change in energy for a trial spin flip.

    :param site_index: Perform trial spin flip on site specified by the index.
    :param lattice: Structural information and simulation state.
    :param data: Data container for the simulation.
    :return: Change in energy for trial spin flip.
    """
    change_in_energy: float = -2.0 * ising_compute_site_energy(
        site_index=site_index,
        lattice=lattice,
        data=data,
    )

    return change_in_energy


def ising_save_full_state(lattice: BinaryLattice, data: SimulationData) -> None:
    """Compute the total energy and total magnetization estimators for the lattice.

    :param lattice: Structural information and simulation state.
    :param data: Data container for the simulation.
    """
    data.estimators.energy = ising_total_energy(
        lattice=lattice,
        data=data,
    )
    data.estimators.magnetization = ising_total_magnetization(data=data)


def ising_total_energy(lattice: BinaryLattice, data: SimulationData) -> float:
    """Compute the total energy estimator for the lattice.

    :param lattice: Structural information and simulation state.
    :param data: Data container for the simulation.
    :return: Total energy of the simulation state.
    """
    total_energy: float = 0
    site_index_list: List[int] = list(range(lattice.number_sites))

    for site_index in site_index_list:
        total_energy += ising_compute_site_energy(
            site_index=site_index,
            lattice=lattice,
            data=data,
        )

    return total_energy / 2.0


def ising_total_magnetization(data: SimulationData) -> float:
    """Compute the total magnetization estimator for the lattice.

    :param lattice: Structural information and simulation state.
    :return: Total magnetization of the simulation state.
    """
    return data.state.sum()


def ising_compute_site_energy(
    site_index: Tuple[int, int],
    lattice: BinaryLattice,
    data: SimulationData,
) -> float:
    """Compute a given site's energy.

    :param site_index: Perform trial spin flip on site specified by these indices.
    :param lattice: Structural information and simulation state.
    :param data: Data container for the simulation.
    :return: Energy of site specified by ``site_index``.
    """
    row_index: int = int(site_index // lattice.dimensions[1])
    column_index: int = int(site_index % lattice.dimensions[1])
    site_spin: float = data.state[site_index]
    site_neighbors: np.ndarray = lattice.get_neighbors_states(
        site_index=(row_index, column_index),
        state=data.state,
        neighborhood=data.parameters.neighborhood,
    )
    energy: float = (
        data.parameters.interaction_coefficients[0] * site_spin *
        np.sum(site_neighbors[:4])
    )

    if data.parameters.neighborhood == "Moore":
        energy += (
            data.parameters.interaction_coefficients[1] * site_spin *
            np.sum(site_neighbors[4:])
        )

    return energy
