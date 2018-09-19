# -*- coding: utf-8 -*-

from typing import List, Tuple

import numpy as np

from isingmodel.data import SimulationData
from isingmodel.distributions import BinaryLattice


def ising_test_flip(
    site_index: Tuple[int, int],
    lattice: BinaryLattice,
    data: SimulationData,
) -> float:
    """Compute the change in energy for a trial spin flip.

    :param site_index: Perform trial spin flip on site specified by these indices.
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
    data.state["energy"] = ising_total_energy(lattice=lattice, data=data)
    data.state["magnetization"] = ising_total_magnetization(lattice=lattice)


def ising_total_energy(lattice: BinaryLattice, data: SimulationData) -> float:
    """Compute the total energy estimator for the lattice.

    :param lattice: Structural information and simulation state.
    :param data: Data container for the simulation.
    :return: Total energy of the simulation state.
    """
    total_energy: float = 0
    site_index_list: List[Tuple[int, int]] = \
        [(index1, index2)
         for index1 in range(data.parameters.dimensions[0])
         for index2 in range(data.parameters.dimensions[1])]

    for site_index in site_index_list:
        total_energy += ising_compute_site_energy(
            site_index=site_index,
            lattice=lattice,
            data=data,
        )

    return total_energy / 2.0


def ising_total_magnetization(lattice: BinaryLattice) -> float:
    """Compute the total magnetization estimator for the lattice.

    :param lattice: Structural information and simulation state.
    :return: Total magnetization of the simulation state.
    """
    return lattice.state.sum()


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
    site_spin: float = lattice.state[site_index]
    site_neighbors: np.ndarray = lattice.get_neighbors_states(
        site_index=site_index,
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
