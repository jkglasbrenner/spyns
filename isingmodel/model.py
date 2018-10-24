# -*- coding: utf-8 -*-

from typing import Tuple

import numpy as np

from isingmodel.data import SimulationData
from isingmodel.lattice import BinaryLattice


def ising_test_flip(
    site_index: int,
    lattice: BinaryLattice,
    data: SimulationData,
) -> float:
    """Compute the change in energy for a trial spin flip.

    :param site_index: Perform trial spin flip on site specified by the index.
    :param lattice: Structural information and neighbor tables.
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

    :param lattice: Structural information and neighbor tables.
    :param data: Data container for the simulation.
    """
    data.estimators.energy = ising_total_energy(
        lattice=lattice,
        data=data,
    )
    data.estimators.magnetization = ising_total_magnetization(data=data)
    data.estimators.magnetization_even_sites = ising_total_magnetization_even_sites(
        lattice=lattice,
        data=data,
    )
    data.estimators.magnetization_odd_sites = ising_total_magnetization_odd_sites(
        lattice=lattice,
        data=data,
    )


def ising_total_energy(lattice: BinaryLattice, data: SimulationData) -> float:
    """Compute the total energy estimator for the lattice.

    :param lattice: Structural information and neighbor tables.
    :param data: Data container for the simulation.
    :return: Total energy of the simulation state.
    """
    total_energy: float = 0

    for site_index in range(lattice.number_sites):
        total_energy += ising_compute_site_energy(
            site_index=site_index,
            lattice=lattice,
            data=data,
        )

    return total_energy / 2.0


def ising_total_magnetization(data: SimulationData) -> float:
    """Compute the total magnetization estimator for the lattice.

    :param data: Data container for the simulation.
    :return: Total magnetization of the simulation state.
    """
    return data.state.sum()


def ising_total_magnetization_even_sites(
    lattice: BinaryLattice,
    data: SimulationData,
) -> float:
    """Compute the total magnetization estimator for the even sites on the lattice.

    :param lattice: Structural information and neighbor tables.
    :param data: Data container for the simulation.
    :return: Total magnetization of the simulation state.
    """
    return data.state[lattice.even_site_indices].sum()


def ising_total_magnetization_odd_sites(
    lattice: BinaryLattice,
    data: SimulationData,
) -> float:
    """Compute the total magnetization estimator for the even sites on the lattice.

    :param lattice: Structural information and neighbor tables.
    :param data: Data container for the simulation.
    :return: Total magnetization of the simulation state.
    """
    return data.state[lattice.odd_site_indices].sum()


def ising_compute_site_energy(
    site_index: int,
    lattice: BinaryLattice,
    data: SimulationData,
) -> float:
    """Compute a given site's energy.

    :param site_index: Site whose energy you want to compute.
    :param lattice: Structural information and neighbor tables.
    :param data: Data container for the simulation.
    :return: Energy of site specified by ``site_index``.
    """
    site_spin: float = data.state[site_index]
    neighbor_states: Tuple[np.ndarray, np.ndarray] = lattice.get_neighbor_states(
        site_index=site_index,
        state=data.state,
    )
    energy: float = np.sum(neighbor_states[1] * site_spin * neighbor_states[0])

    return energy
