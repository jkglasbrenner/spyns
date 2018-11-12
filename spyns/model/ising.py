# -*- coding: utf-8 -*-

from typing import Tuple

import numpy as np

from spyns.data import SimulationData
import spyns


def sample_random_state(number_sites: int) -> np.ndarray:
    """Generate sample of random states on the Ising lattice.

    :param number_sites: Number of sites in the lattice.
    :return: Array of random states on the Ising lattice.
    """
    return np.random.choice(a=[-1, 1], size=number_sites, replace=True)


def flip(
    site_index: int,
    data: SimulationData,
) -> float:
    """Compute the change in energy for a trial spin flip.

    :param site_index: Perform trial spin flip on site specified by the index.
    :param data: Data container for the simulation.
    :return: Change in energy for trial spin flip.
    """
    change_in_energy: float = -2.0 * compute_site_energy(
        site_index=site_index,
        data=data,
    )

    return change_in_energy


def keep_flip_and_update_state(
    data: SimulationData,
    site_index: int,
    energy_difference: float,
) -> None:
    """Keep flip and update simulation state.

    :param data: Data container for the simulation.
    :param site_index: Index for randomly chosen site.
    :param energy_difference: Energy difference for the trial spin flip.
    """
    sublattice_index: int = data.lookup_tables.sublattice_table[site_index]

    data.state[site_index] *= -1
    magnetization_change = 2 * data.state[site_index]

    data.estimators.energy += energy_difference
    data.estimators.magnetization[sublattice_index] += magnetization_change


def save_full_state(data: SimulationData) -> None:
    """Compute the total energy and total magnetization estimators for the lattice.

    :param data: Data container for the simulation.
    """
    data.estimators.energy = compute_total_energy(data=data)
    data.estimators.magnetization[:] = compute_sublattice_magnetization(data=data)


def compute_total_energy(data: SimulationData) -> float:
    """Compute the total energy estimator for the lattice.

    :param data: Data container for the simulation.
    :return: Total energy of the simulation state.
    """
    total_energy: float = 0

    for site_index in range(data.lookup_tables.number_sites):
        total_energy += compute_site_energy(
            site_index=site_index,
            data=data,
        )

    return total_energy / 2.0


def compute_sublattice_magnetization(data: SimulationData) -> np.ndarray:
    """Compute the total magnetization estimator for the lattice.

    :param data: Data container for the simulation.
    :return: Total magnetization of the simulation state.
    """
    sublattice_indices: np.ndarray = data.lookup_tables.sublattice_table
    magnetization: np.ndarray = np.zeros(shape=data.lookup_tables.number_sublattices)

    for sublattice in range(data.lookup_tables.number_sublattices):
        magnetization[sublattice] = data.state[sublattice_indices == sublattice].sum()

    return magnetization


def compute_site_energy(
    site_index: int,
    data: SimulationData,
) -> float:
    """Compute a given site's energy.

    :param site_index: Site whose energy you want to compute.
    :param data: Data container for the simulation.
    :return: Energy of site specified by ``site_index``.
    """
    site_spin: float = data.state[site_index]
    neighbor_states: Tuple[np.ndarray, np.ndarray] = \
        spyns.model.base.lookup_neighbor_states(
            site_index=site_index,
            data=data,
        )
    energy: float = np.sum(neighbor_states[1] * site_spin * neighbor_states[0])

    return energy
