# -*- coding: utf-8 -*-

import numpy as np

from isingmodel.data import SimulationData
from isingmodel.lattice import BinaryLattice
import isingmodel


def metropolis(lattice: BinaryLattice, data: SimulationData) -> None:
    """Update system state using the Metropolis algorithm.

    :param lattice: Structural information and neighbor tables.
    :param data: Data container for the simulation.
    """
    site_index: int = metropolis_pick_site(lattice=lattice)
    energy_difference: float = isingmodel.model.ising_test_flip(
        site_index=site_index,
        lattice=lattice,
        data=data,
    )
    accept_state: bool = metropolis_accept_or_reject(
        temperature=data.parameters.temperature,
        energy_difference=energy_difference,
    )
    if accept_state:
        metropolis_update_state(
            lattice=lattice,
            data=data,
            site_index=site_index,
            energy_difference=energy_difference,
        )


def metropolis_pick_site(lattice: BinaryLattice) -> int:
    """Pick a lattice site at random for the metropolis algorithm.

    :param lattice: Structural information and neighbor tables.
    :return: Index for randomly chosen site.
    """
    site_index: int = np.random.randint(low=0, high=lattice.number_sites)

    return site_index


def metropolis_accept_or_reject(temperature: float, energy_difference: float) -> bool:
    """Accept or reject trial spin flip using the Metropolis algorithm.

    :param temperature: Simulation temperature.
    :param energy_difference: Energy difference for the trial spin flip.
    :return: Boolean specifying if trial flip was accepted or not.
    """
    accept: bool = True

    if energy_difference >= 0:
        acceptance_probability: float = metropolis_proposal_distribution(
            energy_difference=energy_difference, temperature=temperature
        )
        random_number: float = np.random.uniform()
        accept = random_number <= acceptance_probability

    return accept


def metropolis_update_state(
    lattice: BinaryLattice,
    data: SimulationData,
    site_index: int,
    energy_difference: float,
) -> None:
    """Accept trial flip and update simulation state.

    :param lattice: Structural information and neighbor tables.
    :param data: Data container for the simulation.
    :param site_index: Index for randomly chosen site.
    :param energy_difference: Energy difference for the trial spin flip.
    """
    data.state[site_index] *= -1
    data.estimators.energy += energy_difference
    magnetization_change = 2 * data.state[site_index]
    data.estimators.magnetization += magnetization_change

    if site_index in lattice.even_site_indices:
        data.estimators.magnetization_even_sites += magnetization_change

    else:
        data.estimators.magnetization_odd_sites += magnetization_change


def metropolis_proposal_distribution(
    energy_difference: float, temperature: float
) -> float:
    """Compute proposal distribution for energy difference and simulation temperature.

    :param energy_difference: Energy difference for the trial sample.
    :param temperature: Temperature of the simulation.
    :return: Probability of accepting trial sample.
    """
    return np.exp(-energy_difference / temperature)
