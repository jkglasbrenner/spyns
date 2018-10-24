# -*- coding: utf-8 -*-

from typing import Tuple

import numpy as np

from isingmodel.data import SimulationData
from isingmodel.distributions import BinaryLattice
import isingmodel


def metropolis(lattice: BinaryLattice, data: SimulationData) -> None:
    """Update system state using the Metropolis algorithm.

    :param lattice: Structural information and simulation state.
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

    :param data: Data container for the simulation.
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
        acceptance_probability: float = isingmodel.distributions.proposal_distribution(
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
    """Method to update simulation state when trial flip is accepted. 

    :param lattice: Structural information and simulation state.
    :param data: Data container for the simulation.
    :param site_index: Index for randomly chosen site.
    :param energy_difference: Energy difference for the trial spin flip.
    """
    data.state[site_index] *= -1
    data.estimators.energy += energy_difference
    data.estimators.magnetization += 2 * data.state[site_index]
