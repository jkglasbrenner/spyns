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
    site_index: Tuple[int, int] = metropolis_pick_site(data=data)
    energy_difference: float = isingmodel.model.ising_test_flip(
        site_index=site_index, lattice=lattice, data=data
    )
    accept_state: bool = metropolis_accept_or_reject(
        temperature=data.parameters.temperature, energy_difference=energy_difference
    )
    if accept_state:
        metropolis_update_state(
            lattice=lattice, data=data, site_index=site_index,
            energy_difference=energy_difference
        )


def metropolis_pick_site(data: SimulationData) -> Tuple[int, int]:
    """Pick a lattice site at random for the metropolis algorithm.

    :param data: Data container for the simulation.
    :return: Indices for randomly chosen site.
    """
    index1: int = np.random.randint(low=0, high=data.parameters.dimensions[0])
    index2: int = np.random.randint(low=0, high=data.parameters.dimensions[1])

    return index1, index2


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
    site_index: Tuple[int, int],
    energy_difference: float,
) -> None:
    """Method to update simulation state when trial flip is accepted. 

    :param lattice: Structural information and simulation state.
    :param data: Data container for the simulation.
    :param site_index: Indices for randomly chosen site.
    :param energy_difference: Energy difference for the trial spin flip.
    """
    lattice.state[site_index] *= -1
    data.state["energy"] += energy_difference
    data.state["magnetization"] += 2 * lattice.state[site_index]
