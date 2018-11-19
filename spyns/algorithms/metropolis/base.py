# -*- coding: utf-8 -*-

import numpy as np


def pick_site(number_sites: int) -> int:
    """Pick a lattice site at random for the metropolis algorithm.

    :param number_sites: Number of sites in the lattice.
    :return: Index for randomly chosen site.
    """
    site_index: int = np.random.randint(low=0, high=number_sites)

    return site_index


def accept_or_reject(temperature: float, energy_difference: float) -> bool:
    """Accept or reject trial flip using the Metropolis algorithm.

    :param temperature: Simulation temperature.
    :param energy_difference: Energy difference for the trial spin flip.
    :return: Boolean specifying if trial flip was accepted or not.
    """
    accept: bool = True

    if energy_difference >= 0:
        acceptance_probability: float = proposal_distribution(
            energy_difference=energy_difference, temperature=temperature
        )
        random_number: float = np.random.uniform()
        accept = random_number <= acceptance_probability

    return accept


def proposal_distribution(energy_difference: float, temperature: float) -> float:
    """Compute proposal distribution for energy difference and simulation temperature.

    :param energy_difference: Energy difference for the trial sample.
    :param temperature: Temperature of the simulation.
    :return: Probability of accepting trial sample.
    """
    return np.exp(-energy_difference / temperature)
