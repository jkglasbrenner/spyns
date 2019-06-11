from libc.math cimport exp
from spyns.data_cython cimport SimulationHeisenbergData_t

import cython


cdef long pick_site(SimulationHeisenbergData_t data):
    """Pick a lattice site at random for the metropolis algorithm.

    :param number_sites: Number of sites in the lattice.
    :return: Index for randomly chosen site.
    """
    cdef long site_index = data.random_number_generator.randint()

    return site_index


cdef bint accept_or_reject(double temperature, double energy_difference,
                           SimulationHeisenbergData_t data):
    """Accept or reject trial flip using the Metropolis algorithm.

    :param temperature: Simulation temperature.
    :param energy_difference: Energy difference for the trial spin flip.
    :return: Boolean specifying if trial flip was accepted or not.
    """
    cdef double acceptance_probability
    cdef double random_number

    cdef bint accept = True

    if energy_difference >= 0:
        acceptance_probability = proposal_distribution(
            energy_difference=energy_difference, temperature=temperature
        )
        random_number = data.random_number_generator.uniform()
        accept = random_number <= acceptance_probability

    return accept


cdef double proposal_distribution(double energy_difference, double temperature):
    """Compute proposal distribution for energy difference and simulation temperature.

    :param energy_difference: Energy difference for the trial sample.
    :param temperature: Temperature of the simulation.
    :return: Probability of accepting trial sample.
    """
    cdef double acceptance_probability = exp(-energy_difference / temperature)

    return acceptance_probability
