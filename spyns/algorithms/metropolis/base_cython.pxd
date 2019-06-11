from spyns.random_numbers.distribution cimport RandomNumberGenerator
from spyns.data_cython cimport SimulationHeisenbergData_t

cdef long pick_site(SimulationHeisenbergData_t data)
cdef bint accept_or_reject(double temperature, double energy_difference,
                           SimulationHeisenbergData_t data)
cdef double proposal_distribution(double energy_difference, double temperature)
