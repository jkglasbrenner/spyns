from libcpp.vector cimport vector

from spyns.data_cython cimport SimulationHeisenbergData_t


cdef class TrialFlip_t:
    cdef double energy_difference
    cdef vector[double] current_spin_vector
    cdef vector[double] trial_spin_vector


cdef class NeighborStates_t:
    cdef vector[double] x
    cdef vector[double] y
    cdef vector[double] z
    cdef vector[double] interaction_parameters


cdef object sample_random_state(long number_sites)
cdef vector[double] sample_random_spin_vector(SimulationHeisenbergData_t data)
cdef TrialFlip_t flip(long site_index, SimulationHeisenbergData_t data)
cdef void keep_flip_and_update_state(SimulationHeisenbergData_t data, long site_index,
                                     TrialFlip_t trial_flip)
cpdef void save_full_state(SimulationHeisenbergData_t data)
cpdef double compute_total_energy(SimulationHeisenbergData_t data)
cpdef vector[vector[double]] sum_spin_vectors_within_sublattices(SimulationHeisenbergData_t data)
cdef vector[double] get_site_spin_vector(long site_index, SimulationHeisenbergData_t data)
cdef double compute_site_energy(long site_index, SimulationHeisenbergData_t data)
cdef double compute_energy_of_spin_vector_at_site(vector[double] site_spin, long site_index,
                                                  SimulationHeisenbergData_t data)
cdef NeighborStates_t lookup_neighbor_states(long site_index, SimulationHeisenbergData_t data)
