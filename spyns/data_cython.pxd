from spyns.random_numbers.distribution cimport RandomNumberGenerator

cdef class SimulationParameters_t:
    cdef long sample_interval
    cdef double temperature
    cdef long sweeps
    cdef long equilibration_sweeps


cdef class LookupTables_t:
    cdef long[:] sublattice_table
    cdef long[:] neighbors_table
    cdef long[:] neighbors_count
    cdef long[:] neighbors_lookup_index
    cdef double[:] interaction_parameters_table
    cdef long number_sites
    cdef long number_sublattices


cdef class HeisenbergState_t:
    cdef double[:] x
    cdef double[:] y
    cdef double[:] z


cdef class Estimators_t:
    cdef long[:] number_samples
    cdef double[:] energy
    cdef double[:, :] spin_vector
    cdef double[:] magnetization


cdef class SimulationTrace_t:
    cdef long[:] sweep
    cdef double[:] energy
    cdef double[:, :, :] spin_vector
    cdef double[:] magnetization


cdef class SimulationHeisenbergData_t:
    cdef RandomNumberGenerator random_number_generator
    cdef SimulationParameters_t parameters
    cdef LookupTables_t lookup_tables
    cdef HeisenbergState_t state
    cdef SimulationTrace_t trace
    cdef Estimators_t estimators
    cdef object _data
