from spyns.data_cython cimport SimulationHeisenbergData_t
from spyns.random_numbers.distribution cimport RandomNumberGenerator

cdef void step(SimulationHeisenbergData_t data)
cdef void sweep(SimulationHeisenbergData_t data, long sweep_index,
                bint equilibration_run)
cpdef void run_sweeps(SimulationHeisenbergData_t data, bint equilibration_run)
