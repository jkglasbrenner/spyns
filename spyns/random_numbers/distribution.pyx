import cython

from random_cpp cimport mt19937, uniform_int_distribution, uniform_real_distribution


cdef class RandomNumberGenerator:
    
    def __cinit__(self, long seed, long number_sites):
        self._engine = mt19937(seed)
        self._uniform = uniform_real_distribution[double](0.0, 1.0)
        self._randint = uniform_int_distribution[long](0, number_sites - 1)
        self._uniform_low = 0.0
        self._uniform_high = 1.0
        self._randint_low = 0
        self._randint_high = number_sites

    cdef double uniform(self):
        return self._uniform(self._engine)

    cdef long randint(self):
        return self._randint(self._engine)

    @property
    def uniform_bounds(self) -> (cython.double, cython.double):
        return (self._uniform_low, self._uniform_high)
    
    @uniform_bounds.setter
    def uniform_bounds(self, (double, double) value) -> cython.void:
        self._uniform_low = value[0]
        self._uniform_high = value[1]
        self._uniform = uniform_real_distribution[double](value[0], value[1])
    
    @property
    def randint_bounds(self) -> (long, long):
        return (self._randint_low, self._randint_high)
    
    @randint_bounds.setter
    def randint_bounds(self, (long, long) value) -> cython.void:
        self._randint_low = value[0]
        self._randint_high = value[1]
        self._randint = uniform_int_distribution[long](value[0], value[1])
