import cython
from libcpp.pair cimport pair

from random_cpp cimport mt19937, uniform_int_distribution, uniform_real_distribution

cdef class RandomNumberGenerator:
    cdef mt19937 _engine
    cdef uniform_int_distribution[long] _randint
    cdef uniform_real_distribution[double] _uniform
    cdef long _randint_low
    cdef long _randint_high
    cdef double _uniform_low
    cdef double _uniform_high

    cdef double uniform(self)
    cdef long randint(self)
