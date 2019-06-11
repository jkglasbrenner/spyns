from libc.math cimport sin, cos, acos, pi
from libcpp.vector cimport vector
from spyns.data_cython cimport SimulationHeisenbergData_t

import cython
import numpy as np

from spyns.data import HeisenbergState


cdef class NeighborStates_t:
    def __cinit__(self, long number_neighbors):
        cdef long neighbor

        for neighbor in range(number_neighbors):
            self.x.push_back(0.0)
            self.y.push_back(0.0)
            self.z.push_back(0.0)
            self.interaction_parameters.push_back(0.0)


cdef class TrialFlip_t:
    def __cinit__(self):
        cdef long axis

        self.energy_difference = 0.0

        for axis in range(3):
            self.current_spin_vector.push_back(0.0)
            self.trial_spin_vector.push_back(0.0)


cdef object sample_random_state(long number_sites):
    """Generate sample of random spin vectors on the Heisenberg lattice.

    :param number_sites: Number of sites in the lattice.
    :return: Container of random states on the Heisenberg lattice.
    """
    theta = 2 * np.pi * np.random.uniform(size=number_sites)
    phi = np.arccos(np.random.uniform(low=-1, high=1, size=number_sites))
    sin_phi = np.sin(phi)

    return HeisenbergState(
        x=sin_phi * np.cos(theta),
        y=sin_phi * np.sin(theta),
        z=np.cos(phi),
    )


cdef vector[double] sample_random_spin_vector(
    SimulationHeisenbergData_t data,
):
    """Generate sample of a random spin vector.

    :return: Random spin vector as a three component array.
    """
    cdef double theta = 2.0 * pi * data.random_number_generator.uniform()
    cdef double phi = acos(2.0 * data.random_number_generator.uniform() - 1.0)
    cdef double sin_phi = sin(phi)
    cdef double cos_phi = cos(phi)
    cdef double sin_theta = sin(theta)
    cdef double cos_theta = cos(theta)

    cdef vector[double] trial_spin_vector
    trial_spin_vector.push_back(sin_phi * cos_theta)
    trial_spin_vector.push_back(sin_phi * sin_theta)
    trial_spin_vector.push_back(cos_phi)

    return trial_spin_vector


cdef TrialFlip_t flip(
    long site_index,
    SimulationHeisenbergData_t data,
 ):
    """Compute the change in energy for a trial spin flip.

    :param site_index: Perform trial spin flip on site specified by the index.
    :param data: Data container for the simulation.
    :return: Data container storing the current spin vector, the trial spin vector, and
        the trial flip's energy difference.
    """
    cdef TrialFlip_t trial_flip = TrialFlip_t()
    
    trial_flip.current_spin_vector = get_site_spin_vector(
        site_index=site_index, data=data
    )

    cdef double site_energy_preflip = compute_energy_of_spin_vector_at_site(
        site_spin=trial_flip.current_spin_vector,
        site_index=site_index,
        data=data,
    )

    trial_flip.trial_spin_vector = sample_random_spin_vector(
        data=data,
    )

    cdef double site_energy_postflip = compute_energy_of_spin_vector_at_site(
        site_spin=trial_flip.trial_spin_vector,
        site_index=site_index,
        data=data,
    )

    trial_flip.energy_difference = site_energy_postflip - site_energy_preflip

    return trial_flip


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void keep_flip_and_update_state(
    SimulationHeisenbergData_t data,
    long site_index,
    TrialFlip_t trial_flip,
):
    """Keep flip and update simulation state.

    :param data: Data container for the simulation.
    :param site_index: Index for randomly chosen site.
    :param trial_flip: Data container storing the current spin vector, the trial spin
        vector, and the trial flip's energy difference.
    """
    cdef long axis
    cdef vector[double] spin_vector_change
    cdef long sublattice_index = data.lookup_tables.sublattice_table[site_index]

    for axis in range(3):
        spin_vector_change.push_back(0.0)

    for axis in range(3):
        spin_vector_change[axis] = (
            trial_flip.trial_spin_vector[axis] -
            trial_flip.current_spin_vector[axis]
        )

    data.state.x[site_index] = trial_flip.trial_spin_vector[0]
    data.state.y[site_index] = trial_flip.trial_spin_vector[1]
    data.state.z[site_index] = trial_flip.trial_spin_vector[2]

    data.estimators.energy[0] += trial_flip.energy_difference
    
    for axis in range(3):
        data.estimators.spin_vector[sublattice_index, axis] += spin_vector_change[axis]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void save_full_state(SimulationHeisenbergData_t data):
    """Compute the total energy and total magnetization estimators for the lattice.

    :param data: Data container for the simulation.
    """
    cdef long sublattice
    cdef long axis
    
    data.estimators.energy[0] = compute_total_energy(data=data)

    cdef vector[vector[double]] spin_vector = \
        sum_spin_vectors_within_sublattices(data=data)

    for sublattice in range(data.lookup_tables.number_sublattices):
        for axis in range(3):
            data.estimators.spin_vector[sublattice, axis] = spin_vector[sublattice][axis]


cpdef double compute_total_energy(SimulationHeisenbergData_t data):
    """Compute the total energy estimator for the lattice.

    :param data: Data container for the simulation.
    :return: Total energy of the simulation state.
    """
    cdef long site_index

    cdef double total_energy = 0

    for site_index in range(data.lookup_tables.number_sites):
        total_energy += compute_site_energy(
            site_index=site_index,
            data=data,
        )

    return total_energy / 2.0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef vector[vector[double]] sum_spin_vectors_within_sublattices(SimulationHeisenbergData_t data):
    """Sum the spin vectors within each sublattice.

    :param data: Data container for the simulation.
    :return: Array of summed spin vectors grouped by sublattice.
    """
    cdef long sublattice
    cdef long axis
    cdef long site_index
    cdef vector[double] sublattice_spin_vector
    cdef vector[vector[double]] spin_vector

    for axis in range(3):
        sublattice_spin_vector.push_back(0.0)

    for sublattice in range(data.lookup_tables.number_sublattices):
        spin_vector.push_back(sublattice_spin_vector)

    for site_index in range(data.lookup_tables.number_sites):
        sublattice = data.lookup_tables.sublattice_table[site_index]
        
        spin_vector[sublattice][0] += data.state.x[site_index]
        spin_vector[sublattice][1] += data.state.y[site_index]
        spin_vector[sublattice][2] += data.state.z[site_index]

    return spin_vector


@cython.boundscheck(False)
@cython.wraparound(False)
cdef vector[double] get_site_spin_vector(
    long site_index,
    SimulationHeisenbergData_t data,
):
    """Read and return the spin vector at a site.

    :param site_index: Site whose spin vector you want to read.
    :param data: Data container for the simulation.
    :return: Site's spin vector as a three component array.
    """
    cdef vector[double] site_spin

    site_spin.push_back(data.state.x[site_index])
    site_spin.push_back(data.state.y[site_index])
    site_spin.push_back(data.state.z[site_index])

    return site_spin


cdef double compute_site_energy(
    long site_index,
    SimulationHeisenbergData_t data,
):
    """Compute a given site's energy.

    :param site_index: Site whose energy you want to compute.
    :param data: Data container for the simulation.
    :return: Energy of site specified by ``site_index``.
    """
    cdef vector[double] site_spin = get_site_spin_vector(site_index=site_index, data=data)

    cdef double energy = compute_energy_of_spin_vector_at_site(
        site_spin=site_spin,
        site_index=site_index,
        data=data,
    )

    return energy


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double compute_energy_of_spin_vector_at_site(
    vector[double] site_spin,
    long site_index,
    SimulationHeisenbergData_t data,
):
    """Compute a spin vector's energy when placed within a given site's neighborhood.

    :return: Spin vector as a three component array.
    :param site_index: Site at which to place spin vector.
    :param data: Data container for the simulation.
    :return: Energy of site specified by ``site_index``.
    """
    cdef long neighbor

    cdef long number_neighbors = data.lookup_tables.neighbors_count[site_index]
    cdef double energy = 0.0
    cdef double interaction_parameter = 0.0
    cdef double site_spin_x = site_spin[0]
    cdef double site_spin_y = site_spin[1]
    cdef double site_spin_z = site_spin[2]
    cdef double neighbor_spin_x = 0.0
    cdef double neighbor_spin_y = 0.0
    cdef double neighbor_spin_z = 0.0

    cdef NeighborStates_t neighbor_states = lookup_neighbor_states(
        site_index=site_index,
        data=data,
    )

    for neighbor in range(number_neighbors):
        interaction_parameter = neighbor_states.interaction_parameters[neighbor]
        neighbor_spin_x = neighbor_states.x[neighbor]
        neighbor_spin_y = neighbor_states.y[neighbor]
        neighbor_spin_z = neighbor_states.z[neighbor]
    
        energy += interaction_parameter * (
            site_spin_x * neighbor_spin_x + site_spin_y * neighbor_spin_y +
            site_spin_z * neighbor_spin_z
        )

    return energy


@cython.boundscheck(False)
@cython.wraparound(False)
cdef NeighborStates_t lookup_neighbor_states(
    long site_index,
    SimulationHeisenbergData_t data,
):
    """Get the states and interaction parameters of a site's neighbors.

    :param site_index: Site index whose neighbor states you want to query.
    :param data: Data container for the simulation.
    :return: Data container of the site's neighbor's states and interaction parameters.
    """
    cdef long neighbor
    cdef long neighbor_index

    cdef long number_neighbors = data.lookup_tables.neighbors_count[site_index]
    cdef long lookup_start = data.lookup_tables.neighbors_lookup_index[site_index]
    cdef long lookup_end = lookup_start + number_neighbors

    cdef NeighborStates_t neighbor_states = NeighborStates_t(number_neighbors=number_neighbors)

    cdef long[:] neighbor_indices = \
        data.lookup_tables.neighbors_table[lookup_start:lookup_end]

    cdef double[:] interaction_parameters = \
            data.lookup_tables.interaction_parameters_table[lookup_start:lookup_end]

    for neighbor in range(number_neighbors):
        neighbor_index = neighbor_indices[neighbor]

        neighbor_states.x[neighbor] = data.state.x[neighbor_index]
        neighbor_states.y[neighbor] = data.state.y[neighbor_index]
        neighbor_states.z[neighbor] = data.state.z[neighbor_index]

        neighbor_states.interaction_parameters[neighbor] = interaction_parameters[neighbor]

    return neighbor_states
