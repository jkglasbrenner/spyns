# -*- coding: utf-8 -*-

from dataclasses import dataclass

import numpy as np

from spyns.data import HeisenbergState, SimulationData


@dataclass(frozen=True)
class TrialFlip(object):
    energy_difference: float
    current_spin_vector: np.ndarray
    trial_spin_vector: np.ndarray
    __slots__ = [
        "energy_difference",
        "current_spin_vector",
        "trial_spin_vector",
    ]


@dataclass(frozen=True)
class NeighborStates(object):
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    interaction_parameters: np.ndarray
    __slots__ = [
        "x",
        "y",
        "z",
        "interaction_parameters",
    ]


def sample_random_state(number_sites: int) -> HeisenbergState:
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


def sample_random_spin_vector() -> np.ndarray:
    """Generate sample of a random spin vector.

    :return: Random spin vector as a three component array.
    """
    theta: float = 2 * np.pi * np.random.uniform()
    phi: float = np.arccos(np.random.uniform(low=-1, high=1))
    sin_phi: float = np.sin(phi)

    trial_spin_vector: np.ndarray = np.array([
        sin_phi * np.cos(theta),
        sin_phi * np.sin(theta),
        np.cos(phi),
    ])

    return trial_spin_vector


def flip(
    site_index: int,
    data: SimulationData,
) -> TrialFlip:
    """Compute the change in energy for a trial spin flip.

    :param site_index: Perform trial spin flip on site specified by the index.
    :param data: Data container for the simulation.
    :return: Data container storing the current spin vector, the trial spin vector, and
        the trial flip's energy difference.
    """
    current_spin_vector: np.ndarray = get_site_spin_vector(
        site_index=site_index, data=data
    )

    site_energy_preflip: float = compute_energy_of_spin_vector_at_site(
        site_spin=current_spin_vector,
        site_index=site_index,
        data=data,
    )

    trial_spin_vector: np.ndarray = sample_random_spin_vector()

    site_energy_postflip: float = compute_energy_of_spin_vector_at_site(
        site_spin=trial_spin_vector,
        site_index=site_index,
        data=data,
    )

    energy_difference: float = site_energy_postflip - site_energy_preflip

    return TrialFlip(
        energy_difference=energy_difference,
        current_spin_vector=current_spin_vector,
        trial_spin_vector=trial_spin_vector,
    )


def keep_flip_and_update_state(
    data: SimulationData,
    site_index: int,
    trial_flip: TrialFlip,
) -> None:
    """Keep flip and update simulation state.

    :param data: Data container for the simulation.
    :param site_index: Index for randomly chosen site.
    :param trial_flip: Data container storing the current spin vector, the trial spin
        vector, and the trial flip's energy difference.
    """
    sublattice_index: int = data.lookup_tables.sublattice_table[site_index]

    spin_vector_change: np.ndarray = \
        trial_flip.trial_spin_vector - trial_flip.current_spin_vector

    data.state.x[site_index] = trial_flip.trial_spin_vector[0]
    data.state.y[site_index] = trial_flip.trial_spin_vector[1]
    data.state.z[site_index] = trial_flip.trial_spin_vector[2]

    data.estimators.energy += trial_flip.energy_difference
    data.estimators.spin_vector[sublattice_index] += spin_vector_change


def save_full_state(data: SimulationData) -> None:
    """Compute the total energy and total magnetization estimators for the lattice.

    :param data: Data container for the simulation.
    """
    data.estimators.energy = compute_total_energy(data=data)
    data.estimators.spin_vector[:, :] = sum_spin_vectors_within_sublattices(data=data)


def compute_total_energy(data: SimulationData) -> float:
    """Compute the total energy estimator for the lattice.

    :param data: Data container for the simulation.
    :return: Total energy of the simulation state.
    """
    total_energy: float = 0

    for site_index in range(data.lookup_tables.number_sites):
        total_energy += compute_site_energy(
            site_index=site_index,
            data=data,
        )

    return total_energy / 2.0


def sum_spin_vectors_within_sublattices(data: SimulationData) -> np.ndarray:
    """Sum the spin vectors within each sublattice.

    :param data: Data container for the simulation.
    :return: Array of summed spin vectors grouped by sublattice.
    """
    sublattice_indices: np.ndarray = data.lookup_tables.sublattice_table
    spin_vector: np.ndarray = np.zeros(
        shape=(data.lookup_tables.number_sublattices, 3),
        dtype=np.float,
    )

    for sublattice in range(data.lookup_tables.number_sublattices):
        spin_vector[sublattice] = np.array([
            data.state.x[sublattice_indices == sublattice].sum(),
            data.state.y[sublattice_indices == sublattice].sum(),
            data.state.z[sublattice_indices == sublattice].sum(),
        ])

    return spin_vector


def get_site_spin_vector(site_index: int, data: SimulationData) -> np.ndarray:
    """Read and return the spin vector at a site.

    :param site_index: Site whose spin vector you want to read.
    :param data: Data container for the simulation.
    :return: Site's spin vector as a three component array.
    """
    site_spin: np.ndarray = np.array([
        data.state.x[site_index],
        data.state.y[site_index],
        data.state.z[site_index],
    ])

    return site_spin


def compute_site_energy(
    site_index: int,
    data: SimulationData,
) -> float:
    """Compute a given site's energy.

    :param site_index: Site whose energy you want to compute.
    :param data: Data container for the simulation.
    :return: Energy of site specified by ``site_index``.
    """
    site_spin: np.ndarray = get_site_spin_vector(site_index=site_index, data=data)

    energy: float = compute_energy_of_spin_vector_at_site(
        site_spin=site_spin,
        site_index=site_index,
        data=data,
    )

    return energy


def compute_energy_of_spin_vector_at_site(
    site_spin: np.ndarray,
    site_index: int,
    data: SimulationData,
) -> float:
    """Compute a spin vector's energy when placed within a given site's neighborhood.

    :return: Spin vector as a three component array.
    :param site_index: Site at which to place spin vector.
    :param data: Data container for the simulation.
    :return: Energy of site specified by ``site_index``.
    """
    neighbor_states: NeighborStates = lookup_neighbor_states(
        site_index=site_index,
        data=data,
    )

    energy: float = np.sum(
        neighbor_states.interaction_parameters * (
            site_spin[0] * neighbor_states.x + site_spin[1] * neighbor_states.y +
            site_spin[2] * neighbor_states.z
        )
    )

    return energy


def lookup_neighbor_states(
    site_index: int,
    data: SimulationData,
) -> NeighborStates:
    """Get the states and interaction parameters of a site's neighbors.

    :param site_index: Site index whose neighbor states you want to query.
    :param data: Data container for the simulation.
    :return: Data container of the site's neighbor's states and interaction parameters.
    """
    neighbors_count: int = data.lookup_tables.neighbors_count[site_index]
    lookup_start: int = data.lookup_tables.neighbors_lookup_index[site_index]
    lookup_end: int = lookup_start + neighbors_count

    neighbor_indices: np.ndarray = \
        data.lookup_tables.neighbors_table[lookup_start:lookup_end]

    neighbors_states_x: np.ndarray = data.state.x[neighbor_indices]
    neighbors_states_y: np.ndarray = data.state.y[neighbor_indices]
    neighbors_states_z: np.ndarray = data.state.z[neighbor_indices]
    interaction_parameters: np.ndarray = \
        data.lookup_tables.interaction_parameters_table[lookup_start:lookup_end]

    return NeighborStates(
        x=neighbors_states_x,
        y=neighbors_states_y,
        z=neighbors_states_z,
        interaction_parameters=interaction_parameters,
    )
