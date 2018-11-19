# -*- coding: utf-8 -*-

from typing import Tuple

import numpy as np

from spyns.data import SimulationData


def lookup_neighbor_states(
    site_index: int,
    data: SimulationData,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the states and interaction parameters of a site's neighbors.

    :param site_index: Site index whose neighbor states you want to query.
    :param data: Data container for the simulation.
    :return: Tuple containing the site's neighbor's states and interaction
        parameters.
    """
    neighbors_count: int = data.lookup_tables.neighbors_count[site_index]
    lookup_start: int = data.lookup_tables.neighbors_lookup_index[site_index]
    lookup_end: int = lookup_start + neighbors_count

    neighbor_indices: np.ndarray = \
        data.lookup_tables.neighbors_table[lookup_start:lookup_end]

    neighbors_states: np.ndarray = data.state[neighbor_indices]
    interaction_parameters: np.ndarray = \
        data.lookup_tables.interaction_parameters_table[lookup_start:lookup_end]

    return (neighbors_states, interaction_parameters)
