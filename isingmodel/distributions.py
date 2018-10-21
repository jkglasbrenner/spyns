# -*- coding: utf-8 -*-

from typing import Tuple, Union

import attr
import numpy as np


@attr.s(slots=True)
class BinaryLattice(object):
    """Structural information and simulation state assuming a simple square lattice.

    :ivar dimensions: Number of sites along x and y dimensions of square lattice.
    :ivar number_sites: Total sites in the simulation.
    :ivar state: Simulation state.
    """
    dimensions: Tuple[int, int] = attr.ib()
    number_sites: int = attr.ib(init=False)
    state: np.ndarray = attr.ib(init=False)

    @dimensions.validator
    def limit_dimensions(self, attribute, value):
        """Enforce two-dimensional simulation."""
        if len(value) != 2:
            raise ValueError("Only two-dimensional lattices are supported.")

    def __attrs_post_init__(self):
        """Initialize ``number_sites`` and ``state`` attributes using ``dimensions``."""
        self.number_sites = np.prod(self.dimensions)
        self.state = self._initialize_state()

    def _initialize_state(self):
        """Internal method used to randomize states on binary lattice."""
        return np.random.choice(a=[-1, 1], size=self.dimensions, replace=True)

    def get_neighbors_states(self, site_index, neighborhood="Neumann") -> np.array:
        """Get the states of a site's neighbors.

        :param site_index: Indices for site whose neighbor states you want to query.
        :param neighborhood: Controls the number of neighbors returned, defaults to
            "Neumann"
        :return: Array of neighbor states.
        """
        neighbor_indices: np.array = self._get_neighbor_indices(
            site_index=site_index,
            neighborhood=neighborhood,
        )

        return self.state[neighbor_indices[:, 0], neighbor_indices[:, 1]]

    def _get_neighbor_indices(
        self,
        site_index: Union[Tuple[int, int], np.ndarray],
        neighborhood: str,
    ) -> np.ndarray:
        """Get the neighbor indices for the specified site.

        :param site_index: Indices for site whose neighbor states you want to query.
        :param neighborhood: Controls the number of neighbors returned.
        :return: Array of neighbor indices.
        """
        neighbor_indices: Union[np.array, None] = None
        row_index = site_index[0]
        col_index = site_index[1]

        if neighborhood == "Neumann":
            neighbor_indices = np.array(
                [
                    [row_index + 1, col_index],
                    [row_index - 1, col_index],
                    [row_index, col_index + 1],
                    [row_index, col_index - 1],
                ],
                dtype="i8",
            )

        elif neighborhood == "Moore":
            neighbor_indices = np.array(
                [
                    [row_index + 1, col_index],
                    [row_index - 1, col_index],
                    [row_index, col_index + 1],
                    [row_index, col_index - 1],
                    [row_index + 1, col_index + 1],
                    [row_index + 1, col_index - 1],
                    [row_index - 1, col_index + 1],
                    [row_index - 1, col_index - 1],
                ],
                dtype="i8",
            )

        self._apply_periodic_boundary(indices=neighbor_indices)

        return neighbor_indices

    def _apply_periodic_boundary(self, indices: np.array) -> None:
        """Enforce periodic boundaries after getting neighbor indices.

        :param indices: Array of neighbor indices.
        """
        indices[indices[:, 0] < 0, 0] = self.dimensions[0] - 1
        indices[indices[:, 0] >= self.dimensions[0], 0] = 0
        indices[indices[:, 1] < 0, 1] = self.dimensions[1] - 1
        indices[indices[:, 1] >= self.dimensions[1], 1] = 0


def proposal_distribution(energy_difference: float, temperature: float) -> float:
    """Compute proposal distribution for a given energy difference and simulation
    temperature.

    :param energy_difference: Energy difference for the trial sample.
    :param temperature: Temperature of the simulation.
    :return: Probability of accepting trial sample.
    """
    return np.exp(-energy_difference / temperature)
