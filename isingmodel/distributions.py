# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np


@dataclass
class BinaryLattice(object):
    """Structural information and simulation state assuming a simple square lattice.

    :ivar dimensions: Number of sites along x and y dimensions of square lattice.
    """

    dimensions: Tuple[int, int]
    __slots__ = ["dimensions"]

    @property
    def number_sites(self):
        """Total sites in the simulation."""
        return np.prod(self.dimensions)

    def sample_random_state(self):
        """Generate sample of random states on the binary lattice."""
        return np.random.choice(a=[-1, 1], size=np.prod(self.dimensions), replace=True)

    def get_neighbors_states(
        self,
        site_index: Union[Tuple[int, int], np.ndarray],
        state: np.ndarray,
        neighborhood="Neumann",
    ) -> np.array:
        """Get the states of a site's neighbors.

        :param site_index: Indices for site whose neighbor states you want to query.
        :param state: Array of lattice spins.
        :param neighborhood: Controls the number of neighbors returned, defaults to
            'Neumann'
        :return: Array of neighbor states.
        """
        neighbor_indices: np.array = self._get_neighbor_indices(
            site_index=site_index,
            neighborhood=neighborhood,
        )

        return state[neighbor_indices]

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

        return neighbor_indices[:, 1] + neighbor_indices[:, 0] * self.dimensions[1]

    def _apply_periodic_boundary(self, indices: np.array) -> None:
        """Enforce periodic boundaries after getting neighbor indices.

        :param indices: Array of neighbor indices.
        """
        indices[indices[:, 0] < 0, 0] = self.dimensions[0] - 1
        indices[indices[:, 0] >= self.dimensions[0], 0] = 0
        indices[indices[:, 1] < 0, 1] = self.dimensions[1] - 1
        indices[indices[:, 1] >= self.dimensions[1], 1] = 0


def proposal_distribution(energy_difference: float, temperature: float) -> float:
    """Compute proposal distribution for energy difference and simulation temperature.

    :param energy_difference: Energy difference for the trial sample.
    :param temperature: Temperature of the simulation.
    :return: Probability of accepting trial sample.
    """
    return np.exp(-energy_difference / temperature)
