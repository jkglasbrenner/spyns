# -*- coding: utf-8 -*-

from typing import Iterable, List, Tuple, Union

import numpy as np


class BinaryLattice(object):
    """Structural information and neighbor tables assuming a simple square lattice.

    :ivar dimensions: Number of sites along x and y dimensions of square lattice.
    :ivar neighborhood: Controls the number of neighbors returned.
    """

    __slots__ = [
        "dimensions",
        "neighborhood",
        "_neighbor_table",
        "_neighbor_count_list",
        "_neighbor_table_lookup_index",
        "_interaction_parameters_table",
        "_even_site_indices",
        "_odd_site_indices",
        "_number_sites",
    ]

    def __init__(self, dimensions, neighborhood, interaction_parameters):
        """Initialize attributes, build neighbor tables, and cache even/odd indices.
        
        :param dimensions: Number of sites along x and y dimensions of square lattice.
        :param neighborhood: Controls the number of neighbors returned.
        :param interaction_parameters: Interaction parameters for site neighbors.
        """
        self.dimensions: Tuple[int, int] = dimensions
        self.neighborhood: str = neighborhood
        self._number_sites: int = int(np.prod(dimensions))
        self._build_and_cache_neighbor_table()
        self._build_and_cache_interaction_parameter_table(interaction_parameters)
        self._cache_even_and_odd_site_indices()

    @property
    def number_sites(self):
        """Total sites in the simulation."""
        return self._number_sites

    @property
    def even_site_indices(self):
        """Even site indices on the lattice."""
        return self._even_site_indices

    @property
    def odd_site_indices(self):
        """Odd site indices on the lattice."""
        return self._odd_site_indices

    def sample_random_state(self) -> np.ndarray:
        """Generate sample of random states on the binary lattice."""
        return np.random.choice(a=[-1, 1], size=np.prod(self.dimensions), replace=True)

    def get_neighbor_states(
        self,
        site_index: int,
        state: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the states and interaction parameters of a site's neighbors.

        :param site_index: Site index whose neighbor states you want to query.
        :param state: Array of lattice states.
        :return: Tuple containing the site's neighbor's states and interaction
            parameters.
        """
        lookup_start_index: np.ndarray = self._neighbor_table_lookup_index[site_index]
        lookup_end_index: np.ndarray = \
            lookup_start_index + self._neighbor_count_list[site_index]
        neighbor_indices: np.ndarray = \
            self._neighbor_table[lookup_start_index:lookup_end_index]

        return (
            state[neighbor_indices],
            self._interaction_parameters_table[neighbor_indices],
        )

    def get_neighbor_states_for_multiple_sites(
        self,
        site_index: np.ndarray,
        state: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the states and interaction parameters of the neighbors of multiple sites.

        :param site_index: One-dimensional array of site indices whose neighbor states
            you want to query.
        :param state: Array of lattice spins.
        :return: Tuple containing the site's neighbor's states and interaction
            parameters.
        """
        lookup_start_index: np.ndarray = self._neighbor_table_lookup_index[site_index]
        lookup_end_index: np.ndarray = \
            lookup_start_index + self._neighbor_count_list[site_index]
        lookup_indices: Iterable = zip(lookup_start_index, lookup_end_index)

        count_neighbors_states: int = np.sum(
            self._neighbor_count_list[lookup_start_index],
            dtype=np.int,
        )

        neighbor_indices: np.ndarray = np.zeros(count_neighbors_states)

        begin_idx: int = 0
        for loop_index, (start_index, end_index) in enumerate(lookup_indices):
            number_neighbors: int = end_index - start_index
            end_idx: int = begin_idx + number_neighbors
            neighbor_indices[begin_idx:end_idx] = \
                self._neighbor_table[start_index:end_index]
            begin_idx = end_idx

        return (
            state[neighbor_indices],
            self._interaction_parameters_table[neighbor_indices],
        )

    def _build_and_cache_neighbor_table(self) -> None:
        """Build and save neighbor tables for a two-dimensional square lattice."""
        neighbor_table: List[Union[None, int]] = []
        neighbor_count_list: List[Union[None, int]] = []
        neighbor_table_lookup_index: List[Union[None, int]] = []

        site_index_list: np.ndarray = np.arange(self._number_sites)

        number_neighbors: int = self._get_number_neighbors(self.neighborhood)

        self._build_neighbor_table(
            number_neighbors=number_neighbors,
            site_index_list=site_index_list,
            neighbor_table=neighbor_table,
            neighbor_count_list=neighbor_count_list,
            neighbor_table_lookup_index=neighbor_table_lookup_index,
        )

        self._neighbor_table: List[int] = neighbor_table
        self._neighbor_count_list: List[int] = neighbor_count_list
        self._neighbor_table_lookup_index: List[int] = neighbor_table_lookup_index

    def _build_and_cache_interaction_parameter_table(
        self,
        interaction_parameters: List[float],
    ) -> None:
        """Build and save interaction tables for a two-dimensional square lattice.

        :param interaction_parameters: Interaction parameters for site neighbors.
        """
        interaction_parameters_table: List[Union[int, None]] = []

        for _ in range(self._number_sites):
            interaction_parameters_table.extend(4 * [interaction_parameters[0]])

            if self.neighborhood == "Moore":
                interaction_parameters_table.extend(4 * [interaction_parameters[1]])

        self._interaction_parameters_table: np.ndarray = \
            np.array(interaction_parameters_table)

    @staticmethod
    def _get_number_neighbors(neighborhood: str) -> int:
        """Get neighbor count for two-dimensional Neumann or Moore neighborhoods.
        
        :param neighborhood: Controls the number of neighbors returned, can be either
            'Neumann' or 'Moore'.
        :return: Neighbor count per site.
        """
        if neighborhood == "Neumann":
            return 4
        elif neighborhood == "Moore":
            return 8

    def _build_neighbor_table(
        self,
        number_neighbors: int,
        site_index_list: List[int],
        neighbor_table: List[int],
        neighbor_count_list: List[int],
        neighbor_table_lookup_index: List[int],
    ) -> None:
        """Build the neighbor table for a square two-dimensional lattice.

        :param number_neighbors: Neighbor count per site.
        :param site_index_list: [description]
        :param neighbor_table: A list of neighbor indices for each site.
        :param neighbor_count_list: A list of the number of neighbors each site has.
        :param neighbor_table_lookup_index: A list of the lookup index to use on
            ``neighbor_table`` when querying neighbor indices.
        """
        for site_index in site_index_list:
            row_index: int = site_index // self.dimensions[1]
            column_index: int = site_index % self.dimensions[1]

            neighbor_index_list: np.ndarray = self._get_neighbor_indices(
                site_index=(row_index, column_index),
                neighborhood=self.neighborhood,
            )

            neighbor_table.extend(neighbor_index_list.tolist())
            neighbor_count_list.append(number_neighbors)
            neighbor_table_lookup_index.append(site_index * number_neighbors)

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
        row_index: int = site_index[0]
        col_index: int = site_index[1]

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

    def _apply_periodic_boundary(self, indices: np.ndarray) -> None:
        """Enforce periodic boundaries after getting neighbor indices.

        :param indices: Array of neighbor indices.
        """
        indices[indices[:, 0] < 0, 0] = self.dimensions[0] - 1
        indices[indices[:, 0] >= self.dimensions[0], 0] = 0
        indices[indices[:, 1] < 0, 1] = self.dimensions[1] - 1
        indices[indices[:, 1] >= self.dimensions[1], 1] = 0

    def _cache_even_and_odd_site_indices(self):
        """Find and cache indices for even and odd lattice sites."""
        self._even_site_indices: List[int] = [
            row * self.dimensions[1] + column
            for row in range(0, self.dimensions[0], 2)
            for column in range(0, self.dimensions[1], 2)
        ] + [
            row * self.dimensions[1] + column
            for row in range(1, self.dimensions[0], 2)
            for column in range(1, self.dimensions[1], 2)
        ]

        self._odd_site_indices: List[int] = [
            row * self.dimensions[1] + column
            for row in range(0, self.dimensions[0], 2)
            for column in range(1, self.dimensions[1], 2)
        ] + [
            row * self.dimensions[1] + column
            for row in range(1, self.dimensions[0], 2)
            for column in range(0, self.dimensions[1], 2)
        ]
