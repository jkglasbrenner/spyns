# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pymatgen as pmg

import spyns
from spyns.lattice.neighborhood import NeighborsDataFrames

Neighbor = Tuple[pmg.PeriodicSite, float, int]
SiteNeighbors = List[Optional[Neighbor]]
AllNeighborDistances = List[SiteNeighbors]
NeighborDistances = Dict[str, Union[List[str], List[float], List[int]]]


class Lattice(object):
    """Pymatgen ``Structure`` object and neighbor tables defining the system lattice.

    :ivar neighbors_data_frame: Data frame of neighbors grouped over site-index pairs
        and distances.
    :ivar sublattice_pairs_data_frame: Data frame of unique sublattice pairs.
    :ivar sublattice_table: Lookup table of sublattice indices.
    :ivar sublattice_labels: Sublattice labels corresponding to the factorized
        sublattice indices.
    :ivar neighbors_table: Lookup table of neighbor indices.
    :ivar neighbors_count: Lookup table of neighbor counts.
    :ivar neighbors_lookup_index: Lookup starting index for site's neighbors in
        ``neighbors_table``.
    :ivar interaction_parameters_table: Lookup table of interaction parameters.
    :ivar number_sites: Total sites in the lattice.
    :ivar number_sublattices: Total unique sublattices defined in the lattice.
    """

    __slots__ = [
        "_structure",
        "_neighbor_table",
        "_neighbor_count_list",
        "_neighbor_table_lookup_index",
        "_sublattice_table",
        "_sublattice_labels",
        "_interaction_parameters_table",
        "_number_sites",
        "_number_sublattices",
        "_neighbor_count_df",
        "_sublattice_pairs_df",
        "_sublattice_pairs_interaction_df",
        "_r",
    ]

    def __init__(self, structure: pmg.Structure, r: float):
        """Initialize attributes, build neighbor tables, and cache even/odd indices.

        :param structure: Unit cell in pymatgen structure format.
        :param r: Radius of sphere.
        :param interaction_parameters: Interaction parameters for site neighbors.
        """
        self._structure: pmg.Structure = structure
        self._r: float = r
        self._number_sites = self._structure.num_sites
        self._build_and_cache_neighbor_table()
        self._build_and_cache_sublattice_table()

    @property
    def neighbors_data_frame(self):
        """Data frame of neighbors grouped over site-index pairs and distances."""
        try:
            return self._neighbor_count_df

        except AttributeError:
            self._cache_neighbor_data_frames()

        return self._neighbor_count_df

    @property
    def sublattice_pairs_data_frame(self):
        """Data frame of unique sublattice pairs."""
        try:
            return self._sublattice_pairs_interaction_df

        except AttributeError:
            pass

        try:
            return self._sublattice_pairs_df

        except AttributeError:
            self._cache_neighbor_data_frames()

        return self._sublattice_pairs_df

    @property
    def sublattice_table(self):
        """Lookup table of sublattice indices."""
        return self._sublattice_table

    @property
    def sublattice_labels(self):
        """Sublattice labels corresponding to the factorized sublattice indices."""
        return self._sublattice_labels

    @property
    def neighbors_table(self):
        """Lookup table of neighbor indices."""
        return self._neighbor_table

    @property
    def neighbors_count(self):
        """Lookup table of neighbor counts."""
        return self._neighbor_count_list

    @property
    def neighbors_lookup_index(self):
        """Lookup starting index for site's neighbors in ``neighbors_table``."""
        return self._neighbor_table_lookup_index

    @property
    def interaction_parameters_table(self):
        """Lookup table of interaction parameters."""
        try:
            return self._interaction_parameters_table

        except AttributeError:
            raise AttributeError("Interaction parameters not set.")

    @property
    def number_sites(self):
        """Total sites in the lattice."""
        return self._number_sites

    @property
    def number_sublattices(self):
        """Total unique sublattices defined in the lattice."""
        return self._number_sublattices

    def set_sublattice_pair_interactions(self, interaction_df: pd.DataFrame) -> None:
        """Set the pairwise interaction coefficients.

        :param interaction_df: ``sublattice_pairs_data_frame`` transformed to add a
            ``J_ij`` column defining the interaction coefficients for site pairs
            as a function of sublattices and neighbor distances.
        :raises KeyError: An error will be raised if ``interaction_df`` does not have
            a column named ``J_ij`` or is missing one or more of the columns
            ``subspecies_i``, ``subspecies_j``, ``distance_bin``, and
            ``subspecies_ij_distance_rank`` needed for a data frame merge.
        """
        merge_columns = [
            "subspecies_i",
            "subspecies_j",
            "distance_bin",
            "subspecies_ij_distance_rank",
        ]
        try:
            interaction_df = interaction_df[merge_columns + ["J_ij"]]

        except KeyError:
            raise KeyError(
                "interaction_df must have a column named J_ij containing the "
                "interaction parameters."
            )

        try:
            self._sublattice_pairs_interaction_df = self._sublattice_pairs_df \
                .merge(interaction_df, on=merge_columns)

        except KeyError:
            raise KeyError(
                "interaction_df must have 'subspecies_i', 'subspecies_j' and "
                "'subspecies_ij_distance_rank' columns that match "
                "sublattice_pairs_data_frame."
            )

        self._build_and_cache_interaction_table(self._sublattice_pairs_interaction_df)

    def _cache_neighbor_data_frames(self) -> None:
        """Cache data frame of pairs as a function of distance and sublattice."""
        neighbors_df: NeighborsDataFrames = \
            spyns.lattice.neighborhood.build_neighbors_data_frames(
                structure=self._structure,
                r=self._r,
            )
        self._neighbor_count_df = neighbors_df.neighbor_count
        self._sublattice_pairs_df = neighbors_df.sublattice_pairs

    def _build_and_cache_neighbor_table(self) -> None:
        """Build and save neighbor tables for lattice."""
        neighbor_count_df: pd.DataFrame = self.neighbors_data_frame

        neighbor_table: List[int] = neighbor_count_df \
            .loc[:, "j"] \
            .values
        neighbor_count_list: List[int] = neighbor_count_df \
            .loc[:, ["i", "n"]] \
            .groupby("i") \
            .sum() \
            .values \
            .flatten()
        neighbor_table_lookup_index: List[int] = neighbor_count_df \
            .reset_index() \
            .loc[:, ["i", "index"]] \
            .groupby("i") \
            .first() \
            .values \
            .flatten()

        if neighbor_count_list.sum() == len(neighbor_table):
            self._neighbor_table: np.ndarray = neighbor_table
            self._neighbor_count_list: np.ndarray = neighbor_count_list
            self._neighbor_table_lookup_index: np.ndarray = neighbor_table_lookup_index

        else:
            raise ValueError(
                f"Lattice has too few sites to use neighbor cutoff r={self._r}. "
                "Either reduce neighbor cutoff or add more lattice sites."
            )

    def _build_and_cache_sublattice_table(self) -> None:
        """Build and save sublattice tables for lattice."""
        structure: pmg.Structure = \
            spyns.lattice.generate.add_subspecie_labels_if_missing(
                cell_structure=self._structure,
            )

        sublattice_table, distinct_sublattices = \
            self._factorize_sublattice_labels(
                sublattice_labels=structure.site_properties["subspecie"]
            )

        self._sublattice_table: np.ndarray = sublattice_table
        self._sublattice_labels: np.ndarray = distinct_sublattices
        self._number_sublattices: int = len(distinct_sublattices.tolist())

    def _factorize_sublattice_labels(
        self,
        sublattice_labels: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Associate each unique sublattice label with an integer.
        
        :param sublattice_labels: Sublattice labels associated with all sites in
            the structure.
        :return: Tuple where first element is the sublattice integers and the second
            element is the unique sublattice labels.
        """
        return pd.factorize(sublattice_labels)

    def _build_and_cache_interaction_table(
        self,
        interaction_parameters: List[float],
    ) -> None:
        """Build and save interaction tables for a two-dimensional square lattice.

        :param interaction_parameters: Interaction parameters for site neighbors.
        """
        try:
            interaction_df: pd.DataFrame = self._sublattice_pairs_interaction_df

        except AttributeError:
            raise AttributeError("Sublattice interactions not set.")

        self._interaction_parameters_table: np.ndarray = self.neighbors_data_frame \
            .merge(interaction_df) \
            .sort_values(["i", "distance_bin", "j"]) \
            .loc[:, "J_ij"] \
            .values
