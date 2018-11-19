# -*- coding: utf-8 -*-

from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pymatgen as pmg

import spyns

Neighbor = Tuple[pmg.PeriodicSite, float, int]
SiteNeighbors = List[Optional[Neighbor]]
AllNeighborDistances = List[SiteNeighbors]
NeighborDistances = Dict[str, Union[List[str], List[float], List[int]]]


class NeighborsDataFrames(NamedTuple):
    neighbor_count: pd.DataFrame
    sublattice_pairs: pd.DataFrame


def build_neighbors_data_frames(
    structure: pmg.Structure,
    r: float,
) -> NeighborsDataFrames:
    """Find neighbor and sublattice pairs in a structure within a cutoff distance.

    :param structure: A pymatgen ``Structure`` object.
    :param r: Cutoff radius for finding neighbors in sphere.
    :return: A ``NeighborsDataFrames`` named tuple with two field names:

        ``neighbor_count``
            A pandas ``DataFrame`` of neighbor counts aggregated over site-index pairs
            and separation distances.

        ``sublattice_pairs``
            A pandas ``DataFrame`` of neighbor distances mapped to unique bin
            intervals.
    """
    cell_structure = spyns.lattice.generate.add_subspecie_labels_if_missing(
        cell_structure=structure,
    )

    neighbor_distances_df: pd.DataFrame = get_neighbor_distances_data_frame(
        cell_structure=cell_structure,
        r=r,
    )

    distance_bins_df: pd.DataFrame = neighbor_distances_df \
        .pipe(define_bins_to_group_and_sort_by_distance)

    neighbor_count_df: pd.DataFrame = neighbor_distances_df \
        .pipe(group_site_index_pairs_by_distance,
              distance_bins_df=distance_bins_df) \
        .pipe(count_neighbors_within_distance_groups) \
        .pipe(sort_neighbors_by_site_index_i)

    sublattice_pairs_df: pd.DataFrame = neighbor_count_df \
        .pipe(sort_and_rank_unique_sublattice_pairs)

    return NeighborsDataFrames(
        neighbor_count=neighbor_count_df,
        sublattice_pairs=sublattice_pairs_df,
    )


def sort_and_rank_unique_sublattice_pairs(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Group, sort, and rank unique subspecies_ij and distance_bin columns.

    :param neighbor_distances_df: A pandas ``DataFrame`` of pairwise neighbor
        distances.
    :return: A pandas ``DataFrame`` of unique sublattice pairs.
    """
    subspecies_columns = ["subspecies_i", "subspecies_j"]
    sublattice_columns = subspecies_columns + ["distance_bin"]

    return data_frame \
        .loc[:, sublattice_columns] \
        .drop_duplicates(subset=sublattice_columns) \
        .sort_values(sublattice_columns) \
        .assign(subspecies_ij_distance_rank=lambda x: x.groupby(subspecies_columns)
                                                       .cumcount()) \
        .reset_index(drop=True)


def sort_neighbors_by_site_index_i(neighbor_count_df: pd.DataFrame) -> pd.DataFrame:
    """Sort by site index i, then neighbor distances, then neighbor index j.

    :param neighbor_count_df: A data frame of neighbor counts aggregated over
        site-index pairs and separation distances.
    :return: A pandas ``DataFrame`` of neighbor counts aggregated over site-index
        pairs and separation distances sorted by site index i, then neighbor
        distances, then neighbor index j.
    """
    return neighbor_count_df \
        .sort_values(by=["i", "distance_bin", "j"]) \
        .reset_index(drop=True)


def count_neighbors_within_distance_groups(
    grouped_distances: pd.core.groupby.DataFrameGroupBy,
) -> pd.DataFrame:
    """Count number of neighbors within each group of same-distance site-index pairs.

    :param grouped_distances: A data frame grouped over site-index pairs, subspecies
        pairs, and bin intervals.
    :return: A pandas ``DataFrame`` of neighbor counts aggregated over site-index pairs
        and separation distances.
    """
    return grouped_distances \
        .apply(lambda x: pd.to_numeric(arg=x["distance_ij"].count(),
                                       downcast="integer")) \
        .rename("n") \
        .reset_index()


def group_site_index_pairs_by_distance(
    neighbor_distances_df: pd.DataFrame,
    distance_bins_df: pd.DataFrame,
) -> pd.core.groupby.DataFrameGroupBy:
    """Iterate over all sites, grouping by site-index pairs, subspecies pairs, and
    bin intervals.

    :param neighbor_distances_df: A pandas ``DataFrame`` containing all pairwise
        neighbor distances.
    :param distance_bins_df: A pandas ``DataFrame`` of neighbor distances mapped to
        unique bin intervals.
    :return: A data frame grouped over site-index pairs, subspecies pairs, and
        bin intervals.
    """
    binned_distances: pd.Series = \
        pd.cut(x=neighbor_distances_df["distance_ij"], bins=distance_bins_df.index) \
          .rename("distance_bin")

    return neighbor_distances_df \
        .groupby(["i", "j", "subspecies_i", "subspecies_j", binned_distances])


def define_bins_to_group_and_sort_by_distance(
    neighbor_distances_df: pd.DataFrame,
) -> pd.DataFrame:
    """Defines bin intervals to group and sort neighbor pairs by distance.

    :param neighbor_distances_df: A pandas ``DataFrame`` of pairwise neighbor
        distances.
    :return: A pandas ``DataFrame`` of neighbor distances mapped to unique bin
        intervals.
    """
    unique_distances: np.ndarray = find_unique_distances(
        distance_ij=neighbor_distances_df["distance_ij"]
    )

    bin_intervals: pd.IntervalIndex = define_bin_intervals(
        unique_distances=unique_distances
    )

    return pd.DataFrame(
        data={
            "distance_bin": pd.Categorical(values=bin_intervals, ordered=True),
            "distance_ij": pd.Categorical(values=unique_distances, ordered=True),
        },
        index=bin_intervals,
    )


def find_unique_distances(distance_ij: pd.Series) -> np.ndarray:
    """Finds the unique distances that define the neighbor groups.

    :param distance_ij: A pandas ``Series`` of pairwise neighbor distances.
    :return: An array of unique neighbor distances.
    """
    unique_floats: np.ndarray = np.sort(distance_ij.unique())

    next_distance_not_close: np.ndarray = np.logical_not(
        np.isclose(unique_floats[1:], unique_floats[:-1])
    )

    return np.concatenate((
        unique_floats[:1],
        unique_floats[1:][next_distance_not_close],
    ))


def define_bin_intervals(unique_distances: np.ndarray) -> pd.IntervalIndex:
    """Constructs bin intervals used to group over neighbor distances.

    This binning procedure provides a robust method for grouping data based on a
    variable with a float data type.

    :param unique_distances: An array of neighbor distances returned by asking
        pandas to return the unique distances.
    :return: A pandas ``IntervalIndex`` defining bin intervals can be used to sort
        and group neighbor distances.
    """
    bin_centers: np.ndarray = np.concatenate(([0], unique_distances))

    bin_edges: np.ndarray = np.concatenate([
        bin_centers[:-1] + (bin_centers[1:] - bin_centers[:-1]) / 2,
        bin_centers[-1:] + (bin_centers[-1:] - bin_centers[-2:-1]) / 2,
    ])

    return pd.IntervalIndex.from_breaks(breaks=bin_edges)


def get_neighbor_distances_data_frame(
    cell_structure: pmg.Structure,
    r: float,
) -> pd.DataFrame:
    """Get data frame of pairwise neighbor distances for each atom in the unit cell,
    out to a distance ``r``.

    :param cell_structure: A pymatgen ``Structure`` object.
    :param r: Cut-off distance to use when detecting site neighbors.
    :return: A pandas ``DataFrame`` of pairwise neighbor distances.
    """
    all_neighbors: AllNeighborDistances = cell_structure.get_all_neighbors(
        r=r,
        include_index=True,
    )

    neighbor_distances: NeighborDistances = extract_neighbor_distance_data(
        cell_structure=cell_structure,
        all_neighbors=all_neighbors,
    )

    return pd.DataFrame(data=neighbor_distances)


def extract_neighbor_distance_data(
    cell_structure: pmg.Structure,
    all_neighbors: AllNeighborDistances,
) -> NeighborDistances:
    """Extracts the site indices, site species, and neighbor distances for each pair
    and stores it in a dictionary.

    :param cell_structure: A pymatgen ``Structure`` object.
    :param all_neighbors: A list of lists containing the neighbors for each site in
        the structure.
    :return: A dictionary of site indices, site species, and neighbor distances for
        each pair.
    """
    neighbor_distances: NeighborDistances = {
        "i": [],
        "j": [],
        "subspecies_i": [],
        "subspecies_j": [],
        "distance_ij": [],
    }

    site_i_index: int
    site_i_neighbors: SiteNeighbors
    for site_i_index, site_i_neighbors in enumerate(all_neighbors):
        append_site_i_neighbor_distance_data(
            site_i_index=site_i_index,
            site_i_neighbors=site_i_neighbors,
            cell_structure=cell_structure,
            neighbor_distances=neighbor_distances,
        )

    return neighbor_distances


def append_site_i_neighbor_distance_data(
    site_i_index: int,
    site_i_neighbors: SiteNeighbors,
    cell_structure: pmg.Structure,
    neighbor_distances: NeighborDistances,
) -> None:
    """Helper function to append indices, species, and distances in the
    ``neighbor_distances`` dictionary.

    :param site_i_index: Site index of first site in neighbor pair.
    :param site_i_neighbors: A list of site i's neighbors.
    :param cell_structure: The pymatgen ``Structure`` object that defines the crystal
        structure.
    :param neighbor_distances: A dictionary of site indices, site species, and neighbor
        distances for each pair.
    """
    site_j: Neighbor
    for site_j in site_i_neighbors:
        subspecies_pair: List[str] = [
            cell_structure[site_i_index].properties["subspecie"],
            cell_structure[site_j[2]].properties["subspecie"],
        ]

        index_pair: List[str] = [site_i_index, site_j[2]]

        neighbor_distances["i"].append(index_pair[0])
        neighbor_distances["j"].append(index_pair[1])
        neighbor_distances["subspecies_i"].append(subspecies_pair[0])
        neighbor_distances["subspecies_j"].append(subspecies_pair[1])
        neighbor_distances["distance_ij"].append(site_j[1])
