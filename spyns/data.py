# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ScalingMatrix = \
    Tuple[Tuple[int, int, int],
          Tuple[int, int, int],
          Tuple[int, int, int]]


@dataclass(frozen=True)
class StructureParameters(object):
    abc: Tuple[float, float, float]
    ang: Tuple[float, float, float]
    spacegroup: int
    species: List[str]
    coordinates: List[List[float]]
    __slots__ = [
        "abc",
        "ang",
        "spacegroup",
        "species",
        "coordinates",
    ]


@dataclass(frozen=True)
class StructureFile(object):
    path: str
    __slots__ = [
        "path",
    ]


@dataclass(frozen=True)
class SimulationParameters(object):
    seed: int
    mode: str
    trace_filepath: Optional[str]
    sweeps: int
    equilibration_sweeps: int
    sample_interval: int
    temperature: float
    __slots__ = [
        "seed",
        "mode",
        "trace_filepath",
        "sweeps",
        "equilibration_sweeps",
        "sample_interval",
        "temperature",
    ]


@dataclass(frozen=True)
class LookupTables(object):
    sublattice_table: np.ndarray
    sublattice_labels: np.ndarray
    neighbors_table: np.ndarray
    neighbors_count: np.ndarray
    neighbors_lookup_index: np.ndarray
    interaction_parameters_table: np.ndarray
    number_sites: int
    number_sublattices: int


@dataclass
class Estimators(object):
    number_samples: int
    energy: float
    energy_1st_moment: float
    energy_2nd_moment: float
    energy_3rd_moment: float
    energy_4th_moment: float
    magnetization: np.ndarray
    magnetization_1st_moment: np.ndarray
    magnetization_2nd_moment: np.ndarray
    magnetization_3rd_moment: np.ndarray
    magnetization_4th_moment: np.ndarray
    __slots__ = [
        "number_samples",
        "energy",
        "energy_1st_moment",
        "energy_2nd_moment",
        "energy_3rd_moment",
        "energy_4th_moment",
        "magnetization",
        "magnetization_1st_moment",
        "magnetization_2nd_moment",
        "magnetization_3rd_moment",
        "magnetization_4th_moment",
    ]


@dataclass
class SimulationTrace(object):
    sweep: np.ndarray
    energy_1st_moment: np.ndarray
    energy_2nd_moment: np.ndarray
    energy_3rd_moment: np.ndarray
    energy_4th_moment: np.ndarray
    magnetization_1st_moment: np.ndarray
    magnetization_2nd_moment: np.ndarray
    magnetization_3rd_moment: np.ndarray
    magnetization_4th_moment: np.ndarray
    __slots__ = [
        "sweep",
        "energy_1st_moment",
        "energy_2nd_moment",
        "energy_3rd_moment",
        "energy_4th_moment",
        "magnetization_1st_moment",
        "magnetization_2nd_moment",
        "magnetization_3rd_moment",
        "magnetization_4th_moment",
    ]


@dataclass
class SimulationData(object):
    parameters: SimulationParameters
    lookup_tables: LookupTables
    state: np.ndarray
    trace: SimulationTrace
    estimators: Estimators
    __slots__ = [
        "parameters",
        "lookup_tables",
        "state",
        "trace",
        "estimators",
    ]


def setup_containers(
    parameters: SimulationParameters,
    state: np.ndarray,
    lattice: "Lattice",
) -> SimulationData:
    """Initialize the data container for the simulation.

    :param parameters: Parameters to use in simulation.
    :return: Data container for the simulation.
    """
    mag_trace_shape: Tuple[int, int] = (parameters.sweeps, lattice.number_sublattices)
    return SimulationData(
        parameters=parameters,
        lookup_tables=LookupTables(
            sublattice_table=lattice.sublattice_table,
            sublattice_labels=lattice.sublattice_labels,
            neighbors_table=lattice.neighbors_table,
            neighbors_count=lattice.neighbors_count,
            neighbors_lookup_index=lattice.neighbors_lookup_index,
            interaction_parameters_table=lattice.interaction_parameters_table,
            number_sites=lattice.number_sites,
            number_sublattices=lattice.number_sublattices,
        ),
        state=state,
        trace=SimulationTrace(
            np.arange(start=1, stop=parameters.sweeps + 1),
            np.zeros(shape=parameters.sweeps),
            np.zeros(shape=parameters.sweeps),
            np.zeros(shape=parameters.sweeps),
            np.zeros(shape=parameters.sweeps),
            np.zeros(shape=mag_trace_shape),
            np.zeros(shape=mag_trace_shape),
            np.zeros(shape=mag_trace_shape),
            np.zeros(shape=mag_trace_shape),
        ),
        estimators=Estimators(
            0,
            0,
            0,
            0,
            0,
            0,
            np.zeros(shape=lattice.number_sublattices),
            np.zeros(shape=lattice.number_sublattices),
            np.zeros(shape=lattice.number_sublattices),
            np.zeros(shape=lattice.number_sublattices),
            np.zeros(shape=lattice.number_sublattices),
        ),
    )


def write_trace_to_disk(
    data: SimulationData,
    number_sublattices: int,
) -> None:
    """Save simulation history to disk.

    :param data: Data container for the simulation.
    """
    trace: Dict[str, np.ndarray] = {
        "sweep": data.trace.sweep,
        "E": data.trace.energy_1st_moment,
        "E**2": data.trace.energy_2nd_moment,
        "E**3": data.trace.energy_3rd_moment,
        "E**4": data.trace.energy_4th_moment,
    }

    for sublattice_index in range(number_sublattices):
        trace[f"M{sublattice_index + 1}"] = \
            data.trace.magnetization_1st_moment[:, sublattice_index]
        trace[f"M{sublattice_index + 1}**2"] = \
            data.trace.magnetization_2nd_moment[:, sublattice_index]
        trace[f"M{sublattice_index + 1}**3"] = \
            data.trace.magnetization_3rd_moment[:, sublattice_index]
        trace[f"M{sublattice_index + 1}**4"] = \
            data.trace.magnetization_4th_moment[:, sublattice_index]

    trace_df: pd.DataFrame = pd.DataFrame(trace)

    if data.parameters.trace_filepath:
        trace_df.to_csv(path_or_buf=data.parameters.trace_filepath, index=False)
