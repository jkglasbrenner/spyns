# -*- coding: utf-8 -*-

import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

ScalingMatrix = Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]


@dataclass(frozen=True)
class StructureParameters(object):
    abc: Tuple[float, float, float]
    ang: Tuple[float, float, float]
    spacegroup: int
    species: List[str]
    coordinates: List[List[float]]
    __slots__ = ["abc", "ang", "spacegroup", "species", "coordinates"]


@dataclass(frozen=True)
class StructureFile(object):
    path: str
    __slots__ = ["path"]


@dataclass(frozen=True)
class SimulationParameters(object):
    seed: int
    mode: str
    trace_filepath: Optional[str]
    snapshot_filepath: Optional[str]
    sweeps: int
    equilibration_sweeps: int
    sample_interval: int
    temperature: Optional[float]
    __slots__ = [
        "seed",
        "mode",
        "trace_filepath",
        "snapshot_filepath",
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
    interaction_parameters_table: Optional[np.ndarray]
    number_sites: int
    number_sublattices: int


@dataclass
class HeisenbergState(object):
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    __slots__ = ["x", "y", "z"]


@dataclass
class Estimators(object):
    number_samples: np.ndarray
    energy: np.ndarray
    spin_vector: np.ndarray
    magnetization: np.ndarray
    __slots__ = ["number_samples", "energy", "spin_vector", "magnetization"]


@dataclass
class SimulationTrace(object):
    sweep: np.ndarray
    energy: np.ndarray
    spin_vector: np.ndarray
    magnetization: np.ndarray
    __slots__ = ["sweep", "energy", "spin_vector", "magnetization"]


@dataclass
class SimulationData(object):
    parameters: SimulationParameters
    lookup_tables: LookupTables
    state: Union[np.ndarray, HeisenbergState]
    trace: SimulationTrace
    estimators: Estimators
    data_frame: Optional[pd.DataFrame]
    __slots__ = [
        "parameters",
        "lookup_tables",
        "state",
        "trace",
        "estimators",
        "data_frame",
    ]


def setup_containers(
    parameters: SimulationParameters,
    state: Union[np.ndarray, HeisenbergState],
    lattice: "Lattice",
) -> SimulationData:
    """Initialize sPyns simulation data container.

    :param parameters: Parameters to use for setting up and running the simulation.
    :param state: One-dimensional array of the simulation state.
    :param lattice: Neighbor and interaction tables that define the system under
        simulation.
    :return: Data container for the simulation.
    """
    spin_components: int = 1

    if parameters.mode.strip().lower() in ["heisenberg", "heisenberg_cython"]:
        spin_components = 3

    spin_vector_estimator_shape: Tuple[int, int] = (
        lattice.number_sublattices,
        spin_components,
    )
    spin_vector_trace_shape: Tuple[int, int, int] = (
        parameters.sweeps,
        lattice.number_sublattices,
        spin_components,
    )

    interaction_parameters_table: Optional[np.ndarray] = None

    if parameters.mode.strip().lower() in ["ising", "heisenberg", "heisenberg_cython"]:
        interaction_parameters_table = lattice.interaction_parameters_table

    return SimulationData(
        parameters=parameters,
        lookup_tables=LookupTables(
            sublattice_table=lattice.sublattice_table,
            sublattice_labels=lattice.sublattice_labels,
            neighbors_table=lattice.neighbors_table,
            neighbors_count=lattice.neighbors_count,
            neighbors_lookup_index=lattice.neighbors_lookup_index,
            interaction_parameters_table=interaction_parameters_table,
            number_sites=lattice.number_sites,
            number_sublattices=lattice.number_sublattices,
        ),
        state=state,
        trace=SimulationTrace(
            np.arange(start=1, stop=parameters.sweeps + 1, dtype=np.int),
            np.zeros(shape=parameters.sweeps, dtype=np.float),
            np.zeros(shape=spin_vector_trace_shape, dtype=np.float),
            np.zeros(shape=parameters.sweeps, dtype=np.float),
        ),
        estimators=Estimators(
            np.zeros(shape=1, dtype=np.int),
            np.zeros(shape=1, dtype=np.float),
            np.zeros(shape=spin_vector_estimator_shape, dtype=np.float),
            np.zeros(shape=1, dtype=np.float),
        ),
        data_frame=None,
    )


def make_trace_data_frame(data: SimulationData) -> None:
    """Make data frame of the trace history and store in simulation data container.

    :param data: Data container for the simulation.
    """
    trace: Dict[str, np.ndarray] = {
        "sweep": data.trace.sweep,
        "E": data.trace.energy,
        "M": data.trace.magnetization,
    }

    if data.parameters.mode.strip().lower() in ["ising", "voter"]:
        for sublattice in range(data.lookup_tables.number_sublattices):
            trace[f"S{sublattice}"] = data.trace.spin_vector[:, sublattice, 0]

    elif data.parameters.mode.strip().lower() == "heisenberg":
        for sublattice in range(data.lookup_tables.number_sublattices):
            trace[f"S{sublattice}x"] = data.trace.spin_vector[:, sublattice, 0]
            trace[f"S{sublattice}y"] = data.trace.spin_vector[:, sublattice, 1]
            trace[f"S{sublattice}z"] = data.trace.spin_vector[:, sublattice, 2]
            trace[f"theta{sublattice}"] = np.arctan(
                (trace[f"S{sublattice}x"] ** 2 + trace[f"S{sublattice}y"] ** 2)
                / trace[f"S{sublattice}z"]
            )
            trace[f"phi{sublattice}"] = np.arctan(
                trace[f"S{sublattice}y"] / trace[f"S{sublattice}x"]
            )

    data.data_frame = pd.DataFrame(trace)


def write_trace_history_to_disk(data: SimulationData) -> None:
    """Save simulation history to disk.

    :param data: Data container for the simulation.
    """
    if data.parameters.trace_filepath:
        data.data_frame.to_csv(path_or_buf=data.parameters.trace_filepath, index=False)


def dump_state_snapshot_to_disk(data: SimulationData, sweep_index: int) -> None:
    """Save snapshot of simulation state to disk.

    :param data: Data container for the simulation.
    :param sweep_index: Sweep index for the simulation.
    """
    components: Optional[List[str]] = []
    snapshot: List[Union[int, float]] = [sweep_index]

    if data.parameters.mode.strip().lower() in ["ising", "voter"]:
        components.extend([""])
        snapshot += data.state.tolist()

    elif data.parameters.mode.strip().lower() == "heisenberg":
        components.extend(["x", "y", "z"])
        snapshot += (
            data.state.x.tolist() + data.state.y.tolist() + data.state.z.tolist()
        )

    if sweep_index == 1:
        fieldnames: List[str] = [f"sweep"] + [
            f"site{site_index}{component}"
            for component in components
            for site_index in range(data.lookup_tables.number_sites)
        ]

        with open(data.parameters.snapshot_filepath, "w", newline="") as csvfile:
            dictwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
            dictwriter.writeheader()

    with open(data.parameters.snapshot_filepath, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(snapshot)
