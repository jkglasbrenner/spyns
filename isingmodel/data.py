# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationParameters(object):
    seed: int
    trace_filepath: Union[str, None]
    dimensions: Tuple[int, int]
    sweeps: int
    equilibration_sweeps: int
    sample_interval: int
    temperature: float
    interaction_coefficients: Union[Tuple[float], Tuple[float, float]]
    neighborhood: str
    __slots__ = [
        "seed",
        "trace_filepath",
        "dimensions",
        "sweeps",
        "equilibration_sweeps",
        "sample_interval",
        "temperature",
        "interaction_coefficients",
        "neighborhood",
    ]


@dataclass
class Estimators(object):
    energy: float
    energy_1st_moment: float
    energy_2nd_moment: float
    energy_3rd_moment: float
    energy_4th_moment: float
    magnetization: float
    magnetization_1st_moment: float
    magnetization_2nd_moment: float
    magnetization_3rd_moment: float
    magnetization_4th_moment: float
    __slots__ = [
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
    sweep: List[Union[int, None]]
    energy_1st_moment: List[Union[float, None]]
    energy_2nd_moment: List[Union[float, None]]
    energy_3rd_moment: List[Union[float, None]]
    energy_4th_moment: List[Union[float, None]]
    magnetization_1st_moment: List[Union[float, None]]
    magnetization_2nd_moment: List[Union[float, None]]
    magnetization_3rd_moment: List[Union[float, None]]
    magnetization_4th_moment: List[Union[float, None]]
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
    state: np.ndarray
    trace: SimulationTrace
    estimators: Estimators
    __slots__ = [
        "parameters",
        "state",
        "trace",
        "estimators",
    ]


def setup_containers(
    parameters: SimulationParameters,
    state: np.ndarray,
) -> SimulationData:
    """Initialize the data container for the simulation.

    :param parameters: Parameters to use in simulation.
    :return: Data container for the simulation.
    """
    return SimulationData(
        parameters=parameters,
        state=state,
        trace=SimulationTrace([], [], [], [], [], [], [], [], []),
        estimators=Estimators(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )


def update_trace(data: SimulationData, sweep_index: int, number_sites: int) -> None:
    """Add estimators sample to the simulation trace.

    :param data: Data container for the simulation.
    :param sweep_index: Sweep index for the simulation.
    :param number_sites: Number of lattice sites in the simulation.
    """
    data.trace.sweep.append(sweep_index)
    data.trace.energy_1st_moment.append(
        data.estimators.energy_1st_moment / number_sites
    )
    data.trace.energy_2nd_moment.append(
        data.estimators.energy_2nd_moment / number_sites
    )
    data.trace.energy_3rd_moment.append(
        data.estimators.energy_3rd_moment / number_sites
    )
    data.trace.energy_4th_moment.append(
        data.estimators.energy_4th_moment / number_sites
    )
    data.trace.magnetization_1st_moment.append(
        data.estimators.magnetization_1st_moment / number_sites
    )
    data.trace.magnetization_2nd_moment.append(
        data.estimators.magnetization_2nd_moment / number_sites
    )
    data.trace.magnetization_3rd_moment.append(
        data.estimators.magnetization_3rd_moment / number_sites
    )
    data.trace.magnetization_4th_moment.append(
        data.estimators.magnetization_4th_moment / number_sites
    )


def write_trace_to_disk(data: SimulationData) -> None:
    """Save simulation history to disk.

    :param data: Data container for the simulation.
    """
    trace_df: pd.DataFrame = pd.DataFrame({
        "sweep": data.trace.sweep,
        "E^1": data.trace.energy_1st_moment,
        "E^2": data.trace.energy_2nd_moment,
        "E^3": data.trace.energy_3rd_moment,
        "E^4": data.trace.energy_4th_moment,
        "M^1": data.trace.magnetization_1st_moment,
        "M^2": data.trace.magnetization_2nd_moment,
        "M^3": data.trace.magnetization_3rd_moment,
        "M^4": data.trace.magnetization_4th_moment,
    })

    if data.parameters.trace_filepath:
        trace_df.to_csv(path_or_buf=data.parameters.trace_filepath, index=False)
