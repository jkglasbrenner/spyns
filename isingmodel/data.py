# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List, NamedTuple, Tuple, Union

import numpy as np
import pandas as pd


class SimulationParameters(NamedTuple):
    seed: int
    trace_filepath: Union[str, None]
    dimensions: Tuple[int, int]
    sweeps: int
    equilibration_sweeps: int
    sample_interval: int
    temperature: float
    interaction_coefficients: Union[Tuple[float], Tuple[float, float]]
    neighborhood: str


@dataclass
class SimulationTrace(object):
    sweep: List[Union[int, None]]
    energy: List[Union[int, None]]
    magnetization: List[Union[int, None]]
    E_1st_moment: List[Union[float, None]]
    E_2nd_moment: List[Union[float, None]]
    E_3rd_moment: List[Union[float, None]]
    E_4th_moment: List[Union[float, None]]
    Mag_1st_moment: List[Union[float, None]]
    Mag_2nd_moment: List[Union[float, None]]
    Mag_3rd_moment: List[Union[float, None]]
    Mag_4th_moment: List[Union[float, None]]
    __slots__ = [
        "sweep", "energy", "magnetization", "E_1st_moment", "E_2nd_moment",
        "E_3rd_moment", "E_4th_moment", "Mag_1st_moment", "Mag_2nd_moment",
        "Mag_3rd_moment", "Mag_4th_moment"
    ]


@dataclass
class SimulationData(object):
    parameters: SimulationParameters
    trace: SimulationTrace
    estimators: np.ndarray
    state: np.ndarray
    __slots__ = ["parameters", "trace", "estimators", "state"]


def setup_containers(parameters: SimulationParameters) -> SimulationData:
    """Initialize the data container for the simulation.

    :param parameters: Parameters to use in simulation.
    :return: Data container for the simulation.
    """
    return SimulationData(
        parameters=parameters,
        trace=SimulationTrace([], [], [], [], [], [], [], [], [], [], []),
        state=np.array(
            [(0, 0)],
            dtype=np.dtype(
                {
                    "names": ["energy", "magnetization"],
                    "formats": ["f8", "f8"],
                    "offsets": [0, 8],
                },
                align=True,
            ),
        ),
        estimators=np.array(
            [(0, 0, 0, 0, 0, 0, 0, 0)],
            dtype=np.dtype(
                {
                    "names": [
                        "E_1st_moment", "E_2nd_moment", "E_3rd_moment", "E_4th_moment",
                        "Mag_1st_moment", "Mag_2nd_moment", "Mag_3rd_moment",
                        "Mag_4th_moment"
                    ],
                    "formats": ["f8", "f8", "f8", "f8", "f8", "f8", "f8", "f8"],
                    "offsets": [0, 8, 16, 24, 32, 40, 48, 56]
                },
                align=True,
            ),
        ),
    )


def update_trace(data: SimulationData, sweep_index: int, number_sites: int) -> None:
    """Add estimators sample to the simulation trace.

    :param data: Data container for the simulation.
    :param sweep_index: Sweep index for the simulation.
    :param number_sites: Number of lattice sites in the simulation.
    """
    data.trace.sweep.append(sweep_index)
    data.trace.energy.append(data.state["energy"][0])
    data.trace.magnetization.append(data.state["magnetization"][0])
    data.trace.E_1st_moment.append(data.estimators["E_1st_moment"][0] / number_sites)
    data.trace.E_2nd_moment.append(data.estimators["E_2nd_moment"][0] / number_sites)
    data.trace.E_3rd_moment.append(data.estimators["E_3rd_moment"][0] / number_sites)
    data.trace.E_4th_moment.append(data.estimators["E_4th_moment"][0] / number_sites)
    data.trace.Mag_1st_moment.append(
        data.estimators["Mag_1st_moment"][0] / number_sites
    )
    data.trace.Mag_2nd_moment.append(
        data.estimators["Mag_2nd_moment"][0] / number_sites
    )
    data.trace.Mag_3rd_moment.append(
        data.estimators["Mag_3rd_moment"][0] / number_sites
    )
    data.trace.Mag_4th_moment.append(
        data.estimators["Mag_4th_moment"][0] / number_sites
    )


def write_trace_to_disk(data: SimulationData) -> None:
    """Save simulation history to disk.

    :param data: Data container for the simulation.
    """
    trace_df: pd.DataFrame = pd.DataFrame({
        "sweep": data.trace.sweep,
        "energy": data.trace.energy,
        "magnetization": data.trace.magnetization,
        "E^1": data.trace.E_1st_moment,
        "E^2": data.trace.E_2nd_moment,
        "E^3": data.trace.E_3rd_moment,
        "E^4": data.trace.E_4th_moment,
        "M^1": data.trace.Mag_1st_moment,
        "M^2": data.trace.Mag_2nd_moment,
        "M^3": data.trace.Mag_3rd_moment,
        "M^4": data.trace.Mag_4th_moment,
    })

    if data.parameters.trace_filepath:
        trace_df.to_csv(path_or_buf=data.parameters.trace_filepath, index=False)
