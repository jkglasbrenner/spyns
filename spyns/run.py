# -*- coding: utf-8 -*-

import numpy as np

from spyns.data import HeisenbergState, SimulationData, SimulationParameters
from spyns.lattice import Lattice
import spyns


def simulation(lattice: Lattice, parameters: SimulationParameters) -> SimulationData:
    """Run a sPyns simulation.

    sPyns currently supports two models of spin simulations on a periodic lattice, the
    Ising model and the Heisenberg model.

    :param lattice: Neighbor and interaction tables that define the system under
        simulation.
    :param parameters: Parameters to use for setting up and running the simulation.
    :return: Data container of results for the sPyns simulation.
    """
    np.random.seed(parameters.seed)

    if parameters.mode.strip().lower() == "ising":
        data: SimulationData = spyns.data.setup_containers(
            parameters=parameters,
            state=spyns.model.ising.sample_random_state(lattice.number_sites),
            lattice=lattice,
        )

    elif parameters.mode.strip().lower() == "heisenberg":
        heisenberg_state: HeisenbergState = \
            spyns.model.heisenberg.sample_random_state(lattice.number_sites)
        data: SimulationData = spyns.data.setup_containers(
            parameters=parameters,
            state=heisenberg_state,
            lattice=lattice,
        )

    else:
        raise AttributeError(
            f"Simulation mode {parameters.mode.strip()} is not supported."
        )

    pre_simulation(data=data)
    main_simulation(data=data)
    post_simulation(data=data)

    return data


def pre_simulation(data: SimulationData) -> None:
    """Run equilibration sweeps.

    :param data: Data container for the simulation.
    """
    if data.parameters.mode.strip().lower() == "ising":
        for sweep_index in range(data.parameters.equilibration_sweeps):
            spyns.algorithms.metropolis.ising.sweep(
                data=data,
                sweep_index=sweep_index,
                equilibration_run=True,
            )

    elif data.parameters.mode.strip().lower() == "heisenberg":
        for sweep_index in range(data.parameters.equilibration_sweeps):
            spyns.algorithms.metropolis.heisenberg.sweep(
                data=data,
                sweep_index=sweep_index,
                equilibration_run=True,
            )

    else:
        raise AttributeError(
            f"Simulation mode {data.parameters.mode.strip()} is not supported."
        )


def main_simulation(data: SimulationData) -> None:
    """Run the production sweeps for the sPyns simulation.

    :param data: Data container for the simulation.
    """
    if data.parameters.mode.strip().lower() == "ising":
        spyns.model.ising.save_full_state(data=data)
        for sweep_index in range(data.parameters.sweeps):
            spyns.algorithms.metropolis.ising.sweep(
                data=data,
                sweep_index=sweep_index,
                equilibration_run=False,
            )

    elif data.parameters.mode.strip().lower() == "heisenberg":
        spyns.model.heisenberg.save_full_state(data=data)
        for sweep_index in range(data.parameters.sweeps):
            spyns.algorithms.metropolis.heisenberg.sweep(
                data=data,
                sweep_index=sweep_index,
                equilibration_run=False,
            )

    else:
        raise AttributeError(
            f"Simulation mode {data.parameters.mode.strip()} is not supported."
        )


def post_simulation(data: SimulationData) -> None:
    """Make (and optionally save) a trace history data frame and print estimators.

    :param data: Data container for the simulation.
    """
    spyns.data.make_trace_data_frame(data=data)
    spyns.data.write_trace_history_to_disk(data=data)

    average_energy: float = \
        data.estimators.energy_1st_moment / data.lookup_tables.number_sites
    magnetization: np.ndarray = \
        data.estimators.magnetization_1st_moment / data.lookup_tables.number_sites

    print(f"Average energy = {average_energy}")
    print(f"Average magnetization = {magnetization}")
