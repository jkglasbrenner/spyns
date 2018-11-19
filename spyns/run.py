# -*- coding: utf-8 -*-

import numpy as np

from spyns.data import SimulationData, SimulationParameters
from spyns.lattice import Lattice
import spyns


def simulation(lattice: Lattice, parameters: SimulationParameters) -> SimulationData:
    """Run the Ising model simulation.

    :param lattice: Pymatgen ``Structure`` object and neighbor tables defining the
        system lattice.
    :param parameters: Parameters to use for setting up and running the simulation.
    :return: Output of the Ising model simulation.
    """
    np.random.seed(parameters.seed)

    if parameters.mode.strip().lower() == "ising":
        data: SimulationData = spyns.data.setup_containers(
            parameters=parameters,
            state=spyns.model.ising.sample_random_state(lattice.number_sites),
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

    else:
        raise AttributeError(
            f"Simulation mode {data.parameters.mode.strip()} is not supported."
        )


def main_simulation(data: SimulationData) -> None:
    """Run the production sweeps for the Ising model simulation.

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

    else:
        raise AttributeError(
            f"Simulation mode {data.parameters.mode.strip()} is not supported."
        )


def post_simulation(data: SimulationData) -> None:
    """Write the simulation history to disk and print estimators.

    :param data: Data container for the simulation.
    """
    if data.parameters.mode.strip().lower() == "ising":
        spyns.data.write_trace_to_disk(
            data=data,
            number_sublattices=data.lookup_tables.number_sublattices,
        )

        average_energy: float = \
            data.estimators.energy_1st_moment / data.lookup_tables.number_sites
        total_magnetization: float = \
            data.estimators.magnetization_1st_moment.sum() / data.lookup_tables.number_sites
        print(f"Average energy = {average_energy}")
        print(f"Total magnetization = {total_magnetization}")

    else:
        raise AttributeError(
            f"Simulation mode {data.parameters.mode.strip()} is not supported."
        )
