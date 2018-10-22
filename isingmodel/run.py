# -*- coding: utf-8 -*-

import numpy as np

from isingmodel.data import SimulationData, SimulationParameters
from isingmodel.distributions import BinaryLattice
import isingmodel


def simulation(parameters: SimulationParameters) -> SimulationData:
    """Run the Ising model simulation.

    :param parameters: Parameters to use for setting up and running the simulation.
    :return: Output of the Ising model simulation.
    """
    np.random.seed(parameters.seed)

    lattice: BinaryLattice = BinaryLattice(parameters.dimensions)
    data: SimulationData = isingmodel.data.setup_containers(
        parameters=parameters,
        state=lattice.sample_random_state(),
    )

    pre_simulation(
        lattice=lattice,
        data=data,
    )
    main_simulation(
        lattice=lattice,
        data=data,
    )
    post_simulation(
        lattice=lattice,
        data=data,
    )

    return data


def pre_simulation(
    lattice: BinaryLattice,
    data: SimulationData,
) -> None:
    """Run equilibration sweeps.

    :param lattice: Structural information and simulation state.
    :param data: Data container for the simulation.
    """
    for sweep_index in range(data.parameters.equilibration_sweeps):
        isingmodel.sampling.sweep_grid(
            lattice=lattice,
            data=data,
            sweep_index=sweep_index,
            equilibration_run=True,
        )


def main_simulation(
    lattice: BinaryLattice,
    data: SimulationData,
) -> None:
    """Run the production sweeps for the Ising model simulation.

    :param lattice: Structural information and simulation state.
    :param data: Data container for the simulation.
    """
    isingmodel.model.ising_save_full_state(lattice=lattice, data=data)
    for sweep_index in range(data.parameters.sweeps):
        isingmodel.sampling.sweep_grid(
            lattice=lattice,
            data=data,
            sweep_index=sweep_index,
            equilibration_run=False,
        )


def post_simulation(
    lattice: BinaryLattice,
    data: SimulationData,
) -> None:
    """Write the simulation history to disk and print energy and magnetization
    estimators.

    :param lattice: Structural information and simulation state.
    :param data: Data container for the simulation.
    """
    isingmodel.data.write_trace_to_disk(data=data)
    print(f"Energy = {data.estimators.energy_1st_moment / lattice.number_sites}")
    print(
        "Magnetization = "
        f"{data.estimators.magnetization_1st_moment / lattice.number_sites}"
    )
