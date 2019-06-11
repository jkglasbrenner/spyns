# -*- coding: utf-8 -*-

import numpy as np

from spyns.data import HeisenbergState, SimulationData, SimulationParameters
from spyns.lattice import Lattice
import spyns
import spyns.model.heisenberg
import spyns.algorithms.metropolis.heisenberg_cython
import spyns.model.heisenberg_cython
from spyns.data_cython import SimulationHeisenbergData_t
from spyns.random_numbers.distribution import RandomNumberGenerator


def simulation(lattice: Lattice, parameters: SimulationParameters) -> SimulationData:
    """Run a sPyns simulation.

    sPyns currently supports one model of spin simulations on a periodic lattice, the
    Heisenberg model.

    :param lattice: Neighbor and interaction tables that define the system under
        simulation.
    :param parameters: Parameters to use for setting up and running the simulation.
    :return: Data container of results for the sPyns simulation.
    """
    np.random.seed(parameters.seed)

    heisenberg_state: HeisenbergState = spyns.model.heisenberg.sample_random_state(
        lattice.number_sites
    )
    data_object: SimulationData = spyns.data.setup_containers(
        parameters=parameters, state=heisenberg_state, lattice=lattice
    )
    random_number_generator: RandomNumberGenerator = RandomNumberGenerator(
        seed=data_object.parameters.seed,
        number_sites=data_object.lookup_tables.number_sites,
    )
    data: SimulationHeisenbergData_t = SimulationHeisenbergData_t(
        data=data_object, random_number_generator=random_number_generator
    )

    pre_simulation(data=data)
    main_simulation(data=data)
    post_simulation(data=data)

    return data


def pre_simulation(data: SimulationData) -> None:
    """Run equilibration sweeps.

    :param data: Data container for the simulation.
    """
    spyns.algorithms.metropolis.heisenberg_cython.run_sweeps(
        data=data, equilibration_run=True
    )


def main_simulation(data: SimulationData) -> None:
    """Run the production sweeps for the sPyns simulation.

    :param data: Data container for the simulation.
    """
    spyns.model.heisenberg.save_full_state(data=data.container)
    spyns.algorithms.metropolis.heisenberg_cython.run_sweeps(
        data=data, equilibration_run=False
    )


def post_simulation(data: SimulationData) -> None:
    """Make (and optionally save) a trace history data frame and print estimators.

    :param data: Data container for the simulation.
    """
    spyns.data.make_trace_data_frame(data=data.container)

    for estimator in ["E", "M"]:
        spyns.statistics.compute_estimator_moments(
            trace_df=data.container.data_frame, estimator_name=estimator
        )

        for power in range(1, 5):
            spyns.statistics.compute_running_average(
                trace_df=data.container.data_frame,
                estimator_name=f"{estimator}**{power}",
            )

    if data.container.parameters.mode.strip().lower() in ["heisenberg_cython"]:
        for fluctuation_name, estimator_name, temperature_power in [
            ("C", "E", 2),
            ("X", "M", 1),
        ]:
            spyns.statistics.compute_estimator_fluctuations(
                trace_df=data.container.data_frame,
                fluctuation_name=fluctuation_name,
                estimator_name=estimator_name,
                number_sites=data.container.lookup_tables.number_sites,
                coefficient=(
                    1 / data.container.parameters.temperature ** temperature_power
                ),
            )

        spyns.statistics.compute_binder_parameter(
            trace_df=data.container.data_frame, estimator_name="M"
        )

    spyns.data.write_trace_history_to_disk(data=data.container)

    average_energy: float = data.container.data_frame["<E**1>"].values[
        -1
    ] / data.container.lookup_tables.number_sites
    magnetization: float = data.container.data_frame["<M**1>"].values[
        -1
    ] / data.container.lookup_tables.number_sites

    print(f"Average energy = {average_energy}")
    print(f"Average magnetization = {magnetization}")
