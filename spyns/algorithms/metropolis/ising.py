# -*- coding: utf-8 -*-

from spyns.data import SimulationData
import spyns


def step(data: SimulationData) -> None:
    """Update system state using the Metropolis algorithm.

    :param data: Data container for the simulation.
    """
    site_index: int = spyns.algorithms.metropolis.base.pick_site(
        number_sites=data.lookup_tables.number_sites
    )
    energy_difference: float = spyns.model.ising.flip(
        site_index=site_index,
        data=data,
    )
    accept_state: bool = spyns.algorithms.metropolis.base.accept_or_reject(
        temperature=data.parameters.temperature,
        energy_difference=energy_difference,
    )
    if accept_state:
        spyns.model.ising.keep_flip_and_update_state(
            data=data,
            site_index=site_index,
            energy_difference=energy_difference,
        )


def sweep(
    data: SimulationData,
    sweep_index: int,
    equilibration_run: bool,
) -> None:
    """Sweep the lattice and take a sample if required.

    :param data: Data container for the simulation.
    :param sweep_index: Sweep index for the simulation.
    :param equilibration_run: Whether or not the current sweep is part of equilibration
        run.
    """
    for _ in range(data.lookup_tables.number_sites):
        step(data=data)

    if sweep_index % data.parameters.sample_interval == 0 and not equilibration_run:
        spyns.statistics.update_estimators(data=data)
        spyns.statistics.update_trace(data=data, sweep_index=sweep_index)
