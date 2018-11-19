# -*- coding: utf-8 -*-

import numpy as np

from spyns.data import SimulationData
from spyns.model.heisenberg import TrialFlip
import spyns


def step(data: SimulationData) -> None:
    """Update system state of the Heisenberg model using the Metropolis algorithm.

    :param data: Data container for the simulation.
    """
    site_index: int = spyns.algorithms.metropolis.base.pick_site(
        number_sites=data.lookup_tables.number_sites
    )
    trial_flip: TrialFlip = spyns.model.heisenberg.flip(
        site_index=site_index,
        data=data,
    )
    accept_state: bool = spyns.algorithms.metropolis.base.accept_or_reject(
        temperature=data.parameters.temperature,
        energy_difference=trial_flip.energy_difference,
    )
    if accept_state:
        spyns.model.heisenberg.keep_flip_and_update_state(
            data=data,
            site_index=site_index,
            trial_flip=trial_flip,
        )


def sweep(
    data: SimulationData,
    sweep_index: int,
    equilibration_run: bool,
) -> None:
    """Sweep the Heisenberg lattice and take a sample if required.

    :param data: Data container for the simulation.
    :param sweep_index: Sweep index for the simulation.
    :param equilibration_run: Whether or not the current sweep is part of equilibration
        run.
    """
    for _ in range(data.lookup_tables.number_sites):
        step(data=data)

    if sweep_index % data.parameters.sample_interval == 0 and not equilibration_run:
        data.estimators.magnetization = np.linalg.norm(
            data.estimators.spin_vector.sum(axis=0)
        )
        spyns.statistics.update_estimators(data=data)
        spyns.statistics.update_trace(data=data, sweep_index=sweep_index)

        if data.parameters.snapshot_filepath:
            spyns.data.dump_state_snapshot_to_disk(
                data=data,
                sweep_index=sweep_index + 1,
            )
