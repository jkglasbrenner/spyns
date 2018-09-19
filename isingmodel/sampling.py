# -*- coding: utf-8 -*-

from isingmodel.distributions import BinaryLattice
from isingmodel.data import SimulationData
import isingmodel


def sweep_grid(
    lattice: BinaryLattice,
    data: SimulationData,
    sweep_index: int,
    equilibration_run: bool,
) -> None:
    """Sweep the lattice and take a sample if required.

    :param lattice: Structural information and simulation state.
    :param data: Data container for the simulation.
    :param sweep_index: Sweep index for the simulation.
    :param equilibration_run: Whether or not the current sweep is part of equilibration
        run.
    """
    for _ in range(lattice.number_sites):
        isingmodel.algorithms.metropolis(lattice=lattice, data=data)

    if sweep_index % data.parameters.sample_interval == 0 and not equilibration_run:
        update_simulation_data(data=data, lattice=lattice, sweep_index=sweep_index)


def update_simulation_data(
    data: SimulationData,
    lattice: BinaryLattice,
    sweep_index: int,
) -> None:
    """Take a sample and update the simulation estimators and trace.
    
    :param data: Data container for the simulation.
    :param lattice: Structural information and simulation state.
    :param sweep_index: Sweep index for the simulation.
    """
    number_samples: int = len(data.trace.sweep)
    isingmodel.statistics.update_estimators(data=data, number_samples=number_samples)
    isingmodel.data.update_trace(
        data=data,
        sweep_index=sweep_index,
        number_sites=lattice.number_sites,
    )
