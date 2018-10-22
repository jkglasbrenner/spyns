# -*- coding: utf-8 -*-

import numpy as np
import pytest

from isingmodel.data import SimulationParameters, SimulationData
import isingmodel.run as run


@pytest.fixture()
def neumann_fm_parameters() -> SimulationParameters:
    return SimulationParameters(
        seed=np.random.randint(100000),
        trace_filepath=None,
        dimensions=(10, 10),
        sweeps=100,
        equilibration_sweeps=10,
        sample_interval=1,
        temperature=1,
        interaction_coefficients=(-1, ),
        neighborhood="Neumann",
    )


@pytest.fixture()
def moore_fm_parameters() -> SimulationParameters:
    return SimulationParameters(
        seed=np.random.randint(100000),
        trace_filepath=None,
        dimensions=(10, 10),
        sweeps=100,
        equilibration_sweeps=10,
        sample_interval=1,
        temperature=1,
        interaction_coefficients=(-1, -1),
        neighborhood="Moore",
    )


def test_ising_run_neumann_ferromagnet(neumann_fm_parameters) -> None:
    data: SimulationData = run.simulation(parameters=neumann_fm_parameters)
    number_sites: int = np.prod(data.parameters.dimensions, dtype="i8")
    energy: np.ndarray = data.estimators.energy / number_sites
    magnetization: np.ndarray = data.estimators.magnetization / number_sites

    assert energy >= -2.0 and energy <= 2.0
    assert magnetization >= -1.0 and magnetization <= 1.0


def test_ising_run_moore_ferromagnet(moore_fm_parameters) -> None:
    data: SimulationData = run.simulation(parameters=moore_fm_parameters)
    number_sites: int = np.prod(data.parameters.dimensions, dtype="i8")
    energy: np.ndarray = data.estimators.energy / number_sites
    magnetization: np.ndarray = data.estimators.magnetization / number_sites

    assert energy >= -4.0 and energy <= 4.0
    assert magnetization >= -1.0 and magnetization <= 1.0
