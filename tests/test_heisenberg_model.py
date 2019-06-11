# -*- coding: utf-8 -*-

import numpy as np
import pymatgen as pmg
import pytest

from spyns.data import (
    HeisenbergState,
    StructureParameters,
    SimulationParameters,
    SimulationData,
)
from spyns.lattice import Lattice
import spyns
import spyns.model.heisenberg
from spyns.data_cython import SimulationHeisenbergData_t
from spyns.random_numbers.distribution import RandomNumberGenerator
import spyns.model.heisenberg_cython


@pytest.fixture()
def simple_cubic_lattice() -> pmg.Structure:
    structure_parameters: StructureParameters = StructureParameters(
        abc=(2.0, 2.0, 2.0),
        ang=3 * (90,),
        spacegroup=1,
        species=8 * ["Fe"],
        coordinates=[
            [0.00, 0.00, 0.00],
            [0.50, 0.00, 0.00],
            [0.00, 0.50, 0.00],
            [0.50, 0.50, 0.00],
            [0.00, 0.00, 0.50],
            [0.50, 0.00, 0.50],
            [0.00, 0.50, 0.50],
            [0.50, 0.50, 0.50],
        ],
    )
    structure: pmg.Structure = spyns.lattice.generate.from_parameters(
        structure_parameters=structure_parameters
    )
    structure = spyns.lattice.generate.label_subspecies(
        structure=structure,
        subspecies_labels={
            0: "1",
            1: "2",
            2: "2",
            3: "1",
            4: "2",
            5: "1",
            6: "1",
            7: "2",
        },
    )
    structure = spyns.lattice.generate.make_supercell(
        cell_structure=structure, scaling_factors=(5, 5, 5)
    )

    return structure


@pytest.fixture()
def simulation_parameters() -> SimulationParameters:
    return SimulationParameters(
        seed=np.random.randint(100000),
        mode="heisenberg_cython",
        trace_filepath=None,
        snapshot_filepath=None,
        sweeps=400,
        equilibration_sweeps=100,
        sample_interval=1,
        temperature=1,
    )


@pytest.fixture()
def sc_heisenberg_model(
    simple_cubic_lattice: pmg.Structure, simulation_parameters: SimulationParameters
) -> SimulationData:
    lattice: Lattice = Lattice(structure=simple_cubic_lattice, r=1.2)

    lattice.set_sublattice_pair_interactions(
        interaction_df=lattice.sublattice_pairs_data_frame.assign(J_ij=2 * (-1.0,))
    )

    heisenberg_state: HeisenbergState = spyns.model.heisenberg.sample_random_state(
        lattice.number_sites
    )

    return spyns.data.setup_containers(
        parameters=simulation_parameters, state=heisenberg_state, lattice=lattice
    )


def test_sc_heisenberg_fm_energy_and_magnetization_computation(
    sc_heisenberg_model: SimulationData
) -> None:
    total_energy_random: float = spyns.model.heisenberg.compute_total_energy(
        data=sc_heisenberg_model
    )
    total_energy_random /= sc_heisenberg_model.lookup_tables.number_sites

    magnetization_random: float = np.sum(
        spyns.model.heisenberg.sum_spin_vectors_within_sublattices(
            data=sc_heisenberg_model
        ),
        axis=0,
    )
    magnetization_random /= sc_heisenberg_model.lookup_tables.number_sites

    print(f"Average energy (random) = {total_energy_random}")
    print(f"Magnetization (random) = {magnetization_random}")

    assert np.abs(total_energy_random) <= 3.0
    assert np.all(np.abs(magnetization_random) <= 1.0)

    for vector_align in np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]):
        sc_heisenberg_model.state.x[:] = vector_align[0]
        sc_heisenberg_model.state.y[:] = vector_align[1]
        sc_heisenberg_model.state.z[:] = vector_align[2]

        total_energy_aligned: float = spyns.model.heisenberg.compute_total_energy(
            data=sc_heisenberg_model
        )
        total_energy_aligned /= sc_heisenberg_model.lookup_tables.number_sites

        magnetization_aligned: float = np.sum(
            spyns.model.heisenberg.sum_spin_vectors_within_sublattices(
                data=sc_heisenberg_model
            ),
            axis=0,
        )
        magnetization_aligned /= sc_heisenberg_model.lookup_tables.number_sites
        magnetization_aligned = np.linalg.norm(magnetization_aligned)

        print(f"Average energy ({vector_align} aligned) = {total_energy_aligned}")
        print(f"Magnetization ({vector_align} z-aligned) = {magnetization_aligned}")

        assert np.abs(total_energy_aligned) <= 3.0
        assert np.abs(magnetization_aligned) <= 1.0


def test_sc_heisenberg_cython_fm_energy_and_magnetization_computation(
    sc_heisenberg_model: SimulationData
) -> None:
    random_number_generator = RandomNumberGenerator(
        seed=sc_heisenberg_model.parameters.seed,
        number_sites=sc_heisenberg_model.lookup_tables.number_sites,
    )

    data = SimulationHeisenbergData_t(
        data=sc_heisenberg_model, random_number_generator=random_number_generator
    )


def test_sc_heisenberg_cython_fm_energy_and_magnetization_computation(
    sc_heisenberg_model: SimulationData
) -> None:
    random_number_generator = RandomNumberGenerator(
        seed=sc_heisenberg_model.parameters.seed,
        number_sites=sc_heisenberg_model.lookup_tables.number_sites,
    )

    data = SimulationHeisenbergData_t(
        data=sc_heisenberg_model, random_number_generator=random_number_generator
    )

    total_energy_random: float = spyns.model.heisenberg_cython.compute_total_energy(
        data
    )
    total_energy_random /= data.container.lookup_tables.number_sites

    magnetization_random: float = np.sum(
        spyns.model.heisenberg.sum_spin_vectors_within_sublattices(data=data.container),
        axis=0,
    )
    magnetization_random /= data.container.lookup_tables.number_sites

    print(f"Average energy (random) = {total_energy_random}")
    print(f"Magnetization (random) = {magnetization_random}")

    assert np.abs(total_energy_random) <= 3.0
    assert np.all(np.abs(magnetization_random) <= 1.0)

    for vector_align in np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]):
        data.container.state.x[:] = vector_align[0]
        data.container.state.y[:] = vector_align[1]
        data.container.state.z[:] = vector_align[2]

        total_energy_aligned: float = spyns.model.heisenberg_cython.compute_total_energy(
            data
        )
        total_energy_aligned /= data.container.lookup_tables.number_sites

        magnetization_aligned: float = np.sum(
            spyns.model.heisenberg.sum_spin_vectors_within_sublattices(
                data=data.container
            ),
            axis=0,
        )
        magnetization_aligned /= data.container.lookup_tables.number_sites
        magnetization_aligned = np.linalg.norm(magnetization_aligned)

        print(f"Average energy ({vector_align} aligned) = {total_energy_aligned}")
        print(f"Magnetization ({vector_align} z-aligned) = {magnetization_aligned}")

        assert np.abs(total_energy_aligned) <= 3.0
        assert np.abs(magnetization_aligned) <= 1.0
