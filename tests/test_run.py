# -*- coding: utf-8 -*-

from typing import Tuple, Union

import numpy as np
import pymatgen as pmg
import pytest

from spyns.data import StructureParameters, SimulationParameters, SimulationData
from spyns.lattice import Lattice
import spyns


@pytest.fixture()
def two_dimensional_square_lattice() -> pmg.Structure:
    structure_parameters: StructureParameters = StructureParameters(
        abc=(2.0, 2.0, 20.0),
        ang=3 * (90,),
        spacegroup=1,
        species=4 * ["Fe"],
        coordinates=[
            [0.00, 0.00, 0.00],
            [0.50, 0.00, 0.00],
            [0.00, 0.50, 0.00],
            [0.50, 0.50, 0.00],
        ],
    )
    structure: pmg.Structure = spyns.lattice.generate.from_parameters(
        structure_parameters=structure_parameters
    )
    structure = spyns.lattice.generate.label_subspecies(
        structure=structure, subspecies_labels={0: "1", 1: "2", 2: "2", 3: "1"}
    )
    structure = spyns.lattice.generate.make_supercell(
        cell_structure=structure, scaling_factors=(5, 5, 1)
    )

    return structure


@pytest.fixture()
def cubic_lattice() -> pmg.Structure:
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
def bcc_lattice() -> pmg.Structure:
    structure_parameters: StructureParameters = StructureParameters(
        abc=(2.0, 2.0, 1.0),
        ang=3 * (90,),
        spacegroup=1,
        species=8 * ["Fe"],
        coordinates=[
            [0.00, 0.00, 0.00],
            [0.50, 0.00, 0.00],
            [0.00, 0.50, 0.00],
            [0.50, 0.50, 0.00],
            [0.25, 0.25, 0.50],
            [0.75, 0.25, 0.50],
            [0.25, 0.75, 0.50],
            [0.75, 0.75, 0.50],
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
            4: "3",
            5: "4",
            6: "4",
            7: "3",
        },
    )
    structure = spyns.lattice.generate.make_supercell(
        cell_structure=structure, scaling_factors=(5, 5, 10)
    )

    return structure


@pytest.fixture()
def simulation_parameters_heisenberg_cython() -> SimulationParameters:
    return SimulationParameters(
        seed=np.random.randint(100000),
        mode="heisenberg_cython",
        trace_filepath=None,
        snapshot_filepath=None,
        sweeps=200,
        equilibration_sweeps=100,
        sample_interval=1,
        temperature=1,
    )


# @pytest.mark.parametrize(
#     "r, max_abs_energy, interaction_ij",
#     [
#         (1.2, 2.0, (-1.0, -1.0)),
#         (1.9, 4.0, (-1.0, -1.0, -1.0, -1.0)),
#     ],
# )
# def test_2d_square_ising_ferromagnet_simulation(
#     r: float,
#     max_abs_energy: float,
#     interaction_ij: Union[Tuple[float, float], Tuple[float, float, float, float]],
#     two_dimensional_square_lattice: pmg.Structure,
#     simulation_parameters: SimulationParameters,
# ) -> None:
#     lattice: Lattice = Lattice(structure=two_dimensional_square_lattice, r=r)
#
#     lattice.set_sublattice_pair_interactions(
#         interaction_df=lattice.sublattice_pairs_data_frame.assign(J_ij=interaction_ij)
#     )
#
#     data: SimulationData = spyns.run.simulation(
#         lattice=lattice,
#         parameters=simulation_parameters,
#     )
#
#     energy: float = \
#         data.data_frame["<E**1>"].values[-1] / data.lookup_tables.number_sites
#     magnetization: float = \
#         data.data_frame["<M**1>"].values[-1] / data.lookup_tables.number_sites
#     susceptibility: float = data.data_frame["X"].values[-1]
#     heat_capacity: float = data.data_frame["C"].values[-1]
#     binder_m: float = data.data_frame["Binder_M"].values[-1]
#
#     print(f"Average susceptibility = {susceptibility}")
#     print(f"Average heat capacity = {heat_capacity}")
#     print(f"Binder parameter for M = {binder_m}")
#
#     assert energy >= -max_abs_energy and energy <= max_abs_energy
#     assert magnetization >= 0 and magnetization <= 1.0


# @pytest.mark.parametrize(
#     "r, max_abs_energy, interaction_ij",
#     [
#         (1.2, 2.0, (1.0, 1.0)),
#         (1.9, 4.0, (-1.0, 1.0, 1.0, -1.0)),
#     ],
# )
# def test_2d_square_ising_antiferromagnet_simulation(
#     r: float,
#     max_abs_energy: float,
#     interaction_ij: Union[Tuple[float, float], Tuple[float, float, float, float]],
#     two_dimensional_square_lattice: pmg.Structure,
#     simulation_parameters: SimulationParameters,
# ) -> None:
#     lattice: Lattice = Lattice(structure=two_dimensional_square_lattice, r=r)
#
#     lattice.set_sublattice_pair_interactions(
#         interaction_df=lattice.sublattice_pairs_data_frame.assign(J_ij=interaction_ij)
#     )
#
#     data: SimulationData = spyns.run.simulation(
#         lattice=lattice,
#         parameters=simulation_parameters,
#     )
#
#     spyns.statistics.compute_ising_afm_order_parameter(
#         trace_df=data.data_frame,
#         order_parameter_name="AFM",
#         sublattices1=[0],
#         sublattices2=[1],
#         number_sites=data.lookup_tables.number_sites,
#     )
#
#     energy: float = \
#         data.data_frame["<E**1>"].values[-1] / data.lookup_tables.number_sites
#     magnetization: float = \
#         data.data_frame["<M**1>"].values[-1] / data.lookup_tables.number_sites
#     susceptibility: float = data.data_frame["X"].values[-1]
#     heat_capacity: float = data.data_frame["C"].values[-1]
#     binder_m: float = data.data_frame["Binder_M"].values[-1]
#     antiferromagnetization: float = data.data_frame["AFM"].mean()
#
#     print(f"Average susceptibility = {susceptibility}")
#     print(f"Average heat capacity = {heat_capacity}")
#     print(f"Binder parameter for M = {binder_m}")
#     print(f"Average antiferromagnetization = {antiferromagnetization}")
#
#     assert energy >= -max_abs_energy and energy <= max_abs_energy
#     assert antiferromagnetization <= 1.0
#     assert antiferromagnetization - magnetization > 0.1


# @pytest.mark.parametrize(
#     "r, max_abs_energy, interaction_ij",
#     [
#         (1.2, 3.0, (-1.0, -1.0)),
#         (1.5, 9.0, (-1.0, -1.0, -1.0, -1.0)),
#     ],
# )
# def test_sc_heisenberg_ferromagnet_simulation(
#     r: float,
#     max_abs_energy: float,
#     interaction_ij: Union[Tuple[float, float], Tuple[float, float, float, float]],
#     cubic_lattice: pmg.Structure,
#     simulation_parameters_heisenberg: SimulationParameters,
# ) -> None:
#     lattice: Lattice = Lattice(structure=cubic_lattice, r=r)
#
#     lattice.set_sublattice_pair_interactions(
#         interaction_df=lattice.sublattice_pairs_data_frame.assign(J_ij=interaction_ij)
#     )
#
#     data: SimulationData = spyns.run.simulation(
#         lattice=lattice,
#         parameters=simulation_parameters_heisenberg,
#     )
#
#     energy: float = \
#         data.data_frame["<E**1>"].values[-1] / data.lookup_tables.number_sites
#     magnetization: float = \
#         data.data_frame["<M**1>"].values[-1] / data.lookup_tables.number_sites
#     susceptibility: float = data.data_frame["X"].values[-1]
#     heat_capacity: float = data.data_frame["C"].values[-1]
#     binder_m: float = data.data_frame["Binder_M"].values[-1]
#
#     print(f"Average susceptibility = {susceptibility}")
#     print(f"Average heat capacity = {heat_capacity}")
#     print(f"Binder parameter for M = {binder_m}")
#
#     assert energy >= -max_abs_energy and energy <= max_abs_energy
#     assert magnetization >= -1.0 and magnetization <= 1.0


# @pytest.mark.parametrize(
#     "r, max_abs_energy, interaction_ij",
#     [
#         (0.9, 4.0, (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)),
#     ],
# )
# def test_bcc_heisenberg_ferromagnet_simulation(
#     r: float,
#     max_abs_energy: float,
#     interaction_ij: Union[Tuple[float, float], Tuple[float, float, float, float]],
#     bcc_lattice: pmg.Structure,
#     simulation_parameters_heisenberg: SimulationParameters,
# ) -> None:
#     lattice: Lattice = Lattice(structure=bcc_lattice, r=r)
#
#     lattice.set_sublattice_pair_interactions(
#         interaction_df=lattice.sublattice_pairs_data_frame.assign(J_ij=interaction_ij)
#     )
#
#     data: SimulationData = spyns.run.simulation(
#         lattice=lattice,
#         parameters=simulation_parameters_heisenberg,
#     )
#
#     energy: float = \
#         data.data_frame["<E**1>"].values[-1] / data.lookup_tables.number_sites
#     magnetization: float = \
#         data.data_frame["<M**1>"].values[-1] / data.lookup_tables.number_sites
#     susceptibility: float = data.data_frame["X"].values[-1]
#     heat_capacity: float = data.data_frame["C"].values[-1]
#     binder_m: float = data.data_frame["Binder_M"].values[-1]
#
#     print(f"Average susceptibility = {susceptibility}")
#     print(f"Average heat capacity = {heat_capacity}")
#     print(f"Binder parameter for M = {binder_m}")
#
#     assert energy >= -max_abs_energy and energy <= max_abs_energy
#     assert magnetization >= -1.0 and magnetization <= 1.0


# @pytest.mark.parametrize(
#     "r",
#     [
#         1.2,
#         1.9,
#     ],
# )
# def test_2d_square_voter_model_simulation(
#     r: float,
#     two_dimensional_square_lattice: pmg.Structure,
#     simulation_parameters_voter: SimulationParameters,
# ) -> None:
#     lattice: Lattice = Lattice(structure=two_dimensional_square_lattice, r=r)
#
#     data: SimulationData = spyns.run.simulation(
#         lattice=lattice,
#         parameters=simulation_parameters_voter,
#     )
#
#     magnetization: float = \
#         data.data_frame["<M**1>"].values[-1] / data.lookup_tables.number_sites
#
#     assert magnetization >= 0 and magnetization <= 1.0


@pytest.mark.parametrize(
    "r, max_abs_energy, interaction_ij",
    [(1.2, 3.0, (-1.0, -1.0)), (1.5, 9.0, (-1.0, -1.0, -1.0, -1.0))],
)
def test_sc_heisenberg_cython_ferromagnet_simulation(
    r: float,
    max_abs_energy: float,
    interaction_ij: Union[Tuple[float, float], Tuple[float, float, float, float]],
    cubic_lattice: pmg.Structure,
    simulation_parameters_heisenberg_cython: SimulationParameters,
) -> None:
    lattice: Lattice = Lattice(structure=cubic_lattice, r=r)

    lattice.set_sublattice_pair_interactions(
        interaction_df=lattice.sublattice_pairs_data_frame.assign(J_ij=interaction_ij)
    )

    data: SimulationData = spyns.run.simulation(
        lattice=lattice, parameters=simulation_parameters_heisenberg_cython
    )

    energy: float = data.container.data_frame["<E**1>"].values[
        -1
    ] / data.container.lookup_tables.number_sites
    magnetization: float = data.container.data_frame["<M**1>"].values[
        -1
    ] / data.container.lookup_tables.number_sites
    susceptibility: float = data.container.data_frame["X"].values[-1]
    heat_capacity: float = data.container.data_frame["C"].values[-1]
    binder_m: float = data.container.data_frame["Binder_M"].values[-1]

    print(f"Average susceptibility = {susceptibility}")
    print(f"Average heat capacity = {heat_capacity}")
    print(f"Binder parameter for M = {binder_m}")

    assert energy >= -max_abs_energy and energy <= max_abs_energy
    assert magnetization >= -1.0 and magnetization <= 1.0
