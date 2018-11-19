# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Tuple

import pymatgen as pmg
from pymatgen.transformations.standard_transformations import SupercellTransformation

import spyns
from spyns.data import StructureParameters

ScalingMatrix = Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]


def from_parameters(structure_parameters: StructureParameters) -> pmg.Structure:
    """Generates a pymatgen ``Structure`` object using a material's structural
    parameters.

    :param structure_parameters: A ``StructureParameters`` tuple that specifies a
        material's crystal structure.
    :return: A pymatgen ``Structure`` object.
    """
    cell_lattice: pmg.Lattice = pmg.Lattice.from_lengths_and_angles(
        abc=structure_parameters.abc,
        ang=structure_parameters.ang,
    )

    cell_structure: pmg.Structure = pmg.Structure.from_spacegroup(
        sg=structure_parameters.spacegroup,
        lattice=cell_lattice,
        species=structure_parameters.species,
        coords=structure_parameters.coordinates,
    )

    return cell_structure


def from_file(structure_file: spyns.data.StructureFile) -> pmg.Structure:
    """Generates a pymatgen ``Structure`` object from a supported file format.

    :param structure_file: Path to the structure file. Supported formats include CIF,
        POSCAR/CONTCAR, CHGCAR, LOCPOT, vasprun.xml, CSSR, Netcdf, and pymatgen's
        serialized structures.
    :return: A pymatgen ``Structure`` object.
    """
    cell_structure: pmg.Structure = pmg.Structure.from_file(
        filename=structure_file.path,
        primitive=False,
        sort=False,
        merge_tol=0.01,
    )

    return cell_structure


def label_subspecies(
    structure: pmg.Structure,
    subspecies_labels: Dict[int, str] = {},
) -> pmg.Structure:
    """Groups sites together into sublattices using subspecies labels.

    :param structure: A pymatgen ``Structure`` object.
    :param subspecies_labels: Key-value pairs used to label sites. The key is an integer
        representing the site index, the value is the subspecie label (default {}).
    :return: A pymatgen ``Structure`` object.
    """
    cell_structure: pmg.Structure = structure.copy()

    site_properties_subspecies: List[str] = get_subspecies_labels(
        cell_structure=cell_structure,
        subspecies_labels=subspecies_labels,
    )

    cell_structure.add_site_property(
        property_name="subspecie",
        values=site_properties_subspecies,
    )

    return cell_structure


def get_subspecies_labels(
    cell_structure: pmg.Structure,
    subspecies_labels: Dict[int, str],
) -> List[Optional[str]]:
    """Builds a list of subspecies labels using the provided dictionary. Site indices not
    found as a key in the dictionary are labeled with the atomic species name.

    :param cell_structure: A pymatgen ``Structure`` object.
    :param subspecies_labels: Key-value pairs used to label sites. The key is an integer
        representing the site index, the value is the subspecie label.
    :return: A list of subspecies labels.
    """
    site_properties_subspecies: List[str] = []

    for site_index, site in enumerate(cell_structure):
        specie_name: str = site.specie.name

        if site_index in subspecies_labels:
            site_properties_subspecies.append(
                f"{specie_name}{subspecies_labels[site_index]}"
            )

        else:
            site_properties_subspecies.append(f"{specie_name}")

    return site_properties_subspecies


def make_supercell(
    cell_structure: pmg.Structure,
    scaling_matrix: Optional[ScalingMatrix] = None,
    scaling_factors: Optional[Tuple[int, int, int]] = None,
) -> pmg.Structure:
    """Transforms a pymatgen ``Structure`` object into a supercell according to the
    scaling parameters.

    :param cell_structure: A pymatgen ``Structure`` object.
    :param scaling_matrix: A matrix of transforming the lattice vectors. Has to be all
        integers. e.g., [[2,1,0],[0,3,0],[0,0,1]] generates a new structure with lattice
        vectors a" = 2a + b, b" = 3b, c" = c where a, b, and c are the lattice vectors
        of the original structure.
    :param scaling_factors: A tuple of three numbers used to scale each lattice vector.
        Same as: ``scaling_matrix=[[scale_a, 0, 0], [0, scale_b, 0], [0, 0, scale_c]]``

    :return: A pymatgen ``Structure`` object.
    """
    if scaling_matrix is not None:
        cell_transformation: SupercellTransformation = SupercellTransformation(
            scaling_matrix=scaling_matrix
        )
        return cell_transformation.apply_transformation(cell_structure)

    elif scaling_factors is not None:
        cell_transformation = SupercellTransformation.from_scaling_factors(
            scale_a=scaling_factors[0],
            scale_b=scaling_factors[1],
            scale_c=scaling_factors[2],
        )
        return cell_transformation.apply_transformation(cell_structure)

    else:
        return cell_structure


def add_subspecie_labels_if_missing(
    cell_structure: pmg.Structure,
    subspecies_labels: Dict[int, str] = {},
) -> pmg.Structure:
    """Makes a copy of ``cell_structure`` and then checks if ``cell_structure`` has
    the subspecie site property. If it does, then return the copy as-is, otherwise
    label each site of the copy using the site's atomic specie name and then return
    it.

    :param cell_structure: A pymatgen ``Structure`` object.
    :param subspecies_labels: Key-value pairs used to label sites. The key is an integer
        representing the site index, the value is the subspecie label (Default: {}).
    :return: An exact copy of the input ``cell_structure`` object with subspecie
        labels added, if missing.
    """
    structure = cell_structure.copy()

    if "subspecie" not in structure.site_properties:
        structure = label_subspecies(
            structure=structure,
            subspecies_labels=subspecies_labels,
        )

    return structure
