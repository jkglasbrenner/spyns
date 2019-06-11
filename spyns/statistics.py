# -*- coding: utf-8 -*-

from typing import List

import numpy as np
import pandas as pd

from spyns.data import SimulationData


def compute_running_average(trace_df: pd.DataFrame, estimator_name: str) -> None:
    """Compute running average for a series of estimator samples.

    :param trace_df: Trace history data frame of estimator samples.
    :param estimator_name: Compute running average of this estimator. 
    """
    trace_df[f"<{estimator_name}>"] = trace_df[estimator_name].expanding().mean()


def compute_estimator_moments(
    trace_df: pd.DataFrame, estimator_name: str, max_power: int = 4
) -> None:
    for power in range(1, max_power + 1):
        trace_df[f"{estimator_name}**{power}"] = trace_df[estimator_name] ** power


def compute_estimator_fluctuations(
    trace_df: pd.DataFrame,
    fluctuation_name: str,
    estimator_name: str,
    number_sites: int,
    coefficient: float,
) -> None:
    trace_df[fluctuation_name] = (
        coefficient
        * (trace_df[f"<{estimator_name}**2>"] - trace_df[f"<{estimator_name}**1>"] ** 2)
        / number_sites
    )


def compute_binder_parameter(trace_df: pd.DataFrame, estimator_name: str) -> None:
    trace_df[f"Binder_{estimator_name}"] = 1 - (1 / 3) * (
        trace_df[f"<{estimator_name}**4>"] / trace_df[f"<{estimator_name}**2>"] ** 2
    )


def compute_ising_afm_order_parameter(
    trace_df: pd.DataFrame,
    order_parameter_name: str,
    sublattices1: List[str],
    sublattices2: List[str],
    number_sites: int,
) -> None:
    spin_vector_df = trace_df.copy()
    spin_vector_df["spin1"] = 0.0
    spin_vector_df["spin2"] = 0.0

    for sublattice in sublattices1:
        spin_vector_df["spin1"] += spin_vector_df[f"S{sublattice}"]

    for sublattice in sublattices2:
        spin_vector_df["spin2"] += spin_vector_df[f"S{sublattice}"]

    trace_df[order_parameter_name] = (
        np.abs(spin_vector_df["spin2"] - spin_vector_df["spin1"]) / number_sites
    )


def compute_heisenberg_afm_order_parameter(
    trace_df: pd.DataFrame,
    order_parameter_name: str,
    sublattices1: List[str],
    sublattices2: List[str],
    number_sites: int,
) -> None:
    spin_vector_df = trace_df.copy()
    spin_vector_df["spin1x"] = 0.0
    spin_vector_df["spin1y"] = 0.0
    spin_vector_df["spin1z"] = 0.0
    spin_vector_df["spin2x"] = 0.0
    spin_vector_df["spin2y"] = 0.0
    spin_vector_df["spin2z"] = 0.0

    for sublattice in sublattices1:
        spin_vector_df["spin1x"] += spin_vector_df[f"S{sublattice}x"]
        spin_vector_df["spin1y"] += spin_vector_df[f"S{sublattice}y"]
        spin_vector_df["spin1z"] += spin_vector_df[f"S{sublattice}z"]

    for sublattice in sublattices2:
        spin_vector_df["spin2x"] += spin_vector_df[f"S{sublattice}x"]
        spin_vector_df["spin2y"] += spin_vector_df[f"S{sublattice}y"]
        spin_vector_df["spin2z"] += spin_vector_df[f"S{sublattice}z"]

    trace_df[order_parameter_name] = (
        np.linalg.norm(
            spin_vector_df[["spin1x", "spin1y", "spin1z"]].values
            - spin_vector_df[["spin2x", "spin2y", "spin2z"]].values,
            axis=1,
        )
        / number_sites
    )


def update_trace(data: SimulationData, sweep_index: int) -> None:
    """Save estimators samples in the simulation trace.

    :param data: Data container for the simulation.
    :param sweep_index: Sweep index for the simulation.
    """
    data.trace.energy[sweep_index] = data.estimators.energy[0]
    data.trace.spin_vector[sweep_index] = data.estimators.spin_vector
    data.trace.magnetization[sweep_index] = data.estimators.magnetization[0]
