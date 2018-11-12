# -*- coding: utf-8 -*-

from typing import Iterable, List, Union

import numpy as np

from spyns.data import SimulationData


def update_running_average(
    data: SimulationData,
    estimator_sample_name: str,
    estimator_mean_name: str,
    power: int,
) -> float:
    """Update estimator mean using simple moving average algorithm.

    :param data: Data container for the simulation.
    :param state: Sampled value of estimator.
    :param estimator_sample_name: Name for sampled value of estimator within estimators
        container.
    :param estimator_mean_name: Name for mean value of estimator within estimators
        container.
    :param power: Estimator moment.
    :return: Updated estimator mean.
    """
    estimator_sample: Union[float, np.ndarray] = getattr(
        data.estimators, estimator_sample_name
    )
    estimator_mean: Union[float, np.ndarray] = getattr(
        data.estimators, estimator_mean_name
    )
    return ((estimator_sample**power - estimator_mean) /
            (data.estimators.number_samples + 1))


def update_estimators(data: SimulationData) -> None:
    """Update estimator running averages with new sample.

    :param data: Data container for the simulation.
    :param number_samples: Number of samples currently in mean.
    """
    estimator_means_list: List[str] = ([
        f"{parameter}_{moment}_moment" for parameter in ["energy", "magnetization"]
        for moment in ["1st", "2nd", "3rd", "4th"]
    ])
    estimator_samples_list: List[str] = 4 * ["energy"] + 4 * ["magnetization"]
    powers_list: List[int] = 2 * list(range(1, 5))
    estimators_zip: Iterable = \
        zip(estimator_means_list, estimator_samples_list, powers_list)

    for estimator_mean_name, estimator_sample_name, power in estimators_zip:
        estimator_mean: Union[float, np.ndarray] = getattr(
            data.estimators,
            estimator_mean_name,
        )
        setattr(
            data.estimators,
            estimator_mean_name,
            estimator_mean + update_running_average(
                data=data,
                estimator_sample_name=estimator_sample_name,
                estimator_mean_name=estimator_mean_name,
                power=power,
            ),
        )

    data.estimators.number_samples += 1


def update_trace(
    data: SimulationData,
    sweep_index: int,
) -> None:
    """Add estimators sample to the simulation trace.

    :param data: Data container for the simulation.
    :param sweep_index: Sweep index for the simulation.
    """
    number_sites: int = data.lookup_tables.number_sites

    data.trace.energy_1st_moment[sweep_index] = \
        data.estimators.energy_1st_moment / number_sites
    data.trace.energy_2nd_moment[sweep_index] = \
        data.estimators.energy_2nd_moment / number_sites
    data.trace.energy_3rd_moment[sweep_index] = \
        data.estimators.energy_3rd_moment / number_sites
    data.trace.energy_4th_moment[sweep_index] = \
        data.estimators.energy_4th_moment / number_sites
    data.trace.magnetization_1st_moment[sweep_index] = \
        data.estimators.magnetization_1st_moment / number_sites
    data.trace.magnetization_2nd_moment[sweep_index] = \
        data.estimators.magnetization_2nd_moment / number_sites
    data.trace.magnetization_3rd_moment[sweep_index] = \
        data.estimators.magnetization_3rd_moment / number_sites
    data.trace.magnetization_4th_moment[sweep_index] = \
        data.estimators.magnetization_4th_moment / number_sites
