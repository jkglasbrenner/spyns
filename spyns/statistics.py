# -*- coding: utf-8 -*-

from typing import Iterable, List

from spyns.data import SimulationData


def update_running_average(
    data: SimulationData,
    estimator_sample_name: str,
    estimator_mean_name: str,
    power: int,
    number_samples: int,
) -> float:
    """Update estimator mean using simple moving average algorithm.

    :param data: Data container for the simulation.
    :param state: Sampled value of estimator.
    :param estimator_sample_name: Name for sampled value of estimator within estimators
        container.
    :param estimator_mean_name: Name for mean value of estimator within estimators
        container.
    :param power: Estimator moment.
    :param number_samples: Number of samples currently in mean.
    :return: Updated estimator mean.
    """
    estimator_sample: float = getattr(data.estimators, estimator_sample_name)
    estimator_mean: float = getattr(data.estimators, estimator_mean_name)
    return (estimator_sample**power - estimator_mean) / (number_samples + 1)


def update_estimators(data: SimulationData, number_samples: int) -> None:
    """Update estimator running averages with new sample.

    :param data: Data container for the simulation.
    :param number_samples: Number of samples currently in mean.
    """
    estimator_means_list: List[str] = ([
        f"{parameter}_{moment}_moment" for parameter in [
            "energy",
            "magnetization",
            "magnetization_even_sites",
            "magnetization_odd_sites",
        ] for moment in ["1st", "2nd", "3rd", "4th"]
    ])
    estimator_samples_list: List[str] = (
        4 * ["energy"] + 4 * ["magnetization"] + 4 * ["magnetization_even_sites"] +
        4 * ["magnetization_odd_sites"]
    )
    powers_list: List[int] = 4 * list(range(1, 5))
    estimators_zip: Iterable = \
        zip(estimator_means_list, estimator_samples_list, powers_list)

    for estimator_mean_name, estimator_sample_name, power in estimators_zip:
        estimator_mean: float = getattr(data.estimators, estimator_mean_name)
        setattr(
            data.estimators,
            estimator_mean_name,
            estimator_mean + update_running_average(
                data=data,
                estimator_sample_name=estimator_sample_name,
                estimator_mean_name=estimator_mean_name,
                power=power,
                number_samples=number_samples,
            ),
        )
