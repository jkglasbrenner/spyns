# -*- coding: utf-8 -*-

from typing import List

from isingmodel.data import SimulationData


def update_running_average(
    data: SimulationData,
    state: str,
    estimator: str,
    power: int,
    number_samples: int,
) -> float:
    """Update estimator mean using simple moving average algorithm.

    :param data: Data container for the simulation.
    :param estimator: Estimator name within estimators container.
    :param power: Estimator moment.
    :param number_samples: Number of samples currently in mean.
    :return: Updated estimator mean.
    """
    state_value: float = getattr(data.estimators, state)
    estimator_mean: float = getattr(data.estimators, estimator)
    return (state_value**power - estimator_mean) / (number_samples + 1)


def update_estimators(data: SimulationData, number_samples: int) -> None:
    """Update estimator running averages with new sample.

    :param data: Data container for the simulation.
    :param number_samples: Number of samples currently in mean.
    """
    estimators: List[str] = ([
        f"{parameter}_{moment}_moment" for parameter in ["energy", "magnetization"]
        for moment in ["1st", "2nd", "3rd", "4th"]
    ])
    states: List[str] = 4 * ["energy"] + 4 * ["magnetization"]
    powers: List[int] = 2 * list(range(1, 5))

    for estimator, state, power in zip(estimators, states, powers):
        estimator_mean: float = getattr(data.estimators, estimator)
        setattr(
            data.estimators,
            estimator,
            estimator_mean + update_running_average(
                data=data,
                estimator=estimator,
                state=state,
                power=power,
                number_samples=number_samples,
            ),
        )
