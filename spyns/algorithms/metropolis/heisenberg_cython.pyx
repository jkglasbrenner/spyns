from spyns.random_numbers.distribution cimport RandomNumberGenerator
from spyns.data_cython cimport SimulationHeisenbergData_t
from spyns.model.heisenberg_cython cimport \
    TrialFlip_t, keep_flip_and_update_state, flip
from base_cython cimport pick_site, accept_or_reject

import numpy as np

from spyns.data import dump_state_snapshot_to_disk
from spyns.statistics import update_trace


cdef void step(SimulationHeisenbergData_t data):
    """Update system state of the Heisenberg model using the Metropolis algorithm.

    :param data: Data container for the simulation.
    """
    cdef long site_index = pick_site(
        data=data,
    )
    cdef TrialFlip_t trial_flip = flip(
        site_index=site_index,
        data=data,
    )
    cdef bint accept_state = accept_or_reject(
        temperature=data.parameters.temperature,
        energy_difference=trial_flip.energy_difference,
        data=data,
    )
    if accept_state:
        keep_flip_and_update_state(
            data=data,
            site_index=site_index,
            trial_flip=trial_flip,
        )


cdef void sweep(SimulationHeisenbergData_t data, long sweep_index, bint equilibration_run):
    """Sweep the Heisenberg lattice and take a sample if required.

    :param data: Data container for the simulation.
    :param sweep_index: Sweep index for the simulation.
    :param equilibration_run: Whether or not the current sweep is part of equilibration
        run.
    """
    cdef long _
    
    cdef long number_sites = data.lookup_tables.number_sites

    for _ in range(number_sites):
        step(data=data)

    if sweep_index % data.parameters.sample_interval == 0 and not equilibration_run:
        data.estimators.magnetization[0] = np.linalg.norm(
            data.container.estimators.spin_vector.sum(axis=0)
        )
        data.estimators.number_samples[0] += 1
        update_trace(data=data.container, sweep_index=sweep_index)

        if data.container.parameters.snapshot_filepath:
            dump_state_snapshot_to_disk(
                data=data.container,
                sweep_index=sweep_index + 1,
            )


cpdef void run_sweeps(SimulationHeisenbergData_t data, bint equilibration_run):
    cdef long sweep_index
    cdef long sweeps
    
    if equilibration_run:
        sweeps = data.parameters.equilibration_sweeps
    
    else:
        sweeps = data.parameters.sweeps

    for sweep_index in range(sweeps):
        sweep(
            data=data,
            sweep_index=sweep_index,
            equilibration_run=equilibration_run,
        )
