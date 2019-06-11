from spyns.random_numbers.distribution cimport RandomNumberGenerator


cdef class SimulationHeisenbergData_t:

    def __cinit__(
        self,
        object data,
        RandomNumberGenerator random_number_generator,
    ):
        self.parameters = SimulationParameters_t()
        self.lookup_tables = LookupTables_t()
        self.state = HeisenbergState_t()
        self.trace = SimulationTrace_t()
        self.estimators = Estimators_t()

        self._data = data

        self.random_number_generator = random_number_generator

        self.parameters.sample_interval = self._data.parameters.sample_interval
        self.parameters.temperature = self._data.parameters.temperature
        self.parameters.sweeps = self._data.parameters.sweeps
        self.parameters.equilibration_sweeps = self._data.parameters.equilibration_sweeps

        self.lookup_tables.number_sites = self._data.lookup_tables.number_sites
        self.lookup_tables.number_sublattices = self._data.lookup_tables.number_sublattices
        self.lookup_tables.sublattice_table = self._data.lookup_tables.sublattice_table
        self.lookup_tables.neighbors_table = self._data.lookup_tables.neighbors_table
        self.lookup_tables.neighbors_count = self._data.lookup_tables.neighbors_count
        self.lookup_tables.neighbors_lookup_index = self._data.lookup_tables.neighbors_lookup_index
        self.lookup_tables.interaction_parameters_table = self._data.lookup_tables.interaction_parameters_table

        self.state.x = self._data.state.x
        self.state.y = self._data.state.y
        self.state.z = self._data.state.z

        self.trace.sweep = self._data.trace.sweep
        self.trace.energy = self._data.trace.energy
        self.trace.spin_vector = self._data.trace.spin_vector
        self.trace.magnetization = self._data.trace.magnetization

        self.estimators.number_samples = self._data.estimators.number_samples
        self.estimators.energy = self._data.estimators.energy
        self.estimators.spin_vector = self._data.estimators.spin_vector
        self.estimators.magnetization = self._data.estimators.magnetization

    @property
    def container(self):
        return self._data
