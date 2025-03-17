from datetime import datetime
from decimal import Decimal, getcontext
from functools import partial
import json
import logging
import math
from math import pow
from pathlib import Path
import uuid

import numpy as np
from numpy import array, exp, log
from numpy.random import seed
import pandas as pd
from scipy.stats import ks_2samp
from sympy import exp as sp_exp
from sympy import pprint, symbols
from sympy.utilities.autowrap import ufuncify

seed(2)

SECS_PER_HOUR = 3600

logger = logging.getLogger(__name__)
handler = logging.FileHandler('logging.log')
formatter = logging.Formatter(
    '%(asctime)s %(lineno)-4s %(levelname)-8s %(funcName)20s() %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


class HVACSystemsSimulation:
    MAX_HOURS = 30

    def __init__(self, rng_seed: int, secs_per_time_step: int):
        assert secs_per_time_step in [1, 60]
        self.rng = np.random.default_rng(rng_seed)
        self.callaway = self.CallawayParams(secs_per_time_step)
        self.train_seconds = 3600 * self.callaway.HOURS_TRAIN
        self.test_seconds = 3600 * self.callaway.HOURS_TEST
        self.std_r = self._config_stds(self.callaway.MEAN_R)
        self.std_c = self._config_stds(self.callaway.MEAN_C)
        self.std_energy_transfer = self._config_stds(
            self.callaway.MEAN_ENERGY_TRANSFER)
        # Callaway indicates a range of 40 degrees C without specifying minimum and maximum.
        # 40 degrees C is approximately 104 degrees F.
        self.thermostat_min_c = 0.0
        self.thermostat_max_c = 40.0
        self.thermostat_bits = 14
        self.thermostat_precision = 2
        cools_and_building_params = self.init_lognormal_building_params()
        self.cools = cools_and_building_params['cools']
        self.init_building_params = cools_and_building_params['init_building_params']
        self.sim_building_params = cools_and_building_params['sim_building_params']

    class CallawayParams:
        NUM_DEVS = 60000
        DESCRIPTION = 'Callaway Table 1'
        MEAN_R = 2.0
        MEAN_C = 10.0
        MEAN_ENERGY_TRANSFER = 14.0
        LOAD_EFFICIENCY = 2.5
        AMBIENT = 32.0
        SETPOINT = 20.1
        DEADBAND = 0.5
        STARTING_CONTROL = 20.1
        HOURS_TRAIN = 5
        HOURS_TEST = 5
        STD_DEV_OF_LOGNORMAL_PARAMS = 0.2
        NOISE_DEGREES_STD_DEV_BY_SECS_PER_TIME_STEP = {1: 0.01,
                                                       60: 0.08}

        def __init__(self, h):
            assert isinstance(h, int)
            self.H = h
            self.init_noise_degrees_c_std_dev = self.init_noise_degrees_c_std_dev()
            self.noise_degrees_c_std_dev = self.noise_degrees_c_std_dev_by_secs_per_time_step(h)

        def init_noise_degrees_c_std_dev(self):
            return self.NOISE_DEGREES_STD_DEV_BY_SECS_PER_TIME_STEP[1]
        
        def noise_degrees_c_std_dev_by_secs_per_time_step(self, h):
            return self.NOISE_DEGREES_STD_DEV_BY_SECS_PER_TIME_STEP[h]

    def _config_stds(self, mean):
        # Per Table 1, std dev. of lognormal distributions of R, C and P
        # as fractions of the parameters' mean values are 0.2
        std_value = Decimal(
            self.callaway.STD_DEV_OF_LOGNORMAL_PARAMS) * Decimal(mean)
        getcontext().prec=2
        std_value = Decimal(std_value)
        std_as_float = float(std_value)
        return std_as_float
    
    def init_lognormal_building_params(self, known_energy_transfer=None):
        lognormal_dist = self.create_lognormal_distribution

        # Non-deterministic
        parameter = 'heterogeneous thermal resistance across population'
        cp = self.callaway
        rs = lognormal_dist(mean=cp.MEAN_R, std=self.std_r,
                            parameter=parameter)
        # Non-deterministic
        parameter = 'heterogeneous thermal capacitance across population'
        cs = lognormal_dist(mean=cp.MEAN_C, std=self.std_c,
                            parameter=parameter)

        if self.std_energy_transfer is not None:
            # Heterogeneous, non-deterministic
            assert (cp.MEAN_ENERGY_TRANSFER is not None
                    and known_energy_transfer is None)
            parameter = 'heterogeneous energy transfer rate (cooling) of HVAC across population'
            cooling_ps = lognormal_dist(mean=cp.MEAN_ENERGY_TRANSFER,
                                        std=self.std_energy_transfer, parameter=parameter)
        else:
            assert (known_energy_transfer is not None
                    and cp.MEAN_ENERGY_TRANSFER is None)
            cooling_ps = known_energy_transfer

        # rs corresponds to R_i. Non-deterministic
        # cooling_ps corresponds to P_i. Non-deterministic P = 1 for cooling loads
        # Together, they correspond to Eq. 1 in Callaway
        cools = np.multiply(rs, cooling_ps)  # Product is non-deterministic

        init_building_params = self.building_thermal_params(rs=rs, cs=cs,
                                                                 secs_per_time_step=1)

        sim_secs_per_time_step = self.callaway.H
        sim_building_params = self.building_thermal_params(rs=rs, cs=cs,
                                                            secs_per_time_step=sim_secs_per_time_step,
                                                            display_expr=False)
        return {'cools': cools,
                'init_building_params': init_building_params,
                'sim_building_params': sim_building_params}
        

    def init_simulation(self,
                        known_energy_transfer=None,
                        uniform_temp_dist=True, description=None):
        init_simulation_start = datetime.now()
        logger.info(description)

        hvac_cycle_data = self.get_durations_and_summarize(
            self.cools, display_expr=True)

        num_devs_finished = f"{np.sum(hvac_cycle_data['on_and_off_finished'])}"
        print(f"{num_devs_finished} devs had combined ON and OFF cycles under 24 hours")

        init_simulation_duration = datetime.now() - init_simulation_start
        logger.debug(f'init_simulation ended after {init_simulation_duration}')
        return hvac_cycle_data

    def create_lognormal_distribution(self, mean=None, std=None, parameter=None):
        # See https://blogs.sas.com/content/iml/2014/06/04/simulate-lognormal-data-with-specified-mean-and-variance.html
        # mean and std are mean and variance of lognormally distributed variable Y
        # mu and sigma are the parameters of log(y)
        m = mean
        m_squared = math.pow(m, 2)
        v = math.pow(std, 2)
        mu = log(m_squared / np.sqrt(v + m_squared))
        sigma = np.sqrt(log(1 + v / m_squared))
        # Non-deterministic step
        # Num is provided as first argument for partial
        norm = self.rng.normal(loc=mu, scale=sigma,
                               size=self.callaway.NUM_DEVS)
        distribution = exp(norm)
        logger.info('Created lognormal distribution for ')
        logger.info(f' {parameter}.')
        return distribution

    def building_thermal_params(self, rs=None, cs=None, secs_per_time_step=60,
                                display_expr=True):
        # Declare symbols
        # See https://faculty.bard.edu/~belk/math213/ExponentialDecay.pdf
        h = symbols('h')  # -h is decay constant. k in Eq. 44 equals -h.
        rsym = symbols('rs')
        csym = symbols('cs')
        # Sympy function exp to create a symbolic expression,
        # "Parameters in Eq. (1) are as follows: a governs the thermal
        # characteristics of the the thermal mass and is defined as
        # a = exp(-h/CR) [10], where C is the thermal capacitance
        # (units of kWh/degrees C), and R is the thermal resistance
        # (degrees C/kW)."
        # Source:
        expr = sp_exp(np.divide(-h, np.multiply(csym, rsym)))
        if display_expr:
            # Display is associated with sympy.interactive import printing
            # and printing.init_printing(use_latex=True)
            print("Expression for a (thermal characteristics of thermal mass) in Eq. 1")
            pprint(expr)
            print()
            logger.info(f"Distribution: {expr}")
        # The expression is transformed into a function that takes vectorized
        # arguments in the same order in which symbols appear in the expression
        f = ufuncify((h, csym, rsym), expr)
        # Here the actual arguments are passed into the function
        # The time constant for rs and cs is 60 * 60 seconds
        # Below is equivalent to exp(-h/(60 * 60 * cs * rs))
        # The 60 * 60 indicates the number of seconds per hour.
        # When the units for C and R are multiplied, the overall units
        # become seconds per time step per hour.
        val_of_a_in_eq_1 = f(secs_per_time_step / (60 * 60), cs, rs)
        return val_of_a_in_eq_1

    def init_times_and_temps(self, building_params=None,
                             off_durations=None, on_durations=None,
                             on_and_off_finished=None,
                             cools=None, switch_on_temps=None, switch_off_temps=None):
        init_times, on_off_ones_zeros = self.init_times_in_cycles(self.rng, off_durs=off_durations,
                                                                  on_durs=on_durations)

        logger.debug(
            f"Devices with combined ON and OFF cycle times under 24 hours: {on_and_off_finished}")

        init_temps = self.init_temps_in_cycles(max_temps=switch_on_temps,
                                               min_temps=switch_off_temps,
                                               init_times=init_times,
                                               on_off_bools=on_off_ones_zeros,
                                               building_params=building_params,
                                               cools=cools, ambient=self.callaway.AMBIENT)

        on_states = np.zeros(on_off_ones_zeros.size, dtype=bool)
        on_states[on_off_ones_zeros] = 1
        switched_off = np.logical_and(on_off_ones_zeros,
                                      init_temps <= switch_off_temps)
        on_states[switched_off] = 0
        switched_on = np.logical_and(~on_off_ones_zeros,
                                     init_temps >= switch_on_temps)
        on_states[switched_on] = 1

        return init_temps, on_states, switch_on_temps, switch_off_temps

    def init_times_in_cycles(self, rng, off_durs=None, on_durs=None):
        total_times = np.add(off_durs, on_durs)
        num_devs = total_times.size

        random_times = np.multiply(rng.uniform(size=num_devs), total_times)
        on_off_bools = random_times < on_durs

        init_times = np.zeros(num_devs, dtype=int)

        def round_as_int(x): return np.round(x).astype('int')

        random_times_in_on_cycle = np.multiply(
            rng.uniform(size=num_devs), on_durs)
        random_times_in_off_cycle = np.multiply(
            rng.uniform(size=num_devs), off_durs)
        init_times[on_off_bools] = round_as_int(
            random_times_in_on_cycle[on_off_bools])
        init_times[~on_off_bools] = round_as_int(
            (random_times_in_off_cycle[~on_off_bools]))
        on_off_ones_zeros = np.zeros(on_off_bools.size, dtype=bool)
        on_off_ones_zeros[on_off_bools] = 1
        return init_times, on_off_ones_zeros

    def init_temps_in_cycles(self, max_temps=None, min_temps=None, init_times=None,
                             on_off_bools=None, building_params=None,
                             cools=None, ambient=None):
        # "Pg. 1394 on Eq. 24: ...the assumption has been made that P_i is distributed
        # uniformly with respect to temperature.
        # Convention is that for P > 0 when cooling (upper right-hand side, Pg. 1391)
        cycle_start_temps = np.zeros(on_off_bools.size)
        on_offs = on_off_bools.astype(bool)
        cycle_start_temps[on_offs] = max_temps[on_offs]
        cycle_start_temps[~on_offs] = min_temps[~on_offs]
        logger.debug('starting init_temps_for_on_off_state')

        temps = cycle_start_temps
        on_states = np.zeros(temps.size, dtype=np.int8)
        on_states[on_off_bools] = 1
        init_temps = np.full(temps.size, np.nan)

        time = 0

        temps_update_partial = partial(self.temp_update, on_off_states=on_states,
                                       building_params=building_params,
                                       cool_params=cools, temp_a=ambient)

        while any(np.isnan(init_temps)):
            temps = temps_update_partial(temps=temps)
            time += self.callaway.H
            finished = np.isnan(init_temps) & np.greater_equal(
                time, init_times)
            init_temps[finished] = temps[finished]

        return init_temps

    def operating_states(self, updated_temps=None, on_offs=None, setpoints=None,
                         control=None):
        sp = symbols('theta_s')
        ti = symbols('theta_i')
        u = symbols('u_t')
        sigma = symbols('sigma')

        on_off_states = np.array(on_offs)
        temps = np.round(updated_temps, self.thermostat_precision)

        # Source:
        # Eq. 2, Callaway, "Tapping the Enerfy Storage Potential..." 2009
        # With noise term included
        on_expr = ti > sp + (sigma / 2) + u
        f_on = ufuncify((ti, sp, sigma, u), on_expr)
        m_on = f_on(temps, setpoints, self.callaway.DEADBAND, control)

        off_expr = ti < sp - (sigma / 2) + u
        # Source: Callaway, Eq. 4
        f_off = ufuncify((ti, sp, sigma, u), off_expr)

        # Convert to thermostat precision
        m_off = f_off(temps, setpoints, self.callaway.DEADBAND, control)
        on = m_on.astype(bool)
        on_off_states[on] = 1

        off = m_off.astype(bool)
        on_off_states[off] = 0

        return on_off_states

    def temp_update(self, on_off_states=None, building_params=None, cool_params=None,
                    temp_a=None, temps=None):
        a = symbols('a')
        theta_i = symbols('theta_i')
        theta_a = symbols('theta_a')
        m = symbols('m')
        rp_i = symbols('RP')

        temp_expr = (a * theta_i) + (1 - a) * (theta_a - (m * rp_i))

        f = ufuncify((a, theta_i, theta_a, m, rp_i), temp_expr)
        temps = f(building_params, temps, temp_a, on_off_states, cool_params)
        return temps

    def np_arrays_to_files(self, k, v, data_dir):
        if isinstance(v, np.ndarray):
            logger.debug(f"Creating .npy for {k} array")
            # Create the file name
            random_file_name = str(uuid.uuid4())
            npy_file = data_dir + '/' + random_file_name.upper() + '.npy'
            # Write the file
            np.save(npy_file, v, allow_pickle=False)
            return {"array": npy_file}
        elif isinstance(v, dict):
            v_replacement = {}
            logger.debug(f"Introspecting {k} dict")
            for k_in_v, v_in_v in v.items():
                v_replacement[k_in_v] = np_arrays_to_files(
                    k_in_v, v_in_v, data_dir)
            return v_replacement
        elif isinstance(v, np.int64):
            logger.debug(f"The type of {k} is np.int64")
            return int(v)
        else:
            logger.debug(f"The type of {k} is {type(v)}")
            return v

    def np_files_to_arrays(self, k, v):
        if isinstance(v, dict) and set(v.keys()) == {'array'}:
            np_arr = np.load(v['array'])
            return np_arr
        elif isinstance(v, dict):
            v_with_arrays = {}
            for k_in_v, v_in_v in v.items():
                v_with_arrays[k_in_v] = np_files_to_arrays(k_in_v, v_in_v)
            return v_with_arrays
        else:
            return v

    def get_durations_and_summarize(self, cools, display_expr=True):
        # Determine upper and lower temperature ranges possible with
        # down- and up- changes in temperature
        # 1. Lower temperature
        # 2. Upper temperature
        # 3. Up duration
        # 4. Down duration

        # Determine how long the on/off durations are

        num_devs = self.init_building_params.size
        logger.debug(
            f"Setpoint: {self.callaway.SETPOINT} Deadband: {self.callaway.DEADBAND}")
        lower_deadband_temp = self.callaway.SETPOINT - self.callaway.DEADBAND / 2
        upper_deadband_temp = self.callaway.SETPOINT + self.callaway.DEADBAND / 2
        logger.debug(f"Initial lower deadband temps: {lower_deadband_temp}")
        logger.debug(f"Initial upper deadband temps: {upper_deadband_temp}")
        assert isinstance(self.thermostat_bits, int)
        lower_deadband_temps = np.full(
            self.callaway.NUM_DEVS, lower_deadband_temp)
        upper_deadband_temps = np.full(
            self.callaway.NUM_DEVS, upper_deadband_temp)

        # Figure out where the thermostats might switch from ON to OFF and vice versa.
        # To do this, have them start at one extreme (the upper or lower end of the range)
        # and operate until they have at reached the other temperature extreme or gone past it
        # Now you have a range of possible temperatures, based on your assumed parameters.
        # So you have to store both switching temps.
        # Next, to allow initialization at any temperature and operating state in that range,
        # randomly select a duration that is less than or equal the sum of the durations
        # of the on and off stages.

        theta_i = symbols('theta_i')  # Indoor temp
        a = symbols('a')  # Building thermal params
        theta_a = symbols('theta_a')  # Outdoor temp

        # Initialize 1.
        cycle = 'on'
        rp_i = symbols('RP_i')  # used in on durations only
        # Source: Eq. 1, Callaway
        expr_on = a * theta_i + (1 - a) * (theta_a - rp_i)
        on_func = ufuncify((a, theta_i, theta_a, rp_i), expr_on)

        if display_expr:
            print("Deterministic part of Eq. 1 assuming device is ON:")
            pprint(expr_on)
            print()
            logger.info(f"Equation when m=1: {expr_on}")

        scale = self.callaway.NOISE_DEGREES_STD_DEV_BY_SECS_PER_TIME_STEP[1]
        noise_partial = partial(self.rng.normal, loc=0.0, scale=scale)

        # Take up to 24 hours for this, break at 24
        # Are Eq. 21 and 22 equivalent to Eq. 1 (and to each other)?
        logger.info("Finding initial min temps")
        logger.info(
            f"Temps are all {self.callaway.SETPOINT + self.callaway.DEADBAND / 2} to start")
        logger.info(
            f"Finding switching temps less than {np.max(lower_deadband_temps)}")

        # This is the first of two calls to get_switch_off_temps_and_durations().
        # The function get_switch_on_temps_and_durations() is called in between.
        # The first of the two calls starts the HVAC systems at the upper limit of the deadband.
        # The HVAC systems are ON.
        # Before switching to OFF, the temperatures go slightly beyond the deadband,
        # instead of being precisely at the boundary, and the duration for both OFF and ON cycles to be 
        # more realistic.
        result = self.get_switch_off_temps_and_durations(on_func, self.init_building_params,
                                                         upper_deadband_temps, cools,
                                                         noise_partial, lower_deadband_temps)
        switch_off_temps_init, _ = result

        expr_off = a * theta_i + (1 - a) * theta_a
        off_func = ufuncify((a, theta_i, theta_a), expr_off)

        logger.info('Finding OFF durations from minimum temps')
        logger.info(
            f"Upper deadband temp for population: {self.callaway.SETPOINT + self.callaway.DEADBAND / 2}")
        logger.info(
            f"Max temp {np.max(switch_off_temps_init)} after previous ON cycle")

        result = self.get_switch_on_temps_and_durations(num_devs, off_func,
                                                        self.init_building_params,
                                                        lower_deadband_temps,
                                                        noise_partial,
                                                        upper_deadband_temps)
        switch_on_temps, off_durs = result
        logger.info(f"{np.min(switch_on_temps)=} {np.max(switch_on_temps)=}")

        logger.info('Finding ON durations from max indoor temps')
        logger.info(f"Min starting temp {np.min(switch_on_temps)}")
        logger.info(f"Max starting temp {np.max(switch_on_temps)}")

        result = self.get_switch_off_temps_and_durations(on_func, self.init_building_params,
                                                         switch_on_temps, cools,
                                                         noise_partial, lower_deadband_temps)
        switch_off_temps, on_durs = result
        logger.info(f"{np.min(switch_off_temps)=} {np.max(switch_off_temps)=}")

        total_times = off_durs + on_durs

        max_seconds = 60 * 60 * 24

        on_and_off_finished = total_times < max_seconds
        unfinished = total_times >= max_seconds
        num_finished = np.sum(on_and_off_finished)
        num_unfinished = np.sum(unfinished)
        logger.info(f"{num_finished} finished after 24 hours")
        logger.info(f"{num_unfinished} unfinished after 24 hours")
        assert num_finished + num_unfinished == num_devs
        assert np.max(total_times[on_and_off_finished]) <= max_seconds

        results = {}
        results['off_durations'] = off_durs[on_and_off_finished]
        results['on_durations'] = on_durs[on_and_off_finished]
        results['on_and_off_finished'] = on_and_off_finished
        results['init_building_params'] = self.init_building_params[on_and_off_finished]
        results['sim_building_params'] = self.sim_building_params[on_and_off_finished]
        results['cools'] = cools[on_and_off_finished]
        results['switch_on_temps'] = switch_on_temps[on_and_off_finished]
        results['switch_off_temps'] = switch_off_temps[on_and_off_finished]

        return results

    def simulate_for_train_and_test(self, init_building_params=None,
                                    sim_building_params=None,
                                    off_durations=None, on_durations=None,
                                    on_and_off_finished=None,
                                    cools=None, switch_on_temps=None,
                                    switch_off_temps=None):
        # Training simulation initialization
        logger.debug('Starting simulation')
        starting_simulation_time = datetime.now()
        num_on_and_off_finished = np.sum(on_and_off_finished)
        times_temps_args = {'building_params': init_building_params,
                            'off_durations': off_durations,
                            'on_durations': on_durations,
                            'on_and_off_finished': num_on_and_off_finished,
                            'cools': cools,
                            'switch_on_temps': switch_on_temps,
                            'switch_off_temps': switch_off_temps}
        train_d_result = self.init_times_and_temps(**times_temps_args)
        init_temps, on_off_bools, on_temps, off_temps = train_d_result

        simulate_seconds = self.train_seconds + self.test_seconds
        print(
            f"Simulating total time (training and test) of {simulate_seconds/(SECS_PER_HOUR)} hours")
        secs_per_time_step = self.callaway.H
        print(f"Seconds per time step: {secs_per_time_step}")
        train_stage = 1
        test_stage = 1
        total_stages = train_stage + test_stage
        sim_time_steps = int(simulate_seconds / secs_per_time_step)
        print(f"sim_time_steps: {sim_time_steps}")

        on_offs_at_t = np.zeros((on_off_bools.size, sim_time_steps),
                                dtype=np.int8)
        temps_at_t = np.zeros(on_offs_at_t.shape)
        total_on_at_t = np.zeros(sim_time_steps, dtype=int)
        total_demand_at_t = np.zeros(sim_time_steps, dtype=int)

        t = 0

        # on_off_bools contains 1's and 0's (ons and offs)
        on_offs_at_t[:, t] = on_off_bools
        temps_at_t[:, t] = init_temps
        total_on_at_t[t] = np.sum(on_off_bools)
        total_demand_at_t[t] = np.sum(np.multiply(on_off_bools, cools))

        next_temps = partial(self.temp_update, building_params=sim_building_params,
                             cool_params=cools, temp_a=self.callaway.AMBIENT)

        next_op_states = partial(
            self.operating_states, setpoints=self.callaway.SETPOINT)

        # building_params is based on the secs_per_time_step, correct?
        # then upate should be consistent with that

        # "The ARMAX model parameters were first estimated using the simulation results
        # of the system response to Eq. (20)."
        # Eq. 20
        M = 25

        if self.callaway.H == 60:
            num_control_values = sim_time_steps - 1
        elif self.callaway.H == 1:
            secs_per_min = 60
            assert isinstance(sim_time_steps, int)
            assert isinstance(secs_per_min, int)
            #num_control_values = int(sim_time_steps / secs_per_min) + M - 1
            num_control_values = int(sim_time_steps / secs_per_min) - 1
        else:
            raise ValueError(f"H must be either 1 second or 60 seconds.")
        logger.info(f"{num_control_values=}")

        white_noise = self.rng.normal(loc=0.0, scale=5 * pow(10, -3),
                                      size=num_control_values + M - 1)
        logger.info(f"raw {white_noise.shape=}")
        if self.callaway.H == 1:
            white_noise = np.repeat(white_noise, secs_per_min)
            white_noise = np.divide(white_noise, 60)

        logger.info(f"{white_noise.shape=}")

        if self.callaway.H == 1:
            moving_sum = pd.Series(white_noise).rolling(M * secs_per_min).sum()[(M - 1) * secs_per_min:]
        else:
            moving_sum = pd.Series(white_noise).rolling(M).sum()[M - 1:]
        controls = moving_sum.to_numpy()
        logger.info(f"{controls.shape=}")
        try:
            if self.callaway.H == 60:
                assert controls.size == (sim_time_steps - 1)
            elif self.callaway.H == 1:
                assert controls.size == (sim_time_steps - 60)
        except AssertionError as e:
            e.add_note(f"{controls.size=} and {sim_time_steps=}")
            raise

        logger.info(f"shape of controls: {controls.shape}")

        temps, on_offs = init_temps, on_off_bools
        logger.debug(f"Size of temps: {len(temps)}")
        logger.debug(f"Size of on_offs: {len(on_offs)}")
        # "Note that the thermostat set point is assumed inifinitely adjustable in these
        # simulations. Clearly, in practice, sensor resolution will limit effective
        # set point adjustability." (p. 1393, just after Eq. 20)
        num_devs_sampled = 50  # Not related to the number of bins (also 50)
        num_bins_per_dev = 50
        edges_of_temp_bins = np.zeros((num_devs_sampled, num_bins_per_dev + 1))
        logger.info(f"edges_of_temp_bins.size: {edges_of_temp_bins.shape}")
        on_temp_bin_counts = np.zeros((num_devs_sampled, num_bins_per_dev))
        off_temp_bin_counts = np.zeros((num_devs_sampled, num_bins_per_dev))
        num_devs = np.sum(on_and_off_finished)
        interval_size = int(num_devs / num_devs_sampled)
        logger.info(f"interval size: {interval_size}")

        logger.debug(f"size of on_temps: {on_temps.size}")
        logger.debug(f"size of off_temps: {off_temps.size}")
        full_range = np.arange(on_and_off_finished.size)
        random_devices = self.rng.choice(full_range[on_and_off_finished], size=num_devs_sampled)

        logger.debug(
            f"Size of temps[random_devices]: {temps[random_devices].size}")
        logger.debug(
            f"size of on_temps[random_devices]: {on_temps[random_devices].size}")
        logger.debug(
            f"size of off_temps[random_devices]: {off_temps[random_devices].size}")

        for dev, max_temp, min_temp in zip(np.arange(len(random_devices)),
                                           on_temps[random_devices],
                                           off_temps[random_devices]):
            try:
                assert min_temp < max_temp
            except AssertionError:
                raise ValueError(f"for dev {dev}, {min_temp=} and {max_temp=}")
            edges_of_temp_bins[dev, :] = np.histogram(np.array([max_temp, min_temp]),
                                                      range=(
                                                          min_temp, max_temp),
                                                      bins=num_bins_per_dev)[1]

        scale = 0.01 if secs_per_time_step == 1 else 0.08
        noise = partial(self.rng.normal, loc=0.0, scale=scale)

        controls = np.round(controls, decimals=self.thermostat_precision)
        control_exog = np.nditer(controls)
        for t, u in zip(np.arange(1, sim_time_steps), control_exog):
            prev_temps = temps
            temps = next_temps(temps=prev_temps, on_off_states=on_offs)

            noise_vector = noise(size=num_devs)

            temp_changes_plus_noise = np.add(temps, noise_vector)
            temps = temp_changes_plus_noise

            prev_temps = np.round(prev_temps, self.thermostat_precision)
            on_offs = next_op_states(updated_temps=prev_temps, on_offs=on_offs,
                                     control=u)
            total_on_at_t[t] = np.sum(on_offs)
            total_demand_at_t[t] = np.sum(np.multiply(
                on_offs, 1/self.callaway.LOAD_EFFICIENCY * cools))

        #     for dev, temp in enumerate(temps[random_devices]):
        #         try:
        #             bin_num = np.where(np.histogram(temp,
        #                                             bins=edges_of_temp_bins[dev, :])[0])
        #         except IndexError as exc:
        #             print(f"dev: {dev} has IndexError")
        #             print(
        #                 f"len of temps[::interval_size]: {len(temps[::interval_size])}")
        #             print(f"interval_size: {interval_size}")
        #             print(
        #                 f"size of edges_of_temp_bins: {edges_of_temp_bins.shape}")
        #         if on_offs[dev]:
        #             on_temp_bin_counts[dev, bin_num] += 1
        #         else:
        #             off_temp_bin_counts[dev, bin_num] += 1

        # # Calculate probability densities
        # bin_width = 1 / num_bins_per_dev
        # avg_devs_on_per_bin = np.divide(
        #     np.sum(on_temp_bin_counts, axis=0), sim_time_steps)
        # on_prob_densities = np.divide(
        #     avg_devs_on_per_bin, bin_width * num_devs_sampled)

        # avg_devs_off_per_bin = np.divide(
        #     np.sum(off_temp_bin_counts, axis=0), sim_time_steps)
        # off_prob_densities = np.divide(
        #     avg_devs_off_per_bin, bin_width * num_devs_sampled)

        # print(f"prob densities, on: {on_prob_densities}")
        # print(f"prob densities, off: {off_prob_densities}")
        # print(
        #     f"Sum of prob densities, on, lower 50%: {np.sum(on_prob_densities[0:25])}")
        # print(
        #     f"Sum of prob densitites, on, upper 50%: {np.sum(on_prob_densities[25:])}")
        # print(
        #     f"Sum of prob densities, off, lower 50%: {np.sum(off_prob_densities[0:25])}")
        # print(
        #     f"Sum of prob densitites, off, upper 50%: {np.sum(off_prob_densities[25:])}")

        simulation_duration = (
            datetime.now() - starting_simulation_time).seconds
        logger.debug(f'Simulation ended after {simulation_duration}')

        total_on_at_t = total_on_at_t.reshape(-1, 1)
        total_demand_at_t = total_demand_at_t.reshape(-1, 1)
        return controls[120:], total_on_at_t[120:], total_demand_at_t[120:]

    def get_switch_off_temps_and_durations(self, on_func, building_params, temps, cools, noise_partial,
                                           lower_deadband_temps):
        time = 0
        num_devs = self.callaway.NUM_DEVS
        switch_off_temps = np.zeros(num_devs)
        on_durs = np.zeros(num_devs, dtype=int)
        ambient = self.callaway.AMBIENT

        temps = self.update_temperatures_for_on_cycle(on_func, building_params, temps,
                                                      ambient, cools, noise_partial, num_devs)
        next_temps = self.update_temperatures_for_on_cycle(on_func, building_params, temps,
                                                           ambient, cools, noise_partial, num_devs)
        time += 1
        temps_with_thermostat_resolution = np.round(
            temps, self.thermostat_precision)
        lower_deadband_passed = np.logical_and(switch_off_temps == 0.0,
                                               np.less(temps_with_thermostat_resolution, lower_deadband_temps))
        switch_off_temps[lower_deadband_passed] = next_temps[lower_deadband_passed]
        on_durs[lower_deadband_passed] = time + 1

        while any(switch_off_temps == 0):
            temps = next_temps
            next_temps = self.update_temperatures_for_on_cycle(on_func, building_params, temps,
                                                               ambient, cools, noise_partial, num_devs)
            time += 1

            temps_with_thermostat_resolution = np.round(
                temps, self.thermostat_precision)
            lower_deadband_passed = np.logical_and(switch_off_temps == 0.0,
                                                   np.less(temps_with_thermostat_resolution, lower_deadband_temps))
            switch_off_temps[lower_deadband_passed] = next_temps[lower_deadband_passed]
            on_durs[lower_deadband_passed] = time + 1

            unfinished = switch_off_temps == 0
            num_unfinished = np.sum(unfinished)
            if time == 60 * 60 * 24:
                logger.info(f"{num_unfinished} 'on' cycles unfinished after "
                            f"24 hours.")
                break
        return switch_off_temps, on_durs

    def update_temperatures_for_on_cycle(self, on_func, building_params, temps, ambient, cools, noise, num_devs):
        new_temps_before_noise = on_func(
            building_params, temps, ambient, cools)
        noise_vector = noise(size=num_devs)
        temps = np.add(new_temps_before_noise, noise_vector)
        return temps

    def update_temperatures_for_off_cycle(self, off_func, building_params, temps, ambient, noise, num_devs):
        new_temps_before_noise = off_func(building_params, temps, ambient)
        noise_vector = noise(size=num_devs)
        temps = np.add(new_temps_before_noise, noise_vector)
        return temps

    def get_switch_on_temps_and_durations(self, num_devs, off_func, building_params, lower_deadband_temps, noise,
                                          upper_deadband_temps):
        time = 0
        switch_on_temps = np.zeros(num_devs)
        off_durs = np.zeros(num_devs, dtype=int)

        ambient = self.callaway.AMBIENT
        temps = self.update_temperatures_for_off_cycle(off_func, building_params, lower_deadband_temps,
                                                       ambient, noise, num_devs)
        next_temps = self.update_temperatures_for_off_cycle(off_func, building_params, temps,
                                                            ambient, noise, num_devs)
        time += 1
        temps_with_thermostat_resolution = np.round(
            temps, self.thermostat_precision)
        upper_deadband_passed = np.logical_and(switch_on_temps == 0.0,
                                               np.greater(temps_with_thermostat_resolution, upper_deadband_temps))
        switch_on_temps[upper_deadband_passed] = next_temps[upper_deadband_passed]
        off_durs[upper_deadband_passed] = time + 1

        while any(switch_on_temps == 0):
            temps = next_temps
            next_temps = self.update_temperatures_for_off_cycle(off_func, building_params, temps,
                                                                ambient, noise, num_devs)
            time += 1
            temps_with_thermostat_resolution = np.round(
                temps, self.thermostat_precision)
            upper_deadband_passed = np.logical_and(switch_on_temps == 0.0,
                                                   np.greater(temps_with_thermostat_resolution, upper_deadband_temps))
            switch_on_temps[upper_deadband_passed] = next_temps[upper_deadband_passed]
            off_durs[upper_deadband_passed] = time + 1

            unfinished = switch_on_temps == 0
            num_unfinished = np.sum(unfinished)
            if time == 60 * 60 * 24:
                logger.info(f"{num_unfinished} 'off' cycles unfinished after "
                            f"24 hours.")
                break
        return switch_on_temps, off_durs
