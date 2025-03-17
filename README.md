# Simulation of sixty thousand home AC units

### Python packages supporting simulation - NumPy, SymPy

NumPy is used to randomly determine the thermal parameters of a set of 60,000 home air conditioners and homes, such that the parameters have lognormal distributions.

SymPy is used to vectorize functions that simulate the temperature changes within the homes and the operating states based on thermostatic setpoint control.

Setpoint control is simulated across a 10-hour scenario with 1-second time steps. All of the air conditioners are subject to the same setpoint signal, which is derived from a white noise process.

### Source of assumptions and original equations

The specific values that were used to initialize the thermal parameters and the formulae for the functions described above originated in _Calloway, Tapping the energy storage potential in electric loads to deliver load following and regulation, with application to wind energy (2019)_.

### Running simulation and obtaining output

A Jupyter notebook was run to initialize the population of air conditioners and simulate their behavior. The simulation of the AC units over 10 simulated hours took 58 seconds on an M1 laptop. At the end of the notebook, a plot was generated of the total number of air conditioners in the ON state over the course of the simulation.

An HTML version of the Jupyter notebook is at https://htmlpreview.github.io/?https://github.com/nickpowersys/sixty-thousand-homes-simulation/blob/main/home_hvacs_simulation.html.

To create a virtual environment locally and run the notebook, first clone the repository and use `uv venv && uv sync`. The kernel at .venv/bin/python should be selected.

To launch the notebook from the local venv,

`uv run --with jupyter jupyter lab` and open the notebook `home_hvacs_simulation.ipynb`.
