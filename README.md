# Explanation Battery Branch.

This branch was made to create the battery environment. This ReadMe file is created
only to explain how to use the battery environment.

## Using the battery environment

Copy the following files from this branch to the branch you're using to create your agent:

- `__init__.py`
- `environment_battery.py`
- `train.py`

If you've done this, running train.py as normal will train your agent in the battery environment.
If you want to change the capacity of the battery you can do this by setting the `--battery_size`
arg in the command line. So, for example, running `python train.py --battery_size 200` will give the
agent a battery with a capacity of 200 steps. **The default value for the battery size is 1000 steps**.

## Using the default environment

If you want to train your agent on the default (old) environment, you can do this by adding the arg
`--no_battery` to the command line, e.g. `python train.py --no_battery`. Adding this arg will make
everything work exactly the same way as it did before the battery environment was created.

## Note about the the reward function

I created a new reward function for the battery environment punishing the agent for running out of
battery or going to the charger too soon. I'm not completely sure if all values i chose make sense,
so please look at this critically and change it if you think that'll make it better.
