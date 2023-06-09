# Explanation Battery Branch.

This branch was made to create the battery environment. This ReadMe file is created
only to explain how to use the battery environment. This is based on an environment with 1 agent, so you'll probably have to make some changes for the multi-agent environment.

## Using the battery environment

Copy the following files from this branch to the branch you're using to create your agent:

- `world/__init__.py`
- `world/environment_battery.py`
- `train_deep_q.py`
- Change the agent on line 210 of `train_deep_q.py` to your agent.

If you've done this, running `train_deep_q.py` as normal will train your agent in the battery environment.
If you want to change the capacity of the battery you can do this by setting the `--battery_size`
arg in the command line. So, for example, running `python train_deep_q.py  --battery_size 200` will give the
agent a battery with a capacity of 200 steps. **The default value for the battery size is 1000 steps**.

All changes in the `train_deep_q.py` file related to the battery are marked with a comment `# BatteryRelated`, so that'll hopefully help in knowing where to look.

## What is different in the battery environment?

- The agent always starts the run at the charger
- The reward function is different (also see the note at the bottom of this ReadME)
- If the agent runs out of battery while walking around the room, the battery is recharged to full capacity in order to be able to continue the training, however the agent receives a very negative reward in order to discourage this behaviour.
- If the agent goes to the charger when not all dirt is gone, the battery is recharged to full capacity in just 1 step (i guess we first make sure everything works like this, if it does me might make some tweaks to do this as we discussed after the meeting with the tutor). Other than that, this is treated as a normal move (with this I mean that agent_moved = True and total_agent_moves += 1).

## New info/world_stats variables related to battery

Info:

- `info["battery_left"]` - The amount of steps left in the battery.
- `info["quick_charging"]` - True if the agent is recharging at the charging station (so when not all dirt is gone), False otherwise.

World Stats:

- `world_stats["battery_left"]` - The amout of steps left in the battery
- `world_stats["quick_charged"]` - counts the number of times the agent went to charging station to recharge (without all dirt being gone).
- `world_stats["empty_battery_counter"]` - counts the number of times the agent ran out of battery, and we had to recharge it for him.

## Using the default environment

If you want to train your agent on the default (old) environment, you can do this by adding the arg
`--no_battery` to the command line, e.g. `python train_deep_q.py --no_battery`. Adding this arg will make
everything work exactly the same way as it did before the battery environment was created.

## Note about the the reward function

I created a new reward function for the battery environment punishing the agent for running out of
battery or going to the charger too soon. I'm not completely sure if all values i chose make sense,
so please look at this critically and change it if you think that'll make it better.
