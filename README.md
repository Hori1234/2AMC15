# 2AMC15-2023-DIC

Welcome to 2AMC15 Data Intelligence Challenge solution of Group 2!
This is the repository containing our solution.

## Test our agents

Our agents are tested and evaluated on six different setups. For details about these setups, please refer to our report.
To run the tests, please run the following command: `$ python train.py --out results/`.

Running this line will prompt all agents to be trained and evaluated on the six different setups.
The results of this example command will be saved in the `results/` folder.

```bash
usage: train.py [-h] [--no_gui] [--fps FPS] [--iter ITER]
                [--random_seed RANDOM_SEED] [--out OUT]

DIC Reinforcement Learning Trainer.

options:
  -h, --help            show this help message and exit
  --no_gui              Disables rendering to train faster
  --fps FPS             Frames per second to render at. Only used if no_gui is
                        not set.
  --iter ITER           Number of iterations to go through.
  --random_seed RANDOM_SEED
                        Random seed value for the environment.
  --out OUT             Where to save training results.
```

## Requirements

- python ~= 3.10
- numpy >= 1.24
- tqdm ~= 4
- pygame ~= 2.3
- flask ~= 2.2
- flask-socketio ~= 5.3
- pillow ~= 9.4

## Reproducibility
Run the following command in the command-line to reproduce the results.
$ python train.py --iter 500000 --no_gui