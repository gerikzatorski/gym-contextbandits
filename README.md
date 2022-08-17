# Contextual Bandit Environments

## Environments

### PinchHitterFixed-v0

A simulated baseball scenario. It's bottom of the ninth with two outs and the bases are loaded in a tie ballgame. You are the manager and have to decide who is going to pinch hit. Walks do not count.

The only context given is whether the pitcher is a lefty or righty. Your pinch hitting options are all batters from the 2021 NL all-star roster.

## Installation

This package uses `setup.py` to manage installations. Pip is the recommended tool to create a local installation on your system or in a virtual environment.

```
cd gym-contextbandits
pip install -e .
```

Environment instances can then be created.

```
import gym_contextbandits
env = gym.make("PinchHitterFixed-v0")
```
