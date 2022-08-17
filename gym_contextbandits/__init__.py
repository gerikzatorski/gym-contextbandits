from gym.envs.registration import register

register(
    id='gym_contextbandits/PinchHitterFixed-v0',
    entry_point='gym_contextbandits.envs:PinchHitterFixedEnv',
)
