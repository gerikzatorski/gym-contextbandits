from gym.envs.registration import register

register(
    id='ContextBandits-v0',
    entry_point='gym_contextbandits.envs:ContextBanditsEnv',
)
