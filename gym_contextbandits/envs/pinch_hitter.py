import numpy as np
import gym
from gym import spaces

class PinchHitterFixedEnv(gym.Env):
    """
    Contextual bandit environment based on selecting a pinch hitter vs lefty/righty pitcher.
    """
    
    # 2021 National League All Star Batters
    # List of tuples (vs RHP, vs LHP, name)
    batting_splits = [
        (0.280, 0.368, 'Buster Posey'),
        (0.271, 0.241, 'JT Realmuto'),
        (0.247, 0.278, 'Yadier Molina'),
        (0.289, 0.152, 'Omar Narvaez'),
        (0.317, 0.257, 'Freddie Freeman'),
        (0.240, 0.276, 'Max Muncy'),
        (0.317, 0.274, 'Adam Frazier'),
        (0.237, 0.323, 'Ozzie Albies'),
        (0.265, 0.270, 'Jake Cronenworth'),
        (0.282, 0.285, 'Fernando Tatis Jr.'),
        (0.319, 0.244, 'Brandon Crawford'),
        (0.305, 0.392, 'Trea Turner'),
        (0.245, 0.295, 'Nolan Arenado'),
        (0.259, 0.284, 'Kris Bryant'),
        (0.238, 0.295, 'Eduardo Escobar'),
        (0.286, 0.258, 'Justin Turner'),
        (0.288, 0.246, 'Manny Machado'),
        (0.278, 0.302, 'Ronald Acuna Jr.'),
        (0.346, 0.177, 'Jessie Winker'),
        (0.310, 0.306, 'Nick Castellanos'),
        (0.263, 0.266, 'Mookie Betts'),
        (0.293, 0.325, 'Bryan Reynolds'),
        (0.265, 0.268, 'Kyle Schwarber'),
        (0.333, 0.280, 'Juan Soto'),
        (0.237, 0.296, 'Chris Taylor'),
    ]

    def __init__(self):
        self.action_space = spaces.Discrete(len(self.batting_splits))
        self.observation_space = spaces.Discrete(2) # lefty/righty
        self._ab_info = {}

    def _get_obs(self):
        return self._pitcher_handedness

    def _get_info(self):
        return self._ab_info

    def reset(self, seed=None, return_info=False, options=None):
        # Required to seed self.np_random
        super().reset(seed=seed)
        
        # New pitcher selected for every episode
        self._pitcher_handedness = self.np_random.integers(2)

        # Hitter selected by agent
        self._ab_info["hitter"] = None
        self._ab_info["pitcher"] = "LHP" if self._pitcher_handedness else "RHP"
        self._ab_info["outcome"] = None
        
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action):
        assert self.action_space.contains(action)

        reward = int(np.random.uniform() <
                     self.batting_splits[action][self._pitcher_handedness])
        terminated = True
        truncated = False

        self._ab_info["hitter"] = self.batting_splits[action][2]
        self._ab_info["outcome"] = "hit" if reward else "out"

        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
