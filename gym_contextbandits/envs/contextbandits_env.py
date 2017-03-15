import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class ContextBanditsEnv(gym.Env):
    """ A Context Bandit environment
    
    2 bandits with 3 arms
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.viewer = None

        self.bandits = np.array([[10.0,    5.0,    0.0],
                                 [4.0,    3.0,    10.0]])
        self.num_bandits = self.bandits.shape[0] # = 2 states
        self.num_arms = self.bandits.shape[1] # = 3 actions
        
        self.action_space = spaces.Discrete(self.num_arms)
        self.observation_space = spaces.Discrete(self.num_bandits) # todo: is this right?
        
        self._seed()
        self._reset() # initializes anything that is reset at episodes

    def _reset(self):
        self.state = np.random.randint(self.num_bandits) # random starting state
        # return np.array([self.state, 0])
        return np.array(self.state)

    # modeled after classic_control envs
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _render(self, mode='human', close=False):
        pass

    # def _close(self):
    
    def _step(self, action):
        """ steps forward one trial
        
        Args:
            action - action comes from the algorithm file

        Returns:
            a reward of 1 or -1
        """
        assert self.action_space.contains(action)
        reward = 0
        done = True
        bandit = self.bandits[self.state,action] # function of state and action
        result = np.random.randn(1)
        # if result < bandit:
        #     #return a positive reward.
        #     reward = 1
        # else:
        #     #return a negative reward.
        #     reward = -1
        reward = self.bandits[self.state,action]
        
        if self.state == 0: self.state = 1
        else: self.state = 0
        
        return np.array(self.state), reward, done, {}
        # return self.state, reward, done, {}

if __name__ == '__main__':
    env = ContextBanditsEnv()
