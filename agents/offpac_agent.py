#!/usr/bin/env python

import argparse
import logging
import sys
import numpy as np

import gym
from gym import wrappers
import gym_contextbandits


class OffPAC(object):
    """ State of the art agent """
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        self.Nv = observation_space.n
        self.Nu = action_space.n

        self.ev = np.zeros((self.Nv, 1))
        self.eu = np.zeros((self.Nu, 1))
        self.w = np.zeros((self.Nv, 1))
        self.v = np.zeros((self.Nv, 1)) # all weight vectors initialized to zero
        self.u = np.zeros((self.Nu, 1)) # all weight vectors initialized to zero

        self.pi = np.zeros((self.Nu, self.Nv)) # A x S -> [0,1]
        self.b = np.full((self.Nu, self.Nv), 1.0/self.Nu)
        # self.gamma = 1 # todo: assumed right now

        # do we need these?
        self.phi = 0
        self.psi = np.zeros((self.Nu, self.Nv))
        self.x = np.identity(2)

    def act(self, observation, reward, done):
        action = np.random.randint(self.Nu) # todo: works for now because self.b is uniform
        return action

    def update(self, s, a, s_prime, r):
        """ updates the Off-PAC algorithm """
        # parameters for TD error, critic, and actor update
        lam = 0.5 # eligibility trace parameter lambda
        alpha_v = 0.001 # step parameters
        alpha_w = 0.001 # step parameters
        alpha_u = 0.001 

        gamma_s = 1
        gamma_sprime = 1

        xs = np.transpose([self.x[:,s]])
        xsprime = np.transpose([self.x[:,s_prime]])
        
        delta = r + gamma_sprime * np.squeeze(np.dot(np.transpose(self.v), xsprime)) - np.squeeze(np.dot(np.transpose(self.v), xs))
        roe = self.pi[a][s]/self.b[a][s]
        roe = self.u[a]/self.b[a][s]

        #critic update
        self.ev = roe * (xs + gamma_s * lam * self.ev)
        self.v = self.v + alpha_v * (np.multiply(delta, self.ev) - gamma_sprime * (1 - lam) * np.multiply(self.w, self.ev) * xs)
        self.w = self.w + alpha_w * (np.multiply(delta, self.ev) - np.multiply(np.multiply(self.w, xs), xs))

        #actor update
        # grad_s =
        # self.eu = roe * (grad/self.pi[a][xs] + gamma_s * np.multiply(lam, self.eu))

        # update eligibility trace matrix
        self.psi = self.psi * gamma_s * lam
        self.eu = roe * (self.psi[a][s] + gamma_s * lam * self.eu)
        print "roe",
        print roe

        # self.u = self.u + alpha_u * np.multiply(np.squeeze(delta), self.eu)
        # self.u = self.u + np.multiply(alpha_u, np.multiply(np.squeeze(delta), self.eu))
        print "self.eu"
        print self.eu
        self.u = self.u + alpha_u * np.squeeze(delta) * self.eu
        print self.u


def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()

    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = OffPAC(env.observation_space, env.action_space)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            curr_state = ob
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action) 
            agent.update(curr_state, action, ob, reward) # ob = next_state
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    logger.info("Successfully ran OffPAC. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    print agent.u
    gym.upload(outdir)


if __name__ == '__main__':
    main()

    
    # main()
    # env = gym.make('ContextBandits-v0')
    # test = OffPAC(env.observation_space, env.action_space)
    # test.act(1, 5, False)
