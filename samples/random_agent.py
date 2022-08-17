import gym
import gym_contextbandits

SEED_VAL = None
NUM_STEPS = 10

def main():
    env = gym.make("gym_contextbandits/PinchHitterFixed-v0", new_step_api=True)
    obs = env.reset(seed=SEED_VAL)
    for step in range(NUM_STEPS):
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(info)
            obs = env.reset(seed=SEED_VAL)
    env.close()

if __name__ == "__main__":
    main()
