import gymnasium as gym
import numpy as np
import time

env = gym.make("CartPole-v1")
obs, _ = env.reset()

STEPS = 10000
start = time.time()
for _ in range(STEPS):
    _ = env.step(0)
end = time.time()

sps = STEPS / (end - start)
print(f"Python version SPS: {sps}")
