#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gymnasium",
#     "numpy",
# ]
# ///

import gymnasium as gym
import numpy as np
import time

env = gym.make("CartPole-v1")
obs, _ = env.reset()

STEPS = 10000
start = time.time()
for _ in range(STEPS):
    _, _, done, _, _ = env.step(0)
    if done:
        env.reset()
end = time.time()

sps = STEPS / (end - start)
print(f"Python version SPS: {sps}")
