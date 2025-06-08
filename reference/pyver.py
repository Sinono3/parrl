import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Return logits


def compute_returns(rewards, dones, gamma=0.99):
    """Compute discounted returns and normalize them."""
    returns = torch.zeros_like(rewards)
    running_return = 0.0

    # Compute returns backwards
    for i in reversed(range(len(rewards))):
        if dones[i]:
            running_return = 0.0
        running_return = rewards[i] + gamma * running_return
        returns[i] = running_return

    # Normalize returns
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


def get_average_episode_reward(rewards, dones):
    """Calculate average reward per episode."""
    total_reward = rewards.sum().item()
    episode_count = dones.sum().item()
    if episode_count == 0:
        episode_count = 1
    return total_reward / episode_count


def collect_experience(seed_offset, steps_per_sim=2000):
    """Collect experience from one simulation."""
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed_offset)
    np.random.seed(seed_offset)
    torch.manual_seed(seed_offset)

    observations = []
    actions = []
    rewards = []
    dones = []

    obs, _ = env.reset(seed=seed_offset)

    for _ in range(steps_per_sim):
        observations.append(obs.copy())

        # This will be filled by the main process
        action = np.random.randint(2)  # Placeholder
        actions.append(action)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        rewards.append(reward)
        dones.append(done)

        if done:
            obs, _ = env.reset()

    # Set last step as done for proper return calculation
    dones[-1] = True

    env.close()
    return {
        "observations": np.array(observations),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "dones": np.array(dones),
    }


def train_single_seed(seed):
    """Train policy for a single seed."""
    ACTIONS = 2
    EPOCHS = 10000
    STEPS = 10000
    SIMS = 5
    STEPS_PER_SIM = STEPS // SIMS
    TARGET_AVG_REWARD = 500.0
    LEARNING_RATE = 0.1

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")  # Keep on CPU for consistency with C++ version
    policy = PolicyNetwork().to(device)

    # Initialize weights with He initialization (similar to C++ version)
    for layer in policy.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    optimizer = torch.optim.SGD(policy.parameters(), lr=LEARNING_RATE)

    # Timing
    sim_forward_time = 0.0
    backward_time = 0.0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Experience collection
        sim_start = time.time()

        # Collect data from parallel simulations
        all_observations = []
        all_actions = []
        all_rewards = []
        all_dones = []

        for sim in range(SIMS):
            env = gym.make("CartPole-v1")
            sim_seed = (seed << 16) + sim
            sim_seed = hash(sim_seed) % (2**32)
            env.action_space.seed(sim_seed)

            obs, _ = env.reset(seed=sim_seed)

            sim_observations = []
            sim_actions = []
            sim_rewards = []
            sim_dones = []

            for _ in range(STEPS_PER_SIM):
                sim_observations.append(obs.copy())

                # Forward pass
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    logits = policy(obs_tensor)
                    probs = F.softmax(logits, dim=1)
                    action = torch.multinomial(probs, 1).item()

                sim_actions.append(action)

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                sim_rewards.append(reward)
                sim_dones.append(done)

                if done:
                    obs, _ = env.reset()

            # Set last step as done
            sim_dones[-1] = True

            all_observations.extend(sim_observations)
            all_actions.extend(sim_actions)
            all_rewards.extend(sim_rewards)
            all_dones.extend(sim_dones)

            env.close()

        sim_forward_time += time.time() - sim_start

        # Convert to tensors
        observations = torch.FloatTensor(all_observations).to(device)
        actions = torch.LongTensor(all_actions).to(device)
        rewards = torch.FloatTensor(all_rewards).to(device)
        dones = torch.BoolTensor(all_dones).to(device)

        # Compute returns
        returns = compute_returns(rewards, dones)

        # Training
        backward_start = time.time()

        optimizer.zero_grad()

        # Forward pass
        logits = policy(observations)
        log_probs = F.log_softmax(logits, dim=1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Policy gradient loss (negative because we want to maximize)
        loss = -(selected_log_probs * returns).mean()

        # Backward pass
        loss.backward()
        optimizer.step()

        backward_time += time.time() - backward_start

        # Calculate average reward
        avg_reward = get_average_episode_reward(rewards, dones)

        if epoch % 50 == 0:
            print(f"Batch {epoch} over. Avg reward: {avg_reward:.6f}.")

        if avg_reward >= TARGET_AVG_REWARD:
            total_time = time.time() - start_time
            return {
                "success": True,
                "epoch": epoch,
                "total_time": total_time,
                "sim_forward_time": sim_forward_time,
                "backward_time": backward_time,
            }

    return {
        "success": False,
        "epoch": 0,
        "total_time": 0.0,
        "sim_forward_time": 0.0,
        "backward_time": 0.0,
    }


def main():
    SEEDS = 1

    total_time = 0.0

    for seed in range(SEEDS):
        result = train_single_seed(seed)

        print(f"seed {seed}: ", end="")
        if result["success"]:
            print(
                f"finished in {result['epoch']} epochs "
                f"(total = {result['total_time']:.6f} s, "
                f"sim+forward = {result['sim_forward_time']:.6f}s, "
                f"backward = {result['backward_time']:.6f}s)"
            )
        else:
            print("failed...")

        total_time += result["total_time"]

    average_time = total_time / SEEDS
    print(f"average time across {SEEDS} seeds: {average_time:.6f}")


if __name__ == "__main__":
    main()
