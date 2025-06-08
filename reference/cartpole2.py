import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import copy

device = torch.device("cpu")
sm = nn.Softmax(dim=0)


def policy_from_net(net, state):
    return torch.distributions.Categorical(probs=sm(net(state)))


def choose_action(policy):
    action = policy.sample()
    log_prob = policy.log_prob(action)
    action = action.item()
    return action, log_prob


def returns_from_rewards(rewards):
    GAMMA = 0.99
    gammavec = GAMMA ** torch.arange(len(rewards))
    returns = (
        torch.flip(torch.cumsum(torch.flip(rewards, [0]) * gammavec, 0), [0]) / gammavec
    )
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


net = nn.Sequential(
    nn.Linear(4, 128),
    nn.ReLU(),
    nn.Linear(128, 2),
).to(device)
net.train()

loss = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(params=net.parameters(), lr=0.1)

env = gym.make("CartPole-v1")

EP_BATCHES = 1000000
EPS_TO_TRY = 100
EPS_TO_LEARN = 2

best_net = copy.deepcopy(net)
best_reward = 0.0

for batch in range(1, EP_BATCHES + 1):
    net.train()
    episodes = []

    for ep in range(1, EPS_TO_TRY + 1):
        total_reward = 0.0
        rewards = []
        log_probs = []

        state, _ = env.reset()
        reward = 0.0

        for _ in range(10000):
            total_reward += float(reward)
            state = torch.tensor(state, device=device)
            action, log_prob = choose_action(policy_from_net(net, state))

            state, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            if done or truncated:
                break

        episodes.append((total_reward, rewards, log_probs))

    avg_reward = sum(map(lambda x: x[0], episodes)) / len(episodes)

    # if avg_reward > (best_reward - 5):
    #     best_net = copy.deepcopy(net)
    #     best_reward = avg_reward
    # else:
    #     # Revert to best known policy if current one is worse
    #     net = copy.deepcopy(best_net)

    # Discard those with reward below PERCENTILE
    episodes.sort(key=(lambda r: r[0]))
    episodes = episodes[EPS_TO_TRY - EPS_TO_LEARN : EPS_TO_TRY]

    losses = torch.zeros(1)

    # Learn
    for _, rewards, log_probs in episodes:
        rewards = torch.tensor(rewards, device=device)
        log_probs = torch.stack(log_probs)

        returns = returns_from_rewards(rewards)
        losses += (-returns * log_probs).sum()

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    print(f"Batch {batch} over. Avg reward: {avg_reward}.")
    if batch % 20 == 0:
        # torch.save(net.state_dict(), "cartpole.pt")
        net.eval()
        renv = gym.make("CartPole-v1", render_mode="human")
        state, _ = renv.reset()

        for _ in range(10000):
            state = torch.tensor(state, device=device)
            state, reward, done, truncated, _ = renv.step(
                choose_action(policy_from_net(net, state))[0]
            )
            if done or truncated:
                break

        del renv
