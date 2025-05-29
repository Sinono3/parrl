# Fast Reinforcement Learning from Scratch (Parallel and Distributed Computing Final Project)

## Dependencies

The dependencies required for the project are:

- SFML >=3.0.0
- libtorch (C++ API) >= 2.7

They can all be installed in Ubuntu with the following command:

```sh
sudo apt-get install -y build-essential libsfml-dev libtorch-dev
```

## Building from source

```sh
git clone https://github.com/Sinono3/parrl.git
cd parrl
make
```

## Usage

Once the project has built, you can run the following binaries:

```sh
# To train the agent
./train
# To see the agent in action, visually
./test
# For the SPS (steps per second) benchmark
./benchmark
# To see the environment and control it manually
./human
```

## Objective

Reduce average convergence time to the barest minimum possible with parallelization.

## Roadmap

- [x] Cartpole environment implementation
- [x] Reinforcement learning implementation with REINFORCE
  - The agent converges (and quickly)
- [ ] Implement neural network from scratch (CPU)
- [ ] Implement parallel batched episode generation (vectorization)
  - e.g. instead of simulating 10000 steps serially, have 10 cores running 1000 steps
- [ ] Implement a way to measure improvement in training speed by benchmarking average convergence time with lots of different random seeds.
- [ ]
- [ ] (Optional) Use a more complex environment than CartPole
- [ ] (Optional) Move everything to GPU, the NN, the logs,  he environment
