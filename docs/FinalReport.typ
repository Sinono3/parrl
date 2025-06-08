#import "@preview/zebraw:0.5.5": *
#show: zebraw
#show link: underline
#show link: set text(blue)

#let ntustheader(course, assignment, deadline) = {[
  *National Taiwan University of Science and Technology*

  #grid(
    columns: 2,
    row-gutter: 0.7em,
    column-gutter: 2em,
    [*Student*], [Aldo Acevedo],
    [*ID*], [F11315101],
    [*Course*], course,
    [*Deadline*], deadline,
  )
  #line(length: 100%)

  = #assignment
]}
#ntustheader("Parallel and Distributed Compututing", [Term Project --- Final Report], "2025-06-09")

== Title

Fast Parallel Reinforcement Learning on the Cartpole Environment

== Objective

// Full description of the problem you want to solve.

Up to this date, most reinforcement learning models are built mostly in Python, by using
C/C++ wrapper libraries for the tensor operations e.g. PyTorch, TensorFlow.
Depending on the complexity of the projects, the environments used to train these models
rely to varying degrees on their native and Python code.
Modern ML architectures often have bad performance due to
(1) overhead of data transfer between Python and native code
(2) overhead of data transfer between CPU and GPU
(3) Python’s inherent bad performance
(4) general lack of optimization (unnecessary operations, cache misses, data duplication).

*Solution:* create a fully-fledged self-contained program that trains an RL agent and
(1) use fully native code for all components used during training (environment, optimization, backprop/gradient descent)
(2) reduce data transfers between devices and data conversions between formats to a minimum
(3) take advantage of multithreading to simulate multiple environments at once.

== Methods and Algorithms

// - Clearly list how do you solve the problem. If the solution method comes from some references you found, refer to the reference clearly.
// - Make sure that you can convince everyone that you understand the problem, and you can solve the problem by these methods/algorithms.
// - Do some sample calculation will be helpful.
// - Draw a flowchart for your serial program.
// - Updated the flowchart to reflect your serial program.

Reinforcement learning consists of _an agent_, taking _actions_ in an _environment_ given _observations_, then receiving a _positive or negative reward_ depending on the current _state_ of the environment. Then, sequences of actions that lead to greater aggregate rewards are _reinforced_ with machine learning. How precisely reward-optimal behaviors are reinforced is defined by the chosen _RL method_ @sutton1998.

The chosen *environment* for this project is an identical implementation of the Cartpole-v1 environment found in OpenAI's Gym library, originally described by Sutton and Barto @sutton1998.
The objective of this environment is for the agent to balance a pole on a cart.
In this environment, the agent's observation space is a 4-float vector, the position of the cart, the velocity of the cart, the angle of the pole and the angular velocity of the pole.
This environment has a discrete and finite action space. For every step of the environment, it expects a single integer $a$ as an action, where $a=0$ moves the cart left, and $a=1$ moves the cart right.

For this project, *the agent's policy* (i.e. the criterion on which our agent will take actions, a function producing actions based on observations)
will be a standard multi-layer perceptron (MLP), also known as a feedforward neural network.
The input of the MLP is the observation from the environment, in this case a 4-element float vector.
The output of the MLP is a discrete probability mass function of which action to take, a 2-element float vector.
The actual action to be taken is sampled from this distribution produced by the agent.

The chosen *RL method* is simple vanilla policy gradient, also known as the REINFORCE method @sutton1998.

We can then cleanly divide the program into three independent sections: the environment, the training infrastructure, and the optimization.

==== Environment

The environment needs to (1) receive inputs (2) perform a _step_ or simulate the game environment based on the inputs (3) generate observations and a reward for the agent.

#figure(caption: "Graphical representation of the CartPole-v1 environment", image(width: 50%, "./cartpole.gif")))

For the purposes of this project, it is only important to know that all the environment step function does is
approximately integrate a few state variables, to simulate the physics of the cart and the pole.

During the training no actual graphics are produced, as they are not needed.
Our agent only requires the observations.
The graphics are only for human visualization and testing.

==== Training infrastructure


#figure(caption: "Main training algorithm", ```
void train(seed) {
	initialize RNG on seed

	for epoch in 0..EPOCHS {
		observation = environment.reset()
		experience = []

		for step in 0..STEPS {
			action = agent.forward_pass(observation);
			observation, reward, done = environment.step(action)
			add (observation, action, reward, done) to experience

			if (done)
				observation = environment.reset()
		}

		returns = calculateReturns(experience);
		agent.backward_pass(returns, experience);
	}
```)

In modern implementations of reinforcement learning, we consistently switch between a environment phase, and an optimization phase.
In the environment phase, we let the agent produce enough experiences to train on, and in the optimization phase, we train our policy
on the experiences. _Experiences_ refers to a sequence of observation-action-reward-done tuples.

The environment is run for a maximum of `STEPS` total steps, from program start to finish, resetting when the environment is "done."
Every one of these steps and its associated data (observation, action, reward, done), is appended into the experience buffer.

==== Optimization

Once we have collected enough experience in the rollout buffer, we use this information to optimize the policy.
In our case, since the policy is a neural network, we need to perform gradient descent. To perform gradient descent,
we need to calculate the gradient, and we do that with backpropagation.
The implementation of the entire MLP is very similar to those appearing in the assignments.

== Application

// potential applications of your term project

- Fast versions of environments for RL training.
- Infrastructure for mass-scale simultaneous RL training.

== Implementation

// - Input of your program: If your program takes an input file, document your input file format; if your program interactively get some input from users, document these inputs.
// - Execution time: Create some example problems and run them. Report its execution time.
// - Output of your program: Show outputs of your program.
// - Implementation: your code should be in working condition in our group cluster system, with a makefile to build executables from your source code. Let me know where you put your programs and makefile.
// - Validations: Give some pseudo input, prepare expected output (e.g. analytical solutions), and compare your actual output vs. expected output.

There are four executables produced in the build:

1. `train` performs the actual reinforcement learning described above.
2. `benchmark` runs the enviroment for 20000 steps and calculates the average SPS (steps per second) of the environment alone. This is used to compare the performance with the original Python version and to measure speedup before and after optimizations.
3. `human` allows a human to test the environment manually using the keyboard, without any neural network or policy.

Usage of `train`:

```
Usage: ./train <operation>
Operations:
bench <N> <target_avg_reward>: runs N training runs reaching up to the target and returns the average training time
once <target_avg_reward>: trains the agent once and then visualizes the result graphically
```

=== Output

Executing `./train once 1900`:

```
Batch 50 over. Avg reward: 92.59259.
Batch 100 over. Avg reward: 161.29033.
Batch 150 over. Avg reward: 227.27272.
Batch 200 over. Avg reward: 312.5.
Batch 250 over. Avg reward: 303.0303.
Batch 300 over. Avg reward: 384.6154.
Batch 350 over. Avg reward: 416.66666.
Batch 400 over. Avg reward: 416.66666.
Batch 450 over. Avg reward: 454.54544.
Batch 500 over. Avg reward: 588.2353.
Batch 550 over. Avg reward: 833.3333.
Batch 600 over. Avg reward: 666.6667.
Batch 650 over. Avg reward: 625.
Batch 700 over. Avg reward: 714.2857.
Batch 750 over. Avg reward: 769.2308.
Batch 800 over. Avg reward: 1250.
Batch 850 over. Avg reward: 1111.1111.
Batch 900 over. Avg reward: 1250.
```

After the agent reaches the 1900.0 reward target, a window is launched showing the agent in action, as seen in @visualization.

#figure(caption: "Visualization of the agent launched after finishing the training.", image(width: 50%, "./after_training.png")) <visualization>

Executing './train bench 15 1900', shows similar outputs to the ones previously but repeated 15 times
for the different 15 seeds tested.

== Parallelization

=== Environment

There isn't anything to parallelize in the environment's step function itself, as it doesn't contain loops.

=== Training

#figure(caption: "Main training algorithm (parallel)", zebraw(
	highlight-lines: (6, 8, 9, 13, 18), ```
void train(seed) {
	initialize RNG on seed

	for epoch in 0..EPOCHS {
		observation = environment.reset()
		experience = [[[]]] # <-- SIMSxSTEPSx4 tensor

		#pragma omp parallel for
		for sim in 0..SIMS {
			for step in 0..STEPS {
				action = agent.forward_pass(observation);
				observation, reward, done = environment.step(action)
				experience[sim][step] = [observation, action, reward, done]

				if (done)
					observation = environment.reset()
			}
		}
		returns = calculateReturns(experience);
		agent.backward_pass(returns, experience);
	}
```))

The parallelization uses OpenMP to run multiple environment simulations 
simultaneously. Each thread maintains its own environment instance and 
has its own place in the experience buffer.
This approach maximizes CPU utilization during the experience collection 
phase, which is the most time-consuming part of training.

=== MLP

The MLP implementation also contains various OpenMP directives to speed up
mainly the backpropagation. The forward passes are already parallel for
free, since they are called from the simulation-specific threads.

For the backprop, we parallelize over batches where possible,
since there's no interdepencies or waiting between the different iterations.
We can reap great speedups from this parallelization since we are working with batch sizes
of over 10000. 

The matrix multiplications in both the backward and forward pass are not really
ideal for parallelization since we are dealing with small ANNs, and the overhead
of spawning threads and the possiblity of false sharing will produce overall
net slowdown. In reinforcement learning, we usually deal with smaller models.

== Validation

Serial and parallel implementations produce equivalent training curves 
when using identical random seeds, confirming correctness of the 
parallel implementation.

Not only that, but the agent consistently is able to reach the accumulated
reward target that is set.

== Performance comparison

=== Setup

All the tests were run in an M1 Pro MacBook Pro (10-core CPU).

=== Environment

The environment alone, by being implemented in C++, will provide a great speedup to training.
The SPS is calculated by simply running environment steps in a loop, and providing dummy inputs on every step
Non-uniform step times are not an issue with this environment. All steps take the same time regardless of the input.

#figure(caption: "Python v. C++ environment comparison", table(
  columns: 3,
  [*Implementation*], [*SPS*],     [*SPS speedup*],
  [Python],           [215189],    [1.0x],
  [C++],              [37950664],  [176.35x],
))

=== Training

We're mainly interested in seeing how much the training time improves on average by using our implementation, with respect
to the Python version, and the serial C++ version.

To calculate the average training time for certain methods in a reliable and reproducible way, we make all random-number
generators deterministic by providing the same seed values on each benchmark. Each benchmark result consists of the
average over 20 different training runs with different seeds.

To evaluate the training time for Python, a Python version of the `src/train.cpp` file was created. The Python version
uses the PyTorch and Gym libraries, and tries to mimick our methods and algorithms as close as possible, while still
adhering to PyTorch scripting conventions.

#figure(caption: "Python v. C++ average training time comparison", table(
  columns: 3,
  [*Implementation*],                    [*Average training time (secs)*], [*Average training time speedup*],
  [Python (serial)],                     [497.754],                        [1.00x],
  [C++ serial],                          [6.073],                          [81.96x],
  [C++ training loop parallel],          [3.166],                          [157.22x],
  [C++ training loop + MLP parallel],    [2.342],                          [212.53x],
))

#figure(caption: "Left: Parallel speedup vs. threads. Right: Parallel efficiency vs. threads", image("plot.png"))

// list references that you found to be related to your term project.
#bibliography("FinalReport_references.bib", full: true, title: "References")
