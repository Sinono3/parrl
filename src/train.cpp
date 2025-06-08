#include "MLP.hpp"
#include <chrono>
#include <random>
#include "Cartpole.hpp"
#include "test.hpp"
#include <print>

static std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

void softmax(size_t n, float *logits, float* probs_out) {
	// Find max logit
	float max_logit = logits[0];
	for (size_t i = 0; i < n; i++)
		if (logits[i] > max_logit)
			max_logit = logits[i];

	float sumexp = 0.0f;
	for (size_t i = 0; i < n; i++)
		sumexp += std::exp(logits[i] - max_logit); // We subtract the max logit for numerical stability

	float logsumexp = std::log(sumexp);
	for (size_t i = 0; i < n; i++)
		probs_out[i] = std::exp(logits[i] - max_logit - logsumexp); // We subtract the logsumexp it this way for numerical stability
}

int chooseAction(std::mt19937& rng, float* probs, size_t n) {
	auto dist = std::discrete_distribution<int>(probs, probs + n);
	auto action = dist(rng);
	return action;
}

std::vector<float> returnsFromRewards(size_t stepcount, const float *rewards,
									  const bool* dones) {
	constexpr float GAMMA = 0.99f;
	auto returns = std::vector<float>(stepcount);
	returns[stepcount - 1] = rewards[(size_t)stepcount - 1];

	for (int i = (int)stepcount - 2; i >= 0; i--) {
		// If done, we don't add the returns
		if (dones[(size_t)i])
			returns[(size_t)i] = rewards[(size_t)i];
		else
			returns[(size_t)i] = rewards[(size_t)i] + GAMMA * returns[(size_t)i + 1];
	}

	float mean = 0.0f, var = 0.0f;
	for (size_t i = 0; i < stepcount; i++)
		mean += returns[i];
	mean /= (float)stepcount;
	for (size_t i = 0; i < stepcount; i++) {
		float centered = returns[i] - mean;
		var += centered * centered;
	}

	float std = std::sqrtf(var / (float)stepcount);
	// normalize
	for (size_t i = 0; i < stepcount; i++) {
		returns[i] = (returns[i] - mean) / (std + (float)1e-8);
	}
	return returns;
}

float getAverageEpisodeReward(size_t stepcount,
                              const float* rewards,
							  const bool* dones) {
	int totalEpisodeCount = 0;
	float totalReward = 0.0f;

	for (size_t i = 0; i < stepcount; i++) {
		totalReward += rewards[i];
		if (dones[(size_t)i])
			totalEpisodeCount++;
	}
	if (totalEpisodeCount < 1)
		totalEpisodeCount = 1;
	return totalReward / (float)totalEpisodeCount;
}

struct Timing {
	long long sim_forward_time = 0;
	long long backward_time = 0;
	long long total_time = 0;
};

std::tuple<bool, int, Timing> train(unsigned long long seed, bool viz = false) {
	Timing timing;
		
	constexpr size_t ACTIONS = 2;
	constexpr int EPOCHS = 10000;
	constexpr int STEPS = 10000;
	// How many parallel simulations will be running
	constexpr int SIMS = 5;
	constexpr int STEPS_PER_SIM = STEPS/SIMS;
	constexpr float TARGET_AVG_REWARD = 1900.0f;

	std::mt19937 rng((unsigned int)seed);
	auto mlp = MLP(3, new size_t[]{4, 128, ACTIONS}, {ReLU, NoOp}, STEPS);
	mlp.initHe();

	auto observations = new float[STEPS * 4]; // <- hardcoded observation size
	auto rewards = new float[STEPS];
	auto dones = new bool[STEPS];
	auto actions = new int[STEPS];
	auto logits = new float[STEPS * ACTIONS];
	auto probs = new float[STEPS * ACTIONS];
	auto dL_da = new float[STEPS * ACTIONS];

	auto start = std::chrono::high_resolution_clock::now();
	for (int epoch = 1; epoch <= EPOCHS; epoch++) {
		auto s = std::chrono::high_resolution_clock::now();
		#pragma omp parallel for
		for (size_t sim = 0; sim < SIMS; sim++) {
			size_t epoch_sim_seed = (size_t)(seed<<16) + (size_t)sim;
			epoch_sim_seed = std::hash<size_t>{}(epoch_sim_seed);
			Cartpole env(epoch_sim_seed);
			// Experience acquisition
			auto obs = env.reset();

			for (size_t step_idx = sim*STEPS_PER_SIM; step_idx < (sim+1)*STEPS_PER_SIM; step_idx++) {
				observations[step_idx * 4 + 0] = obs.vec[0];
				observations[step_idx * 4 + 1] = obs.vec[1];
				observations[step_idx * 4 + 2] = obs.vec[2];
				observations[step_idx * 4 + 3] = obs.vec[3];

				float* curLogits = &logits[(size_t)step_idx * ACTIONS];
				float* curProbs = &probs[(size_t)step_idx * ACTIONS];
				mlp.forward(observations, curLogits, step_idx);
				softmax(ACTIONS, curLogits, curProbs);

				auto action = chooseAction(rng, curProbs, ACTIONS);
				auto step = env.step((CartpoleAction)action);
				obs = step.obs;
				actions[step_idx] = action;
				rewards[step_idx] = step.reward;
				dones[step_idx] = step.done;
				if (step.done)
					obs = env.reset();
			}
			
			// Set done=true at the of each `sim`
			dones[(sim+1)*STEPS_PER_SIM - 1] = true;
		}
		timing.sim_forward_time += (std::chrono::high_resolution_clock::now() - s).count();

		// Training on experience
		// Calculate loss based on returns
		// Simply increase the probability of actions which have the greatest return,
		// decrease probability of actions which gave us the least return
		auto returns = returnsFromRewards(STEPS, rewards, dones);

		// Backward
		// Calculate dL_da (last layer gradient)
		s = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < STEPS; i++) {
			for (size_t j = 0; j < ACTIONS; j++) {
				float indicator = ((int)j == actions[i]) ? 1.0f : 0.0f;
				dL_da[i*ACTIONS + j] = (probs[i*ACTIONS+j] - indicator) * returns[i];
			}
		}
		mlp.backward_optim(dL_da, observations, STEPS, 0.1f);
		timing.backward_time += (std::chrono::high_resolution_clock::now() - s).count();

		auto avg_reward = getAverageEpisodeReward(STEPS, rewards, dones);

		if ((epoch % 50) == 0) {
			std::println("Batch {} over. Avg reward: {}.", epoch, avg_reward);
		}
		if (avg_reward >= TARGET_AVG_REWARD) {
			auto stop = std::chrono::high_resolution_clock::now();
			timing.total_time = (stop - start).count();
			if (viz)
				testAgent([&](auto obs) {
					mlp.forward(obs.vec.data(), &logits[0], 0);
					return chooseAction(rng, &logits[0], ACTIONS);
				}, false);
			return {true, epoch, timing};
		}
	}
	return {false, 0, timing};
}

void bench(unsigned long long seeds) {
	double totalTime = 0.0;

	for (unsigned long long seed = 0; seed < seeds; seed++) {
		auto [success, epoch, timing] = train(seed);

		std::print("seed {}: ", seed);
		if (success)
			std::println("finished in {} epochs (total = {} s, sim+forward = {}s, backward = {}s)", epoch,
			             (double)timing.total_time / (double) 1e9,
			             (double)timing.sim_forward_time / (double) 1e9,
			             (double)timing.backward_time / (double) 1e9
			         );
		else
			std::println("failed...");

		totalTime += (double)timing.total_time / (double)1e9;
	}

	double averageTime = totalTime / (double)seeds;
	std::println("average time across {} seeds: {}", seeds, averageTime);
}

void once() {
	unsigned long long seed = 42;
	auto [success, epoch, timing] = train(seed, true);

	std::print("seed {}: ", seed);
	if (success)
		std::println("finished in {} epochs (total = {} s, sim+forward = {}s, backward = {}s)", epoch,
		             (double)timing.total_time / (double) 1e9,
		             (double)timing.sim_forward_time / (double) 1e9,
		             (double)timing.backward_time / (double) 1e9
		         );
	else
		std::println("failed...");
}

int main(int argc, const char **argv) {
	std::vector<std::string> args;
	// Convert to C++ strings
	for (int i = 0; i < argc; i++) {
		args.push_back(argv[i]);
	}

	if ((argc == 2 || argc == 3) && args[1] == "bench") {
		unsigned long long seeds = 15;
		if (argc == 3)
			seeds = std::stoull(args[2]);
		bench(seeds);
		return 0;
	} else if (argc == 2 && args[1] == "once") {
		once();
		return 0;
	} else {
		std::println("Usage: ./train <operation>\n"
					 "Operations:\n"
					 "bench <N>: runs N training runs and returns the average "
					 "training time\n"
					 "once: trains the agent once and then visualizes the "
					 "result graphically\n");
		return 1;
	}
}
