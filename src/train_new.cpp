#include "MLP.hpp"
#include <random>
#include "Cartpole.hpp"
#include "test.hpp"

static std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

std::pair<int, float> chooseAction(std::mt19937& rng, float* logits, size_t n) {
	// Find max logit
	float max_logit = logits[0];
	for (size_t i = 0; i < n; i++)
		if (logits[i] > max_logit)
			max_logit = logits[i];

	float sumexp = 0.0f;
	for (size_t i = 0; i < n; i++)
		sumexp += std::exp(logits[i] - max_logit); // We subtract the max logit for numerical stability

	float logsumexp = std::log(sumexp);
	// TODO: Don't allocate here
	auto probs = new float[n];
	for (size_t i = 0; i < n; i++)
		probs[i] = std::exp(logits[i] - logsumexp); // We subtract the logsumexp it this way for numerical stability
	
	auto dist = std::discrete_distribution<int>(probs, probs + n);
	auto action = dist(rng);
	auto log_prob = std::log(probs[action]);
	delete [] probs;
	return {action, log_prob};
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

std::tuple<bool, int, long long> train(unsigned long long seed) {
	constexpr size_t ACTIONS = 2;
	constexpr int EPOCHS = 10000;
	constexpr int STEPS = 1000;
	constexpr float TARGET_AVG_REWARD = 900.0f;

	std::mt19937 rng((unsigned int)seed);
	auto mlp = MLP(3, new size_t[]{4, 128, ACTIONS}, {ReLU, NoOp}, STEPS);
	mlp.initHe();
	Cartpole env(seed);

	auto observations = new float[STEPS * 4]; // <- hardcoded observation size
	auto rewards = new float[STEPS];
	auto dones = new bool[STEPS];
	auto actions = new int[STEPS];
	auto logits = new float[STEPS * ACTIONS];
	auto log_probs = new float[STEPS];
	auto dL_da = new float[STEPS * ACTIONS];

	auto start = std::chrono::high_resolution_clock::now();
	for (int epoch = 1; epoch <= EPOCHS; epoch++) {
		// Experience acquisition
		auto obs = env.reset();
		for (size_t step_idx = 0; step_idx < STEPS; step_idx++) {
			observations[step_idx * 4 + 0] = obs.vec[0];
			observations[step_idx * 4 + 1] = obs.vec[1];
			observations[step_idx * 4 + 2] = obs.vec[2];
			observations[step_idx * 4 + 3] = obs.vec[3];

			float* curLogits = &logits[(size_t)step_idx * ACTIONS];
			mlp.forward(observations, curLogits, step_idx);
			auto [action, log_prob] = chooseAction(rng, curLogits, ACTIONS);

			auto step = env.step((CartpoleAction)action);
			obs = step.obs;
			actions[step_idx] = action;
			rewards[step_idx] = step.reward;
			dones[step_idx] = step.done;
			log_probs[step_idx] = log_prob;

			if (step.done)
				obs = env.reset();
		}

		// Training on experience
		// Calculate loss based on returns
		// Simply increase the probability of actions which have the greatest return,
		// decrease probability of actions which gave us the least return
		auto returns = returnsFromRewards(STEPS, rewards, dones);

		// Calculate dL_da (last layer gradient)
		for (size_t i = 0; i < STEPS; i++) {
			float* logit_ptr = &logits[i * ACTIONS];
			float max_logit = *std::max_element(logit_ptr, logit_ptr + ACTIONS);

			float sumexp = 0.0f;
			for (size_t j = 0; j < ACTIONS; j++)
				sumexp += std::exp(logit_ptr[j] - max_logit);

			for (size_t j = 0; j < ACTIONS; j++) {
				float prob = std::exp(logit_ptr[j] - max_logit) / sumexp;
				float indicator = ((int)j == actions[i]) ? 1.0f : 0.0f;
				dL_da[i*ACTIONS + j] = (prob - indicator) * returns[i];
			}
		}
		
		// Backward
		mlp.backward_optim(dL_da, observations, STEPS, 0.05f);

		auto avg_reward = getAverageEpisodeReward(STEPS, rewards, dones);

		if ((epoch % 50) == 0) {
			std::println("Batch {} over. Avg reward: {}.", epoch, avg_reward);
		}
		if (avg_reward >= TARGET_AVG_REWARD) {
			auto stop = std::chrono::high_resolution_clock::now();
			auto micros = (stop - start).count();
			std::println("Finished in {} epochs ({} µs = {} s)", epoch, micros, (double)micros / (double)(1000000000));
			testAgent([&](auto obs) {
				mlp.forward(obs.vec.data(), &logits[0], 0);
				return chooseAction(rng, &logits[0], ACTIONS);
			}, false);
			return {true, epoch, micros};
		}
	}
	return {false, 0, 0};
}

int main() {
	constexpr unsigned long long SEEDS = 100;

	double totalTime = 0.0;

	for (unsigned long long seed = 0; seed < SEEDS; seed++) {

		auto [success, epoch, micros] = train(seed);
		double time = (double)micros / (double)(1000000000);

		std::print("seed {}: ", seed);
		if (success)
			std::println("finished in {} epochs ({} µs = {} s)", epoch, micros,
						 time);
		else
			std::println("failed...", epoch, micros, time);

		totalTime += time;
	}

	double averageTime = totalTime / (double)SEEDS;
	std::println("average time across {} seeds: {}", SEEDS, averageTime);
	return 0;
}
