#include "MLP.hpp"
#include <random>
#include <span>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <torch/torch.h>
#pragma clang diagnostic pop

#include "Cartpole.hpp"
#include "test.hpp"

static std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

std::pair<int, float> chooseAction(std::mt19937& rng, float* logits, size_t n) {
	// // Generate policy (action prob distribution) from net
	// auto probs = torch::softmax(torch::tensor(at::ArrayRef<float>(logits, n)), -1);
	// // Sample distribution
	// auto action_tensor = torch::multinomial(probs, 1);

	// Find max logit
	float max_logit = logits[0];
	for (size_t i = 0; i < n; i++)
		if (logits[i] > max_logit)
			max_logit = logits[i];

	float sumexp = 0.0f;
	for (size_t i = 0; i < n; i++)
		sumexp += std::exp(logits[i]); // We subtract the max logit for numerical stability

	float logsumexp = std::log(sumexp);
	// TODO: Don't allocate here
	auto probs = new float[n];
	for (size_t i = 0; i < n; i++)
		probs[i] = std::exp(logits[i] - logsumexp); // We subtract the logsumexp it this way for numerical stability
	
	int action = 0;
	float x = uniform01(rng);
	float cumsum = 0.0f;
	for (size_t i = 1; i < n; i++) {
		cumsum += probs[i];
		if (x <= cumsum) {
			action = (int)i;
			break;
		}
	}
	auto log_prob = std::log(probs[action]);
	delete [] probs;
	return {action, log_prob};
}

std::vector<float> returnsFromRewards(const std::vector<float> &rewards,
									  const std::vector<bool> &dones) {
	constexpr float GAMMA = 0.99f;
	auto stepcount = rewards.size();
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
	for (size_t i = 0; i < stepcount; i++) {
		float centered = returns[i] - mean;
		var += centered * centered;
	}
	float std = std::sqrtf(var);

	// normalize
	for (size_t i = 0; i < stepcount; i++) {
		returns[i] = (returns[i] - mean) / (std + (float)1e-8);
	}
	return returns;
}

float getAverageEpisodeReward(const std::vector<float> &rewards,
							  const std::vector<bool> &dones) {
	int totalEpisodeCount = 0;
	float totalReward = 0.0f;

	for (size_t i = 0; i < rewards.size(); i++) {
		totalReward += rewards[i];
		if (dones[(size_t)i])
			totalEpisodeCount++;
	}
	return totalReward / (float)totalEpisodeCount;
}

std::tuple<bool, int, long long> train(unsigned long long seed) {
	constexpr size_t ACTIONS = 2;
	constexpr int EPOCHS = 10000;
	constexpr int STEPS = 1000;
	constexpr float TARGET_AVG_REWARD = 300.0f;

	std::mt19937 rng((unsigned int)seed);
	torch::manual_seed(seed);

	auto mlp = MLP(3, new size_t[]{4, 128, ACTIONS}, {ReLU, NoOp}, STEPS);
	Cartpole env(seed);

	std::vector<float> observations;
	std::vector<float> logits(STEPS * ACTIONS);
	std::vector<int> actions;
	std::vector<float> rewards;
	std::vector<bool> dones;
	std::vector<float> log_probs;

	std::vector<float> dL_da(STEPS * ACTIONS);

	auto start = std::chrono::high_resolution_clock::now();
	for (int epoch = 1; epoch <= EPOCHS; epoch++) {
		observations.clear();
		actions.clear();
		rewards.clear();
		dones.clear();
		log_probs.clear();

		// Experience acquisition
		auto obs = env.reset();
		for (size_t step_idx = 0; step_idx < STEPS; step_idx++) {
			observations.insert(observations.end(), obs.vec.begin(),
								obs.vec.end());
			float* curLogits = &logits[(size_t)step_idx * ACTIONS];
			mlp.forward(obs.vec.data(), curLogits, step_idx, STEPS);
			auto [action, log_prob] = chooseAction(rng, curLogits, ACTIONS);

			auto step = env.step((CartpoleAction)action);
			obs = step.obs;
			actions.push_back(action);
			rewards.push_back(step.reward);
			dones.push_back(step.done);
			log_probs.push_back(log_prob);

			if (step.done)
				obs = env.reset();
		}

		// Training on experience
		// Calculate loss based on returns
		// Simply increase the probability of actions which have the greatest return,
		// decrease probability of actions which gave us the least return
		auto returns = returnsFromRewards(rewards, dones);

		// Calculate dL_da (last layer gradient)
		for (size_t i = 0; i < STEPS; i++) {
			float sum = 0.0f;
			for (size_t j = 0; j < ACTIONS; j++) {
				sum += logits[i * ACTIONS + j];
			}

			for (size_t j = 0; j < ACTIONS; j++) {
				if (j == (size_t)actions[i])
					dL_da[i*ACTIONS + j] = returns[i] * sum;
				else
					dL_da[i*ACTIONS + j] = returns[i] * (sum - 1);
			}
		}
		
		// Backward
		mlp.backward_optim(dL_da.data(), observations.data(), STEPS, 0.01f);

		auto avg_reward = getAverageEpisodeReward(rewards, dones);
		std::println("Batch {} over. Avg reward: {}.", epoch, avg_reward);
		// TODO: testing/validation

		if (avg_reward > TARGET_AVG_REWARD) {
			auto stop = std::chrono::high_resolution_clock::now();
			auto micros = (stop - start).count();
			// std::println("Finished in {} epochs ({} µs = {} s)", epoch, micros, (double)micros / (double)(1000000000));
			// net->eval();
			// testAgent([&](auto obs) {
			// 	auto obs_tensor = torch::tensor(at::ArrayRef<float>(obs.vec));
			// 	return chooseAction(net, obs_tensor);
			// });
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
