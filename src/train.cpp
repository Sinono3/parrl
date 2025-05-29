#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <torch/torch.h>
#pragma clang diagnostic pop

#include "Cartpole.hpp"

std::pair<int, torch::Tensor> chooseAction(torch::nn::Sequential &net,
										   const torch::Tensor &obs) {
	// Generate policy (action prob distribution) from net
	auto logits = net->forward(obs);
	auto probs = torch::softmax(logits, -1);

	// Sample distribution
	auto action_tensor = torch::multinomial(probs, 1);
	int action = action_tensor.item<int>();
	auto log_prob = torch::log(probs[action]);
	return {action, log_prob};
}

torch::Tensor returnsFromRewards(const std::vector<float> &rewards, const std::vector<bool>& dones) {
	constexpr float GAMMA = 0.99f;
	auto stepcount = (long long)rewards.size();
	auto returns = torch::empty({stepcount});
	returns[stepcount - 1] = rewards[(size_t)stepcount - 1];

	for (auto i = stepcount - 2; i >= 0; i--) {
		// If done, we don't add the returns
		if (dones[(size_t)i])
			returns[i] = rewards[(size_t)i];
		else
			returns[i] = rewards[(size_t)i] + GAMMA * returns[i + 1];
	}

	// normalize
	returns = (returns - returns.mean()) / (returns.std() + 1e-8);
	return returns;
}

float getAverageEpisodeReward(const std::vector<float> &rewards, const std::vector<bool>& dones) {
	int totalEpisodeCount = 0;
	float totalReward = 0.0f;

	for (size_t i = 0; i < rewards.size(); i++) {
		totalReward += rewards[i];
		if (dones[(size_t)i])
			totalEpisodeCount++;
	}
	return totalReward / (float)totalEpisodeCount;
}

int main() {
	torch::manual_seed(42);

	auto net =
		torch::nn::Sequential(torch::nn::Linear(4, 128), torch::nn::ReLU(),
							  torch::nn::Linear(128, 2));
	auto opt =
		torch::optim::Adam(net->parameters(), torch::optim::AdamOptions(0.1));

	Cartpole env;

	constexpr int EPOCHS = 1000000;
	constexpr int STEPS = 1000;
	constexpr float TARGET_AVG_REWARD = 150.0f;

	std::vector<float> rewards;
	std::vector<bool> dones;
	std::vector<torch::Tensor> log_probs;

	auto start = std::chrono::high_resolution_clock::now();
	for (int epoch = 1; epoch <= EPOCHS; epoch++) {
		rewards.clear();
		dones.clear();
		log_probs.clear();

		net->train();

		// Experience acquisition
		auto obs = env.reset();
		for (int step_idx = 0; step_idx < STEPS; step_idx++) {
			auto obs_tensor = torch::tensor(at::ArrayRef<float>(obs.vec));
			auto [action, log_prob] = chooseAction(net, obs_tensor);

			auto step = env.step((CartpoleAction)action);
			obs = step.obs;
			rewards.push_back(step.reward);
			dones.push_back(step.done);
			log_probs.push_back(log_prob);

			if (step.done)
				obs = env.reset();
		}

		// Training on experience
		auto returns = returnsFromRewards(rewards, dones);
		auto advantages = returns - returns.mean();
		auto loss = -torch::mean(advantages * torch::stack(log_probs));

		opt.zero_grad();
		loss.backward();
		opt.step();

		auto avg_reward = getAverageEpisodeReward(rewards, dones);
		std::println("Batch {} over. Avg reward: {}.", epoch, avg_reward);
		// TODO: testing/validation

		if (avg_reward > TARGET_AVG_REWARD) {
			auto stop = std::chrono::high_resolution_clock::now();
			auto micros = (stop - start).count();
			std::println("Finished in {} epochs ({} µs = {} s)", epoch, micros,
						 (double)micros / (double)(1000000000));
			
			// TODO: Save model
			return 0;
		}
	}
}
