#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <torch/torch.h>
#pragma clang diagnostic pop

#include "Cartpole.hpp"

// TODO: Reimplement
torch::Tensor policyFromNet(torch::nn::Sequential &net,
							const torch::Tensor &state) {
	auto logits = net->forward(state);
	auto probs = torch::softmax(logits, -1);
	return probs;
}

// TODO: Reimplement
std::pair<int, torch::Tensor> chooseAction(const torch::Tensor &probs) {
	auto action_tensor = torch::multinomial(probs, /*num_samples=*/1);
	int action = action_tensor.item<int>();
	auto log_prob = torch::log(probs[action]);
	return {action, log_prob};
}

// TODO: Reimplement
torch::Tensor returnsFromRewards(const std::vector<float> &rewards) {
	constexpr float GAMMA = 0.99f;

	auto rewards_tensor = torch::tensor(rewards);
	size_t len = rewards.size();

	// Create gamma vector
	auto arange =
		torch::arange((float)len, torch::TensorOptions().dtype(torch::kFloat));
	auto gammavec = torch::pow(GAMMA, arange);

	// Calculate returns
	auto flipped_rewards = torch::flip(rewards_tensor, {0});
	auto weighted = flipped_rewards * gammavec;
	auto cumsum = torch::cumsum(weighted, 0);
	auto returns = torch::flip(cumsum, {0}) / gammavec;

	// Normalize
	auto mean = returns.mean();
	auto std = returns.std();
	returns = (returns - mean) / (std + 1e-8);

	return returns;
}

int main() {
	auto net =
		torch::nn::Sequential(torch::nn::Linear(4, 128), torch::nn::ReLU(),
							  torch::nn::Linear(128, 2));
	auto opt =
		torch::optim::Adam(net->parameters(), torch::optim::AdamOptions(0.01));

	Cartpole env;

	constexpr int EP_BATCHES = 1000000;
	constexpr int EPS_TO_TRY = 100;
	constexpr int EPS_TO_LEARN = 2;

	struct Episode {
		float total_reward;
		std::vector<float> rewards;
		std::vector<torch::Tensor> log_probs;
	};

	for (int batch = 1; batch <= EP_BATCHES; batch++) {
		net->train();

		// Experience acquisition
		std::vector<Episode> episodes;
		for (int ep = 1; ep <= EPS_TO_TRY; ep++) {
			float total_reward = 0.0f;
			std::vector<float> rewards;
			std::vector<torch::Tensor> log_probs;

			auto obs = env.reset();

			for (int step_idx = 0; step_idx < 10000; step_idx++) {
				auto obs_tensor = torch::tensor(at::ArrayRef<float>(obs.vec));
				auto [action, log_prob] =
					chooseAction(policyFromNet(net, obs_tensor));

				auto step = env.step((CartpoleAction)action);
				obs = step.obs;
				rewards.push_back(step.reward);
				log_probs.push_back(log_prob);
				total_reward += step.reward;

				if (step.done)
					break;
			}

			episodes.push_back({total_reward, rewards, log_probs});
		}

		auto total_reward = 0.0f;
		for (auto &ep : episodes)
			total_reward += ep.total_reward;
		auto avg_reward = total_reward / (float)episodes.size();

		std::sort(episodes.begin(), episodes.end(),
				  [](const auto &a, const auto &b) {
					  return a.total_reward > b.total_reward;
				  });

		if (episodes.begin() + EPS_TO_LEARN < episodes.end())
			episodes.erase(episodes.begin() + EPS_TO_LEARN + 1, episodes.end());

		auto losses = torch::zeros({1});

		for (auto &[_, rewards, log_probs] : episodes) {
			auto log_probs2 = torch::stack(log_probs);
			auto returns = returnsFromRewards(rewards);
			losses += torch::sum(-returns * log_probs2);
		}

		opt.zero_grad();
		losses.backward();
		opt.step();

		std::println("Batch {} over. Avg reward: {}.", batch, avg_reward);
		// TODO: testing/validation
	}
}
