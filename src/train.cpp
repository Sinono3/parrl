#include <fstream>
#include <iostream>
#include <filesystem>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <torch/torch.h>
#pragma clang diagnostic pop

#include "CarRacing.hpp"
#include "PPO.hpp"
#include "util.hpp"

int main() {
	auto max_ep_len = 1000;
	auto max_training_timesteps = (int)(5e6);

	// Hyperparameters related to action standard deviation
	float action_std_init = 0.5;
	float action_std_decay_rate = 0.05;
	float action_std_min = 0.05;
	int action_std_decay_freq = (int)(2.5e5);

	// Hyperparameters
	auto update_timestep = max_ep_len * 4;
	auto K_epochs = 1;
	auto eps_clip = 0.2;
	auto gamma = 0.99;
	auto lr = 0.0003;

	// Saving utils
	auto save_freq = max_ep_len * 4;

	// Printing utils
	auto print_freq = max_ep_len * 10;
	auto print_running_reward = 0.0f;
	auto print_running_episodes = 0;
	auto best_running_reward = 0.0f;

	CarRacing env(false);
	PPO ppo(std::vector<int64_t>{3, 96, 96}, (int64_t)3, lr, gamma, K_epochs,
			eps_clip, action_std_init);

	// DEBUG: Load checkpoint
	// torch::load(ppo.policy, "model.pt");

	auto time_step = 0;
	auto i_episode = 0;

	std::filesystem::create_directory("./checkpoint");

	std::ofstream policyLossLog;
	std::ofstream rewardLog;
	policyLossLog.open("checkpoint/loss.csv");
	rewardLog.open("checkpoint/reward.csv");

	if (!policyLossLog.is_open()) {
		std::println("Couldn't open checkpoint/loss.csv");
		return 1;
	}

	if (!rewardLog.is_open()) {
		std::println("Couldn't open checkpoint/reward.csv");
		return 1;
	}

	while (time_step <= max_training_timesteps) {
		auto state = imageToTensor(env.reset());
		auto current_ep_reward = 0.0f;

		for (uint t = 0; t < max_ep_len; t++) {
			auto action = ppo.select_action(state);
			float steer = action[0][0].item<float>();
			float gas = action[0][1].item<float>();
			float brake = action[0][2].item<float>();

			auto [stateImage, reward, done] = env.step({steer, gas, brake});
			state = imageToTensor(stateImage);

			ppo.buffer.rewards.push_back(reward);
			ppo.buffer.is_terminals.push_back(done);

			time_step++;
			current_ep_reward += reward;

			if (time_step % update_timestep == 0) {
				auto loss = ppo.update();
				policyLossLog << time_step << ',' << loss << '\n' << std::flush;
			}

			if (time_step % action_std_decay_freq == 0)
				ppo.decay_action_std(action_std_decay_rate, action_std_min);

			if (time_step % print_freq == 0) {
				auto print_avg_reward =
					print_running_reward / print_running_episodes;
				std::println("Episode : {} \t\t Timestep : {} \t\t Average "
							 "Reward Per Episode : {}",
							 i_episode, time_step, print_avg_reward);
				if (print_running_reward > best_running_reward) {
					std::string base = "checkpoint/";
					std::println(
						"Saving best to best_model.pt,best_opt.pt to {}", base);
					torch::save(ppo.policy, base + "best_model.pt");
					torch::save(ppo.opt, base + "best_opt.pt");
					best_running_reward = print_running_reward;
				}

				print_running_reward = 0;
				print_running_episodes = 0;
			}

			if (time_step % save_freq == 0) {
				std::string base = "checkpoint/";
				std::println("Saving model.pt,opt.pt to {}", base);
				torch::save(ppo.policy, base + "model.pt");
				torch::save(ppo.opt, base + "opt.pt");
			}

			if (done)
				break;
		}
		print_running_reward += current_ep_reward;
		print_running_episodes++;
		i_episode++;
	}

	policyLossLog.close();
	rewardLog.close();
	return 0;
}
