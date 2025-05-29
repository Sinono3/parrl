#pragma once
#include <ranges>
#include <tuple>
#include <algorithm>
#include <numbers>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <torch/torch.h>
#pragma clang diagnostic pop
#include <vector>

constexpr float PI = std::numbers::pi;

using VT = std::vector<torch::Tensor>;

// Source: https://stackoverflow.com/questions/74166710/how-can-i-copy-the-parameters-of-one-model-to-another-in-libtorch
template<typename M>
void copy_module(M& to, M& from) {
	torch::NoGradGuard _guard;
	{
		auto new_params = from->named_parameters();
		auto params = to->named_parameters(true /*recurse*/);
		auto buffers = to->named_buffers(true /*recurse*/);
		for (auto &val : new_params) {
			auto name = val.key();
			auto *t = params.find(name);
			if (t != nullptr) {
				t->copy_(val.value());
			} else {
				t = buffers.find(name);
				if (t != nullptr) {
					t->copy_(val.value());
				}
			}
		}
	}
}

struct RolloutBuffer {
	VT actions, states, logprobs, state_values;
	std::vector<float> rewards;
	// Simply a vector of `done`: whether this was last step of an episode
	std::vector<bool> is_terminals;

	void clear() {
		actions.clear();
		states.clear();
		logprobs.clear();
		rewards.clear();
		state_values.clear();
		is_terminals.clear();
	}
};

// Continuous action actor-critic model
class ActorCriticImpl : public torch::nn::Module {
	torch::nn::Sequential conv{nullptr};
	torch::nn::Sequential actor{nullptr};
	torch::nn::Sequential critic{nullptr};
	torch::Tensor action_var;
	int64_t action_dim;

	std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor state) {
		throw std::exception();
	}

public:
	ActorCriticImpl(std::vector<int64_t> input_shape, int64_t action_dim, float action_std_init) {
		this->action_dim = action_dim;
		conv = register_module(
			"conv",
			torch::nn::Sequential(
				torch::nn::Conv2d(torch::nn::Conv2dOptions(input_shape[0], 4, 8).stride(4)),
				torch::nn::ReLU(),
				torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 8, 8).stride(2)),
				torch::nn::ReLU(),
				torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 16, 8).stride(2)),
				torch::nn::ReLU()));

		torch::Tensor dummy_input =
			torch::zeros({1, input_shape[0], input_shape[1], input_shape[2]});
		torch::Tensor conv_out = conv->forward(dummy_input);
		int64_t flattened_size = conv_out.numel() / conv_out.size(0);

		action_var = register_buffer(
			"action_var",
			torch::full({action_dim}, action_std_init * action_std_init));
		actor = register_module(
			"actor",
			torch::nn::Sequential(
				torch::nn::Flatten(),
				torch::nn::Linear(flattened_size, 128),
				torch::nn::ReLU(),
				torch::nn::Linear(128, action_dim),
				torch::nn::Tanh()));
		critic = register_module(
			"critic",
			torch::nn::Sequential(torch::nn::Flatten(),
								  torch::nn::Linear(flattened_size, 128),
								  torch::nn::ReLU(),
								  torch::nn::Linear(128, 1)
		));
	}

	void set_action_std(float new_action_std) {
		action_var = torch::full({action_dim}, new_action_std * new_action_std);
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> act(torch::Tensor state) {
		auto x = this->conv->forward(state);
		auto mean = this->actor->forward(x);
		auto var = action_var;
		auto std = torch::sqrt(action_var);

		// (1) Normally, we would use torch.distributions.MultivariateNormal, but
		// that only exists in PyTorch, not in C++ libtorch. So we have to
		// manually do the distribution sampling.
		auto eps = torch::randn_like(mean);       // [B, D], ~N(0,1)
		auto action = mean + eps * std;

		// (2) Log-prob
		auto log_term = torch::log(2 * PI * var);    // [B, D]
		auto log_prob = -0.5 * (
		    ((action - mean).pow(2) / var) + log_term
		).sum(-1, true);                              // [B, 1]

		// (3) Get critic's value
		auto state_value = this->critic->forward(x);
		return {action.detach(), log_prob.detach(), state_value.detach()};
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluate(torch::Tensor state, torch::Tensor action) {
		auto x = this->conv->forward(state);
		auto mean = this->actor->forward(x);
		auto var = action_var;
		auto std = torch::sqrt(action_var);
		
		// (1) Logprob from action
		auto log_term = torch::log(2 * PI * var);
		auto log_prob = -0.5 * (
		    ((action - mean).pow(2) / var) + log_term
		).sum(-1, true);
			
		// (2) Get critic's value
		auto state_value = this->critic->forward(x);

		// (3) Entropy
		// H = 0.5 * sum( log(2 π e σ^2) )
		auto entropy = 0.5 * (torch::log(2 * PI * var) + 1).sum(-1);
		return {log_prob, state_value, entropy};
	}

	
};

TORCH_MODULE(ActorCritic);

using Shape = std::vector<int64_t>;
constexpr float ACTION_STD_INIT = 0.6f;

struct PPO {
	float action_std, gamma, eps_clip;
	int K_epochs;
	RolloutBuffer buffer;
	ActorCritic policy, policy_old;
	torch::optim::Adam opt;

	PPO(Shape state_dim, int64_t action_dim, float lr, float gamma, int K_epochs, float eps_clip, float action_std_init=ACTION_STD_INIT) :
		action_std(action_std_init),
		gamma(gamma),
		eps_clip(eps_clip),
		K_epochs(K_epochs),
		policy(ActorCritic(state_dim, action_dim, action_std_init)),
		policy_old(ActorCritic(state_dim, action_dim, action_std_init)),
		opt(torch::optim::Adam(this->policy->parameters()))
	{
		copy_module(this->policy_old, this->policy);
	}

	void set_action_std(float new_action_std) {
		this->action_std = new_action_std;
		this->policy->set_action_std(new_action_std);
	}

	void decay_action_std(float action_std_decay_rate, float min_action_std) {
		action_std = action_std - action_std_decay_rate;
		// TODO: Round
		if (action_std <= min_action_std)
			action_std = min_action_std;
		set_action_std(action_std);
	}

	torch::Tensor select_action(torch::Tensor state) {
		auto _guard = torch::NoGradGuard();
		auto [action, action_logprob, state_value] = this->policy_old->act(state);
		this->buffer.states.push_back(state.detach());
		this->buffer.actions.push_back(action.detach());
		this->buffer.logprobs.push_back(action_logprob.detach());
		this->buffer.state_values.push_back(state_value.detach());
		return action.detach();
	}

	// Returns the total loss
	float update() {
		std::vector<float> returnsVec;
		float discounted_reward = 0.0f;

		for (auto [reward, is_terminal] :
			 std::views::zip(std::views::reverse(buffer.rewards),
							 std::views::reverse(buffer.is_terminals))) {
			if (is_terminal)
				discounted_reward = 0.0f;
			discounted_reward = reward + (this->gamma * discounted_reward);
			returnsVec.insert(returnsVec.begin(), discounted_reward);
		}

		// Convert returns vector to torch::Tensor, and normalize
		auto returns = torch::tensor(returnsVec, torch::kF32);
		returns = (returns - returns.mean()) / (returns.std() + 1e-7);

		// Rollout buffer to tensor
		auto old_states = torch::squeeze(torch::row_stack(buffer.states));
		auto old_actions = torch::squeeze(torch::row_stack(buffer.actions));
		auto old_logprobs = torch::squeeze(torch::row_stack(buffer.logprobs));
		auto old_state_values = torch::squeeze(torch::row_stack(buffer.state_values));
		auto advantages = returns - old_state_values;

		auto total_loss = 0.0f;
		for (int epoch = 0; epoch < this->K_epochs; epoch++) {
			std::println("Optimizing on rollout... {}/{}", epoch, K_epochs);
			auto [logprobs, state_values, entropy] = this->policy->evaluate(old_states, old_actions);
			state_values = state_values.squeeze();

			// Perform the actual PPO optimization
			auto ratios = (logprobs - old_logprobs).exp();
			auto surr1 = ratios * advantages;
			auto surr2 = ratios.clamp(1 - this->eps_clip, 1 + this->eps_clip) *
						 advantages;
			auto loss = -torch::min(surr1, surr2) +
						0.5f * torch::mse_loss(state_values, returns) -
						0.01f * entropy;

			this->opt.zero_grad();
			auto loss_mean = loss.mean();
			loss_mean.backward();
			total_loss += loss_mean.item<float>();
			this->opt.step();
		}

		copy_module(this->policy_old, this->policy);
		buffer.clear();
		return total_loss;
	}
};
