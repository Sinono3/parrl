#include "Env.hpp"
#include <array>
#include <numbers>
#include <random>

constexpr float PI = static_cast<float>(std::numbers::pi);

enum class CartpoleAction { Left = 0, Right = 1 };
struct CartpoleObs {
	std::array<float, 4> vec;
};

std::mt19937 rng(42);
std::uniform_real_distribution initial_state_dist(-0.05, 0.05);

class Cartpole final : public Env<CartpoleObs, CartpoleAction> {
	std::array<float, 4> state = {};
	bool already_done = false;

  public:
	static constexpr float GRAVITY = 9.8f;
	static constexpr float MASSCART = 1.0f;
	static constexpr float MASSPOLE = 0.1f;
	static constexpr float TOTAL_MASS = MASSPOLE + MASSCART;
	static constexpr float LENGTH = 0.5f;
	static constexpr float POLEMASS_LENGTH = MASSPOLE + LENGTH;
	static constexpr float FORCE_MAG = 10.0f;
	static constexpr float TAU = 0.02f;
	static constexpr float THETA_THRESHOLD_RADIANS = 12.0f * 2.0f * PI / 360.0f;
	static constexpr float X_THRESHOLD = 2.4f;

	// Returns initial observation
	CartpoleObs reset() override {
		for (auto &x : state) {
			x = (float)initial_state_dist(rng);
		}
		return {.vec = state};
	}

	Step<CartpoleObs> step(CartpoleAction action) override {
		auto [x, x_dot, theta, theta_dot] = state;
		auto force = (action == CartpoleAction::Right) ? FORCE_MAG : -FORCE_MAG;
		auto costheta = std::cos(theta), sintheta = std::sin(theta);
		auto temp =
			(force + POLEMASS_LENGTH * (theta_dot * theta_dot) * sintheta) /
			TOTAL_MASS;
		auto thetaacc =
			(GRAVITY * sintheta - costheta * temp) /
			(LENGTH * (4.0f / 3.0f - MASSPOLE * costheta * costheta) /
			 TOTAL_MASS);
		auto xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS;

		// Integrate and update state
		x += TAU * x_dot;
		x_dot += TAU * xacc;
		theta += TAU * theta_dot;
		theta_dot += TAU * thetaacc;
		state = {x, x_dot, theta, theta_dot};

		// Check if done. Set up reward
		auto done = (x < -X_THRESHOLD) || (x > X_THRESHOLD) ||
					(theta < -THETA_THRESHOLD_RADIANS) ||
					(theta > THETA_THRESHOLD_RADIANS);
		auto reward = 0.0f;

		if (!done) {
			reward = 1.0f;
		} else if (!already_done) {
			reward = 1.0f;
		} else {
			reward = 0.0f;
		}
		already_done = done;

		return {
			.obs = {.vec = state},
			.reward = reward,
			.done = done,
		};
	}
};
