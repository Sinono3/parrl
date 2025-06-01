#include "Cartpole.hpp"

Cartpole::Cartpole(unsigned long long seed)
	: rng((unsigned int)seed), initial_state_dist(-0.05f, 0.05f) {}

// Returns initial observation
CartpoleObs Cartpole::reset() {
	for (auto &x : state) {
		x = (float)initial_state_dist(rng);
	}
	return {.vec = state};
}

Step<CartpoleObs> Cartpole::step(CartpoleAction action) {
	auto [x, x_dot, theta, theta_dot] = state;
	auto force = (action == CartpoleAction::Right) ? FORCE_MAG : -FORCE_MAG;
	auto costheta = std::cos(theta), sintheta = std::sin(theta);
	auto temp = (force + POLEMASS_LENGTH * (theta_dot * theta_dot) * sintheta) /
				TOTAL_MASS;
	auto thetaacc =
		(GRAVITY * sintheta - costheta * temp) /
		(LENGTH * (4.0f / 3.0f - MASSPOLE * costheta * costheta) / TOTAL_MASS);
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
