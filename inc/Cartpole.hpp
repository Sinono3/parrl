#pragma once
#include "Env.hpp"
#include <array>
#include <numbers>

constexpr float PI = static_cast<float>(std::numbers::pi);

enum class CartpoleAction { Left = 0, Right = 1 };
struct CartpoleObs {
	std::array<float, 4> vec;
};

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
	CartpoleObs reset() override;
	Step<CartpoleObs> step(CartpoleAction action) override;
};
