#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "Cartpole.hpp"
#include <print>

int main()
{
    // create the window
    Cartpole env;
    env.reset();

    constexpr int STEPS = 20000;

    sf::Clock clock; 
    clock.start();
    for (int i = 0; i < STEPS; i++) {
        // Assumes time taken is the same independent of the action.
        auto [state, step_reward, done] = env.step(CartpoleAction::Left);
        if (done)
            env.reset();
	}
	double sps = STEPS / clock.getElapsedTime().asSeconds();
	std::println("C++ version SPS: {}", sps);
}
