#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "CarRacing.hpp"

int main()
{
    // create the window
    CarRacing env;
    env.reset();

    constexpr int STEPS = 20000;

    sf::Clock clock; 
    clock.start();
    for (int i = 0; i < STEPS; i++) {
        auto [state, step_reward, done] = env.step({0.0f, 0.0f, 0.0f});
        if (done)
            env.reset();
	}
	double sps = STEPS / clock.getElapsedTime().asSeconds();
	std::println("C++ version SPS: {}", sps);
}
