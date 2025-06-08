#pragma once
#include "CartpoleRenderer.hpp"
#include <concepts>
#include <print>

template <typename F>
concept ActionChooser = std::invocable<F, CartpoleObs>;

// Spawns a window and runs the agent with the specified policy until the window is closed
template <ActionChooser F> void testAgent(F policy, bool print = true) {
	// create the window
	sf::RenderWindow window(sf::VideoMode({600, 400}), "parrl",
							sf::Style::Default, sf::State::Windowed,
							sf::ContextSettings(32));
	window.setFramerateLimit(60);

	Cartpole env;
	CartpoleRenderer renderer;

	auto obs = env.reset();

	// run the main loop
	bool running = true;
	while (running) {
		// handle events
		while (const std::optional event = window.pollEvent()) {
			if (event->is<sf::Event::Closed>()) {
				// end the program
				running = false;
			}
		}

		auto action = policy(obs);
		auto step = env.step((CartpoleAction)action);
		obs = step.obs;

		if (print)
			std::println("act={} -> obs={}, reward={}, done={}", (int)action,
						 step.obs.vec, step.reward, step.done);

		if (step.done)
			env.reset();

		renderer.render(obs);
		renderer.drawToWindow(window);
		window.display();
	}
	window.close();
}
