#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Shape.hpp>
#include <SFML/Graphics/StencilMode.hpp>
#include <SFML/OpenGL.hpp>
#include <SFML/System/Angle.hpp>
#include <SFML/Window.hpp>
#include <SFML/Window/Keyboard.hpp>
#include <SFML/Window/Window.hpp>

#include "Cartpole.hpp"
#include "CartpoleRenderer.hpp"

int main() {
	// create the window
	sf::RenderWindow window(sf::VideoMode({600, 400}), "parrl",
							sf::Style::Default, sf::State::Windowed,
							sf::ContextSettings(32));
	window.setFramerateLimit(60);

	Cartpole env;
	env.reset();

	CartpoleRenderer renderer;

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

		CartpoleAction action = CartpoleAction::Right;
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A))
			action = CartpoleAction::Left;
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D))
			action = CartpoleAction::Right;

		auto [obs, reward, done] = env.step(action);
		std::println("act={} -> obs={}, reward={}, done={}", (int)action,
					 obs.vec, reward, done);

		if (done)
			env.reset();

		renderer.render(obs);
		renderer.drawToWindow(window);
		window.display();
	}
}
