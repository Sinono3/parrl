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
#include <box2d/box2d.h>
#include <box2d/math_functions.h>
#include <box2d/types.h>

#include "Cartpole.hpp"

int main() {
	// create the window
	constexpr unsigned int SCREEN_WIDTH = 600;
	constexpr unsigned int SCREEN_HEIGHT = 400;

	sf::RenderWindow window(sf::VideoMode({SCREEN_WIDTH, SCREEN_HEIGHT}),
							"parrl", sf::Style::Default, sf::State::Windowed,
							sf::ContextSettings(32));
	window.setFramerateLimit(60);
	sf::View view;
	view.setCenter(sf::Vector2f((float)SCREEN_WIDTH / 2.0f, (float)SCREEN_HEIGHT / 2.0f));
	view.setSize(sf::Vector2f((float)SCREEN_WIDTH, -(float)SCREEN_HEIGHT));
	window.setView(view);

	Cartpole env;
	env.reset();

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

		float world_width = Cartpole::X_THRESHOLD * 2.0f;
		float scale = SCREEN_WIDTH / world_width;
		float polewidth = 10.0f;
		float polelen = scale * (2.0f * Cartpole::LENGTH);
		float cartwidth = 50.0f, cartheight = 30.0f;

		window.clear(sf::Color::White);

		sf::ConvexShape shape(4);

		// Cart
		auto axleoffset = cartheight / 4.0f;
		auto cartx = obs.vec[0] * scale + SCREEN_WIDTH / 2.0f;
		auto carty = 100.0f;
		{
			auto l = -cartwidth / 2.0f, r = cartwidth / 2.0f,
				 t = cartheight / 2.0f, b = -cartheight / 2.0f;
			auto cart_coords = std::array<std::pair<float, float>, 4>{
				std::pair(l, b), std::pair(l, t), std::pair(r, t),
				std::pair(r, b)};

			for (size_t i = 0; i < 4; i++) {
				auto [cx, cy] = cart_coords[i];
				cx += cartx;
				cy += carty;
				shape.setPoint(i, sf::Vector2f(cx, cy));
			}
		}
		// TODO: Draw
		shape.setFillColor(sf::Color(0, 0, 0));
		window.draw(shape);

		// Pole
		auto [l, r, t, b] =
			std::array{-polewidth / 2.0f, polewidth / 2.0f,
					   polelen - polewidth / 2.0f, -polewidth / 2.0f};
		std::vector<sf::Vector2f> pole_coords;
		std::array<sf::Vector2f, 4> it = {
			sf::Vector2f(l, b), sf::Vector2f(l, t), sf::Vector2f(r, t),
			sf::Vector2f(r, b)};
		for (size_t i = 0; i < 4; i++) {
			auto coord = it[i];
			coord = coord.rotatedBy(sf::radians(-obs.vec[2]));
			coord.x += cartx;
			coord.y += carty + axleoffset;
			shape.setPoint(i, coord);
		}
		shape.setFillColor(sf::Color(202, 152, 101));
		window.draw(shape);

		// Decorations (axle, hline)
		sf::CircleShape circle(polewidth / 2.0f);
		circle.setPosition(sf::Vector2f(cartx, carty + axleoffset));
		circle.setFillColor(sf::Color(129, 132, 203));

		sf::VertexArray line(sf::PrimitiveType::Lines, 2);
		line.append(sf::Vertex{.position = sf::Vector2f(0.0f, carty),
							   .color = sf::Color::Black});
		line.append(sf::Vertex{.position = sf::Vector2f(SCREEN_WIDTH, carty),
							   .color = sf::Color::Black});
		window.draw(line);
		window.display();
	}
}
