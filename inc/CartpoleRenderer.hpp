#pragma once
#include "Cartpole.hpp"
#include <SFML/Graphics.hpp>

class CartpoleRenderer {
	static constexpr unsigned int SCREEN_WIDTH = 600;
	static constexpr unsigned int SCREEN_HEIGHT = 400;
	static constexpr sf::Vector2u SCREEN_SIZE =
		sf::Vector2u(SCREEN_WIDTH, SCREEN_HEIGHT);

	sf::RenderTexture target;
	sf::Sprite sprite;

  public:
	CartpoleRenderer();
	// `target` should be a 600x400 texture/window
	void render(CartpoleObs &obs);
	void drawToWindow(sf::RenderWindow &window);
};
