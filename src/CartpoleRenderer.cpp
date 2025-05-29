#include "CartpoleRenderer.hpp"

CartpoleRenderer::CartpoleRenderer()
	: target(SCREEN_SIZE), sprite(target.getTexture()) {
	sf::View view;
	view.setCenter(
		sf::Vector2f((float)SCREEN_WIDTH / 2.0f, (float)SCREEN_HEIGHT / 2.0f));
	view.setSize(sf::Vector2f((float)SCREEN_WIDTH, -(float)SCREEN_HEIGHT));
	target.setView(view);
}

// `target` should be a 600x400 texture/window
void CartpoleRenderer::render(CartpoleObs &obs) {
	float world_width = Cartpole::X_THRESHOLD * 2.0f;
	float scale = SCREEN_WIDTH / world_width;
	float polewidth = 10.0f;
	float polelen = scale * (2.0f * Cartpole::LENGTH);
	float cartwidth = 50.0f, cartheight = 30.0f;

	target.clear(sf::Color::White);

	sf::ConvexShape shape(4);

	// Cart
	auto axleoffset = cartheight / 4.0f;
	auto cartx = obs.vec[0] * scale + SCREEN_WIDTH / 2.0f;
	auto carty = 100.0f;
	{
		auto l = -cartwidth / 2.0f, r = cartwidth / 2.0f, t = cartheight / 2.0f,
			 b = -cartheight / 2.0f;
		auto cart_coords = std::array<std::pair<float, float>, 4>{
			std::pair(l, b), std::pair(l, t), std::pair(r, t), std::pair(r, b)};

		for (size_t i = 0; i < 4; i++) {
			auto [cx, cy] = cart_coords[i];
			cx += cartx;
			cy += carty;
			shape.setPoint(i, sf::Vector2f(cx, cy));
		}
	}
	// TODO: Draw
	shape.setFillColor(sf::Color(0, 0, 0));
	target.draw(shape);

	// Pole
	auto [l, r, t, b] =
		std::array{-polewidth / 2.0f, polewidth / 2.0f,
				   polelen - polewidth / 2.0f, -polewidth / 2.0f};
	std::vector<sf::Vector2f> pole_coords;
	std::array<sf::Vector2f, 4> it = {sf::Vector2f(l, b), sf::Vector2f(l, t),
									  sf::Vector2f(r, t), sf::Vector2f(r, b)};
	for (size_t i = 0; i < 4; i++) {
		auto coord = it[i];
		coord = coord.rotatedBy(sf::radians(-obs.vec[2]));
		coord.x += cartx;
		coord.y += carty + axleoffset;
		shape.setPoint(i, coord);
	}
	shape.setFillColor(sf::Color(202, 152, 101));
	target.draw(shape);

	// Decorations (axle, hline)
	sf::CircleShape circle(polewidth / 2.0f);
	circle.setPosition(sf::Vector2f(cartx, carty + axleoffset));
	circle.setFillColor(sf::Color(129, 132, 203));

	sf::VertexArray line(sf::PrimitiveType::Lines, 2);
	line.append(sf::Vertex{.position = sf::Vector2f(0.0f, carty),
						   .color = sf::Color::Black});
	line.append(sf::Vertex{.position = sf::Vector2f(SCREEN_WIDTH, carty),
						   .color = sf::Color::Black});
	target.draw(line);
}

void CartpoleRenderer::drawToWindow(sf::RenderWindow &window) {
	sprite.setScale(sf::Vector2f(1.0f, -1.0f));
	sprite.setOrigin(sf::Vector2f(0.0f, (float)window.getSize().y));
	window.draw(sprite);
}
