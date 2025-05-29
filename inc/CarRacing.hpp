#pragma once
#include <OpenGL/OpenGL.h>
#include <vector>
#include "Car.hpp"
#include "SFML/Graphics/ConvexShape.hpp"
#include "SFML/Graphics/Image.hpp"
#include "SFML/Graphics/RectangleShape.hpp"
#include "SFML/Graphics/RenderStates.hpp"
#include "SFML/Graphics/RenderTarget.hpp"
#include "SFML/Graphics/RenderTexture.hpp"
#include "SFML/Graphics/RenderWindow.hpp"
#include "SFML/Graphics/Texture.hpp"
#include "SFML/Graphics/VertexBuffer.hpp"
#include "box2d/id.h"
#include "RoadSegment.hpp"

constexpr int STATE_W = 96;  // less than Atari 160x192
constexpr int STATE_H = 96;
constexpr int VIDEO_W = 600;
constexpr int VIDEO_H = 400;
constexpr int WINDOW_W = 1000;
constexpr int WINDOW_H = 800;

constexpr float SCALE = 6.0;  // Track scale
constexpr float TRACK_RAD = 900 / SCALE;  // Track is heavily morphed circle with this radius
constexpr float PLAYFIELD = 2000 / SCALE;  // Game over boundary
constexpr int FPS = 50;  // Frames per second
constexpr float ZOOM = 2.7;  // Camera zoom
constexpr bool ZOOM_FOLLOW = true;  // Set to False for fixed view (don't use zoom)

constexpr float TRACK_DETAIL_STEP = 21 / SCALE;
constexpr float TRACK_TURN_RATE = 0.31;
constexpr float TRACK_WIDTH = 40 / SCALE;
constexpr float BORDER = 8 / SCALE;
constexpr int BORDER_MIN_COUNT = 4;

constexpr std::array<float, 3> ROAD_COLOR = {0.4, 0.4, 0.4};
constexpr sf::Color BG_COLOR(102, 204, 102);
constexpr sf::Color GRASS_COLOR(102, 230, 102);

using Observation = sf::Image;
using Action = std::array<float, 3>;

struct CarRacing {
	int np_random, seed;
	float reward, prevReward;
	int tileVisitedCount;
	float t;
	std::vector<RoadSegment> road;
	std::vector<sf::ConvexShape> roadPoly;
	b2WorldId world = b2_nullWorldId;
	Car car;
	float start_alpha;
	std::vector<std::tuple<float, float, float, float>> track;
	bool verbose = true;
	sf::RenderTexture stateTexture;

	// Render util
	sf::VertexBuffer roadPolyVBO;
	sf::ConvexShape wheelShape, hullShape1, hullShape2, hullShape3, hullShape4;
	sf::RectangleShape grassRect;

	// TODO: This constructor is clunky
	CarRacing(bool verbose = true);
	void setSeed();
	bool create_track();
	sf::Image reset();
	std::tuple<Observation, float, bool> step(Action action);

	// TODO: Not a 100% replica of the render, but should be enough to train
	// a model.
	void render(sf::RenderTarget& target);
	void render_car(sf::RenderTarget& target, const sf::RenderStates& states, sf::Vector2f hullPos, float hullAngle);
	void render_road(sf::RenderTarget& target, const sf::RenderStates& states);
	void render_indicators(sf::RenderTarget& target, float W, float H);
	void destroy();
	~CarRacing();
};
