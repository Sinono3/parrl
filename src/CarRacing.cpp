#include <SFML/Graphics/PrimitiveType.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <numbers>
#include <random>
#include "util.hpp"

#include "CarRacing.hpp"

std::default_random_engine rng;
b2WorldId createWorld() {
	b2WorldDef worldDef = b2DefaultWorldDef();
	worldDef.gravity = {0.0f, 0.0f};
	return b2CreateWorld(&worldDef);
}

CarRacing::CarRacing(bool verbose) : world(createWorld()), car(world, 0.0f, 0.0f, 0.0f), stateTexture(sf::Vector2u(STATE_W, STATE_H)) {
	setSeed();	
	// TODO: Viewer and invisible ...
	this->road.clear();
	this->reward = 0.0f;
	this->prevReward = 0.0f;
	this->verbose = verbose;
	// TODO: Action space
	// TODO: Observation space

    wheelShape = arrayToShape(WHEEL_POLY, SIZE);
    hullShape1 = arrayToShape(HULL_POLY1, SIZE);
    hullShape2 = arrayToShape(HULL_POLY2, SIZE);
    hullShape3 = arrayToShape(HULL_POLY3, SIZE);
    hullShape4 = arrayToShape(HULL_POLY4, SIZE);
    wheelShape.setFillColor(sf::Color::Black);
    hullShape1.setFillColor(sf::Color(204, 0, 0));
    hullShape2.setFillColor(sf::Color(204, 0, 0));
    hullShape3.setFillColor(sf::Color(204, 0, 0));
    hullShape4.setFillColor(sf::Color(204, 0, 0));
	grassRect.setPosition(sf::Vector2f(-PLAYFIELD, -PLAYFIELD));
	grassRect.setSize(sf::Vector2f(2.0f * PLAYFIELD, 2.0f * PLAYFIELD));
	grassRect.setFillColor(GRASS_COLOR);
}
void CarRacing::setSeed() {
	// TODO: Make it actually random
	np_random = seed = 123;
}

bool CarRacing::create_track() {
	constexpr int CHECKPOINTS = 12;
	constexpr float PI = static_cast<float>(std::numbers::pi);

	std::uniform_real_distribution noise_dist(0.0f, 2.0f * PI * 1.0f / CHECKPOINTS);
	std::uniform_real_distribution rad_dist(TRACK_RAD / 3.0f, TRACK_RAD);

	std::vector<std::tuple<float, float, float>> checkpoints;
	for (int c = 0; c < CHECKPOINTS; c++) {
		auto noise = noise_dist(rng);
		auto alpha = 2.0f * PI * c / CHECKPOINTS + noise;
		auto rad = rad_dist(rng);

		if (c == 0) {
			alpha = 0.0f;
			rad = 1.5f * TRACK_RAD;
		}
		if (c == CHECKPOINTS - 1) {
			alpha = 2.0f * PI * c / CHECKPOINTS;
			this->start_alpha = 2.0f * PI * (-0.5f)  / CHECKPOINTS;
			rad = 1.5f * TRACK_RAD;
		}

		checkpoints.push_back(std::make_tuple(alpha, rad * std::cos(alpha),
											  rad * std::sin(alpha)));
	}

	this->road.clear();

	float x = 1.5f * TRACK_RAD, y = 0.0f, beta = 0.0f;
	int dest_i = 0;
	int laps = 0;
	std::vector<std::tuple<float, float, float, float>> track;
	int no_freeze = 2500;
	bool visited_other_side = false;

	while (true) {
		float alpha = std::atan2(y, x);
		if (visited_other_side && alpha > 0.0f) {
			laps += 1;
			visited_other_side = false;
		}
		if (alpha < 0.0f) {
			visited_other_side = true;
			alpha += 2.0f * PI;
		}

		float dest_alpha, dest_x, dest_y;

		while(true) {
			bool failed = true;

			while (true) {
				auto [_dest_alpha, _dest_x, _dest_y] = checkpoints[dest_i % checkpoints.size()];
				dest_alpha = _dest_alpha;
				dest_x = _dest_x;
				dest_y = _dest_y;

				if (alpha <= dest_alpha) {
					failed = false;
					break;
				}

				dest_i += 1;
				if ((dest_i % checkpoints.size()) == 0) {
					break;
				}
			}

			if (!failed)
				break;

			alpha -= 2.0f * PI;
			continue;
		}

		auto r1x = std::cos(beta), r1y = std::sin(beta);
		auto p1x = -r1y, p1y = r1x;
		auto dest_dx = dest_x - x, dest_dy = dest_y - y;
		auto proj = r1x * dest_dx + r1y * dest_dy;

		while ((beta - alpha) > (1.5f * PI))
			beta -= 2.0f * PI;

		while ((beta - alpha) < (-1.5f * PI))
			beta += 2.0f * PI;

		auto prev_beta = beta;
		proj *= SCALE;
		if (proj > 0.3f)
			beta -= std::min(TRACK_TURN_RATE, std::abs(0.001f * proj));
		if (proj < -0.3f)
			beta += std::min(TRACK_TURN_RATE, std::abs(0.001f * proj));

		x += p1x * TRACK_DETAIL_STEP;
		y += p1y * TRACK_DETAIL_STEP;
		track.push_back(std::make_tuple(alpha, prev_beta * 0.5f + beta * 0.5f, x, y));
		if (laps > 4) {
			break;
		}
		no_freeze -= 1;
		if (no_freeze == 0) {
			break;
		}
	}

	auto i1 = -1, i2 = -1;
	auto i = track.size();

	while(true) {
		i -= 1;
		if (i == 0)
			return false;

		bool pass_through_start = (std::get<0>(track[i]) > this->start_alpha) &&
		                          (std::get<0>(track[(i - 1 + track.size()) % track.size()]) <= this->start_alpha);

		if (pass_through_start && (i2 == -1)) {
			i2 = i;
		}
		else if (pass_through_start && (i1 == -1)) {
			i1 = i;
			break;
		}
	}

       if (verbose)
           std::println("Track generation: {}..{} -> {}-tiles track", i1, i2, i2 - i1);
       
	assert(i1 != -1);
	assert(i2 != -1);

	// Crop track
	track = std::vector(track.begin() + i1, track.begin() + i2 + 1);
	
	// Check for well glued togetherness
	{
		auto first_beta = std::get<1>(track[0]);
		auto first_perp_x = std::cos(first_beta), first_perp_y = std::sin(first_beta);
		auto x = first_perp_x * (std::get<2>(track[0]) - std::get<2>(track[track.size()-1]));
		auto y = first_perp_y * (std::get<3>(track[0]) - std::get<3>(track[track.size()-1]));
		auto well_glued_together = std::sqrt(x * x + y * y);
		if (well_glued_together > TRACK_DETAIL_STEP)
			return false;
	}

	// TODO: Red white border on hard turns

	std::vector<sf::Vertex> roadPolyVertices;

	// Create tiles
	for (int i = 1; i < track.size(); i++) {
		auto [alpha1, beta1, x1, y1] = track[i];
		auto [alpha2, beta2, x2, y2] = track[(i - 1 + track.size()) % track.size()];
		std::array vertices = {
			b2Vec2{
				x1 - TRACK_WIDTH * std::cos(beta1),
				y1 - TRACK_WIDTH * std::sin(beta1),
			},
			b2Vec2{
				x1 + TRACK_WIDTH * std::cos(beta1),
				y1 + TRACK_WIDTH * std::sin(beta1),
			},
			b2Vec2{
				x2 + TRACK_WIDTH * std::cos(beta2),
				y2 + TRACK_WIDTH * std::sin(beta2),
			},
			b2Vec2{
				x2 - TRACK_WIDTH * std::cos(beta2),
				y2 - TRACK_WIDTH * std::sin(beta2),
			},
		};

		b2ShapeDef tileShapeDef = b2DefaultShapeDef();
		tileShapeDef.isSensor = true;
		b2Hull tileHull = b2ComputeHull(vertices.data(), vertices.size());
		b2Polygon tilePolygon = b2MakePolygon(&tileHull, 0.0f);

		b2BodyDef tileBodyDef = b2DefaultBodyDef();
		tileBodyDef.type = b2_staticBody;
		b2BodyId tileBody = b2CreateBody(world, &tileBodyDef);
		b2CreatePolygonShape(tileBody, &tileShapeDef, &tilePolygon) ;

		// Game
		RoadSegment t;
		t.body = tileBody;
		t.roadVisited = false;
		t.roadFriction = 1.0f;
		this->road.push_back(t);
		b2Body_SetUserData(tileBody, (void *)(road.size() - 1));

		// Graphical
		auto c = 0.01f * (i % 3);
		auto color = sf::Color(
		                       (ROAD_COLOR[0] + c) * 255,
		                       (ROAD_COLOR[1] + c) * 255,
		                       (ROAD_COLOR[2] + c) * 255
		                   );
		roadPolyVertices.push_back({{vertices[0].x, vertices[0].y}, color, {}});
		roadPolyVertices.push_back({{vertices[1].x, vertices[1].y}, color, {}});
		roadPolyVertices.push_back({{vertices[2].x, vertices[2].y}, color, {}});
		roadPolyVertices.push_back({{vertices[2].x, vertices[2].y}, color, {}});
		roadPolyVertices.push_back({{vertices[3].x, vertices[3].y}, color, {}});
		roadPolyVertices.push_back({{vertices[0].x, vertices[0].y}, color, {}});
	}

	this->roadPolyVBO.setPrimitiveType(sf::PrimitiveType::Triangles);
	this->roadPolyVBO.create(roadPolyVertices.size());
	this->roadPolyVBO.update(roadPolyVertices.data());
	this->track = track;
	return true;
}

sf::Image CarRacing::reset() {
	destroy();
	world = createWorld();

	reward = 0.0f;
	prevReward = 0.0f;
	tileVisitedCount = 0;
	t = 0.0f;
	roadPoly.clear();

	for (;;) {
		auto success = create_track();
		if (success) {
			break;
		} else {
			std::println("retry to generate track (normal if there are not many "
				   "instances of this message)");
		}
	}

	// Set car initial position
	// TODO: Do this correctly
	{
		auto [_, angle, x, y] = this->track[0];
		car.destroy();
		car = Car(world, angle, x, y);
	}
	
	this->render(stateTexture);
	sf::Image stateImage = stateTexture.getTexture().copyToImage();
	return stateImage;
}

std::tuple<Observation, float, bool> CarRacing::step(Action action) {;
	this->car.steer(-action[0]);
	this->car.gas(action[1]);
	this->car.brake(action[2]);

	this->car.step(1.0f / FPS, this->road);
	b2World_Step(this->world, 1.0f / FPS, 4);

	{
		auto events = b2World_GetSensorEvents(world);
		auto contact = [&](b2ShapeId shape_sensor, b2ShapeId shape_visitor, bool begin){
			auto body_sensor = b2Shape_GetBody(shape_sensor);
			auto body_visitor = b2Shape_GetBody(shape_visitor);
			long u1 = (long)b2Body_GetUserData(body_sensor);	// Tile idx
			long u2 = (long)b2Body_GetUserData(body_visitor);	// Wheel idx
			RoadSegment* tile = (RoadSegment*)(&road[u1]);
			Wheel* wheel = (Wheel*)(&car.wheels[u2]);
			
			// TODO: Change tile color on exploration
			
			if (begin) {
				wheel->tiles.insert(u1);
				if (!tile->roadVisited) {
					tile->roadVisited = true;
					this->reward += 1000.0 / this->track.size();
					this->tileVisitedCount += 1;
				}
			} else {
				wheel->tiles.erase(u1);
			}
		};

		for (auto i = 0; i < events.beginCount; i++) {
			auto& event = events.beginEvents[i];
			contact(event.sensorShapeId, event.visitorShapeId, true);
		}

		for (auto i = 0; i < events.endCount; i++) {
			auto& event = events.endEvents[i];
			contact(event.sensorShapeId, event.visitorShapeId, false);
		}
	}

	this->t += 1.0f / FPS;
	
	auto step_reward = 0.0f;
	auto done = false;

	this->reward -= 0.1f;
	this->car.fuelSpent = 0.0f;
	step_reward = this->reward - this->prevReward;
	
	// End if all track tiles were visited
	if (this->tileVisitedCount == this->track.size())
		done = true;

	// Check out of bounds access (outside PLAYFIELD)
	auto carPos = b2Body_GetPosition(this->car.hull);
	if ((std::abs(carPos.x) > PLAYFIELD) || (std::abs(carPos.y) > PLAYFIELD)) {
		done = true;
		step_reward = -100.0f;
	}

	// Render and create state buffer
	this->render(stateTexture);
	sf::Image stateImage = stateTexture.getTexture().copyToImage();
	return std::make_tuple(stateImage, step_reward, done);
}

// TODO: Not a 100% replica of the render, but should be enough to train
// a model.
void CarRacing::render(sf::RenderTarget& target) {
       auto hullPosb2 = b2Body_GetPosition(car.hull);
       auto hullPos = sf::Vector2f(hullPosb2.x, hullPosb2.y);
       auto hullAngle = b2Rot_GetAngle(b2Body_GetRotation(car.hull));

       auto zoom = 0.1f * SCALE * std::max(1.0f - this->t, 0.0f) + ZOOM * SCALE * std::min(this->t, 1.0f);
       auto scroll_x = hullPos.x, scroll_y = hullPos.y;
       auto angle = -hullAngle;
       auto vel = b2Body_GetLinearVelocity(car.hull);

       if (b2Length(vel) > 0.5f)
       	angle = std::atan2(vel.x, vel.y);
       sf::RenderStates states;
       states.transform = sf::Transform();
	states.transform.translate(sf::Vector2f(
		(WINDOW_W / 2.0f) - (scroll_x * zoom * std::cos(angle) -
							 scroll_y * zoom * std::sin(angle)),
		(WINDOW_H / 4.0f) - (scroll_x * zoom * std::sin(angle) +
							 scroll_y * zoom * std::cos(angle))));
       states.transform.scale(sf::Vector2f(zoom, zoom));
	states.transform.rotate(sf::radians(angle));
	
    auto size = target.getSize();
	sf::View gameView(
		sf::FloatRect(sf::Vector2f(0.0f, 0.0f),
					  sf::Vector2f((float)WINDOW_W, (float)WINDOW_H)));
	// TODO: Magic values
	gameView.setViewport(
		sf::FloatRect(sf::Vector2f(0.0f, 0.15f),
					  sf::Vector2f(1.0f, 0.85f)));

	target.clear();
	target.setView(gameView);
	render_road(target, states);
	render_car(target, states, hullPos, hullAngle);

	target.setView(target.getDefaultView());
       render_indicators(target, size.x, size.y);
}
void CarRacing::render_car(sf::RenderTarget& target, const sf::RenderStates& states, sf::Vector2f hullPos, float hullAngle) {
	hullShape1.setPosition(hullPos);
	hullShape2.setPosition(hullPos);
	hullShape3.setPosition(hullPos);
	hullShape4.setPosition(hullPos);
	hullShape1.setRotation(sf::radians(hullAngle));
	hullShape2.setRotation(sf::radians(hullAngle));
	hullShape3.setRotation(sf::radians(hullAngle));
	hullShape4.setRotation(sf::radians(hullAngle));
	target.draw(hullShape1, states);
	target.draw(hullShape2, states);
	target.draw(hullShape3, states);
	target.draw(hullShape4, states);

	for (const auto &wheel : car.wheels) {
		auto wheelPos = b2Body_GetPosition(wheel.body);
		auto wheelAngle = b2Rot_GetAngle(b2Body_GetRotation(wheel.body));
		wheelShape.setPosition(sf::Vector2f(wheelPos.x, wheelPos.y));
		wheelShape.setRotation(sf::radians(wheelAngle));
		target.draw(wheelShape, states);
	}
}
void CarRacing::render_road(sf::RenderTarget& target, const sf::RenderStates& states) {
	// Grass
	target.draw(grassRect, states);
	// TODO: Prettier grass
	target.draw(roadPolyVBO, states);
}
void CarRacing::render_indicators(sf::RenderTarget& target, float W, float H) {
	float s = W / 40.0;
	float h = H / 40.0;

	auto vertical_ind = [&](float place, float val, std::array<float, 3> color) {
		sf::RectangleShape shape;
		shape.setPosition(sf::Vector2f(place * s, h));
		shape.setSize(sf::Vector2f(s, h * val));
		shape.setFillColor(sf::Color((float)(color[0] * 255),
									 (float)(color[1] * 255),
									 (float)(color[2] * 255)));
		target.draw(shape);
	};
	auto horiz_ind = [&](float place, float val, std::array<float, 3> color) {
		sf::RectangleShape shape;
		shape.setPosition(sf::Vector2f(place * s, 2 * h));
		shape.setSize(sf::Vector2f(s * val, 2 * h));
		shape.setFillColor(sf::Color((float)(color[0] * 255),
									 (float)(color[1] * 255),
									 (float)(color[2] * 255)));
		target.draw(shape);
	};

	auto hullVel = b2Body_GetLinearVelocity(this->car.hull);
	float trueSpeed = b2Length(hullVel);
	auto hullAngularVel = b2Body_GetAngularVelocity(this->car.hull);
	auto carWheel0JointAngle = b2RevoluteJoint_GetAngle(this->car.wheels[0].joint);

	vertical_ind(5.0f, 0.02f * trueSpeed, {1, 1, 1});
	vertical_ind(7.0f, 0.01f * this->car.wheels[0].omega, {0.0, 0, 1});
	vertical_ind(8.0f, 0.01f * this->car.wheels[1].omega, {0.0, 0, 1});
	vertical_ind(9.0f, 0.01f * this->car.wheels[2].omega, {0.2, 0, 1});
	vertical_ind(10.0f, 0.01f * this->car.wheels[3].omega, {0.2, 0, 1});
	horiz_ind(20.0f, -10.0f * carWheel0JointAngle, {0.0f, 1, 0});
	horiz_ind(30.0f, -0.8f * hullAngularVel, {1.0f, 0, 0});
}

void CarRacing::destroy() {
	car.destroy();

	for (auto& segment : road) {
		if (B2_IS_NON_NULL(segment.body)) {
			b2DestroyBody(segment.body);
			segment.body = b2_nullBodyId;
		}
	}
	road.clear();

	if (B2_IS_NON_NULL(world)) {
		b2DestroyWorld(world);
		world = b2_nullWorldId;
	}
}

CarRacing::~CarRacing() {
	destroy();
}
