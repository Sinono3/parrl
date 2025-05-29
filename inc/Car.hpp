#pragma once
#include "SFML/Graphics/Color.hpp"
#include "box2d/collision.h"
#include "box2d/id.h"
#include "box2d/types.h"
#include <algorithm>
#include <array>
#import <box2d/box2d.h>
#include <span>
#include <set>
#include <print>
#include <vector>
#include "RoadSegment.hpp"

const float SIZE = 0.02;
const float ENGINE_POWER = 100000000 * SIZE * SIZE;
const float WHEEL_MOMENT_OF_INERTIA = 4000 * SIZE * SIZE;
const float FRICTION_LIMIT =
	(1000000 * SIZE *
	 SIZE); // friction ~= mass ~= size^2 (calculated implicitly using density)
const float WHEEL_R = 27;
const float WHEEL_W = 14;
const std::array<b2Vec2, 4> WHEELPOS = {b2Vec2{-55, +80}, b2Vec2{+55, +80}, b2Vec2{-55, -82}, b2Vec2{+55, -82}};
const std::array<b2Vec2, 4> HULL_POLY1 = {
	b2Vec2{-60, +130}, b2Vec2{+60, +130}, b2Vec2{+60, +110}, b2Vec2{-60, +110}};
const std::array<b2Vec2, 4> HULL_POLY2 = {
	b2Vec2{-15, +120}, b2Vec2{+15, +120}, b2Vec2{+20, +20}, b2Vec2{-20, 20}};
const std::array<b2Vec2, 8> HULL_POLY3 = {
	b2Vec2{+25, +20}, b2Vec2{+50, -10}, b2Vec2{+50, -40}, b2Vec2{+20, -90},
	b2Vec2{-20, -90}, b2Vec2{-50, -40}, b2Vec2{-50, -10}, b2Vec2{-25, +20},
};
const b2Vec2 HULL_POLY4[4] = {
	{-50, -120}, {+50, -120}, {+50, -90}, {-50, -90}};
const sf::Color WHEEL_COLOR = {0, 0, 0};
const float WHEEL_WHITE[3] = {77, 77, 77};
const float MUD_COLOR[3] = {102, 102, 0};
const std::array<b2Vec2, 4> WHEEL_POLY = {
	b2Vec2{-WHEEL_W, +WHEEL_R},
	b2Vec2{+WHEEL_W, +WHEEL_R},
	b2Vec2{+WHEEL_W, -WHEEL_R},
	b2Vec2{-WHEEL_W, -WHEEL_R},
};

struct Wheel {
	b2BodyId body = b2_nullBodyId;
	float wheelRad;
	sf::Color color;
	float gas, brake, steer, phase, omega;
	b2JointId joint = b2_nullJointId;
	std::set<int> tiles;
};

struct Car {
	b2WorldId world = b2_nullWorldId;
	b2BodyId hull = b2_nullBodyId;
	std::vector<Wheel> wheels;
	float fuelSpent;

	Car(b2WorldId _world, float initAngle, float initX, float initY);
	void gas(float gas);
	void brake(float b);
	void steer(float s);
	void step(float dt, std::span<const RoadSegment> tiles);
	void render();
	void destroy();
};
