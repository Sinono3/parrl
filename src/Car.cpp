#include "Car.hpp"

inline float sign(float x) {
	return (x > 0.0f) - (x < 0.0f);
}

Car::Car(b2WorldId _world, float initAngle, float initX, float initY) {
	world = _world;
	fuelSpent = 0.0f;

	b2BodyDef hullDef = b2DefaultBodyDef();
	hullDef.type = b2_dynamicBody;
	hullDef.position = (b2Vec2){initX, initY};
	hullDef.rotation = b2MakeRot(initAngle);
	hull = b2CreateBody(world, &hullDef);

	std::array<b2Vec2, 4> hull1_poly, hull2_poly, hull4_poly, wheel_poly;
	std::array<b2Vec2, 8> hull3_poly;
	std::transform(std::begin(HULL_POLY1), std::end(HULL_POLY1), hull1_poly.begin(), [](b2Vec2 v) { return SIZE * v; });
	std::transform(std::begin(HULL_POLY2), std::end(HULL_POLY2), hull2_poly.begin(), [](b2Vec2 v) { return SIZE * v; });
	std::transform(std::begin(HULL_POLY3), std::end(HULL_POLY3), hull3_poly.begin(), [](b2Vec2 v) { return SIZE * v; });
	std::transform(std::begin(HULL_POLY4), std::end(HULL_POLY4), hull4_poly.begin(), [](b2Vec2 v) { return SIZE * v; });
	std::transform(std::begin(WHEEL_POLY), std::end(WHEEL_POLY), wheel_poly.begin(), [](b2Vec2 v) { return SIZE * v; });

	b2Hull hull1_h = b2ComputeHull(hull1_poly.cbegin(), 4);
	b2Polygon hull1_p = b2MakePolygon(&hull1_h, 0.0f);
	b2ShapeDef hull1_d = b2DefaultShapeDef();
	hull1_d.density = 1.0f;
	b2CreatePolygonShape(hull, &hull1_d, &hull1_p);

	b2Hull hull2_h = b2ComputeHull(hull2_poly.cbegin(), 4);
	b2Polygon hull2_p = b2MakePolygon(&hull2_h, 0.0f);
	b2ShapeDef hull2_d = b2DefaultShapeDef();
	hull2_d.density = 1.0f;
	b2CreatePolygonShape(hull, &hull2_d, &hull2_p);

	b2Hull hull3_h = b2ComputeHull(hull3_poly.cbegin(), 8);
	b2Polygon hull3_p = b2MakePolygon(&hull3_h, 0.0f);
	b2ShapeDef hull3_d = b2DefaultShapeDef();
	hull3_d.density = 1.0f;
	b2CreatePolygonShape(hull, &hull3_d, &hull3_p);

	b2Hull hull4_h = b2ComputeHull(hull4_poly.cbegin(), 4);
	b2Polygon hull4_p = b2MakePolygon(&hull4_h, 0.0f);
	b2ShapeDef hull4_d = b2DefaultShapeDef();
	hull4_d.density = 1.0f;
	b2CreatePolygonShape(hull, &hull4_d, &hull4_p);
	
	for (auto wheelPos : WHEELPOS) {
		b2BodyDef wheelDef = b2DefaultBodyDef();
		wheelDef.position = {initX + wheelPos.x * SIZE, initY + wheelPos.y * SIZE};
		wheelDef.rotation = b2MakeRot(initAngle);
		wheelDef.type = b2_dynamicBody;

		Wheel wheel;
		wheel.body = b2CreateBody(world, &wheelDef);

		b2Hull wheel_h = b2ComputeHull(wheel_poly.cbegin(), 4);
		b2Polygon wheel_p = b2MakePolygon(&wheel_h, 0.0f);
		b2ShapeDef wheel_d = b2DefaultShapeDef();
		wheel_d.density = 0.1f;
		wheel_d.material.restitution = 0.0f;
		b2Filter filter = b2DefaultFilter();
		filter.categoryBits = 0x0020;
		filter.maskBits = 0x001;
		wheel_d.filter = filter;
		b2CreatePolygonShape(wheel.body, &wheel_d, &wheel_p);
		
		wheel.wheelRad =  WHEEL_R * SIZE;
		wheel.color = WHEEL_COLOR;
		wheel.gas = 0.0f;
		wheel.brake = 0.0f;
		wheel.steer = 0.0f;
		wheel.phase = 0.0f;
		wheel.omega = 0.0f;
		//TODO wheel.skid_start, wheel.skid_particle
		
		auto rjd = b2DefaultRevoluteJointDef();
		rjd.bodyIdA = hull;
		rjd.bodyIdB = wheel.body;
		rjd.localAnchorA = {wheelPos.x * SIZE, wheelPos.y * SIZE};
		rjd.localAnchorB = {0.0f, 0.0f};
		rjd.enableMotor = true;
		rjd.enableLimit = true;
		rjd.maxMotorTorque = 180.0f * 900.0f * SIZE * SIZE;
		rjd.motorSpeed = 0.0f;
		rjd.lowerAngle = -0.4f;
		rjd.upperAngle = +0.4f;
		wheel.joint = b2CreateRevoluteJoint(world, &rjd);
		wheel.tiles = std::set<int>();
		wheels.push_back(wheel);
		b2Body_SetUserData(wheel.body, (void*)(wheels.size() - 1));
	}
}

void Car::gas(float gas) {
	gas = std::clamp(gas, 0.0f, 1.0f);
	for (int i = 2; i < 4; i++) {
		auto& w = wheels[i];
		auto diff = gas - w.gas;
		if (diff > 0.1f) 
			diff = 0.1f;
		w.gas += diff;
	}
}

void Car::brake(float b) {
	for (auto& w : wheels)
		w.brake = b;
}

void Car::steer(float s) {
	wheels[0].steer = s;
	wheels[1].steer = s;
}

void Car::step(float dt, std::span<const RoadSegment> tiles) {
	for (auto& w : wheels) {
		auto diff = (w.steer - b2RevoluteJoint_GetAngle(w.joint));
		auto dir = sign(diff);
		auto val = std::abs(diff);

		b2RevoluteJoint_SetMotorSpeed(w.joint, dir * std::min(50.0f * val, 3.0f));

		// Change friction depending on whether wheel is on grass or not
		auto grass = true;
		auto frictionLimit = FRICTION_LIMIT * 0.6f;
		for (auto tileIdx : w.tiles) {
			const auto& tile = tiles[tileIdx];
			frictionLimit = std::max(frictionLimit, FRICTION_LIMIT * tile.roadFriction);
			grass = false;
		}

		auto forw = b2Body_GetWorldVector(w.body, {0, 1});
		auto side = b2Body_GetWorldVector(w.body, {1, 0});
		auto v = b2Body_GetLinearVelocity(w.body);
		auto vf = forw.x * v.x + forw.y * v.y;
		auto vs = side.x * v.x + side.y * v.y;

		w.omega += dt * ENGINE_POWER * w.gas / WHEEL_MOMENT_OF_INERTIA / (std::abs(w.omega) + 5.0f);
		fuelSpent += dt * ENGINE_POWER * w.gas;

		if (w.brake >= 0.9f) {
			w.omega = 0.0f;
		} else if (w.brake > 0.0f) {
			constexpr float BRAKE_FORCE = 15;
			dir = -sign(w.omega);
			val = BRAKE_FORCE * w.brake;
			if (std::abs(val) > std::abs(w.omega)) {
				val = std::abs(w.omega);
			}
			w.omega += dir * val;
		}
		w.phase += w.omega * dt;

		auto vr = w.omega * w.wheelRad;
		auto f_force = -vf + vr;
		auto p_force = -vs;

		f_force *= 205000 * SIZE * SIZE;
		p_force *= 205000 * SIZE * SIZE;
		auto force = std::sqrt(f_force * f_force + p_force * p_force);
		
		// TODO: Skid trace

		if (std::abs(force) > frictionLimit) {
			f_force /= force;
			p_force /= force;
			force = frictionLimit;
			f_force *= force;
			p_force *= force;
		}

		w.omega -= dt * f_force * w.wheelRad / WHEEL_MOMENT_OF_INERTIA;

		b2Vec2 forceVector = {
			p_force * side.x + f_force * forw.x,
			p_force * side.y + f_force * forw.y,
		};
		b2Body_ApplyForceToCenter(w.body, forceVector, true);
	}
}

void Car::render() {
	// TODO
}

void Car::destroy() {
	if (B2_IS_NON_NULL(hull)) {
		b2DestroyBody(hull);
		hull = b2_nullBodyId;
	}

	for (auto& wheel : wheels) {
		if (B2_IS_NON_NULL(wheel.body)) {
			b2DestroyBody(wheel.body);
			wheel.body = b2_nullBodyId;
			wheel.joint = b2_nullJointId;
		}
	}
	wheels.clear();
}
