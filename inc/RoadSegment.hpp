#pragma once
#include "box2d/id.h"

struct RoadSegment {
	b2BodyId body;
	bool roadVisited;
	float roadFriction;
};

