#pragma once
#include "Cartpole.hpp"
#include <functional>

// Spawns a window and runs the agent with the specified policy until the window is closed
void testAgent(std::function<int(CartpoleObs)> policy, bool print = true);
