#include "MLP.hpp"
#include <iostream>

// testWeights function initialize weights according to the Excel spreadsheets
// given to you so that it will be easy for you to debug/test your implementation.
void setTestWeights(MLP &model) {
	float weights1[2 * 3] = {0.1f, 0.2f, 0.3f, 1.1f, 1.2f, 1.3f};
	memcpy(model.layers[0].weights, weights1, sizeof(float) * 2 * 3);
	float bias1[3] = {-1.0f, -2.0f, -3.0f};
	memcpy(model.layers[0].biases, bias1, sizeof(float) * 3);

	float weights2[3 * 2] = {-0.1f, -0.2f, -0.3f, -0.4f, -0.5f, -0.6f};
	memcpy(model.layers[1].weights, weights2, sizeof(float) * 3 * 2);
	float bias2[2] = {-0.7f, -0.8f};
	memcpy(model.layers[1].biases, bias2, sizeof(float) * 2);
}

int main() {
	float X[] = {1.00f, 2.00f, 0.00f, 1.00f}; // 2 input, 2 numbers each
	float T[] = {0.25f, 0.75f, 0.50f, 0.25f}; // 2 targets, 2 numbers each
	float o[4];								  // output

	// Build ANN model
	size_t sizes[] = {2, 3, 2};
	std::vector<ActivationFunction> acts = {
		Sigmoid, Sigmoid}; // an array of function pointers
	MLP model(3, sizes, acts, 2);
	model.initWeights();
	setTestWeights(model);

	for (auto i = 0; i < 1000; ++i) {
		model.forward(X, o, 2);
		auto SSE = model.backprop(T, X, 2, 1.0f);
		std::cout << "\n"
				  << i << ":" << SSE << "[ " << o[0] << ", " << o[1] << ", "
				  << o[2] << ", " << o[3] << " ]" << std::flush;
	}
}
