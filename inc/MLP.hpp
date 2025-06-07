#pragma once
#include <vector>

using Func = void (*)(size_t n, const float *z, float *a);
using FuncBack = void (*)(size_t n, const float *z, const float *a, const float *dL_da, float* dL_dz);

struct ActivationFunction {
	Func func;
	FuncBack backprop;
};

void sigmoid(size_t n, const float *z, float *a);
void sigmoid_back(size_t n, const float *z, const float *a, const float *dL_da, float* dL_dz);
void relu(size_t n, const float *z, float *a);
void relu_back(size_t n, const float *z, const float *a, const float *dL_da, float* dL_dz);
void softmax(size_t n, const float *z, float *a);
void softmax_back(size_t n, const float *z, const float *a, const float *dL_da, float* dL_dz);

constexpr ActivationFunction Sigmoid = {sigmoid, sigmoid_back};
constexpr ActivationFunction ReLU = {relu, relu_back};
constexpr ActivationFunction Softmax = {softmax, softmax_back};

struct MLP {
	struct Layer {
		size_t inputSize, outputSize;
		// inputSize * outputSize
		float *weights;
		// outputSize
		float *biases;
		float *a;
		float *z;
		ActivationFunction activation;
	};

	std::vector<Layer> layers;
	size_t batchSize;

	// Buffers used in backprop
	// They're here to prevent reallocation on every backward pass
	float *dL_da, *dL_dz, *SSE;

	MLP(const size_t noLayers, const size_t *layerSizes,
		std::vector<ActivationFunction> aFunctions, size_t maxBatchSize = 1);
	~MLP();

	void initUniform();
	void initXavier();
	void initHe();
	// Assumes a batched input/output
	void forward(float *input, float *output, size_t miniBatchSize);
	// Assumes a batched input/output
	// Returns the loss
	float backprop(float *target, float *input, size_t miniBatchSize, float learningRate = 0.001f);
};
