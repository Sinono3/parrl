#pragma once
#include <vector>

using Func = void (*)(size_t n, const float *input, float *output);

struct ActivationFunction {
	Func func;
	Func derivative;
};

void sigmoid(size_t n, const float *in, float *out);
void sigmoid_d(size_t n, const float *in, float *out);
void relu(size_t n, const float *in, float *out);
void relu_d(size_t n, const float *in, float *out);
void softmax(size_t n, const float *in, float *out);
void softmax_d(size_t n, const float *in, float *out);

constexpr ActivationFunction Sigmoid = {sigmoid, sigmoid_d};
constexpr ActivationFunction ReLU = {relu, relu_d};
constexpr ActivationFunction Softmax = {softmax, softmax_d};

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

	void initWeights();
	// Assumes a batched input/output
	void forward(float *input, float *output, size_t miniBatchSize);
	// Assumes a batched input/output
	// Returns the loss
	float backprop(float *target, float *input, size_t miniBatchSize, float learningRate = 0.001f);
};
