#include "MLP.hpp"
#include <algorithm>
#include <cmath>
#include <print>
#include <random>
#include <string.h>

using namespace std;

// vector * matrix (NOT matrix * vector)
void mul_vec_mat(size_t rows, size_t cols, const float *vec, const float *mat,
				 float *out) {
	for (size_t j = 0; j < cols; j++)
		out[j] = 0.0f;

	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			out[j] += vec[i] * mat[i * cols + j];
		}
	}
}

// matrix * vector
void mul_mat_vec(size_t rows, size_t cols, const float *mat, const float *vec,
				 float *out) {
	// Approach: dot product of matrix rows with the vector
	for (size_t i = 0; i < rows; i++) {
		out[i] = 0.0f;
		for (size_t j = 0; j < cols; j++) {
			out[i] += vec[j] * mat[i * cols + j];
		}
	}
}

// Scale a matrix
void mul_mat_scalar(size_t m, size_t n, const float *mat, float scalar,
					float *out) {
	for (size_t i = 0; i < (m * n); i++) {
		out[i] = scalar * mat[i];
	}
}

void add_mat_mat(size_t m, size_t n, const float *mat_a, const float *mat_b,
				 float *out) {
	for (size_t i = 0; i < (m * n); i++) {
		out[i] = mat_a[i] + mat_b[i];
	}
}

void add_mat_mat_scalar(size_t m, size_t n, const float *mat_a,
						const float *mat_b, float scalar, float *out) {
	for (size_t i = 0; i < (m * n); i++) {
		out[i] = mat_a[i] + scalar * mat_b[i];
	}
}

// Scale a vector
void mul_vec_scalar(size_t cols, const float *vec, float scalar, float *out) {
	for (size_t i = 0; i < cols; i++) {
		out[i] = scalar * vec[i];
	}
}

void add_vec_vec(size_t n, const float *vec_a, const float *vec_b, float *out) {
	for (size_t i = 0; i < n; i++) {
		out[i] = vec_a[i] + vec_b[i];
	}
}

void sub_vec_vec(size_t n, const float *vec_a, const float *vec_b, float *out) {
	for (size_t i = 0; i < n; i++) {
		out[i] = vec_a[i] - vec_b[i];
	}
}

void add_vec_vec_scalar(size_t n, const float *vec_a, const float *vec_b,
						float scalar, float *out) {
	for (size_t i = 0; i < n; i++) {
		out[i] = vec_a[i] + scalar * vec_b[i];
	}
}

// Component-wise multiplication, a.k.a. hadamard product
void mul_vec_vec(size_t n, const float *vec_a, const float *vec_b, float *out) {
	for (size_t i = 0; i < n; i++) {
		out[i] = vec_a[i] * vec_b[i];
	}
}

// "outer" product of two row vectors, i.e. a^T * b
//
// vec_a and vec_b are expected to be row vectors.
// vec_a: (1, m)
// vec_b: (1, n)
// out: (m, n)
void vec_outer(size_t m, size_t n, const float *vec_a, const float *vec_b,
			   float *out) {
	for (size_t i = 0; i < m; i++)
		for (size_t j = 0; j < n; j++)
			out[i * n + j] = vec_a[i] * vec_b[j];
}

void print_mat(size_t rows, size_t cols, const float *mat) {
	std::println("");
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			std::print("{} ", mat[i * cols + j]);
		}
		std::println("");
	}
}

void print_vec(size_t n, const float *vec) { print_mat(1, n, vec); }

void sigmoid(size_t n, const float *x, float *y) {
	for (size_t i = 0; i < n; i++) {
		y[i] = 1.0f / (1.0f + exp(-x[i]));
	}
}

void sigmoid_d(size_t n, const float *x, float *y) {
	sigmoid(n, x, y);

	for (size_t i = 0; i < n; i++) {
		y[i] = y[i] * (1 - y[i]);
	}
}

void relu(size_t n, const float *x, float *y) {
	throw std::logic_error("Function not yet implemented");
}
void relu_d(size_t n, const float *x, float *y) {
	throw std::logic_error("Function not yet implemented");
}

void softmax(size_t n, const float *x, float *y) {
	throw std::logic_error("Function not yet implemented");
}
void softmax_d(size_t n, const float *x, float *y) {
	throw std::logic_error("Function not yet implemented");
}

// build your NN model based on the passed in parameters. Hopefully all memory allocation is done here.
MLP::MLP(const size_t noLayers, const size_t *layerSizes,
		 std::vector<ActivationFunction> acts, size_t batchSize) {
	this->layers.reserve(noLayers - 1);
	this->batchSize = batchSize;

	// We will find the max amount of neurons through all layer
	// to know how much memory we need for our backprop buffers.
	size_t maxNeurons = 1;

	for (size_t layerIdx = 0; layerIdx < (noLayers - 1); layerIdx++) {
		size_t inputSize = layerSizes[layerIdx];
		size_t outputSize = layerSizes[layerIdx + 1];

		Layer layer;
		layer.inputSize = inputSize;
		layer.outputSize = outputSize;
		layer.weights = new float[inputSize * outputSize];
		layer.biases = new float[outputSize];
		layer.z = new float[outputSize * batchSize];
		layer.a = new float[outputSize * batchSize];
		layer.activation = acts[layerIdx];
		this->layers.push_back(layer);

		maxNeurons = (maxNeurons > (outputSize * batchSize))
						 ? maxNeurons
						 : (outputSize * batchSize);
		maxNeurons = (maxNeurons > (inputSize * batchSize))
						 ? maxNeurons
						 : (inputSize * batchSize);
	}

	// Create backprop buffers
	this->dL_da = new float[maxNeurons];
	this->dL_dz = new float[maxNeurons];
	this->SSE = new float[batchSize];
}

// de-allocate all allocated memory (hsize_t: each new must comes with a delete)
MLP::~MLP() {
	for (size_t layerIdx = 0; layerIdx < layers.size(); layerIdx++) {
		delete[] layers[layerIdx].weights;
		delete[] layers[layerIdx].biases;
		delete[] layers[layerIdx].z;
		delete[] layers[layerIdx].a;
	}

	delete[] dL_da;
	delete[] dL_dz;
	delete[] SSE;
}

static std::mt19937 rng(1);
static std::uniform_real_distribution<float> uniformDist(-1.0f, 1.0f);

template <typename T> inline void fillWithUniform(T *a, size_t size) {
	for (size_t i = 0; i < size; i++)
		a[i] = uniformDist(rng);
}

void MLP::initWeights() {
	for (auto &layer : layers) {
		fillWithUniform(layer.weights, layer.inputSize * layer.outputSize);
		fillWithUniform(layer.biases, layer.outputSize);
	}
}

void layerForwardBatch(float *input, MLP::Layer &layer, size_t batchSize) {
#pragma omp parallel for schedule(static)
	for (size_t b = 0; b < batchSize; b++) {
		auto b_input = &input[b * layer.inputSize];
		auto b_z = &layer.z[b * layer.outputSize];
		auto b_a = &layer.a[b * layer.outputSize];

		mul_vec_mat(layer.inputSize, layer.outputSize, b_input, layer.weights,
					b_z);
		add_vec_vec(layer.outputSize, b_z, layer.biases, b_z);
		layer.activation.func(layer.outputSize, b_z, b_a);
	}
}

// implement the feed-forward process to do inferencing.
void MLP::forward(float *input, float *output, size_t miniBatchSize) {
	layerForwardBatch(input, this->layers[0], miniBatchSize);

	for (size_t i = 1; i < this->layers.size(); i++) {
		layerForwardBatch(this->layers[i - 1].a, this->layers[i],
						  miniBatchSize);
	}

	// Copy layer's `a` to output
	size_t lastIdx = this->layers.size() - 1;
	auto &lastLayer = this->layers[lastIdx];
	std::copy(&lastLayer.a[0],
			  &lastLayer.a[lastLayer.outputSize * miniBatchSize], output);
}

// TODO: implement the mini-batched back-propagate process to train your NN. With OpenMP!
float MLP::backprop(float *T, float *X, size_t miniBatchSize, float alpha) {
	auto batchedScale = 1.0f / (float)miniBatchSize;
	auto &lastLayer = this->layers[this->layers.size() - 1];

	// Calculate loss
	// TODO: Pre-allocate this

	for (size_t b = 0; b < miniBatchSize; b++) {
		auto offset = b * lastLayer.outputSize;
		auto b_output = &lastLayer.a[offset];
		auto b_T = &T[offset];
		auto &b_SSE = SSE[b];
		b_SSE = 0.0f;

		for (size_t i = 0; i < lastLayer.outputSize; i++) {
			auto diff = b_output[i] - b_T[i];
			b_SSE += 0.5f * (diff * diff);
		}
	}

	// Max neurons per layer
	auto dL_da = this->dL_da;
	auto dL_dz = this->dL_dz;

// We must set up dL_da for the last layer (the first in the iteration)
// because it depends on the loss function
#pragma omp parallel for schedule(static)
	for (size_t b = 0; b < miniBatchSize; b++) {
		auto offset = b * lastLayer.outputSize;
		auto b_a = &lastLayer.a[offset];
		auto b_t = &T[offset];
		auto b_dL_da = &dL_da[offset];
		sub_vec_vec(lastLayer.outputSize, b_a, b_t, b_dL_da);
	}

	for (int layerIdx = (int)(this->layers.size() - 1); layerIdx >= 0;
		 layerIdx--) {
		auto &curLayer = this->layers[(size_t)layerIdx];

		auto m = curLayer.inputSize;
		auto n = curLayer.outputSize;

#pragma omp parallel for schedule(static)
		for (size_t b = 0; b < miniBatchSize; b++) {
			auto offset = b * n;
			auto b_dL_dz = &dL_dz[offset];
			auto b_dL_da = &dL_da[offset];
			auto b_z = &curLayer.z[offset];

			curLayer.activation.derivative(n, b_z, b_dL_dz);
			for (size_t i = 0; i < n; i++) {
				b_dL_dz[i] = b_dL_da[i] * b_dL_dz[i];
			}
		}

		// Calculate gradient for the previous layer's output (setup for the next iteration)
		// (This will be used in the next iteration, except on the last one [thus the if statement])
		// NOTE: why this works: involves a bit of a mathematical/memory-contiguity hack
		if (layerIdx > 0) {
#pragma omp parallel for schedule(static)
			for (size_t b = 0; b < miniBatchSize; b++) {
				auto b_dL_dz = &dL_dz[b * n]; // <-- cur layer offset
				auto b_dL_da = &dL_da[b * m]; // <-- last layer offset

				mul_mat_vec(m, n, curLayer.weights, b_dL_dz, b_dL_da);
			}
		}

		// If we're on the first layer, check the input instead (there's no i-1 layer)
		auto prevLayerOutput =
			(layerIdx > 0) ? this->layers[(size_t)layerIdx - 1].a : X;

		// TODO: #pragma omp parallel for schedule(static)
		// We cannot modify parameters parallely. What can we do?
		// critical is too expensive. Atomic operations will likely be slower.
		// Maybe this part needs to just be serial.
		for (size_t b = 0; b < miniBatchSize; b++) {
			auto b_prevLayerOutput =
				&prevLayerOutput[b * m]; // <-- b * M, not N
			auto b_dL_dz = &dL_dz[b * n];

			// Performs outer product and gradient descent
			// Does what these did combined:
			// - `vec_outer(m, n, X, dL_dz, dL_dw);`
			// - `add_mat_mat_scalar(m, n, curLayer.weights, dL_dw, -alpha, curLayer.weights);`
			for (size_t i = 0; i < m; i++)
				for (size_t j = 0; j < n; j++)
					curLayer.weights[i * n + j] -=
						batchedScale * alpha *
						(b_prevLayerOutput[i] * b_dL_dz[j]);
		}

		// The previous if-else clauses are almost identical except for how we index `prevLayerOutput`
		// We must do this because X is provided as a float**, and our actual implementation uses
		// explicitly contiguous arrays (float*)
		// There's some template-trickery that can be done to prevent this code duplication, but
		// I think it's not necessary.

		// gradient of bias (dL_db) is just gradient of z (dL_dz)
		// TODO: #pragma omp parallel for schedule(static)
		// We cannot modify parameters parallely. What can we do?
		// `omp critical` is too expensive. `omp atomic` operations will likely be slower.
		// Maybe this part needs to just be serial.
		for (size_t b = 0; b < miniBatchSize; b++) {
			auto b_dL_dz = &dL_dz[b * n];
			add_vec_vec_scalar(n, curLayer.biases, b_dL_dz,
							   batchedScale * -alpha, curLayer.biases);
		}
	}

	// Calculate total SSE
	auto total_SSE = 0.0f;
	for (size_t b = 0; b < miniBatchSize; b++) {
		total_SSE += SSE[b];
	}
	return total_SSE;
}
