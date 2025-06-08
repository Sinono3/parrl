#include "MLP.hpp"
#include <algorithm>
#include <cmath>
#include <print>
#include <random>
#include <stdexcept>
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
		float x = 0.0f;
		for (size_t j = 0; j < cols; j++) {
			x += vec[j] * mat[i * cols + j];
		}
		out[i] = x;
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

void sigmoid(size_t n, const float *z, float *a) {
	for (size_t i = 0; i < n; i++) {
		a[i] = 1.0f / (1.0f + exp(-z[i]));
	}
}

void sigmoid_back(size_t n, const float *z, const float *a, const float *dL_da,
			   float *dL_dz) {
	for (size_t i = 0; i < n; i++) {
		float a_i = a[i];
		dL_dz[i] = dL_da[i] * a_i * (1 - a_i);
	}
}

void relu(size_t n, const float *z, float *a) {
	for (size_t i = 0; i < n; i++) {
		a[i] = max(0.0f, z[i]);
	}
}

void relu_back(size_t n, const float *z, const float *a, const float *dL_da,
			float *dL_dz) {
	for (size_t i = 0; i < n; i++) {
		dL_dz[i] = dL_da[i] * (float)(z[i] > 0.0f);
	}
}

void softmax(size_t n, const float *z, float *a) {
	float sumexp = 0.0f;
	for (size_t i = 0; i < n; i++) {
		sumexp += exp(z[i]);
	}
	float logsumexp = log(sumexp);
	for (size_t i = 0; i < n; i++) {
		a[i] = exp(z[i] - logsumexp);
	}
}
void softmax_back(size_t n, const float *z, const float *a, const float *dL_da, float *dL_dz) {
	throw std::logic_error("not implemented");
}

void nop(size_t n, const float *z, float *a) {
	std::copy(z, z + n, a);
}
void nop_back(size_t n, const float *z, const float *a, const float *dL_da,
			   float *dL_dz) {
	std::copy(dL_da, dL_da + n, dL_dz);
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
static std::uniform_real_distribution<float> stdUniformDist(-1.0f, 1.0f);

template <typename T> inline void fillWithUniform(T *a, size_t size) {
	for (size_t i = 0; i < size; i++)
		a[i] = stdUniformDist(rng);
}

void MLP::initUniform() {
	for (auto &layer : layers) {
		fillWithUniform(layer.weights, layer.inputSize * layer.outputSize);
	}
}

template <typename T> inline void fillWithXavier(T *a, size_t n, size_t in, size_t out) {
	float bounds = std::sqrtf(6.0f / (float)(in + out));
	for (size_t i = 0; i < n; i++)
		a[i] = bounds * stdUniformDist(rng);
}

inline void fillWithZeros(float *a, size_t size) {
	std::fill(a, a + size, 0.0f);
}

void MLP::initXavier() {
	for (auto &layer : layers) {
		auto m = layer.inputSize;
		auto n = layer.outputSize;
		fillWithXavier(layer.weights, m * n, m, n);
		fillWithZeros(layer.biases, n);
	}
}


void MLP::initHe() {
	for (auto &layer : layers) {
		auto m = layer.inputSize;
		auto n = layer.outputSize;

        float std_dev = std::sqrtf(2.0f / (float)m);
        std::normal_distribution<float> normal(0.0f, std_dev);
        
        for (size_t i = 0; i < m * n; i++) {
            layer.weights[i] = normal(rng);
            layer.weights[i] = std::clamp(layer.weights[i], -5.0f, 5.0f);
        }
            
        fillWithZeros(layer.biases, n);
	}
}

void layerForwardBatch(float *input, MLP::Layer &layer, size_t b) {
	auto b_input = &input[b * layer.inputSize];
	auto b_z = &layer.z[b * layer.outputSize];
	auto b_a = &layer.a[b * layer.outputSize];

	mul_vec_mat(layer.inputSize, layer.outputSize, b_input, layer.weights,
				b_z);
	add_vec_vec(layer.outputSize, b_z, layer.biases, b_z);
	layer.activation.func(layer.outputSize, b_z, b_a);
}

// implement the feed-forward process to do inferencing.
void MLP::forward(float *input, float *output, size_t b) {
	layerForwardBatch(input, this->layers[0], b);

	for (size_t i = 1; i < this->layers.size(); i++) {
		layerForwardBatch(this->layers[i - 1].a, this->layers[i], b);
	}

	// Copy layer's `a` to output
	size_t lastIdx = this->layers.size() - 1;
	auto &lastLayer = this->layers[lastIdx];
	std::copy(&lastLayer.a[b * lastLayer.outputSize],
			  &lastLayer.a[(b + 1) * lastLayer.outputSize], output);
}

// Backpropagation and optimization all-in-one.
// Need to provide the loss.
void MLP::backward_optim(float* dL_dOUT, float *X, size_t miniBatchSize, float alpha) {
	auto batchedScale = 1.0f / (float)miniBatchSize;
	auto parameterUpdateScale = -batchedScale * alpha;
	auto dL_dz = this->dL_dz;

	// Max neurons per layer
	auto lastLayerIdx =(int)(this->layers.size() - 1);
	for (int layerIdx = lastLayerIdx; layerIdx >= 0; layerIdx--) {
		auto &curLayer = this->layers[(size_t)layerIdx];
		auto m = curLayer.inputSize;
		auto n = curLayer.outputSize;

		{		
			auto dL_da = (layerIdx == lastLayerIdx) ? dL_dOUT : this->dL_da;
			for (size_t b = 0; b < miniBatchSize; b++) {
				auto offset = b * n;
				auto b_dL_dz = &dL_dz[offset];
				auto b_dL_da = &dL_da[offset];
				auto b_z = &curLayer.z[offset];
				auto b_a = &curLayer.a[offset];
				curLayer.activation.backprop(n, b_z, b_a, b_dL_da, b_dL_dz);
			}
		}

		// Calculate gradient for the previous layer's output (setup for the next iteration)
		// (This will be used in the next iteration, except on the last one [thus the if statement])
		// NOTE: why this works: involves a bit of a mathematical/memory-contiguity hack
		if (layerIdx > 0) {
			for (size_t b = 0; b < miniBatchSize; b++) {
				auto b_dL_dz = &dL_dz[b * n]; // <-- cur layer offset
				auto b_dL_da = &dL_da[b * m]; // <-- last layer offset
				mul_mat_vec(m, n, curLayer.weights, b_dL_dz, b_dL_da);
			}
		}

		// If we're on the first layer, check the input instead (there's no i-1 layer)
		auto prevLayerOutput =
			(layerIdx > 0) ? this->layers[(size_t)layerIdx - 1].a : X;
		// UPDATE WEIGHTS
		for (size_t b = 0; b < miniBatchSize; b++) {
			auto b_prevLayerOutput =
				&prevLayerOutput[b * m]; // <-- b * M, not N
			auto b_dL_dz = &dL_dz[b * n];

			for (size_t i = 0; i < m; i++)
				for (size_t j = 0; j < n; j++)
					curLayer.weights[i * n + j] +=
						parameterUpdateScale *
						(b_prevLayerOutput[i] * b_dL_dz[j]);
		}
		// UPDATE BIASES
		for (size_t b = 0; b < miniBatchSize; b++) {
			auto b_dL_dz = &dL_dz[b * n];
			add_vec_vec_scalar(n, curLayer.biases, b_dL_dz,
							   parameterUpdateScale, curLayer.biases);
		}
	}
}
