#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>
#include <math.h>
#include <exception>
#include <vector>
#include <time.h>
#include <memory>
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <chrono>
#include <cassert>
using namespace std::chrono;

// The Shape struct is just a way to represent an x and y rectangular scale
struct Shape
{
    size_t x, y, z;
    
    
    Shape(size_t x = 1, size_t y = 1, size_t z = 1) : x(x), y(y), z(z) {}

};

// The matrix class contains methods to allocate itself to host and device, initialize, getting indices, and more.
class Matrix
{
private:
    bool device_allocated;
    bool host_allocated;

    void allocateHipMemory() {
        if (!device_allocated)
        {
            float *device_memory = nullptr;
            hipMalloc(&device_memory, shape.x * shape.y * shape.z * sizeof(float));
            data_device = std::shared_ptr<float>(device_memory,
                                                [&](float *ptr)
                                                { hipFree(ptr); });
            device_allocated = true;
        }
    }

    void allocateHostMemory() {
        if (!host_allocated)
        {
            data_host = std::shared_ptr<float>(new float[shape.x * shape.y * shape.z],
                                            [&](float *ptr)
                                            { delete[] ptr; });
            host_allocated = true;
        }
    }

public:
    Shape shape;

    std::shared_ptr<float> data_device;
    std::shared_ptr<float> data_host;

    Matrix(size_t x_dim = 1, size_t y_dim = 1, size_t z_dim = 1) :
        shape(x_dim, y_dim, z_dim), data_device(nullptr), data_host(nullptr),
        device_allocated(false), host_allocated(false)
    { }
    Matrix(Shape shape) : Matrix(shape.x, shape.y, shape.z)
    { }

    void allocateMemory() {
        allocateHipMemory();
        allocateHostMemory();
    }

    void allocateMemoryIfNotAllocated(Shape shape) {
        if (!device_allocated && !host_allocated) {
            this->shape = shape;
            allocateMemory();
        }
    }

    void copyHostToDevice() {
        if (device_allocated && host_allocated) {
            hipMemcpy(data_device.get(), data_host.get(), shape.x * shape.y * shape.z * sizeof(float), hipMemcpyHostToDevice);
        }
        else {
        }
    }
    void copyDeviceToHost() {
        if (device_allocated && host_allocated) {
            hipMemcpy(data_host.get(), data_device.get(), shape.x * shape.y * shape.z * sizeof(float), hipMemcpyDeviceToHost);
        }
        else {
        }
    }

    float &operator[](const int index) {
        return data_host.get()[index];
    }
    const float &operator[](const int index) const {
        return data_host.get()[index];
    }
};

// Generate the coordinates dataset in batches based on the number of batches and the batch size. These are saved as vectors of matrices representing a batch dataset.
class CoordinatesDataset
{
private:
    size_t batch_size;
    size_t number_of_batches;

    std::vector<Matrix> batches;
    std::vector<Matrix> targets;

public:
    CoordinatesDataset(size_t batch_size, size_t number_of_batches) : batch_size(batch_size), number_of_batches(number_of_batches)
    {
        for (int i = 0; i < number_of_batches; i++)
        {
            batches.push_back(Matrix(Shape(batch_size, 2)));
            targets.push_back(Matrix(Shape(batch_size, 1)));

            batches[i].allocateMemory();
            targets[i].allocateMemory();

            for (int k = 0; k < batch_size; k++)
            {
                batches[i][k] = static_cast<float>(rand()) / RAND_MAX - 0.5;
                batches[i][batches[i].shape.x + k] = static_cast<float>(rand()) / RAND_MAX - 0.5;
                ;

                if ((batches[i][k] > 0 && batches[i][batches[i].shape.x + k] > 0) ||
                    ((batches[i][k] < 0 && batches[i][batches[i].shape.x + k] < 0)))
                {
                    targets[i][k] = 1;
                }
                else
                {
                    targets[i][k] = 0;
                }
            }

            batches[i].copyHostToDevice();
            targets[i].copyHostToDevice();
        }
    }

    int getNumOfBatches() {
        return number_of_batches;
    }
    std::vector<Matrix> &getBatches() {
        return batches;
    }
    std::vector<Matrix> &getTargets() {
        return targets;
    }
};

// An abstract class that standardizes the parallel algorithms to have a standardized API / interface to run the neural net with
class NNLayer
{
protected:
    std::string name;

public:
    virtual ~NNLayer() = 0;

    virtual Matrix &forward(Matrix &A) = 0;
    virtual Matrix &backprop(Matrix &dZ, float learning_rate) = 0;

    std::string getName() { return this->name; };
};

inline NNLayer::~NNLayer() {}

__device__ float sigmoid(float x)
{
    return 1.0f / (1 + exp(-x));
}

__global__ void sigmoidActivationForward(float *Z, float *A,
                                         int Z_x_dim, int Z_y_dim)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim)
    {
        A[index] = sigmoid(Z[index]);
    }
}

__global__ void sigmoidActivationBackprop(float *Z, float *dA, float *dZ,
                                          int Z_x_dim, int Z_y_dim)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim)
    {
        dZ[index] = dA[index] * sigmoid(Z[index]) * (1 - sigmoid(Z[index]));
    }
}

// The sigmoid activation layer
class SigmoidActivation : public NNLayer
{
private:
    Matrix A;
    Matrix Z;
    Matrix dZ;

public:
    SigmoidActivation(std::string name) {
        this->name = name;
    }
    ~SigmoidActivation() {
    }

    Matrix &forward(Matrix &Z) {
        this->Z = Z;
        A.allocateMemoryIfNotAllocated(Z.shape);

        dim3 block_size(256);
        dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

        sigmoidActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(),
                                                                Z.shape.x, Z.shape.y);

        return A;
    }

    Matrix &backprop(Matrix &dA, float learning_rate = 0.01) {
        dZ.allocateMemoryIfNotAllocated(Z.shape);

        dim3 block_size(256);
        dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
        sigmoidActivationBackprop<<<num_of_blocks, block_size>>>(Z.data_device.get(), dA.data_device.get(),
                                                                dZ.data_device.get(),
                                                                Z.shape.x, Z.shape.y);

        return dZ;
    }
};


class ReLUActivation : public NNLayer
{
private:
    Matrix A;

    Matrix Z;
    Matrix dZ;

public:
    ReLUActivation(std::string name);
    ~ReLUActivation();

    Matrix &forward(Matrix &Z);
    Matrix &backprop(Matrix &dA, float learning_rate = 0.01);
};

__global__ void reluActivationForward(float *Z, float *A,
                                      int Z_x_dim, int Z_y_dim)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim)
    {
        A[index] = fmaxf(Z[index], 0);
    }
}

__global__ void reluActivationBackprop(float *Z, float *dA, float *dZ,
                                       int Z_x_dim, int Z_y_dim)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim)
    {
        if (Z[index] > 0)
        {
            dZ[index] = dA[index];
        }
        else
        {
            dZ[index] = 0;
        }
    }
}


ReLUActivation::ReLUActivation(std::string name) {
	this->name = name;
}

ReLUActivation::~ReLUActivation() { }

Matrix& ReLUActivation::forward(Matrix& Z) {
	this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	reluActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(),
														 Z.shape.x, Z.shape.y);

	return A;
}

Matrix& ReLUActivation::backprop(Matrix& dA, float learning_rate) {
	dZ.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	reluActivationBackprop<<<num_of_blocks, block_size>>>(Z.data_device.get(), dA.data_device.get(),
													      dZ.data_device.get(),
														  Z.shape.x, Z.shape.y);

	return dZ;
}

class LinearLayer : public NNLayer
{
private:
    const float weights_init_threshold = 0.01;

    Matrix W;
    Matrix b;

    Matrix Z;
    Matrix A;
    Matrix dA;

    void initializeBiasWithZeros();
    void initializeWeightsRandomly();

    void computeAndStoreBackpropError(Matrix &dZ);
    void computeAndStoreLayerOutput(Matrix &A);
    void updateWeights(Matrix &dZ, float learning_rate);
    void updateBias(Matrix &dZ, float learning_rate);

public:
    LinearLayer(std::string name, Shape W_shape);
    ~LinearLayer();

    Matrix &forward(Matrix &A);
    Matrix &backprop(Matrix &dZ, float learning_rate = 0.01);

    int getXDim() const;
    int getYDim() const;

    Matrix getWeightsMatrix() const;
    Matrix getBiasVector() const;
};

__global__ void linearLayerForward(float *W, float *A, float *Z, float *b,
                                   int W_x_dim, int W_y_dim,
                                   int A_x_dim, int A_y_dim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int Z_x_dim = A_x_dim;
    int Z_y_dim = W_y_dim;

    float Z_value = 0;

    if (row < Z_y_dim && col < Z_x_dim)
    {
        for (int i = 0; i < W_x_dim; i++)
        {
            Z_value += W[row * W_x_dim + i] * A[i * A_x_dim + col];
        }
        Z[row * Z_x_dim + col] = Z_value + b[row];
    }
}

__global__ void linearLayerBackprop(float *W, float *dZ, float *dA,
                                    int W_x_dim, int W_y_dim,
                                    int dZ_x_dim, int dZ_y_dim)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // W is treated as transposed
    int dA_x_dim = dZ_x_dim;
    int dA_y_dim = W_x_dim;

    float dA_value = 0.0f;

    if (row < dA_y_dim && col < dA_x_dim)
    {
        for (int i = 0; i < W_y_dim; i++)
        {
            dA_value += W[i * W_x_dim + row] * dZ[i * dZ_x_dim + col];
        }
        dA[row * dA_x_dim + col] = dA_value;
    }
}

__global__ void linearLayerUpdateWeights(float *dZ, float *A, float *W,
                                         int dZ_x_dim, int dZ_y_dim,
                                         int A_x_dim, int A_y_dim,
                                         float learning_rate)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // A is treated as transposed
    int W_x_dim = A_y_dim;
    int W_y_dim = dZ_y_dim;

    float dW_value = 0.0f;

    if (row < W_y_dim && col < W_x_dim)
    {
        for (int i = 0; i < dZ_x_dim; i++)
        {
            dW_value += dZ[row * dZ_x_dim + i] * A[col * A_x_dim + i];
        }
        W[row * W_x_dim + col] = W[row * W_x_dim + col] - learning_rate * (dW_value / A_x_dim);
    }
}

__global__ void linearLayerUpdateBias(float *dZ, float *b,
                                      int dZ_x_dim, int dZ_y_dim,
                                      int b_x_dim,
                                      float learning_rate)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < dZ_x_dim * dZ_y_dim)
    {
        int dZ_x = index % dZ_x_dim;
        int dZ_y = index / dZ_x_dim;
        atomicAdd(&b[dZ_y], -learning_rate * (dZ[dZ_y * dZ_x_dim + dZ_x] / dZ_x_dim));
    }
}


LinearLayer::LinearLayer(std::string name, Shape W_shape) :
	W(W_shape), b(W_shape.y, 1)
{
	this->name = name;
	b.allocateMemory();
	W.allocateMemory();
	initializeBiasWithZeros();
	initializeWeightsRandomly();
}

LinearLayer::~LinearLayer()
{ }

void LinearLayer::initializeWeightsRandomly() {
	std::default_random_engine generator;
	std::normal_distribution<float> normal_distribution(0.0, 1.0);

	for (int x = 0; x < W.shape.x; x++) {
		for (int y = 0; y < W.shape.y; y++) {
			W[y * W.shape.x + x] = normal_distribution(generator) * weights_init_threshold;
		}
	}

	W.copyHostToDevice();
}

void LinearLayer::initializeBiasWithZeros() {
	for (int x = 0; x < b.shape.x; x++) {
		b[x] = 0;
	}

	b.copyHostToDevice();
}

Matrix& LinearLayer::forward(Matrix& A) {
	assert(W.shape.x == A.shape.y);

	this->A = A;
	Shape Z_shape(A.shape.x, W.shape.y);
	Z.allocateMemoryIfNotAllocated(Z_shape);

	computeAndStoreLayerOutput(A);

	return Z;
}

void LinearLayer::computeAndStoreLayerOutput(Matrix& A) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(Z.shape.x + block_size.x - 1) / block_size.x,
						(Z.shape.y + block_size.y - 1) / block_size.y);
	linearLayerForward<<<num_of_blocks, block_size>>>( W.data_device.get(),
													   A.data_device.get(),
													   Z.data_device.get(),
													   b.data_device.get(),
													   W.shape.x, W.shape.y,
													   A.shape.x, A.shape.y);
}

Matrix& LinearLayer::backprop(Matrix& dZ, float learning_rate) {
	dA.allocateMemoryIfNotAllocated(A.shape);

	computeAndStoreBackpropError(dZ);

	updateBias(dZ, learning_rate);

	updateWeights(dZ, learning_rate);

	return dA;
}

void LinearLayer::computeAndStoreBackpropError(Matrix& dZ) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(A.shape.x + block_size.x - 1) / block_size.x,
						(A.shape.y + block_size.y - 1) / block_size.y);
	linearLayerBackprop<<<num_of_blocks, block_size>>>( W.data_device.get(),
														dZ.data_device.get(),
														dA.data_device.get(),
														W.shape.x, W.shape.y,
														dZ.shape.x, dZ.shape.y);
}

void LinearLayer::updateWeights(Matrix& dZ, float learning_rate) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(W.shape.x + block_size.x - 1) / block_size.x,
						(W.shape.y + block_size.y - 1) / block_size.y);
	linearLayerUpdateWeights<<<num_of_blocks, block_size>>>(dZ.data_device.get(),
															A.data_device.get(),
															W.data_device.get(),
															dZ.shape.x, dZ.shape.y,
															A.shape.x, A.shape.y,
															learning_rate);
}

void LinearLayer::updateBias(Matrix& dZ, float learning_rate) {
	dim3 block_size(256);
	dim3 num_of_blocks( (dZ.shape.y * dZ.shape.x + block_size.x - 1) / block_size.x);
	linearLayerUpdateBias<<<num_of_blocks, block_size>>>(dZ.data_device.get(),
														 b.data_device.get(),
														 dZ.shape.x, dZ.shape.y,
														 b.shape.x, learning_rate);
}

int LinearLayer::getXDim() const {
	return W.shape.x;
}

int LinearLayer::getYDim() const {
	return W.shape.y;
}

Matrix LinearLayer::getWeightsMatrix() const {
	return W;
}

Matrix LinearLayer::getBiasVector() const {
	return b;
}

class ConvLayer : public NNLayer
{
private:
    const float weights_init_threshold = 0.01;

    Matrix W;
    Matrix b;

    Matrix Z;
    Matrix A;
    Matrix dA;
    int mask_dim;

    void initializeBiasWithZeros();
    void initializeWeightsRandomly();

    void computeAndStoreBackpropError(Matrix &dZ);
    void computeAndStoreLayerOutput(Matrix &A);
    void updateWeights(Matrix &dZ, float learning_rate);
    void updateBias(Matrix &dZ, float learning_rate);

public:
    ConvLayer(std::string name, Shape W_shape, int mask_dim = 3);
    ~ConvLayer();

    Matrix &forward(Matrix &A);
    Matrix &backprop(Matrix &dZ, float learning_rate = 0.01);

    int getXDim() const;
    int getYDim() const;
    int getZDim() const;

    Matrix getWeightsMatrix() const;
    Matrix getBiasVector() const;
};

__global__ void convLayerForward(float *W, float *A, float *Z, float *b,
                                   int W_x_dim, int W_y_dim, int W_z_dim,
                                   int A_x_dim, int A_y_dim,
                                   int mask_dim)
{
    // Calculate the global thread positions
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.z * blockDim.z + threadIdx.z;

    int mask_offset = mask_dim / 2;

    // Starting index for calculation
    int start_r = row - mask_offset;
    int start_c = col - mask_offset;

    int Z_x_dim = A_x_dim;
    int Z_y_dim = W_y_dim;

    // Temp value for accumulating the result
    int temp = 0;

    // With zero padding
    // Iterate over all the rows
    for (int i = 0; i < mask_dim; i++) {
        // Go over each column
        for (int j = 0; j < mask_dim; j++) {
            // Range check for rows and cols
            if ((start_r + i) >= 0 && (start_r + i) < Z_y_dim && (start_c + j) >= 0 && (start_c + j) < Z_x_dim) {
                // Accumulate result
                temp += A[(start_r + i) * Z_y_dim + (start_c + j)] *
                        W[j + Z_x_dim * (i + Z_y_dim * filter)];
            }
        }
    }
    
    // Write back the result
    Z[col + Z_x_dim * (row + Z_y_dim * filter)] = temp;
}

__global__ void convLayerBackprop(float *W, float *dZ, float *dA,
                                    int W_x_dim, int W_y_dim,
                                    int dZ_x_dim, int dZ_y_dim)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // W is treated as transposed
    int dA_x_dim = dZ_x_dim;
    int dA_y_dim = W_x_dim;

    float dA_value = 0.0f;

    if (row < dA_y_dim && col < dA_x_dim)
    {
        for (int i = 0; i < W_y_dim; i++)
        {
            dA_value += W[i * W_x_dim + row] * dZ[i * dZ_x_dim + col];
        }
        dA[row * dA_x_dim + col] = dA_value;
    }
}

__global__ void convLayerUpdateWeights(float *dZ, float *A, float *W,
                                         int dZ_x_dim, int dZ_y_dim,
                                         int A_x_dim, int A_y_dim,
                                         float learning_rate)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // A is treated as transposed
    int W_x_dim = A_y_dim;
    int W_y_dim = dZ_y_dim;

    float dW_value = 0.0f;

    if (row < W_y_dim && col < W_x_dim)
    {
        for (int i = 0; i < dZ_x_dim; i++)
        {
            dW_value += dZ[row * dZ_x_dim + i] * A[col * A_x_dim + i];
        }
        W[row * W_x_dim + col] = W[row * W_x_dim + col] - learning_rate * (dW_value / A_x_dim);
    }
}

__global__ void convLayerUpdateBias(float *dZ, float *b,
                                      int dZ_x_dim, int dZ_y_dim,
                                      int b_x_dim,
                                      float learning_rate)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < dZ_x_dim * dZ_y_dim)
    {
        int dZ_x = index % dZ_x_dim;
        int dZ_y = index / dZ_x_dim;
        atomicAdd(&b[dZ_y], -learning_rate * (dZ[dZ_y * dZ_x_dim + dZ_x] / dZ_x_dim));
    }
}


ConvLayer::ConvLayer(std::string name, Shape W_shape, int mask_dim) :
	W(W_shape), b(W_shape.y, 1)
{
	this->name = name;
    this->mask_dim = mask_dim;
	b.allocateMemory();
	W.allocateMemory();
	initializeBiasWithZeros();
	initializeWeightsRandomly();
}

ConvLayer::~ConvLayer()
{ }

void ConvLayer::initializeWeightsRandomly() {
	std::default_random_engine generator;
	std::normal_distribution<float> normal_distribution(0.0, 1.0);

	for (int x = 0; x < W.shape.x; x++) {
		for (int y = 0; y < W.shape.y; y++) {
			W[y * W.shape.x + x] = normal_distribution(generator) * weights_init_threshold;
		}
	}

	W.copyHostToDevice();
}

void ConvLayer::initializeBiasWithZeros() {
	for (int x = 0; x < b.shape.x; x++) {
		b[x] = 0;
	}

	b.copyHostToDevice();
}

Matrix& ConvLayer::forward(Matrix& A) {
	assert(W.shape.x == A.shape.y);

	this->A = A;
	Shape Z_shape(A.shape.x, W.shape.y);
	Z.allocateMemoryIfNotAllocated(Z_shape);

	computeAndStoreLayerOutput(A);

	return Z;
}

void ConvLayer::computeAndStoreLayerOutput(Matrix& A) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(Z.shape.x + block_size.x - 1) / block_size.x,
						(Z.shape.y + block_size.y - 1) / block_size.y,
                        (Z.shape.z + block_size.z - 1) / block_size.z );
	convLayerForward<<<num_of_blocks, block_size>>>( W.data_device.get(),
													   A.data_device.get(),
													   Z.data_device.get(),
													   b.data_device.get(),
													   W.shape.x, W.shape.y, W.shape.z,
													   A.shape.x, A.shape.y,
                                                       mask_dim);
}

Matrix& ConvLayer::backprop(Matrix& dZ, float learning_rate) {
	dA.allocateMemoryIfNotAllocated(A.shape);

	computeAndStoreBackpropError(dZ);

	updateBias(dZ, learning_rate);

	updateWeights(dZ, learning_rate);

	return dA;
}

void ConvLayer::computeAndStoreBackpropError(Matrix& dZ) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(A.shape.x + block_size.x - 1) / block_size.x,
						(A.shape.y + block_size.y - 1) / block_size.y);
	convLayerBackprop<<<num_of_blocks, block_size>>>( W.data_device.get(),
														dZ.data_device.get(),
														dA.data_device.get(),
														W.shape.x, W.shape.y,
														dZ.shape.x, dZ.shape.y);
}

void ConvLayer::updateWeights(Matrix& dZ, float learning_rate) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(W.shape.x + block_size.x - 1) / block_size.x,
						(W.shape.y + block_size.y - 1) / block_size.y);
	convLayerUpdateWeights<<<num_of_blocks, block_size>>>(dZ.data_device.get(),
															A.data_device.get(),
															W.data_device.get(),
															dZ.shape.x, dZ.shape.y,
															A.shape.x, A.shape.y,
															learning_rate);
}

void ConvLayer::updateBias(Matrix& dZ, float learning_rate) {
	dim3 block_size(256);
	dim3 num_of_blocks( (dZ.shape.y * dZ.shape.x + block_size.x - 1) / block_size.x);
	convLayerUpdateBias<<<num_of_blocks, block_size>>>(dZ.data_device.get(),
														 b.data_device.get(),
														 dZ.shape.x, dZ.shape.y,
														 b.shape.x, learning_rate);
}

int ConvLayer::getXDim() const {
	return W.shape.x;
}

int ConvLayer::getYDim() const {
	return W.shape.y;
}

int ConvLayer::getZDim() const {
    return W.shape.z;
}

Matrix ConvLayer::getWeightsMatrix() const {
	return W;
}

Matrix ConvLayer::getBiasVector() const {
	return b;
}


class BCECost
{
public:
    float cost(Matrix predictions, Matrix target);
    Matrix dCost(Matrix predictions, Matrix target, Matrix dY);
};

__global__ void binaryCrossEntropyCost(float *predictions, float *target,
                                       int size, float *cost)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        float partial_cost = target[index] * logf(predictions[index]) + (1.0f - target[index]) * logf(1.0f - predictions[index]);
        atomicAdd(cost, -partial_cost / size);
    }
}

__global__ void dBinaryCrossEntropyCost(float *predictions, float *target, float *dY,
                                        int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        dY[index] = -1.0 * (target[index] / predictions[index] - (1 - target[index]) / (1 - predictions[index]));
    }
}

float BCECost::cost(Matrix predictions, Matrix target)
{
    assert(predictions.shape.x == target.shape.x);

    float *cost;
    hipMallocManaged(&cost, sizeof(float));
    *cost = 0.0f;

    dim3 block_size(256);
    dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
    binaryCrossEntropyCost<<<num_of_blocks, block_size>>>(predictions.data_device.get(),
                                                          target.data_device.get(),
                                                          predictions.shape.x, cost);
    hipDeviceSynchronize();

    float cost_value = *cost;
    hipFree(cost);

    return cost_value;
}

Matrix BCECost::dCost(Matrix predictions, Matrix target, Matrix dY)
{
    assert(predictions.shape.x == target.shape.x);

    dim3 block_size(256);
    dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
    dBinaryCrossEntropyCost<<<num_of_blocks, block_size>>>(predictions.data_device.get(),
                                                           target.data_device.get(),
                                                           dY.data_device.get(),
                                                           predictions.shape.x);

    return dY;
}

class NeuralNetwork
{
private:
    std::vector<NNLayer *> layers;
    BCECost bce_cost;

    Matrix Y;
    Matrix dY;
    float learning_rate;

public:
    NeuralNetwork(float learning_rate = 0.01);
    ~NeuralNetwork();

    Matrix forward(Matrix X);
    void backprop(Matrix predictions, Matrix target);

    void addLayer(NNLayer *layer);
    std::vector<NNLayer *> getLayers() const;
};

NeuralNetwork::NeuralNetwork(float learning_rate) : learning_rate(learning_rate)
{
}

NeuralNetwork::~NeuralNetwork()
{
    for (auto layer : layers)
    {
        delete layer;
    }
}

void NeuralNetwork::addLayer(NNLayer *layer)
{
    this->layers.push_back(layer);
}

Matrix NeuralNetwork::forward(Matrix X)
{
    Matrix Z = X;

    for (auto layer : layers)
    {
        Z = layer->forward(Z);
    }

    Y = Z;
    return Y;
}

void NeuralNetwork::backprop(Matrix predictions, Matrix target)
{
    dY.allocateMemoryIfNotAllocated(predictions.shape);
    Matrix error = bce_cost.dCost(predictions, target, dY);

    for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++)
    {
        error = (*it)->backprop(error, learning_rate);
    }

    hipDeviceSynchronize();
}

std::vector<NNLayer *> NeuralNetwork::getLayers() const
{
    return layers;
}

float computeAccuracy(const Matrix &predictions, const Matrix &targets)
{
    int m = predictions.shape.x;
    int correct_predictions = 0;

    for (int i = 0; i < m; i++)
    {
        float prediction = predictions[i] > 0.5 ? 1 : 0;
        if (prediction == targets[i])
        {
            correct_predictions++;
        }
    }

    return static_cast<float>(correct_predictions) / m;
}

int main()
{
    srand(1234);

    
    for (int experiment = 2; experiment < 257; experiment *= 4) {
        CoordinatesDataset dataset(100, 21);
        BCECost bce_cost;

        NeuralNetwork nn;
        nn.addLayer(new LinearLayer("linear_1", Shape(2, experiment)));
        nn.addLayer(new ReLUActivation("relu_1"));
        nn.addLayer(new LinearLayer("linear_2", Shape(experiment, 1)));
        nn.addLayer(new SigmoidActivation("sigmoid_output"));
        
        auto start_time = steady_clock::now();
        auto end_time = steady_clock::now();

        auto forward_time = end_time - start_time;
        auto backprop_time = end_time - start_time;
        auto cost_time = end_time - start_time;
        auto all_time = end_time - start_time; 

        int epochs = 1001;

        // network training
        Matrix Y;
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float cost = 0.0;

            start_time = steady_clock::now();

            for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++)
            {
                Y = nn.forward(dataset.getBatches().at(batch));
                // nn.backprop(Y, dataset.getTargets().at(batch));
                // cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
            }

            end_time = steady_clock::now();
            forward_time += (end_time - start_time);

            
            start_time = steady_clock::now();

            for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++)
            {
                Y = nn.forward(dataset.getBatches().at(batch));
                nn.backprop(Y, dataset.getTargets().at(batch));
                // cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
            }

            end_time = steady_clock::now();
            backprop_time += (end_time - start_time);

            start_time = steady_clock::now();

            for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++)
            {
                Y = nn.forward(dataset.getBatches().at(batch));
                nn.backprop(Y, dataset.getTargets().at(batch));
                cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
            }

            end_time = steady_clock::now();
            all_time += (end_time - start_time);

            if (epoch % 100 == 0 && epoch != 0)
            {
                std::cout << "Epoch: " << epoch
                        << ", Cost: " << cost / dataset.getNumOfBatches()
                        << ", Forward time: " << duration_cast<milliseconds>(forward_time).count() / (epoch / 100)
                        << ", Backprop time: " << duration_cast<milliseconds>(backprop_time - forward_time).count()  / (epoch / 100)
                        << ", Cost time: " << duration_cast<milliseconds>(all_time - backprop_time).count()  / (epoch / 100)
                        << ", All time: " << duration_cast<milliseconds>(all_time).count() / (epoch / 100)
                        << std::endl;
            }
        }

        // compute accuracy
        Y = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
        Y.copyDeviceToHost();

        float accuracy = computeAccuracy(
            Y, dataset.getTargets().at(dataset.getNumOfBatches() - 1));
        std::cout << "Accuracy: " << accuracy 
                    << ", Forward average time: " << duration_cast<milliseconds>(forward_time).count() / (epochs / 100)
                    << ", Backprop average time: " << duration_cast<milliseconds>(backprop_time - forward_time).count() / (epochs / 100)
                    << ", Cost average time: " << duration_cast<milliseconds>(all_time - backprop_time).count() / (epochs / 100)
                    << ", All average time: " << duration_cast<milliseconds>(all_time).count() / (epochs / 100)
                    << std::endl;
    }

    return 0;
}
