
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <math.h>

#define M_SIZE 256
#define N_SIZE 256
#define K_SIZE 256

float* matrixGen(int m, int n);
float* matrixMultiCPU(float *matrixA, float *matrixB, int row_size, int k_size, int col_size);
float* matrixMultiGPU(float *matrixA, float *matrixB, int row_size, int k_size, int col_size);
bool matrixEqual(float *matrixA, float *matrixB, int row_size, int col_size);

__global__ void matMul_kernel(float *_A, float *_B, float *_C, int row_size, int k_size, int col_size) {
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int index = row * blockDim.x*gridDim.x + col;

	_C[index] = 0;
	if (index >= row_size * col_size) {

	}
	else
		for (int k = 0; k < k_size; k++) {
			_C[index] += _A[row*k_size + k] * _B[col + k * col_size];
		}
}

int main()
{
	float *matrixA = matrixGen(M_SIZE, K_SIZE);
	float *matrixB = matrixGen(K_SIZE, N_SIZE);
	std::cout << "gen" << std::endl;

	float *hostMatrix = matrixMultiCPU(matrixA, matrixB, M_SIZE, K_SIZE, N_SIZE);
	std::cout << "host calc is end!" << std::endl;
	float *gpuMatrix = matrixMultiGPU(matrixA, matrixB, M_SIZE, K_SIZE, N_SIZE);
	std::cout << "gpu calc is end!" << std::endl;

	if (matrixEqual(hostMatrix, gpuMatrix, M_SIZE, N_SIZE))
		std::cout << "Two Matrix is equal!" << std::endl;
	else {
		std::cout << "Two Matrix is not equal!" << std::endl;
	}


	delete[] matrixA; delete[] matrixB;
	return 0;
}

float* matrixGen(int m, int n) {
	float *newMatrix = new float[m*n];
#pragma omp parallel for
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			newMatrix[i * n + j] = (float)rand() / (float)RAND_MAX * 10.0;
		}
	}
	return newMatrix;
}
float* matrixMultiCPU(float *matrixA, float *matrixB, int row_size, int k_size, int col_size) {
	float *newMatrix = new float[row_size*col_size];

#pragma omp parallel for
	for (int i = 0; i < row_size; i++) {
		for (int j = 0; j < col_size; j++) {
			newMatrix[i * row_size + j] = 0.0;
			for (int k = 0; k < k_size; k++) {
				newMatrix[i * row_size + j] += matrixA[i * k_size + k] * matrixB[k * col_size + j];
			}
		}
	}
	return newMatrix;
}
bool matrixEqual(float *matrixA, float *matrixB, int row_size, int col_size) {
	int size = row_size * col_size;

	for (int i = 0; i < size; i++) {
		float diff = matrixA[i] - matrixB[i];
		if (fabs(diff) > 0.001) {
			std::cout << diff << std::endl;
			std::cout << "matrixA[" << i << "] : " << matrixA[i] << " != matrixB[" << i << "] : " << matrixB[i] << std::endl;
			return false;
		}
	}
	return true;
}
float* matrixMultiGPU(float *matrixA, float *matrixB, int row_size, int k_size, int col_size) {
	float *newMatrix = new float[row_size*col_size];

	float *dA = NULL, *dB = NULL, *dC = NULL;

	//device memoryAlloc
	cudaMalloc(&dA, sizeof(float)*row_size*k_size);
	cudaMalloc(&dB, sizeof(float)*col_size*k_size);
	cudaMalloc(&dC, sizeof(float)*row_size*col_size);

	//cpy matrix data h to d
	cudaMemcpy(dA, matrixA, sizeof(float)*row_size*k_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, matrixB, sizeof(float)*k_size*col_size, cudaMemcpyHostToDevice);

	dim3 blockDim(16, 16); //256 threads per block
	dim3 gridDim(col_size / 16, row_size / 16);

	matMul_kernel << <gridDim, blockDim >> > (dA, dB, dC, row_size, k_size, col_size);
	cudaThreadSynchronize();

	cudaMemcpy(newMatrix, dC, sizeof(float)*row_size*col_size, cudaMemcpyDeviceToHost);

	cudaFree(&dA); cudaFree(&dB); cudaFree(&dC);

	return newMatrix;
}