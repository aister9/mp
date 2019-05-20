#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cuda.h"
#include "DS_timer.h"
#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <math.h>

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#define M_SIZE 1024
#define N_SIZE 1024
#define K_SIZE 512

DS_timer *timer = new DS_timer(5);

float* matrixGen(int m, int n);
float* matrixMultiCPU(float *matrixA, float *matrixB, int row_size, int k_size, int col_size);
float* matrixMultiGPU(float *matrixA, float *matrixB, int row_size, int k_size, int col_size);
bool matrixEqual(float *matrixA, float *matrixB, int row_size, int col_size);

__global__ void matMul_kernel(float *_A, float *_B, float *_C, int row_size, int k_size, int col_size) {
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int index = row * col_size + col;

	_C[index] = 0;
	if (row >= row_size || col >= col_size) {

	}
	else for (int k = 0; k < k_size; k++) _C[index] += _A[row*k_size + k] * _B[k * col_size + col];
}
__global__ void matMul_kernel_shared(float *_A, float *_B, float *_C, int row_size, int k_size, int col_size) {
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int index = row * col_size + col;
	
	__shared__ float sA[16][K_SIZE]; //2kb(512*4) * 16 = 32
	__shared__ float sB[K_SIZE][8]; //2kb * 8 = 16

	for (int k = 0; k < K_SIZE; k++) {
		sA[threadIdx.y][k] = _A[row*K_SIZE + k];
		sB[k][threadIdx.x] = _B[col + col_size * k];
	}

	__syncthreads();

	_C[index] = 0;
	if (row >= row_size || col >= col_size) {

	}
	else for (int k = 0; k < k_size; k++) _C[index] += sA[threadIdx.y][k] * sB[k][threadIdx.x];
}

int main()
{
	timer->initTimers();
	timer->setTimerName(0, "host calc matrix ");
	timer->setTimerName(1, "gpu calc matrix ");
	timer->setTimerName(2, "gpu calc matrix (shared memory)");
	timer->setTimerName(3, "memcpy host to device ");
	timer->setTimerName(4, "memcpy device to host ");
	float *matrixA = matrixGen(M_SIZE, K_SIZE);
	float *matrixB = matrixGen(K_SIZE, N_SIZE);
	std::cout << "gen" << std::endl;

	timer->onTimer(0);
	float *hostMatrix = matrixMultiCPU(matrixA, matrixB, M_SIZE, K_SIZE, N_SIZE);
	timer->offTimer(0);
	std::cout << "host calc is end!" << std::endl;
	float *gpuMatrix = matrixMultiGPU(matrixA, matrixB, M_SIZE, K_SIZE, N_SIZE);
	std::cout << "gpu calc is end!" << std::endl;
	
	if (matrixEqual(hostMatrix, gpuMatrix, M_SIZE, N_SIZE))
		std::cout << "Two Matrix is equal!" << std::endl;
	else {
		std::cout << "Two Matrix is not equal!" << std::endl;
	}

	timer->printTimer();


	delete[] matrixA; delete[] matrixB;
	return 0;
}

float* matrixGen(int m, int n) {
	float *newMatrix = new float[m*n];
#pragma omp parallel for
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			newMatrix[i * n + j] = rand() % 100;
		}
	}
	return newMatrix;
}
float* matrixMultiCPU(float *matrixA, float *matrixB, int row_size, int k_size, int col_size) {
	float *newMatrix = new float[row_size*col_size];

#pragma omp parallel for
	for (int i = 0; i < row_size; i++) {
		for (int j = 0; j < col_size; j++) {
			newMatrix[i * col_size + j] = 0.0;
			for (int k = 0; k < k_size; k++) {
				newMatrix[i * col_size + j] += matrixA[i * k_size + k] * matrixB[k * col_size + j];
			}
		}
	}
	return newMatrix;
}
bool matrixEqual(float *matrixA, float *matrixB, int row_size, int col_size) {
	int size = row_size * col_size;

	for (int i = 0; i < size; i++) {
		float diff = matrixA[i] - matrixB[i];
		if (fabs(diff) > 0.00001) {
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

	timer->onTimer(3);
	//cpy matrix data h to d
	cudaMemcpy(dA, matrixA, sizeof(float)*row_size*k_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, matrixB, sizeof(float)*k_size*col_size, cudaMemcpyHostToDevice);
	timer->offTimer(3);

	dim3 blockDim(8, 16); //128 threads per block
	dim3 gridDim(ceil((float)col_size / 8.0), ceil((float)row_size / 16.0));

	timer->onTimer(1);
	matMul_kernel << <gridDim, blockDim >> > (dA, dB, dC, row_size, k_size, col_size);
	cudaThreadSynchronize();
	timer->offTimer(1);
	
	timer->onTimer(2);
	matMul_kernel_shared << <gridDim, blockDim >> > (dA, dB, dC, row_size, k_size, col_size);
	cudaThreadSynchronize();
	timer->offTimer(2);
	
	timer->onTimer(4);
	cudaMemcpy(newMatrix, dC, sizeof(float)*row_size*col_size, cudaMemcpyDeviceToHost);
	timer->offTimer(4);

	cudaFree(&dA); cudaFree(&dB); cudaFree(&dC);

	return newMatrix;
}