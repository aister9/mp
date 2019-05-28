#include "DS_timer.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_atomic_functions.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <iostream>

using namespace std;

#define SIZE 1024*1024*1024

double IntegralCPU(double a, double b, int n);
double IntegralGPU(double a, double b, int n);
double IntegralGPU2(double a, double b, int n);
double IntegralGPU3(double a, double b, int n);
__global__ void threadAtomicAdd(double a, double *t, double *res, int n);
__global__ void threadAtomicAddver2(double a, double *t, double *res, int n);
__global__ void threadAtomicAddver2red(double a, double *t, double *res, int n);
__global__ void threadBlockReduction(double *res);

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
	__device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
				__longlong_as_double(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif

void doubleMatch(double val1, double val2) {
	if (abs(val1 - val2) < 0.0000001) {
		cout << "two value is equal is done" << endl;
		cout << val1 << " == " << val2 << endl;
	}
	else {
		cout << val1 << " != " << val2 << endl;
		cout << "val1 - val2 = " << val1 - val2 << endl;
	}
}

int main()
{
	DS_timer timer(4);
	timer.setTimerName(0, "CPU Integral Time");
	timer.setTimerName(1, "GPU Integral Time(Atomic Func)");
	timer.setTimerName(2, "GPU Integral Time(Atomic Func ver2)");
	timer.setTimerName(3, "GPU Integral Time(Atomic Func ver2 reduction func)");

	timer.onTimer(0);
	double val1 = IntegralCPU(-1, 1, SIZE);
	timer.offTimer(0);

	timer.onTimer(1);
	double val2 = IntegralGPU(-1, 1, SIZE);
	timer.offTimer(1);

	timer.onTimer(2);
	double val3 = IntegralGPU2(-1, 1, SIZE);
	timer.offTimer(2);

	timer.onTimer(3);
	double val4 = IntegralGPU3(-1, 1, SIZE);
	timer.offTimer(3);

	cout << "cpu and gpu 1" << endl;
	doubleMatch(val1, val2);
	cout << "cpu and gpu 2" << endl;
	doubleMatch(val1, val3);
	cout << "cpu and gpu 3" << endl;
	doubleMatch(val1, val4);
	
	timer.printTimer();
}

double IntegralCPU(double a, double b, int n) {
	double tokenSize = abs(b-a) / (double)n;
	double result = 0;

#pragma omp parallel for reduction(+:result)
	for (int i = 0; i <= n; i++) {
		double s = a + tokenSize * i;
		double s2 = a + tokenSize * (i + 1);
		result += tokenSize * (s*s + s2 * s2) / 2;
	}

	return result;
}

double IntegralGPU(double a, double b, int n) {
	double tokenSize = abs(b - a) / (double)n;
	double result = 0;

	double  *t; double *resul;

	cudaMalloc((void **)&t, sizeof(double));
	cudaMalloc((void **)&resul, sizeof(double));

	cudaMemcpy(t, &tokenSize, sizeof(double) * 1, cudaMemcpyHostToDevice);

	threadAtomicAdd << <ceil((float)n/1024)+1, 1024>> > (a, t, resul, SIZE);
	cudaDeviceSynchronize();

	cudaMemcpy(&result, resul, sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(t); cudaFree(resul);

	return result;
}

double IntegralGPU2(double a, double b, int n) {
	double tokenSize = abs(b - a) / (double)n;
	double result = 0;

	double  *t; double *resul;

	cudaMalloc((void **)&t, sizeof(double));
	cudaMalloc((void **)&resul, sizeof(double));

	cudaMemcpy(t, &tokenSize, sizeof(double) * 1, cudaMemcpyHostToDevice);

	threadAtomicAddver2 << <1024, 1024 >> > (a, t, resul, SIZE);
	cudaDeviceSynchronize();

	cudaMemcpy(&result, resul, sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(t); cudaFree(resul);

	return result;
}

double IntegralGPU3(double a, double b, int n) {
	double tokenSize = abs(b - a) / (double)n;
	double result = 0;

	double  *t; double *resul;

	cudaMalloc((void **)&t, sizeof(double));
	cudaMalloc((void **)&resul, sizeof(double)*1024);

	cudaMemcpy(t, &tokenSize, sizeof(double) * 1, cudaMemcpyHostToDevice);

	threadAtomicAddver2red << <1024, 1024 >> > (a, t, resul, SIZE);
	threadBlockReduction << <1, 1024 >> > (resul);
	cudaDeviceSynchronize();

	cudaMemcpy(&result, resul, sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(t); cudaFree(resul);

	return result;
}

__global__ void threadAtomicAdd(double a, double *t, double *res, int n) {
	if (threadIdx.x + blockIdx.x*blockDim.x > n) return;
	__shared__ double sa;

	double s = a + __fmul_rn(*t, (threadIdx.x + blockIdx.x*blockDim.x));
	double s2 = a + __fmul_rn(*t,(threadIdx.x + blockIdx.x*blockDim.x + 1));
	double bar = __fmul_rn(*t , (__fmul_rn(s,s) + __fmul_rn(s2,s2))) / 2;

	if (threadIdx.x == 0) {
		sa = 0;
	}
	__syncthreads();

	
	atomicAdd(&sa, bar);
	__syncthreads();

	if (threadIdx.x == 0) atomicAdd(res, sa);
}

__global__ void threadAtomicAddver2(double a, double *t, double *res, int n) {
	if (threadIdx.x + blockIdx.x*blockDim.x > n) return;
	__shared__ double sa;
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (threadIdx.x == 0) {
		sa = 0;
	}
	double bar = 0;
	for (int i = 0; i < 1024; i++) {
		double s = a + __fmul_rn(*t, (tid*1024 + i));
		double s2 = a + __fmul_rn(*t, (tid*1024 + i + 1));
		bar += __fmul_rn(*t, (__fmul_rn(s, s) + __fmul_rn(s2, s2))) / 2;
	}
	if (tid == n - 1) {
		double s = a + __fmul_rn(*t, (tid * 1024 + 1024));
		double s2 = a + __fmul_rn(*t, (tid * 1024 + 1025));
		bar += __fmul_rn(*t, (__fmul_rn(s, s) + __fmul_rn(s2, s2))) / 2;
	}
	__syncthreads();


	atomicAdd(&sa, bar);
	__syncthreads();

	if (threadIdx.x == 0) atomicAdd(res, sa);
}


__global__ void threadAtomicAddver2red(double a, double *t, double *res, int n) {
	if (threadIdx.x + blockIdx.x*blockDim.x > n) return;
	__shared__ double sa[1024];
	int tid = threadIdx.x + blockIdx.x*blockDim.x;


	double bar = 0;
	for (int i = 0; i < 1024; i++) {
		double s = a + __fmul_rn(*t, (tid * 1024 + i));
		double s2 = a + __fmul_rn(*t, (tid * 1024 + i + 1));
		bar += __fmul_rn(*t, (__fmul_rn(s, s) + __fmul_rn(s2, s2))) / 2;
	}
	sa[threadIdx.x] = bar;
	__syncthreads();

	//threads reduction

	for (int offset = 1; offset < 1024; offset *= 2) {
		if (threadIdx.x % (2 * offset) == 0) sa[threadIdx.x] += sa[threadIdx.x + offset];
		__syncthreads();
	}
	__syncthreads();

	res[blockIdx.x] = sa[0];
}

__global__ void threadBlockReduction(double *res) {
	for (int offset = 1; offset < 1024; offset *= 2) {
		if (threadIdx.x % (2 * offset) == 0) res[threadIdx.x] += res[threadIdx.x + offset];
		__syncthreads();
	}
}