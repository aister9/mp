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
__global__ void threadAtomicAdd(double a, double *t, double *res, int n);

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

int main()
{
	DS_timer timer(2);
	timer.setTimerName(0, "CPU Integral Time");
	timer.setTimerName(1, "GPU Integral Time(Atomic Func)");

	timer.onTimer(0);
	double val1 = IntegralCPU(0, 1024, SIZE);
	timer.offTimer(0);

	timer.onTimer(1);
	double val2 = IntegralGPU(0, 1024, SIZE);
	timer.offTimer(1);

	if (abs(val1 - val2) < 0.0000001) {
		cout << "two value is equal is done" << endl;
	}
	else {
		cout << val1 << " != " << val2 << endl;
		cout << "val1 - val2 = " << val1 - val2 << endl;
	}

	timer.printTimer();
}

double IntegralCPU(double a, double b, int n) {
	double tokenSize = (abs(a) + abs(b)) / (double)n;
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
	double tokenSize = (abs(a) + abs(b)) / (double)n;
	double result = 0;

	double  *t; double *resul;

	cudaMalloc((void **)&t, sizeof(double));
	cudaMalloc((void **)&resul, sizeof(double));

	cudaMemcpy(t, &tokenSize, sizeof(double) * 1, cudaMemcpyHostToDevice);

	threadAtomicAdd << <ceil((float)n/512)+1, 512>> > (a, t, resul, SIZE);
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