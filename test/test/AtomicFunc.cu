#include "DS_timer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void threadCounting_noSync(int *a) {
	(*a)++;
}

__global__ void threadCounting_atomicGlobal(int *a) {
	atomicAdd(a, 1);
}

__global__ void threadCounting_atomicShared(int *a) {
	__shared__ int sa;

	if (threadIdx.x == 0) sa = 0;
	__syncthreads();

	atomicAdd(&sa, 1);
	__syncthreads();

	if (threadIdx.x == 0)
		atomicAdd(a, sa);
}


int main()
{
	DS_timer timer(3);
	timer.setTimerName(0, "NoAtomic");
	timer.setTimerName(1, "AtomicGlobal");
	timer.setTimerName(2, "AtomicShared");
	int a = 0; int *d1;
	int b = 0; int *d2;
	int c = 0; int *d3;

	cudaMalloc((void **)&d1, sizeof(int));
	cudaMalloc((void **)&d2, sizeof(int));
	cudaMalloc((void **)&d3, sizeof(int));
	cudaMemset(d1, 0, sizeof(int) * 1);
	cudaMemset(d2, 0, sizeof(int) * 1);
	cudaMemset(d3, 0, sizeof(int) * 1);

	timer.onTimer(0);
	threadCounting_noSync << <10240, 512 >> > (d1);
	cudaDeviceSynchronize();
	timer.offTimer(0);


	timer.onTimer(1);
	threadCounting_atomicGlobal << <10240, 512 >> > (d2);
	cudaDeviceSynchronize();
	timer.offTimer(1);

	timer.onTimer(2);
	threadCounting_atomicShared << <10240, 512 >> > (d3);
	cudaDeviceSynchronize();
	timer.offTimer(2);

	cudaMemcpy(&a, d1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&b, d2, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&c, d3, sizeof(int), cudaMemcpyDeviceToHost);

	printf("[No Atomic ] : %d\n", a);
	printf("[Atomic Global] : %d\n", b);
	printf("[Atomic Shared] : %d\n", c);

	cudaFree(d1);
	cudaFree(d2);
	cudaFree(d3);

	timer.printTimer();
}
