#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DS_timer.h"
#include <stdio.h>
#include <iostream>

#define LOOP_I(x) for(int i = 0 ; i<x; i++)

#define NUM_BLOCK 128*1024
#define NUM_T_IN_B 1024
#define ARRAY_SIZE NUM_T_IN_B*NUM_BLOCK

#define NUM_STREAMS 4

__global__ void myKernel(int *_in, int *_out) {
	int tID = blockDim.x * blockIdx.x + threadIdx.x;

	int temp = 0;
	for (int i = 0; i < 250; i++) {
		temp = (temp + _in[tID] * 5) % 10;
	}
	_out[tID] = temp;
}

void main() {
	DS_timer timer(5);
	timer.setTimerName(0, "Single Stream");
	timer.setTimerName(1, "*host -> device");
	timer.setTimerName(2, "*kernel execution");
	timer.setTimerName(3, "*devcie -> host");
	timer.setTimerName(4, "Multiple Stream");

	int *in = NULL, *out = NULL, *out2 = NULL, *dIn = NULL, *dOut = NULL;

	cudaMallocHost(&in, sizeof(int)*ARRAY_SIZE); memset(in, 0, sizeof(int)*ARRAY_SIZE);
	cudaMallocHost(&out, sizeof(int)*ARRAY_SIZE); memset(out, 0, sizeof(int)*ARRAY_SIZE);
	cudaMallocHost(&out2, sizeof(int)*ARRAY_SIZE); memset(out2, 0, sizeof(int)*ARRAY_SIZE);

	cudaMalloc(&dIn, sizeof(int)*ARRAY_SIZE); cudaMalloc(&dOut, sizeof(int)*ARRAY_SIZE);

	LOOP_I(ARRAY_SIZE) in[i] = rand() % 10;

	//single
	timer.onTimer(0);
	timer.onTimer(1);
	cudaMemcpy(dIn, in, sizeof(int)*ARRAY_SIZE, cudaMemcpyHostToDevice);
	timer.offTimer(1);
	timer.onTimer(2);
	myKernel << <NUM_BLOCK, NUM_T_IN_B >> > (dIn, dOut);
	cudaDeviceSynchronize();
	timer.offTimer(2);
	timer.onTimer(3);
	cudaMemcpy(out, dOut, sizeof(int)*ARRAY_SIZE, cudaMemcpyDeviceToHost);
	timer.offTimer(3);
	timer.offTimer(0);
	cudaStream_t stream[NUM_STREAMS];
	LOOP_I(NUM_STREAMS) cudaStreamCreate(&stream[i]);

	int chunkSize = ARRAY_SIZE / NUM_STREAMS;

	timer.onTimer(4);
	LOOP_I(NUM_STREAMS) {
		int offset = chunkSize * i;
		cudaEvent_t start, stop;
		cudaEventCreate(&start); cudaEventCreate(&stop);

		cudaEventRecord(start);
		cudaMemcpyAsync(dIn + offset, in + offset, sizeof(int)*chunkSize, cudaMemcpyHostToDevice, stream[i]);
		myKernel << <NUM_BLOCK / NUM_STREAMS, NUM_T_IN_B, 0, stream[i] >> > (dIn + offset, dOut + offset);
		cudaMemcpyAsync(out2 + offset, dOut + offset, sizeof(int)*chunkSize, cudaMemcpyDeviceToHost, stream[i]);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		float time;
		cudaEventElapsedTime(&time, start, stop);

		printf("Stream[%d] : %lf ms\n", i, time);

		cudaEventDestroy(start); cudaEventDestroy(stop);
	}
	cudaDeviceSynchronize();
	timer.offTimer(4);

	LOOP_I(NUM_STREAMS) cudaStreamDestroy(stream[i]);

	cudaFree(dIn); cudaFree(dOut);

	cudaFreeHost(in); cudaFreeHost(out); cudaFreeHost(out2);

	timer.printTimer();
	system("pause");
}