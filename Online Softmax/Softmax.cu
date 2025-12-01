#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define CEIL_DIV(x, y) ((x + y - 1) / (y))
#define CUDA_CHECK(err) do { if(err != cudaSuccess) { \
	printf("CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(-1); }} while(0)

__global__ void OnlineSoftmaxCUDA(size_t rows, size_t cols, float *input, float *output)
{
	__shared__ float sharedMemory[1024];
	int warpSize = 32;
	int threadIndex = threadIdx.x;
	int matrixRowIndex = blockIdx.x;
	
	//Ensure threads are working on actual data
	if(matrixRowIndex >= rows){return;}
	float *inputRow  = input  + matrixRowIndex * cols;
	float *outputRow = output + matrixRowIndex * cols;
	float maximumValue   = -INFINITY;
	float sumOfExponents = 0.0f;
	
	//Each thread has its own max and sum
	for(int i = threadIndex; i < cols; i += blockDim.x)
	{
		float x = inputRow[i];
		if(x > maximumValue)
		{
			sumOfExponents = sumOfExponents * expf(maximumValue - x) + 1.0f;
			maximumValue   = x;
		}
		else
		{
			sumOfExponents += expf(x - maximumValue);
		}
	}
	__syncthreads();
	
	//Find warp level maximum
	float warpMax = maximumValue;
	for(int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		warpMax = fmaxf(warpMax, __shfl_down_sync(0xffffffff, warpMax, offset));
	}
	//Store warp's maximum in shared memory
	if(threadIndex % warpSize == 0)
	{
		sharedMemory[threadIndex / warpSize] = warpMax;
	}
	__syncthreads();
	
	//Find block level maximum
	if(threadIndex < warpSize)
	{
		int warpCount = CEIL_DIV(blockDim.x, warpSize);
		float v = (threadIndex < warpCount) ? sharedMemory[threadIndex] : -INFINITY;
		for(int offset = warpSize / 2; offset > 0; offset /= 2){v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));}
		if(threadIndex == 0){sharedMemory[0] = v;}
	}
	__syncthreads();
	float rowMaximum = sharedMemory[0];
	__syncthreads();
	// rescale sum
	float localSum = sumOfExponents * expf(maximumValue - rowMaximum);

	float warpSum = localSum;
	for(int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		warpSum += __shfl_down_sync(0xffffffff, warpSum, offset);
	}

	if(threadIndex % warpSize == 0){sharedMemory[threadIndex / warpSize] = warpSum;}
	__syncthreads();
	// block-level sum
	if(threadIndex < warpSize)
	{
		int warpCount = CEIL_DIV(blockDim.x, warpSize);
		float v = (threadIndex < warpCount) ? sharedMemory[threadIndex] : 0.0f;
		for(int offset = warpSize / 2; offset > 0; offset /= 2)
		{
			v += __shfl_down_sync(0xffffffff, v, offset);		
		}

		if(threadIndex == 0){sharedMemory[0] = v;}
	}
	__syncthreads();
	float rowNormalizer = sharedMemory[0];
	__syncthreads();
	for(int i = threadIndex; i < cols; i += blockDim.x)
	{
      		outputRow[i] = expf(inputRow[i] - rowMaximum) / rowNormalizer;	
	}

}

void BasicSoftmaxCPU(int rows, int cols, float *input, float *output)
{
	clock_t start = clock();
	for (int r = 0; r < rows; r++)
	{
		float *row = input + r*cols;
		float *out = output + r*cols;
		float rowMax = -INFINITY;
		for(int c = 0; c < cols; c++)if (row[c] > rowMax) rowMax = row[c];

		float sum = 0.0f;
		for (int c = 0; c < cols; c++)
		{
			out[c] = expf(row[c] - rowMax);
			sum += out[c];
		}
		for(int c = 0; c < cols; c++)out[c] /= sum;
	}
}

	
int main()
{
	int rows = 1024;     //matrixRows
	int cols = 512;      //matrixColumns
	int BLOCK = 256;  //threads per block

	size_t bytes = rows * cols * sizeof(float);

	float *inputMatrix  = (float*)malloc(bytes);
	float *outputMatrixGPU = (float*)malloc(bytes);
	float *outputMatrixCPU    = (float*)malloc(bytes);

	srand((unsigned)time(NULL));
	for(int i = 0; i < rows*cols; i++){inputMatrix[i] = ((float)rand() / RAND_MAX) * 5.0f - 2.5f;}

	float *gpuCopyInput, *gpuCopyOutput;
	CUDA_CHECK(cudaMalloc(&gpuCopyInput, bytes));
	CUDA_CHECK(cudaMalloc(&gpuCopyOutput, bytes));
	CUDA_CHECK(cudaMemcpy(gpuCopyInput, inputMatrix, bytes, cudaMemcpyHostToDevice));

	/*Time GPU code*/
	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	dim3 grid(rows);
	dim3 block(BLOCK);

	CUDA_CHECK(cudaEventRecord(start));
	OnlineSoftmaxCUDA<<<grid, block>>>(rows, cols, gpuCopyInput, gpuCopyOutput);
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float gpuTimeMs = 0.f;
	CUDA_CHECK(cudaEventElapsedTime(&gpuTimeMs, start, stop));

	CUDA_CHECK(cudaMemcpy(outputMatrixGPU, gpuCopyOutput, bytes, cudaMemcpyDeviceToHost));

	/*Time CPU code*/
	clock_t cpu_start = clock();
	BasicSoftmaxCPU(rows, cols, inputMatrix, outputMatrixCPU);
	clock_t cpu_end = clock();
	float cpuTimeMs = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

	/*Compare accuracy*/

	float maxError = 0.0f;
	for (int i = 0; i < rows*cols; i++)
	{
		float err = fabs(outputMatrixGPU[i] - outputMatrixCPU[i]);
		if(err > maxError) maxError = err;
	}

	printf("GPU softmax time: %.3f ms\n", gpuTimeMs);
	printf("CPU softmax time: %.3f ms\n", cpuTimeMs);
	printf("Max absolute error: %e\n", maxError);

	/*Cleanup*/
	free(inputMatrix); free(outputMatrixGPU); free(outputMatrixCPU);
	CUDA_CHECK(cudaFree(gpuCopyInput));
	CUDA_CHECK(cudaFree(gpuCopyOutput));

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	return 0;
}
