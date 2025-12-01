#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
//clear && gcc Softmax.c -lm -o m.o && ./m.o
typedef enum softmax_type_enum SOFTMAX_TYPE;
enum softmax_type_enum
{
	SOFTMAX_TYPE_NAIVE,
	SOFTMAX_TYPE_SAFE,
	SOFTMAX_TYPE_ONLINE,
	SOFTMAX_ENUM_LENGTH
};

char *GetSoftmaxTypeName(SOFTMAX_TYPE type)
{
	if(type == SOFTMAX_TYPE_NAIVE)
	{
		return "Naive Softmax";
	}
	else if(type == SOFTMAX_TYPE_SAFE)
	{
		return "Safe Softmax";	
	}
	else if(type == SOFTMAX_TYPE_ONLINE)
	{
		return "Online Softmax";	
	}
 	else
 	{
 		printf("Unsupported Softmax");
 		assert(type < SOFTMAX_ENUM_LENGTH);
 	}
}

float FindMaximumFloat(float a, float b)
{
	return fmaxf(a, b);
}

float MeanSquaredError(size_t arrayLength, float *a, float *b)
{
	float mse = 0.0f;
	for(size_t i = 0; i < arrayLength; i++)
	{
		float difference = a[i] - b[i];
		mse += difference * difference;
	}
	mse /= (float)arrayLength;
	return mse;
}

void FillArrayWithRandomValues(size_t arrayLength, float *array)
{
	srand(45654);
	for(size_t i = 0; i < arrayLength; i++)
	{
		array[i] = (float)rand() / RAND_MAX; 
	}
}

void NaiveSoftmax(size_t arrayLength, float *input, float *output)
{
	float sumOfExponents = 0.0f;
	//Compute exponents
	for(size_t i = 0; i < arrayLength; i++)
	{
		output[i] = expf(input[i]);
		sumOfExponents += output[i];
	}	
	//Normalize
	for (size_t i = 0; i < arrayLength; i++)
	{
		output[i] /= sumOfExponents;
	}
}

void SafeSoftmax(size_t arrayLength, float *input, float *output)
{
	float sumOfExponents = 0.0f;
	float maximumValue = input[0];
	for(size_t i = 1; i < arrayLength; i++)
	{
		if(input[i] > maximumValue)
		{
			maximumValue = input[i];
		}
	}
	//Compute exponents and subtract max valu
	for(size_t i = 0; i < arrayLength; i++)
	{
		output[i] = expf(input[i] - maximumValue);
		sumOfExponents += output[i];
	}	
	//Normalize
	for(size_t i = 0; i < arrayLength; i++)
	{
		output[i] /= sumOfExponents;
	}
}

void OnlineSoftmax(size_t arrayLength, float *input, float *output)
{
	float sumOfExponents = 0.0f;
	float maximumValue = -INFINITY;
	for(size_t i = 0; i < arrayLength; i++)
	{
		//Rescale sum of exponents when new max is found
		if(input[i] > maximumValue)
		{
			sumOfExponents = sumOfExponents * expf(maximumValue - input[i]) + 1.0f;
			maximumValue = input[i] ;
		}
		else
		{
			sumOfExponents += expf(input[i] - maximumValue);
		}
	}
	//Find final normalization factor
	float normalizer = 1.0f / sumOfExponents;
	//Normalize
	for(size_t i = 0; i < arrayLength; i++)
	{
		output[i] = expf(input[i] - maximumValue) * normalizer;
	}
}


__global__ void OnlineSoftmaxCUDA(size_t rows, size_t cols, float *input, float *output)
{
	//Shared memory to hold max
	__shared__ float sharedMemory[1024];
	
	int warpSize = 32;
	int threadIndex = threadIdx.x;
	int matrixRowIndex   = blockIdx.x;
	
	if(matrixRowIndex < rows)
	{
		//Pass 1: Find sum of exponents and maximumValue for each thread
		float *inputRow = input + matrixRowIndex * cols;
		float *outputRow = output + matrixRowIndex * cols;
		float maximumValue = -INFINITY;
		float sumOfExponents = 0.0f;
		for(int i = threadIndex; i < cols; i += blockDim.x)
		{
			if(inputRow[i] > maximumValue)
			{
				sumOfExponents = sumOfExponents * expf(maximumValue - inputRow[i]) + 1.0f;
				maximumValue = inputRow[i] ;
			}
			else
			{
				sumOfExponents += expf(inputRow[i] - maximumValue);
			}
		}
		__syncthreads();
		
		//Find the maximum for a single warp
		float currentWarpMax = maximumValue;
		for(int offset = warpSize / 2; offset > 0; offset /= 2)
		{
			currentWarpMax = fmaxf(currentWarpMax, __shfl_down_sync(0xffffffff, currentWarpMax, offset));
		}
		
		//Store currentWarpMax in shared mem for later block level reduction
		if(blockDim.x > warpSize)
		{
			if(threadIndex % warpSize == 0)
			{
				sharedMemory[threadIndex / warpSize] = currentWarpMax;
			}
			__syncthreads();
		
			//Perform global reduction in shared memory
			if(threadIndex < warpSize)
			{
				//Avoid warp divergence
				currentWarpMax = (threadIndex < CEIL_DIV(blockDim.x, warpSize)) ? sharedMemory[threadIndex] : -INFINITY;
				for(int offset = warpSize / 2; offset > 0; offset /= 2)
				{
					currentWarpMax = fmaxf(currentWarpMax, __shfl_down_sync(0xffffffff, currentWarpMax, offset));
				}
				if(threadIndex == 0)
				{
					sharedMemory[0] = currentWarpMax;
				}
			}
			else
			{
				if(threadIndex == 0)
				{
					sharedMemory[0] = currentWarpMax;
				}
			}
			__syncthreads();
			
			//The entire row's max is stored in sharedMemory
			float rowMaximum = sharedMemory[0];
			__syncthreads();
			
			currentWarpMax = sumOfExponents * expf(sumOfExponents - rowMaximum);
			for(int offset = warpSize / 2; offset > 0; offset /= 2)
			{
				currentWarpMax += __shfl_down_sync(0xffffffff, currentWarpMax, offset);
			}
			
			if(blockDim.x > warpSize)
			{
				if(threadIndex % warpSize == 0)
				{
					sharedMemory[threadIndex / warpSize] = currentWarpMax;
				}
				__syncthreads();

				// first warp will do global reduction
				if(threadIndex < warpSize)
				{
					currentWarpMax = (threadIndex < CEIL_DIV(blockDim.x, warpSize)) ? sharedMemory[threadIndex] : 0.0f;
					for(int offset = warpSize / 2; offset > 0; offset /= 2)
					{
						currentWarpMax += __shfl_down_sync(0xffffffff, currentWarpMax, offset);
					}
					if(threadIndex == 0)
					{
						sharedMemory[0] = currentWarpMax;
					}
				}
				else
				{
					if(threadIndex == 0)
					{
						sharedMemory[0] = currentWarpMax;
					}
				}
			}
			__syncthreads();
			float rowNormalizer = sharedMemory[0];
			__syncthreads();
			//Find softmax
			for(int i = threadIndex; i < cols; i += blockDim.x)
			{
				outputRow[i] = expf(inputRow[i] - rowMaximum) / rowNormalizer;
			}
		}
	}
}

//Here we only compare the safe softmax and online softmax
void BenchmarkAccuracy()
{
	printf("Accuracy Test\n");
	size_t test0Length = 10;
	float *test0 = calloc(test0Length, sizeof(float));
	float *test0_Safe = calloc(test0Length, sizeof(float));
	float *test0_Online = calloc(test0Length, sizeof(float));
	FillArrayWithRandomValues(test0Length, test0);
	SafeSoftmax(test0Length, test0, test0_Safe);
	OnlineSoftmax(test0Length, test0, test0_Online);	
	float mse0 = MeanSquaredError(test0Length, test0_Safe, test0_Online);
	printf("Test 0: %3ld values, MSE: %.3f\n",test0Length, mse0);
	
	size_t test1Length = 100;
	float *test1 = calloc(test1Length, sizeof(float));
	float *test1_Safe = calloc(test1Length, sizeof(float));
	float *test1_Online = calloc(test1Length, sizeof(float));
	FillArrayWithRandomValues(test1Length, test1);
	SafeSoftmax(test1Length, test1, test1_Safe);
	OnlineSoftmax(test1Length, test1, test1_Online);	
	float mse1 = MeanSquaredError(test1Length, test1_Safe, test1_Online);
	printf("Test 1: %3ld values, MSE: %.3f\n",test1Length, mse1);
	
	size_t test2Length = 10000;
	float *test2 = calloc(test2Length, sizeof(float));
	float *test2_Safe = calloc(test2Length, sizeof(float));
	float *test2_Online = calloc(test2Length, sizeof(float));
	FillArrayWithRandomValues(test2Length, test2);
	SafeSoftmax(test2Length, test2, test2_Safe);
	OnlineSoftmax(test2Length, test2, test2_Online);	
	float mse2 = MeanSquaredError(test2Length, test2_Safe, test2_Online);
	printf("Test 2: %3ld values, MSE: %.3f\n",test2Length, mse2);
	
	free(test0);free(test0_Safe);free(test0_Online);
	free(test1);free(test1_Safe);free(test1_Online);
	free(test2);free(test2_Safe);free(test2_Online);
}


void BenchmarkSpeed()
{
	printf("\nSpeed Test\n");
	struct timespec start, end;
	double elapsedSafe, elapsedOnline;
	size_t test0Length = 10;
	float *test0 = calloc(test0Length, sizeof(float));
	float *test0_Safe = calloc(test0Length, sizeof(float));
	float *test0_Online = calloc(test0Length, sizeof(float));
	FillArrayWithRandomValues(test0Length, test0);

	//Time SafeSoftmax
	clock_gettime(CLOCK_MONOTONIC, &start);
	SafeSoftmax(test0Length, test0, test0_Safe);
	clock_gettime(CLOCK_MONOTONIC, &end);
	elapsedSafe = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

	//Time OnlineSoftmax
	clock_gettime(CLOCK_MONOTONIC, &start);
	OnlineSoftmax(test0Length, test0, test0_Online);
	clock_gettime(CLOCK_MONOTONIC, &end);
	elapsedOnline = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

	printf("Test 0: %8ld values | Safe = %.6fs | Online = %.6fs\n",test0Length, elapsedSafe, elapsedOnline);

	free(test0);free(test0_Safe);free(test0_Online);

	size_t test1Length = 1000;
	float *test1 = calloc(test1Length, sizeof(float));
	float *test1_Safe = calloc(test1Length, sizeof(float));
	float *test1_Online = calloc(test1Length, sizeof(float));
	FillArrayWithRandomValues(test1Length, test1);

	clock_gettime(CLOCK_MONOTONIC, &start);
	SafeSoftmax(test1Length, test1, test1_Safe);
	clock_gettime(CLOCK_MONOTONIC, &end);
	elapsedSafe = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

	clock_gettime(CLOCK_MONOTONIC, &start);
	OnlineSoftmax(test1Length, test1, test1_Online);
	clock_gettime(CLOCK_MONOTONIC, &end);
	elapsedOnline = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

	printf("Test 1: %8ld values | Safe = %.6fs | Online = %.6fs\n",test1Length, elapsedSafe, elapsedOnline);

	free(test1);free(test1_Safe);free(test1_Online);

	size_t test2Length = 10000000; 
	float *test2 = calloc(test2Length, sizeof(float));
	float *test2_Safe = calloc(test2Length, sizeof(float));
	float *test2_Online = calloc(test2Length, sizeof(float));
	FillArrayWithRandomValues(test2Length, test2);

	clock_gettime(CLOCK_MONOTONIC, &start);
	SafeSoftmax(test2Length, test2, test2_Safe);
	clock_gettime(CLOCK_MONOTONIC, &end);
	elapsedSafe = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

	clock_gettime(CLOCK_MONOTONIC, &start);
	OnlineSoftmax(test2Length, test2, test2_Online);
	clock_gettime(CLOCK_MONOTONIC, &end);
	elapsedOnline = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

	printf("Test 2: %8ld values | Safe = %.6fs | Online = %.6fs\n",test2Length, elapsedSafe, elapsedOnline);
	free(test2);free(test2_Safe);free(test2_Online);
}


int main()
{
	BenchmarkAccuracy();
	BenchmarkSpeed();
	return 0;
}
