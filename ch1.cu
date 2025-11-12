#include <stdio.h>
#include <vector>

// threadIdx is the index of a given thread within a BLOCK, upper bounded by blockDim
// - This means 2 threads could have the same threadIdx when in different blocks
// blockIdx is the index of a given block within a GRID
// blockDim is the total # of threads that go into a block
// - so if blockDim = 256, then the highest threadIdx is 255

// All of these index identifiers have x y & z fields (dependent on how you wanna interpret the flat data)
// - treating them as normal flat arrays is x, 2D is y, 3D is z

// blockIdx = area code
// threadIdx = local phone number

// can construct a unique global index w/ blockIdx & threadIdx
// for example globalIdx = blockIdx.x * blockDim + threadIdx.x
// this basically gets the index by jumping through the giant array by how many blocks/how large each block is + the thread index

// this gets ran per thread (a lot)
__global__ void vecAddKernel(float *a, float *b, float *c, int n)
{
	int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (globalIdx < n)
	{
		c[globalIdx] = a[globalIdx] + b[globalIdx];
	}
}

void vecAdd(std::vector<float> a_h, std::vector<float> b_h, std::vector<float> c_h)
{
	size_t n = c_h.size();

	float *a_d;
	float *b_d;
	float *c_d;

	cudaMalloc((void **)&a_d, n);
	cudaMalloc((void **)&b_d, n);
	cudaMalloc((void **)&c_d, n);

	cudaMemcpy(a_d, a_h.data(), n, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h.data(), n, cudaMemcpyHostToDevice);

	// kernel logic goes here ***
	vecAddKernel(a_d, b_d, c_d, n);

	cudaMemcpy(c_h.data(), c_d, n, cudaMemcpyDeviceToHost);
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
}

int main()
{

	std::vector<float> a = {1, 2, 3, 4};
	std::vector<float> b = {1, 2, 3, 4};
	std::vector<float> c = {0, 0, 0, 0};

	vecAdd(a, b, c);

	for (const float f : c)
	{
		printf("%f\n", f);
	}

	return 0;
}
