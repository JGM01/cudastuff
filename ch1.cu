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
	// this is a leetgpu problem (first one)
	int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (globalIdx < n)
	{
		c[globalIdx] = a[globalIdx] + b[globalIdx];
	}
}

void vecAdd(std::vector<float> &a_h, std::vector<float> &b_h, std::vector<float> &c_h)
{
	size_t bytes = c_h.size() * sizeof(float);

	float *a_d;
	float *b_d;
	float *c_d;

	cudaMalloc((void **)&a_d, bytes);
	cudaMalloc((void **)&b_d, bytes);
	cudaMalloc((void **)&c_d, bytes);

	cudaMemcpy(a_d, a_h.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h.data(), bytes, cudaMemcpyHostToDevice);

	// just make up a reasonable thread count (its the amount of threads in a given block)
	int threads = 32;

	// determine amount of blocks necessary by considering the amount of data needed to process
	// - so if i gotta do 4 floats, then 4 + 32 - 1 is 35, / 32 is 1.whatever which gets us 1 by truncation.
	// - if it was 1000 floats you could choose 256 threads and do (1000 + 255) / 256 = ~4ish
	int blocks = (c_h.size() + threads - 1) / threads;
	vecAddKernel<<<threads, blocks>>>(a_d, b_d, c_d, bytes);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Kernel launch error! : %s\n", cudaGetErrorString(err));
	}
	cudaDeviceSynchronize();

	cudaMemcpy(c_h.data(), c_d, bytes, cudaMemcpyDeviceToHost);
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
