#include <stdio.h>
#include <vector>

void vecAdd(std::vector<float> a_h, std::vector<float> b_h, std::vector<float> c_h) {
	size_t n = c_h.size();

	float* a_d;
	float* b_d;
	float* c_d;

	cudaMalloc((void **) &a_d, n);
	cudaMalloc((void **) &b_d, n);
	cudaMalloc((void **) &c_d, n);

	cudaMemcpy(a_d, a_h.data(), n, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h.data(), n, cudaMemcpyHostToDevice);
	
	// kernel logic goes here ***
	

	cudaMemcpy(c_h.data(), c_d, n, cudaMemcpyDeviceToHost);
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
}

int main() {

	std::vector<float> a = {1, 2, 3, 4};
	std::vector<float> b = {1, 2, 3, 4};
	std::vector<float> c = {0, 0, 0, 0};

	vecAdd(a, b, c );

	for(const float f : c) {
		printf("%f\n", f);
	}

	return 0;
}
