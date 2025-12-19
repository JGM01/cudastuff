// All this stuff is 3D. Grids? 3D rectangular prism of blocks. Blocks? 3D rectangular prism of threads.
//
// When calling a kernel, you need to specify how many blocks'll be in a grid, and how many threads'll be in a
// block. (this is why blockIdx.x * blockDim.x + threadIdx.x gets us a global thread index!)
//
// The 3D-ness is literally just for convenience bc most CUDA tasks are probably on some 2D or 3D dataset (pixels on screen, physics).
//
// 2D is when blockDim.z = 1. 1D is when blockDim.z = 1 & blockDim.y = 1.
//
//

//

void usingDimToCallKernel()
{
    // Each block will have 32 threads laid out in a 1D fashion.
    dim3 block_dimensions(32, 1, 1);

    // The grid will consist of 32 blocks laid out in a 1D fashion.
    dim3 grid_dimensions(32, 1, 1);

    // Usage.
    demoKernel<<<grid_dimensions, block_dimensions>>>();

    // Equivalence, since CUDA lets you use integers and it will assume 1D (appending the y & z as 1).
    demoKernel<<<32, 32>>>();
}

__global__ void demoKernel() {}

int main()
{

    return EXIT_SUCCESS;
}