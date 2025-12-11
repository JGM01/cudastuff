// All this stuff is 3D. Grids? 3D rectangular prism of blocks. Blocks? 3D rectangular prism of threads.
// When calling a kernel, you need to specify how many blocks'll be in a grid, and how many threads'll be in a block. (this is why blockIdx.x * blockDim.x + threadIdx.x gets us a global thread index!)
//
//

int main()
{

    return EXIT_SUCCESS;
}