#include "common.h"
#include "naive.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define block_size 1024

namespace StreamCompaction {
namespace Naive {
using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer &timer() {
    static PerformanceTimer timer;
    return timer;
}
__global__ void kern_scan(int n, int *odata, const int *idata, int offset) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    
    if (index >= offset) {
        odata[index] = idata[index - offset] + idata[index];
    } else {
        odata[index] = idata[index];
    }
}

__global__ void kern_shift(int n, int *odata, const int *idata) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    if (index == 0) {
        odata[index] = 0;
    } else {
        odata[index] = idata[index - 1];
    }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    int *dev_odata;
    int *dev_idata;

    cudaMalloc((void **)&dev_odata, n * sizeof(int));
    checkCUDAError("cudaMalloc dev_odata failed!");
    cudaMemset(dev_odata, 0, n * sizeof(int));
    checkCUDAError("cudaMemset dev_odata failed!");

    cudaMalloc((void **)&dev_idata, n * sizeof(int));
    checkCUDAError("cudaMalloc dev_idata failed!");
    cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy dev_idata failed!");

    timer().startGpuTimer();
    int num_blocks = divup(n, block_size);
    for (int offset = 1; offset < n; offset *= 2) {
        kern_scan<<<num_blocks, block_size>>>(n, dev_odata, dev_idata, offset);
        std::swap(dev_odata, dev_idata);
    }
    kern_shift<<<num_blocks, block_size>>>(n, dev_odata, dev_idata);
    timer().endGpuTimer();

    cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_odata);
    cudaFree(dev_idata);
}
} // namespace Naive
} // namespace StreamCompaction
