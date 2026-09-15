#include "common.h"
#include "efficient.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define block_size 1024

namespace StreamCompaction {
namespace Efficient {
using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer &timer() {
    static PerformanceTimer timer;
    return timer;
}

__global__ void kern_scan(int n, int *data) {
    extern __shared__ int temp[];

    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (thid >= n) {
    //     return;
    // }

    temp[thid << 1] = data[thid << 1];
    temp[(thid << 1) + 1] = data[(thid << 1) + 1];

    // upsweep
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int base = 2 * offset * thid;
            int ai = base + offset - 1;
            int bi = base + 2 * offset - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // downsweep
    if (thid == 0) {
        // have the first thread clear the last element
        temp[n - 1] = 0;
    }

    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int base = 2 * offset * thid;
            int ai = base + offset - 1;
            int bi = base + 2 * offset - 1;

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();
    data[thid << 1] = temp[thid << 1];
    data[(thid << 1) + 1] = temp[(thid << 1) + 1];
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    int size = (block_size << 1);
    int offset = size - n;

    int *dev_data;
    cudaMalloc((void **)&dev_data, size * sizeof(int));
    checkCUDAError("cudaMalloc dev_data failed!");
    cudaMemset(dev_data, 0, offset * sizeof(int));
    cudaMemcpy(dev_data + offset, idata, n * sizeof(int), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy dev_data failed!");

    timer().startGpuTimer();
    int num_blocks = divup(n, size);
    int depth = ilog2ceil(n);
    // each block handles a range of 2 * block_size elements
    kern_scan<<<num_blocks, block_size, size * sizeof(int)>>>(
        size, dev_data);

    timer().endGpuTimer();

    cudaMemcpy(odata, dev_data + offset, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_data);
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int *odata, const int *idata) {
    timer().startGpuTimer();
    // TODO
    timer().endGpuTimer();
    return -1;
}
} // namespace Efficient
} // namespace StreamCompaction
