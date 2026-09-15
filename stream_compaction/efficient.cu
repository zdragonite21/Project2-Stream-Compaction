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

__global__ void kern_scan(int n, int *data, int depth) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid >= n) {
        return;
    }

    // upsweep
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int base = 2 * offset * thid;
            int ai = base + offset - 1;
            int bi = base + 2 * offset - 1;
            data[bi] += data[ai];
        }
        offset <<= 1;
    }

    // downsweep
    if (thid == 0) {
        // have the first thread clear the last element
        data[n - 1] = 0;
    }

    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        int base = 2 * offset * thid;
        int ai = base + offset - 1;
        int bi = base + 2 * offset - 1;

        float t = data[ai];
        data[ai] = data[bi];
        data[bi] += t;
    }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    int *dev_data;
    cudaMalloc((void **)&dev_data, n * sizeof(int));
    checkCUDAError("cudaMalloc dev_data failed!");
    cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy dev_data failed!");

    timer().startGpuTimer();
    int num_blocks = divup(n, block_size);
    int depth = ilog2ceil(n);
    kern_scan<<<num_blocks, block_size>>>(n, dev_data, depth);

    timer().endGpuTimer();

    cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

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
