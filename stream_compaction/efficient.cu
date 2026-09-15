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

__global__ void kern_inc(int n, int data_num_blocks, int *data, int *sums) {
    // a thread is spawned per block
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid >= data_num_blocks) {
        return;
    }
    int base = thid * n;
    int inc = sums[thid];
    for (int i = 0; i < n; ++i) {
        data[base + i] += inc;
    }
}

__global__ void kern_scan(int n, int *data, int *sums, bool write_sum) {
    extern __shared__ int temp[];

    int local_thid = threadIdx.x;
    int block_base = blockIdx.x * n;
    // if (thid >= n) {
    //     return;
    // }

    temp[local_thid << 1] = data[block_base + (local_thid << 1)];
    temp[(local_thid << 1) + 1] = data[block_base + (local_thid << 1) + 1];

    // upsweep
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (local_thid < d) {
            int base = 2 * offset * local_thid;
            int ai = base + offset - 1;
            int bi = base + 2 * offset - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // downsweep
    int total = temp[n - 1];
    if (local_thid == 0) {
        // have the first thread clear the last element
        temp[n - 1] = 0;
    }

    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (local_thid < d) {
            int base = 2 * offset * local_thid;
            int ai = base + offset - 1;
            int bi = base + 2 * offset - 1;

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();
    data[block_base + (local_thid << 1)] = temp[local_thid << 1];
    data[block_base + (local_thid << 1) + 1] = temp[(local_thid << 1) + 1];

    if (write_sum && local_thid == 0) {
        sums[blockIdx.x] = total;
    }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    int size_per_block = (block_size << 1);

    int data_num_blocks = divup(n, size_per_block);
    int data_pad = size_per_block - (n % size_per_block);
    int data_total_size = data_num_blocks * size_per_block;

    int *dev_data;
    cudaMalloc((void **)&dev_data, data_total_size * sizeof(int));
    checkCUDAError("cudaMalloc dev_data failed!");
    cudaMemset(dev_data, 0, data_pad * sizeof(int));
    checkCUDAError("cudaMemset dev_data failed!");
    cudaMemcpy(dev_data + data_pad, idata, n * sizeof(int),
               cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy dev_data failed!");

    int sum_num_blocks = divup(data_num_blocks, size_per_block);
    int sum_pad = size_per_block - (data_num_blocks % size_per_block);
    int sum_total_size = sum_num_blocks * size_per_block;

    int *dev_sums;
    cudaMalloc((void **)&dev_sums, sum_total_size * sizeof(int));
    checkCUDAError("cudaMalloc dev_block_sums failed!");
    cudaMemset(dev_sums, 0, sum_pad * sizeof(int));
    checkCUDAError("cudaMemset dev_block_sums failed!");

    timer().startGpuTimer();
    // int depth = ilog2ceil(n);
    kern_scan<<<data_num_blocks, block_size, size_per_block * sizeof(int)>>>(
        size_per_block, dev_data, dev_sums + sum_pad, true);

    kern_scan<<<sum_num_blocks, block_size, size_per_block * sizeof(int)>>>(
        size_per_block, dev_sums, nullptr, false);

    kern_inc<<<sum_num_blocks, block_size>>>(size_per_block, data_num_blocks, dev_data,
                                             dev_sums);

    timer().endGpuTimer();

    cudaMemcpy(odata, dev_data + data_pad, n * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaFree(dev_data);
    cudaFree(dev_sums);
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
