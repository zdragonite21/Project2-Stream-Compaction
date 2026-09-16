#include "common.h"
#include "efficient.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define threads_per_block 32

namespace StreamCompaction {
namespace Efficient {
using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer &timer() {
    static PerformanceTimer timer;
    return timer;
}

__global__ void kern_inc(int chunk_size, int num_chunks, int *data, int *sums) {
    // a thread is spawned per block
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid >= num_chunks) {
        return;
    }
    int base = thid * chunk_size;
    int inc = sums[thid];
    for (int i = 0; i < chunk_size; ++i) {
        data[base + i] += inc;
    }
}

__global__ void kern_scan(int chunk_size, int *data, int *sums, bool store_sum) {
    extern __shared__ int temp[];

    int local_thid = threadIdx.x;
    int block_base = blockIdx.x * chunk_size;

    temp[local_thid << 1] = data[block_base + (local_thid << 1)];
    temp[(local_thid << 1) + 1] = data[block_base + (local_thid << 1) + 1];

    // upsweep
    int offset = 1;
    for (int d = chunk_size >> 1; d > 0; d >>= 1) {
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
    int total;
    if (local_thid == 0) {
        // have the first thread clear the last element
        total = temp[chunk_size - 1];
        temp[chunk_size - 1] = 0;
    }

    for (int d = 1; d < chunk_size; d <<= 1) {
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

    if (store_sum && local_thid == 0) {
        sums[blockIdx.x] = total;
    }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // one thread handles two elements
    int chunk_size = (threads_per_block << 1);

    int num_chunks = divup(n, chunk_size);
    int pad = (chunk_size - (n % chunk_size)) % chunk_size;
    int padded_size = num_chunks * chunk_size;

    // pad the front with zeros
    int *dev_data;
    cudaMalloc((void **)&dev_data, padded_size * sizeof(int));
    checkCUDAError("cudaMalloc dev_data failed!");
    cudaMemset(dev_data, 0, pad * sizeof(int));
    checkCUDAError("cudaMemset dev_data failed!");
    cudaMemcpy(dev_data + pad, idata, n * sizeof(int),
               cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy dev_data failed!");

    int num_sum_chunks = divup(num_chunks, chunk_size);
    int sum_pad = (chunk_size - (num_chunks % chunk_size)) % chunk_size;
    int sum_padded_size = num_sum_chunks * chunk_size;

    // pad the sum array as well
    int *dev_sums;
    cudaMalloc((void **)&dev_sums, sum_padded_size * sizeof(int));
    checkCUDAError("cudaMalloc dev_block_sums failed!");
    cudaMemset(dev_sums, 0, sum_pad * sizeof(int));
    checkCUDAError("cudaMemset dev_block_sums failed!");

    timer().startGpuTimer();

    kern_scan<<<num_chunks, threads_per_block, chunk_size * sizeof(int)>>>(
        chunk_size, dev_data, dev_sums + sum_pad, true);

    kern_scan<<<num_sum_chunks, threads_per_block, chunk_size * sizeof(int)>>>(
        chunk_size, dev_sums, nullptr, false);

    kern_inc<<<num_sum_chunks, threads_per_block>>>(chunk_size, num_chunks, dev_data,
                                             dev_sums + sum_pad);

    timer().endGpuTimer();

    cudaMemcpy(odata, dev_data + pad, n * sizeof(int),
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
