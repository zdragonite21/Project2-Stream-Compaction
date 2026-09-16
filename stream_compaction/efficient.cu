#include "common.h"
#include "efficient.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define threads_per_block 1024
// one thread handles two elements
#define chunk_size 2 * threads_per_block

namespace StreamCompaction {
namespace Efficient {
using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer &timer() {
    static PerformanceTimer timer;
    return timer;
}

__global__ void kern_inc(int num_chunks, int *data, int *sums) {
    // a thread is spawned per chunk
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid >= num_chunks) {
        return;
    }
    int base = thid * chunk_size;
    int inc = sums[thid];
#pragma unroll
    for (int i = 0; i < chunk_size; ++i) {
        data[base + i] += inc;
    }
}

__global__ void kern_scan(int *data, int *sums, bool store_sum) {
    extern __shared__ int temp[];

    int local_thid = threadIdx.x;
    int block_base = blockIdx.x * chunk_size;

    temp[local_thid << 1] = data[block_base + (local_thid << 1)];
    temp[(local_thid << 1) + 1] = data[block_base + (local_thid << 1) + 1];

    // upsweep
    int offset = 1;
#pragma unroll
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
#pragma unroll
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

struct params {
    int num_chunks;
    int pad;
    int padded_size;
};

params compute_params(int n) {
    int num_chunks = divup(n, chunk_size);
    int pad = (chunk_size - (n % chunk_size)) % chunk_size;
    int padded_size = num_chunks * chunk_size;

    return params{
        num_chunks,
        pad,
        padded_size,
    };
}

void recursive_scan(params p, int *dev_heap) {
    params sp = compute_params(p.num_chunks);

    if (p.num_chunks > 1) {
        int num_inc_blocks = divup(p.num_chunks, threads_per_block);

        int *dev_sums = dev_heap + p.padded_size;

        kern_scan<<<p.num_chunks, threads_per_block,
                    chunk_size * sizeof(int)>>>(dev_heap, dev_sums + sp.pad,
                                                true);
        // checkCUDAError("kern_scan write sum failed");

        recursive_scan(sp, dev_sums);

        kern_inc<<<num_inc_blocks, threads_per_block>>>(p.num_chunks, dev_heap,
                                                        dev_sums + sp.pad);
        // checkCUDAError("kern_inc failed");
    } else {
        kern_scan<<<p.num_chunks, threads_per_block,
                    chunk_size * sizeof(int)>>>(dev_heap, nullptr, false);
        // checkCUDAError("kern_scan failed");
    }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    params p = compute_params(n);

    // compute the exact total amount of memory needed
    int heap_size = p.padded_size;
    params a = p;
    while (a.num_chunks > 1) {
        params b = compute_params(a.num_chunks);
        heap_size += b.padded_size;
        a = b;
    }

    // store all data and block sum arrays in one heap
    int *dev_heap;
    cudaMalloc((void **)&dev_heap, heap_size * sizeof(int));
    checkCUDAError("cudaMalloc heap failed!");
    cudaMemset(dev_heap, 0, heap_size * sizeof(int));
    checkCUDAError("cudaMemset heap failed!");
    cudaMemcpy(dev_heap + p.pad, idata, n * sizeof(int),
               cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy heap failed!");

    timer().startGpuTimer();
    recursive_scan(p, dev_heap);
    timer().endGpuTimer();

    cudaMemcpy(odata, dev_heap + p.pad, n * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaFree(dev_heap);
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
