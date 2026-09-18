#include "common.h"
#include "efficient.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define threads_per_block 128
#define items_per_thread_log2 3 // 8 total per thread (int4, int4)
#define vec_width_log2 2        // int4s
#define items_per_thread (1 << items_per_thread_log2)
#define vec_width (1 << vec_width_log2)
#define chunk_size (threads_per_block << items_per_thread_log2)
#define vec_chunk_size (chunk_size >> vec_width_log2)

#define NUM_BANKS_LOG2 5
#define NUM_BANKS (1 << NUM_BANKS_LOG2)
// floor(n / 32)
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS_LOG2)

// pad shared memory for the additional indices used for elmiinating bank
// conflicts
#define shared_mem_size (vec_chunk_size + vec_chunk_size / NUM_BANKS)

namespace StreamCompaction {
namespace Efficient {
using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer &timer() {
    static PerformanceTimer timer;
    return timer;
}

__global__ void kern_inc(int num_chunks, int *data, int *sums) {
    int chunk = blockIdx.x;
    if (chunk >= num_chunks) {
        return;
    }

    int4 *data4 = reinterpret_cast<int4 *>(data);

    int inc = sums[chunk];

    int vec_base = (chunk * chunk_size) >> 2;
    int vec_ai = threadIdx.x;
    int vec_bi = threadIdx.x + threads_per_block;

    int4 a = data4[vec_base + vec_ai];
    int4 b = data4[vec_base + vec_bi];
    a.x += inc;
    a.y += inc;
    a.z += inc;
    a.w += inc;

    b.x += inc;
    b.y += inc;
    b.z += inc;
    b.w += inc;

    data4[vec_base + vec_ai] = a;
    data4[vec_base + vec_bi] = b;
}

__global__ void kern_scan(int *data, int *sums, bool store_sum) {
    extern __shared__ int temp[];

    int thid = threadIdx.x;
    int vec_block_base = (blockIdx.x * chunk_size) >> 2;

    int4 *data4 = reinterpret_cast<int4 *>(data);

    // organized this way for coalesced memory access
    int vec_ai = thid;
    int vec_bi = thid + (chunk_size >> 3);
    int4 adata = data4[vec_block_base + vec_ai];
    int4 bdata = data4[vec_block_base + vec_bi];

    int a1 = adata.x;
    int a2 = a1 + adata.y;
    int a3 = a2 + adata.z;

    int b1 = bdata.x;
    int b2 = b1 + bdata.y;
    int b3 = b2 + bdata.z;

    int ai = thid;
    int bi = thid + (vec_chunk_size >> 1);
    temp[ai + CONFLICT_FREE_OFFSET(ai)] = a3 + adata.w;
    temp[bi + CONFLICT_FREE_OFFSET(bi)] = b3 + bdata.w;

    // upsweep
    int offset = 1;
#pragma unroll
    for (int d = vec_chunk_size >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int base = (offset << 1) * thid;
            int ai = base + offset - 1;
            int bi = base + (offset << 1) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // downsweep
    if (thid == 0) {
        // have the first thread clear the last element
        int end = vec_chunk_size - 1;
        end += CONFLICT_FREE_OFFSET(end);
        if (store_sum) {
            sums[blockIdx.x] = temp[end];
        }
        temp[end] = 0;
    }
#pragma unroll
    for (int d = 1; d < vec_chunk_size; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int base = (offset << 1) * thid;
            int ai = base + offset - 1;
            int bi = base + (offset << 1) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();
    int a0 = temp[ai + CONFLICT_FREE_OFFSET(ai)];
    a1 += a0;
    a2 += a0;
    a3 += a0;

    int b0 = temp[bi + CONFLICT_FREE_OFFSET(bi)];
    b1 += b0;
    b2 += b0;
    b3 += b0;

    data4[vec_block_base + vec_ai] = make_int4(a0, a1, a2, a3);
    data4[vec_block_base + vec_bi] = make_int4(b0, b1, b2, b3);
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
        int *dev_sums = dev_heap + p.padded_size;

        kern_scan<<<p.num_chunks, threads_per_block,
                    shared_mem_size * sizeof(int)>>>(dev_heap,
                                                     dev_sums + sp.pad, true);

        recursive_scan(sp, dev_sums);

        kern_inc<<<p.num_chunks, threads_per_block>>>(p.num_chunks, dev_heap,
                                                      dev_sums + sp.pad);
    } else {
        kern_scan<<<p.num_chunks, threads_per_block,
                    shared_mem_size * sizeof(int)>>>(dev_heap, nullptr, false);
    }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    if (n == 0) {
        return;
    }

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
    // mem set to 0 so that any padding we have are 0s
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

__global__ void kern_eval(int num_chunks, int *data, int *d_bool) {
    int chunk = blockIdx.x;
    if (chunk >= num_chunks) {
        return;
    }

    int4 *data4 = reinterpret_cast<int4 *>(data);
    int4 *bool4 = reinterpret_cast<int4 *>(d_bool);

    int vec_base = (chunk * chunk_size) >> 2;
    int vec_ai = threadIdx.x;
    int vec_bi = threadIdx.x + threads_per_block;

    int4 abool;
    int4 bbool;

    int4 a = data4[vec_base + vec_ai];
    int4 b = data4[vec_base + vec_bi];
    abool.x = a.x == 0 ? 0 : 1;
    abool.y = a.y == 0 ? 0 : 1;
    abool.z = a.z == 0 ? 0 : 1;
    abool.w = a.w == 0 ? 0 : 1;

    bbool.x = b.x == 0 ? 0 : 1;
    bbool.y = b.y == 0 ? 0 : 1;
    bbool.z = b.z == 0 ? 0 : 1;
    bbool.w = b.w == 0 ? 0 : 1;

    bool4[vec_base + vec_ai] = abool;
    bool4[vec_base + vec_bi] = bbool;
}

__global__ void kern_compact(int num_chunks, int *odata, const int *idata,
                             const int *d_bool) {
    int chunk = blockIdx.x;
    if (chunk >= num_chunks) {
        return;
    }

    const int4 *idata4 = reinterpret_cast<const int4 *>(idata);
    const int4 *bool4 = reinterpret_cast<const int4 *>(d_bool);

    int vec_base = (chunk * chunk_size) >> 2;
    int vec_ai = threadIdx.x;
    int vec_bi = threadIdx.x + threads_per_block;

    int4 ai = bool4[vec_base + vec_ai];
    int4 bi = bool4[vec_base + vec_bi];
    int4 vec_a = idata4[vec_base + vec_ai];
    int4 vec_b = idata4[vec_base + vec_bi];

    if (vec_a.x != 0) {
        odata[ai.x] = vec_a.x;
    }
    if (vec_a.y != 0) {
        odata[ai.y] = vec_a.y;
    }
    if (vec_a.z != 0) {
        odata[ai.z] = vec_a.z;
    }
    if (vec_a.w != 0) {
        odata[ai.w] = vec_a.w;
    }

    if (vec_b.x != 0) {odata[bi.x] = vec_b.x;}
    if (vec_b.y != 0) {odata[bi.y] = vec_b.y;}
    if (vec_b.z != 0) {odata[bi.z] = vec_b.z;}
    if (vec_b.w != 0) {odata[bi.w] = vec_b.w;}
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
    if (n == 0) {
        return -1;
    }

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
    // mem set to 0 so that any padding we have are 0s
    cudaMemset(dev_heap, 0, heap_size * sizeof(int));
    checkCUDAError("cudaMemset heap failed!");
    cudaMemcpy(dev_heap + p.pad, idata, n * sizeof(int),
               cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy heap failed!");

    int *dev_bool;
    cudaMalloc((void **)&dev_bool, p.padded_size * sizeof(int));
    checkCUDAError("cudaMalloc bool failed!");
    cudaMemset(dev_bool, 0, p.padded_size * sizeof(int));
    checkCUDAError("cudaMemset bool failed!");

    int *dev_odata;
    cudaMalloc((void **)&dev_odata, p.padded_size * sizeof(int));
    checkCUDAError("cudaMalloc odata failed!");

    timer().startGpuTimer();
    kern_eval<<<p.num_chunks, threads_per_block>>>(p.num_chunks, dev_heap,
                                                   dev_bool);
    recursive_scan(p, dev_bool);
    kern_compact<<<p.num_chunks, threads_per_block>>>(p.num_chunks, dev_odata,
                                                      dev_heap, dev_bool);

    timer().endGpuTimer();

    int compact_num;
    cudaMemcpy(&compact_num, dev_bool + (p.padded_size - 1), sizeof(int),
               cudaMemcpyDeviceToHost);
    compact_num = compact_num + (idata[n - 1] != 0);
    cudaMemcpy(odata, dev_odata, compact_num * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaFree(dev_heap);
    cudaFree(dev_bool);
    cudaFree(dev_odata);

    return compact_num;
}
} // namespace Efficient
} // namespace StreamCompaction
