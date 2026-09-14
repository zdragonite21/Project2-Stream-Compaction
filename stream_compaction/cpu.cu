#include "cpu.h"
#include <cstdio>

#include "common.h"
#include <vector>

namespace StreamCompaction {
namespace CPU {
using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer &timer() {
    static PerformanceTimer timer;
    return timer;
}

void run_scan(int n, int* odata, const int *idata) {
    odata[0] = 0;
    for (int i = 1; i < n; ++i) {
        odata[i] = odata[i - 1] + idata[i - 1];
    }
}

/**
 * CPU scan (prefix sum).
 * For performance analysis, this is supposed to be a simple for loop.
 * (Optional) For better understanding before starting moving to GPU, you can
 * simulate your GPU scan in this function first.
 */
void scan(int n, int *odata, const int *idata) {
    timer().startCpuTimer();
    run_scan(n, odata, idata);
    timer().endCpuTimer();
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
    timer().startCpuTimer();
    int next = 0;
    for (int i = 0; i < n; ++i) {
        if (idata[i] != 0) {
            odata[next] = idata[i];
            next++;
        }
    }
    timer().endCpuTimer();
    return next;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
    timer().startCpuTimer();
    std::vector<int> tmp(n);
    std::vector<int> scan_res(n);

    for (int i = 0; i < n; ++i) {
        tmp[i] = idata[i] == 0 ? 0 : 1;
    }

    run_scan(n, scan_res.data(), tmp.data());

    for (int i = 0; i < n; ++i) {
        if (tmp[i] == 1) {
            odata[scan_res[i]] = idata[i];
        }
    }
    timer().endCpuTimer();
    // scan_res is exclusive
    return scan_res[n - 1] - tmp[n - 1];
}
} // namespace CPU
} // namespace StreamCompaction
