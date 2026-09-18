/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include "testing_helpers.hpp"
#include <array>
#include <cstdio>
#include <functional>
#include <iostream>
#include <stream_compaction/bank.h>
#include <stream_compaction/chunk.h>
#include <stream_compaction/config.h>
#include <stream_compaction/cpu.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/heap.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/thrust.h>
#include <string>

const int NPOT = TEST_SIZE - NPOT_DIFF;
int *a = new int[TEST_SIZE];
int *b = new int[TEST_SIZE];
int *c = new int[TEST_SIZE];

void profile();

int main(int argc, char *argv[]) {

#if PROFILE
    profile();
#else
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(TEST_SIZE - 1, a,
             50); // Leave a 0 at the end to test that edge case
    a[TEST_SIZE - 1] = 0;
    printArray(TEST_SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your
    // StreamCompaction::CPU::scan is correct. At first all cases passed because
    // b && c are all zeroes.
    zeroArray(TEST_SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(TEST_SIZE, b, a);
    printElapsedTime(
        StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
        "(std::chrono Measured)");
    printArray(TEST_SIZE, b, true);

    zeroArray(TEST_SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(
        StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
        "(std::chrono Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(TEST_SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(TEST_SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer()
                         .getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    // printArray(SIZE, c, true);
    printCmpResult(TEST_SIZE, b, c);

    /* For bug-finding only: Array of 1s to help find bugs in stream compaction
    or scan onesArray(SIZE, c); printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true); */

    zeroArray(TEST_SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer()
                         .getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    // printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(TEST_SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(TEST_SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer()
                         .getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    // printArray(SIZE, c, true);
    printCmpResult(TEST_SIZE, b, c);

    zeroArray(TEST_SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer()
                         .getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    // printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(TEST_SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(TEST_SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer()
                         .getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    // printArray(SIZE, c, true);
    printCmpResult(TEST_SIZE, b, c);

    zeroArray(TEST_SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer()
                         .getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    // printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(TEST_SIZE - 1, a,
             4); // Leave a 0 at the end to test that edge case
    a[TEST_SIZE - 1] = 0;
    printArray(TEST_SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you
    // implement We use b for further comparison. Make sure your
    // StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(TEST_SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(TEST_SIZE, b, a);
    printElapsedTime(
        StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
        "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(TEST_SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(
        StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
        "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(TEST_SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(TEST_SIZE, c, a);
    printElapsedTime(
        StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
        "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(TEST_SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(TEST_SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer()
                         .getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    // printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(TEST_SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer()
                         .getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    // printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    system("pause"); // stop Win32 console from closing on exit
#endif
    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}

struct ScanImpl {
    std::function<void(int, int *, const int *)> scan;
    std::function<float(void)> elapsed;
    std::string name;
};

void profile() {
    std::array<ScanImpl, 7> scan_funcs{
        ScanImpl{StreamCompaction::CPU::scan,
                 [] {
                     return StreamCompaction::CPU::timer()
                         .getCpuElapsedTimeForPreviousOperation();
                 },
                 "cpu scan"},

        ScanImpl{StreamCompaction::Naive::scan,
                 [] {
                     return StreamCompaction::Naive::timer()
                         .getGpuElapsedTimeForPreviousOperation();
                 },
                 "naive gpu scan"},
        ScanImpl{StreamCompaction::Chunk::scan,
                 [] {
                     return StreamCompaction::Chunk::timer()
                         .getGpuElapsedTimeForPreviousOperation();
                 },
                 "recursive shared memory chunk scan"},
        ScanImpl{StreamCompaction::Heap::scan,
                 [] {
                     return StreamCompaction::Heap::timer()
                         .getGpuElapsedTimeForPreviousOperation();
                 },
                 "recursive shared memory heap scan"},
        ScanImpl{StreamCompaction::Bank::scan,
                 [] {
                     return StreamCompaction::Bank::timer()
                         .getGpuElapsedTimeForPreviousOperation();
                 },
                 "bank free conflict scan"},
        ScanImpl{StreamCompaction::Efficient::scan,
                 [] {
                     return StreamCompaction::Efficient::timer()
                         .getGpuElapsedTimeForPreviousOperation();
                 },
                 "vectorized gpu scan"},
        ScanImpl{StreamCompaction::Thrust::scan,
                 [] {
                     return StreamCompaction::Thrust::timer()
                         .getGpuElapsedTimeForPreviousOperation();
                 },
                 "thrust scan"},
    };

    printf("\n");
    printf("****************\n");
    printf("** PROFILING **\n");
    printf("****************\n");

    std::cout << "RUNS = " << RUNS << std::endl;

#if CORRECTNESS
    genArray(TEST_SIZE - 1, a, 50);
    a[TEST_SIZE - 1] = 0;
    printArray(TEST_SIZE, a, true);
    ScanImpl cpu = scan_funcs[0];
    printf("*********** checking correctness... ***********\n");
    zeroArray(TEST_SIZE, b);
    cpu.scan(TEST_SIZE, b, a);

    for (int i = 0; i < scan_funcs.size(); ++i) {
        ScanImpl impl = scan_funcs[i];

        zeroArray(TEST_SIZE, c);
        printDesc((impl.name + ", power-of-two").c_str());
        impl.scan(TEST_SIZE, c, a);
        printElapsedTime(impl.elapsed(), "(ms)");
        printCmpResult(TEST_SIZE, b, c);

        zeroArray(TEST_SIZE, c);
        printDesc((impl.name + ", non-power-of-two").c_str());
        impl.scan(NPOT, c, a);
        printElapsedTime(impl.elapsed(), "(ms)");
        printCmpResult(NPOT, b, c);
    }
#endif

#if WARM_UP
    genArray(TEST_SIZE, a, 50);
    printArray(TEST_SIZE, a, true);
    printf("*********** warm-up (POT) ***********\n");
    for (int i = 0; i < scan_funcs.size(); ++i) {
        ScanImpl impl = scan_funcs[i];
        printDesc((impl.name + ", power-of-two").c_str());

        float total_time{};

        for (int j = 0; j < RUNS; ++j) {
            zeroArray(TEST_SIZE, c);
            impl.scan(TEST_SIZE, c, a);
            total_time += impl.elapsed();
        }
        printElapsedTime(static_cast<float>(total_time / RUNS), "(avg ms)");
    }

    printf("*********** warm-up (NON-POT) ***********\n");
    for (int i = 0; i < scan_funcs.size(); ++i) {
        ScanImpl impl = scan_funcs[i];
        printDesc((impl.name + ", non-power-of-two").c_str());

        float total_time{};

        for (int j = 0; j < RUNS; ++j) {
            zeroArray(TEST_SIZE, c);
            impl.scan(NPOT, c, a);
            total_time += impl.elapsed();
        }
        printElapsedTime(static_cast<float>(total_time / RUNS), "(avg ms)");
    }
#endif

#if BLOCK_SIZE_ALL
    genArray(TEST_SIZE, a, 50);
    constexpr int block_size = threads_per_block;
    printf("\nblock size");
    for (const auto &impl : scan_funcs) {
        printf(",%s", impl.name.c_str());
    }
    printf("\n%d", block_size);
    for (const auto &impl : scan_funcs) {
        float total_time{};
        for (int j = 0; j < RUNS; ++j) {
            zeroArray(TEST_SIZE, c);
            impl.scan(TEST_SIZE, c, a);
            total_time += impl.elapsed();
        }
        printf(",%.3f", total_time / RUNS);
    }
    printf("\n");
#endif

#if ARRAY_SIZE_ALL
    printf("\narray size (all)");
    for (const auto &impl : scan_funcs) {
        printf(",%s (ms)", impl.name.c_str());
    }
    for (int size_log2 = 10; size_log2 <= 26; ++size_log2) {
        int size = 1 << size_log2;
        int npot = size - NPOT_DIFF;
#if ARRAY_SIZE_ALL_NPOT
        size = npot;
#endif
        int *sa = new int[size];
        int *sb = new int[size];
        int *sc = new int[size];
        genArray(size, sa, 50);
        printf("\n%d", size_log2);
        for (int i = 0; i < scan_funcs.size(); ++i) {
            if (size_log2 > 26 && i < 2) {
                printf(",-1");
                continue;
            }
            ScanImpl &impl = scan_funcs[i];
            float total_time{};
            for (int j = 0; j < RUNS; ++j) {
                zeroArray(size, sc);
                impl.scan(size, sc, sa);
                total_time += impl.elapsed();
            }
            printf(",%.3f", total_time / RUNS);
        }
        printf("\n");
        free(sa);
        free(sb);
        free(sc);
    }
#endif

#if ARRAY_SIZE_EFFICIENT_THRUST
    printf("\narray size (efficient vs thrust)");
    for (int i = 5; i < scan_funcs.size(); ++i) {
        printf(",%s (ms)", scan_funcs[i].name.c_str());
    }
    for (int size_log2 = 10; size_log2 <= 28; ++size_log2) {
        int size = 1 << size_log2;
        int npot = size - NPOT_DIFF;
        int *sa = new int[size];
        int *sb = new int[size];
        int *sc = new int[size];
        genArray(size, sa, 50);
        printf("\n%d", size_log2);
        for (int i = 5; i < scan_funcs.size(); ++i) {
            ScanImpl &impl = scan_funcs[i];
            float total_time{};
            for (int j = 0; j < RUNS; ++j) {
                zeroArray(size, sc);
                impl.scan(size, sc, sa);
                total_time += impl.elapsed();
            }
            printf(",%.3f", total_time / RUNS);
        }
        printf("\n");
        free(sa);
        free(sb);
        free(sc);
    }
#endif
}
