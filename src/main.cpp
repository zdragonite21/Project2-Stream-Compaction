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
#include <stream_compaction/cpu.h>
#include <stream_compaction/chunk.h>
#include <stream_compaction/heap.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/bank.h>
#include <string>
#include <iostream>

#define PROFILE 1

const int SIZE = 1 << 25;  // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

void profile();

int main(int argc, char *argv[]) {

#if PROFILE
    profile();
    return 0;
#endif
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50); // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your
    // StreamCompaction::CPU::scan is correct. At first all cases passed because
    // b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(
        StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
        "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(
        StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
        "(std::chrono Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer()
                         .getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    // printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    /* For bug-finding only: Array of 1s to help find bugs in stream compaction
    or scan onesArray(SIZE, c); printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true); */

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer()
                         .getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    // printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer()
                         .getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    // printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer()
                         .getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    // printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer()
                         .getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    // printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
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

    genArray(SIZE - 1, a, 4); // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you
    // implement We use b for further comparison. Make sure your
    // StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(
        StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
        "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(
        StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
        "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(
        StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
        "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer()
                         .getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    // printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer()
                         .getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    // printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
}

struct ScanImpl {
    std::function<void(int, int *, const int *)> scan;
    std::function<float(void)> elapsed;
    std::string name;
};

void profile() {
    constexpr int runs = 10;

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
                 "efficient gpu scan"},
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

    std::cout << "runs = " << runs << std::endl;

    genArray(SIZE - 1, a, 50);
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    ScanImpl cpu = scan_funcs[0];
    printf("*********** checking correctness... ***********\n");
    zeroArray(SIZE, b);
    cpu.scan(SIZE, b, a);

    for (int i = 0; i < scan_funcs.size(); ++i) {
        ScanImpl impl = scan_funcs[i];

        zeroArray(SIZE, c);
        printDesc((impl.name + ", power-of-two").c_str());
        impl.scan(SIZE, c, a);
        printElapsedTime(impl.elapsed(), "(ms)");
        printCmpResult(SIZE, b, c);

        zeroArray(SIZE, c);
        printDesc((impl.name + ", non-power-of-two").c_str());
        impl.scan(NPOT, c, a);
        printElapsedTime(impl.elapsed(), "(ms)");
        printCmpResult(NPOT, b, c);
    }

    printf("*********** profiling (POT) ***********\n");
    for (int i = 0; i < scan_funcs.size(); ++i) {
        ScanImpl impl = scan_funcs[i];
        printDesc((impl.name + ", power-of-two").c_str());

        float total_time{};

        for (int j = 0; j < runs; ++j) {
            zeroArray(SIZE, c);
            impl.scan(SIZE, c, a);
            total_time += impl.elapsed();
        }
        printElapsedTime(static_cast<float>(total_time / runs), "(avg ms)");
    }

    printf("*********** profiling (NON-POT) ***********\n");
    for (int i = 0; i < scan_funcs.size(); ++i) {
        ScanImpl impl = scan_funcs[i];
        printDesc((impl.name + ", non-power-of-two").c_str());

        float total_time{};

        for (int j = 0; j < runs; ++j) {
            zeroArray(SIZE, c);
            impl.scan(NPOT, c, a);
            total_time += impl.elapsed();
        }
        printElapsedTime(static_cast<float>(total_time / runs), "(avg ms)");
    }

    delete[] a;
    delete[] b;
    delete[] c;
}
