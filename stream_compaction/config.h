#define PROFILE 1
#define CORRECTNESS 0
#define WARM_UP 0
#define BLOCK_SIZE_ALL 0
#define ARRAY_SIZE_ALL 0
#define ARRAY_SIZE_ALL_NPOT 0
#define ARRAY_SIZE_EFFICIENT_THRUST 0

#define threads_per_block 128
constexpr int NPOT_DIFF = 3;
constexpr int RUNS = 20;

constexpr int TEST_SIZE = 1 << 25;
