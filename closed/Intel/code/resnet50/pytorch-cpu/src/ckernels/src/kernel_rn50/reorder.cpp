
#include <kernel/kernel_includes.hpp>
static constexpr void *__stream = &sc::runtime::default_stream;

extern int8_t reorder_data[0];
static constexpr int8_t* __module_data = reorder_data;
alignas(64) static int8_t __uninitialized_data[0UL];

static bool reorder_9x64x56x56_ABCD_ACDB(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder_9x2048x7x7_ACDB_ABCD(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder_4x64x56x56_ABCD_ACDB(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder_4x2048x7x7_ACDB_ABCD(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder_2x64x56x56_ABCD_ACDB(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder_2x2048x7x7_ACDB_ABCD(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));



static bool reorder_9x64x56x56_ABCD_ACDB(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_11373___fuseiter_11374_2931 = 0UL; fused_0_fuseiter_11373___fuseiter_11374_2931 < 576UL; fused_0_fuseiter_11373___fuseiter_11374_2931 += 1UL) {
    for (uint64_t _fuseiter_11375 = 0UL; _fuseiter_11375 < 56UL; _fuseiter_11375 += 1UL) {
      for (uint64_t _fuseiter_11376 = 0UL; _fuseiter_11376 < 56UL; _fuseiter_11376 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((fused_0_fuseiter_11373___fuseiter_11374_2931 / 64UL) * 200704UL) + (((fused_0_fuseiter_11373___fuseiter_11374_2931 % 64UL) * 3136UL) + ((_fuseiter_11375 * 56UL) + _fuseiter_11376)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0_fuseiter_11373___fuseiter_11374_2931 / 64UL) * 200704UL) + ((_fuseiter_11375 * 3584UL) + ((_fuseiter_11376 * 64UL) + (fused_0_fuseiter_11373___fuseiter_11374_2931 % 64UL))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder_9x2048x7x7_ACDB_ABCD(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_11377___fuseiter_11378_2932 = 0UL; fused_0_fuseiter_11377___fuseiter_11378_2932 < 63UL; fused_0_fuseiter_11377___fuseiter_11378_2932 += 1UL) {
    for (uint64_t _fuseiter_11379 = 0UL; _fuseiter_11379 < 7UL; _fuseiter_11379 += 1UL) {
      for (uint64_t _fuseiter_11380 = 0UL; _fuseiter_11380 < 2048UL; _fuseiter_11380 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((fused_0_fuseiter_11377___fuseiter_11378_2932 / 7UL) * 100352UL) + (((fused_0_fuseiter_11377___fuseiter_11378_2932 % 7UL) * 14336UL) + ((_fuseiter_11379 * 2048UL) + _fuseiter_11380)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0_fuseiter_11377___fuseiter_11378_2932 / 7UL) * 100352UL) + ((_fuseiter_11380 * 49UL) + (((fused_0_fuseiter_11377___fuseiter_11378_2932 % 7UL) * 7UL) + _fuseiter_11379)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder_4x64x56x56_ABCD_ACDB(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_11381___fuseiter_11382_2933 = 0UL; fused_0_fuseiter_11381___fuseiter_11382_2933 < 256UL; fused_0_fuseiter_11381___fuseiter_11382_2933 += 1UL) {
    for (uint64_t _fuseiter_11383 = 0UL; _fuseiter_11383 < 56UL; _fuseiter_11383 += 1UL) {
      for (uint64_t _fuseiter_11384 = 0UL; _fuseiter_11384 < 56UL; _fuseiter_11384 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((fused_0_fuseiter_11381___fuseiter_11382_2933 / 64UL) * 200704UL) + (((fused_0_fuseiter_11381___fuseiter_11382_2933 % 64UL) * 3136UL) + ((_fuseiter_11383 * 56UL) + _fuseiter_11384)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0_fuseiter_11381___fuseiter_11382_2933 / 64UL) * 200704UL) + ((_fuseiter_11383 * 3584UL) + ((_fuseiter_11384 * 64UL) + (fused_0_fuseiter_11381___fuseiter_11382_2933 % 64UL))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder_4x2048x7x7_ACDB_ABCD(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_11385___fuseiter_11386_2934 = 0UL; fused_0_fuseiter_11385___fuseiter_11386_2934 < 28UL; fused_0_fuseiter_11385___fuseiter_11386_2934 += 1UL) {
    for (uint64_t _fuseiter_11387 = 0UL; _fuseiter_11387 < 7UL; _fuseiter_11387 += 1UL) {
      for (uint64_t _fuseiter_11388 = 0UL; _fuseiter_11388 < 2048UL; _fuseiter_11388 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((fused_0_fuseiter_11385___fuseiter_11386_2934 / 7UL) * 100352UL) + (((fused_0_fuseiter_11385___fuseiter_11386_2934 % 7UL) * 14336UL) + ((_fuseiter_11387 * 2048UL) + _fuseiter_11388)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0_fuseiter_11385___fuseiter_11386_2934 / 7UL) * 100352UL) + ((_fuseiter_11388 * 49UL) + (((fused_0_fuseiter_11385___fuseiter_11386_2934 % 7UL) * 7UL) + _fuseiter_11387)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder_2x64x56x56_ABCD_ACDB(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_11389___fuseiter_11390_2935 = 0UL; fused_0_fuseiter_11389___fuseiter_11390_2935 < 128UL; fused_0_fuseiter_11389___fuseiter_11390_2935 += 1UL) {
    for (uint64_t _fuseiter_11391 = 0UL; _fuseiter_11391 < 56UL; _fuseiter_11391 += 1UL) {
      for (uint64_t _fuseiter_11392 = 0UL; _fuseiter_11392 < 56UL; _fuseiter_11392 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((fused_0_fuseiter_11389___fuseiter_11390_2935 / 64UL) * 200704UL) + (((fused_0_fuseiter_11389___fuseiter_11390_2935 % 64UL) * 3136UL) + ((_fuseiter_11391 * 56UL) + _fuseiter_11392)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0_fuseiter_11389___fuseiter_11390_2935 / 64UL) * 200704UL) + ((_fuseiter_11391 * 3584UL) + ((_fuseiter_11392 * 64UL) + (fused_0_fuseiter_11389___fuseiter_11390_2935 % 64UL))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder_2x2048x7x7_ACDB_ABCD(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_11393___fuseiter_11394_2936 = 0UL; fused_0_fuseiter_11393___fuseiter_11394_2936 < 14UL; fused_0_fuseiter_11393___fuseiter_11394_2936 += 1UL) {
    for (uint64_t _fuseiter_11395 = 0UL; _fuseiter_11395 < 7UL; _fuseiter_11395 += 1UL) {
      for (uint64_t _fuseiter_11396 = 0UL; _fuseiter_11396 < 2048UL; _fuseiter_11396 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((fused_0_fuseiter_11393___fuseiter_11394_2936 / 7UL) * 100352UL) + (((fused_0_fuseiter_11393___fuseiter_11394_2936 % 7UL) * 14336UL) + ((_fuseiter_11395 * 2048UL) + _fuseiter_11396)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0_fuseiter_11393___fuseiter_11394_2936 / 7UL) * 100352UL) + ((_fuseiter_11396 * 49UL) + (((fused_0_fuseiter_11393___fuseiter_11394_2936 % 7UL) * 7UL) + _fuseiter_11395)))] = __cached_1;
      }
    }
  }
  return true;
}

