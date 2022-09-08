
#include <kernel/kernel_includes.hpp>
static constexpr void *__stream = &sc::runtime::default_stream;

extern int8_t rn50_backbone_bs2_data[222400];
static constexpr int8_t* __module_data = rn50_backbone_bs2_data;
alignas(64) static int8_t __uninitialized_data[23657592UL];

static void __init_const_globals(int8_t* __restrict__ backbone_output, int8_t* __restrict__ backbone_input, float* __restrict__ res2a_weight_b, float* __restrict__ res2a_bias_b, float* __restrict__ res2a_weight_0, float* __restrict__ res2a_bias_0, float* __restrict__ res2a_weight_1, float* __restrict__ res2a_bias_1, float* __restrict__ res2a_weight_2, float* __restrict__ res2a_bias_2, float* __restrict__ res2b_weight_0, float* __restrict__ res2b_bias_0, float* __restrict__ res2b_weight_1, float* __restrict__ res2b_bias_1, float* __restrict__ res2b_weight_2, float* __restrict__ res2b_bias_2, float* __restrict__ res2c_weight_0, float* __restrict__ res2c_bias_0, float* __restrict__ res2c_weight_1, float* __restrict__ res2c_bias_1, float* __restrict__ res2c_weight_2, float* __restrict__ res2c_bias_2, float* __restrict__ res3a_weight_b, float* __restrict__ res3a_bias_b, float* __restrict__ res3a_weight_0, float* __restrict__ res3a_bias_0, float* __restrict__ res3a_weight_1, float* __restrict__ res3a_bias_1, float* __restrict__ res3a_weight_2, float* __restrict__ res3a_bias_2, float* __restrict__ res3b_weight_0, float* __restrict__ res3b_bias_0, float* __restrict__ res3b_weight_1, float* __restrict__ res3b_bias_1, float* __restrict__ res3b_weight_2, float* __restrict__ res3b_bias_2, float* __restrict__ res3c_weight_0, float* __restrict__ res3c_bias_0, float* __restrict__ res3c_weight_1, float* __restrict__ res3c_bias_1, float* __restrict__ res3c_weight_2, float* __restrict__ res3c_bias_2, float* __restrict__ res3d_weight_0, float* __restrict__ res3d_bias_0, float* __restrict__ res3d_weight_1, float* __restrict__ res3d_bias_1, float* __restrict__ res3d_weight_2, float* __restrict__ res3d_bias_2, float* __restrict__ res4a_weight_b, float* __restrict__ res4a_bias_b, float* __restrict__ res4a_weight_0, float* __restrict__ res4a_bias_0, float* __restrict__ res4a_weight_1, float* __restrict__ res4a_bias_1, float* __restrict__ res4a_weight_2, float* __restrict__ res4a_bias_2, float* __restrict__ res4b_weight_0, float* __restrict__ res4b_bias_0, float* __restrict__ res4b_weight_1, float* __restrict__ res4b_bias_1, float* __restrict__ res4b_weight_2, float* __restrict__ res4b_bias_2, float* __restrict__ res4c_weight_0, float* __restrict__ res4c_bias_0, float* __restrict__ res4c_weight_1, float* __restrict__ res4c_bias_1, float* __restrict__ res4c_weight_2, float* __restrict__ res4c_bias_2, float* __restrict__ res4d_weight_0, float* __restrict__ res4d_bias_0, float* __restrict__ res4d_weight_1, float* __restrict__ res4d_bias_1, float* __restrict__ res4d_weight_2, float* __restrict__ res4d_bias_2, float* __restrict__ res4e_weight_0, float* __restrict__ res4e_bias_0, float* __restrict__ res4e_weight_1, float* __restrict__ res4e_bias_1, float* __restrict__ res4e_weight_2, float* __restrict__ res4e_bias_2, float* __restrict__ res4f_weight_0, float* __restrict__ res4f_bias_0, float* __restrict__ res4f_weight_1, float* __restrict__ res4f_bias_1, float* __restrict__ res4f_weight_2, float* __restrict__ res4f_bias_2, float* __restrict__ res5a_weight_b, float* __restrict__ res5a_bias_b, float* __restrict__ res5a_weight_0, float* __restrict__ res5a_bias_0, float* __restrict__ res5a_weight_1, float* __restrict__ res5a_bias_1, float* __restrict__ res5a_weight_2, float* __restrict__ res5a_bias_2, float* __restrict__ res5b_weight_0, float* __restrict__ res5b_bias_0, float* __restrict__ res5b_weight_1, float* __restrict__ res5b_bias_1, float* __restrict__ res5b_weight_2, float* __restrict__ res5b_bias_2, float* __restrict__ res5c_weight_0, float* __restrict__ res5c_bias_0, float* __restrict__ res5c_weight_1, float* __restrict__ res5c_bias_1, float* __restrict__ res5c_weight_2, float* __restrict__ res5c_bias_2) noexcept __attribute__((nonnull (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106)));
static bool batchwise_2_fused_res2a_conv_b_cast_mul_add_cast_reorder_res2a_conv_0_cast_mul_add_relu_cast_res2a_conv_1_cast_mul_add_relu_cast_res2a_conv_2_cast_mul_add_cast_add_cast_reorder_res2b_conv_0_cast_mul_add_relu_cast_res2b_conv_1_cast_mul_add_relu_cast_reorder_res2b_conv_2_cast_mul_add_cast_add_cast_reorder_res2c_conv_0_cast_mul_add_relu_cast_reorder_res2c_conv_1_cast_mul_add_relu_cast_reorder_res2c_conv_2_cast_mul_add_cast_add_cast_reorder_res3a_conv_b_cast_mul_add_cast_res3a_conv_0_cast_mul_add_relu_cast_reorder_res3a_conv_1_cast_mul_add_relu_cast_res3a_conv_2_cast_mul_add_cast_add_cast_reorder_res3b_conv_0_cast_mul_add_relu_cast_reorder_res3b_conv_1_cast_mul_add_relu_cast_reorder_res3b_conv_2_cast_mul_add_cast_add_cast_reorder_res3c_conv_0_cast_mul_add_relu_cast_reorder_res3c_conv_1_cast_mul_add_relu_cast_reorder_res3c_conv_2_cast_mul_add_cast_add_cast_reorder_res3d_conv_0_cast_mul_add_relu_cast_reorder_res3d_conv_1_cast_mul_add_relu_cast_res3d_conv_2_cast_mul_add_cast_add_cast_res4a_conv_b_cast_mul_add_cast_reorder_res4a_conv_0_cast_mul_add_relu_cast_res4a_conv_1_cast_mul_add_relu_cast_reorder_res4a_conv_2_cast_mul_add_cast_add_cast_reorder_res4b_conv_0_cast_mul_add_relu_cast_reorder_res4b_conv_1_cast_mul_add_relu_cast_reorder_res4b_conv_2_cast_mul_add_cast_add_cast_reorder_res4c_conv_0_cast_mul_add_relu_cast_reorder_res4c_conv_1_cast_mul_add_relu_cast_reorder_res4c_conv_2_cast_mul_add_cast_add_cast_res4d_conv_0_cast_mul_add_relu_cast_res4d_conv_1_cast_mul_add_relu_cast_reorder_res4d_conv_2_cast_mul_add_cast_add_cast_res4e_conv_0_cast_mul_add_relu_cast_reorder_res4e_conv_1_cast_mul_add_relu_cast_reorder_res4e_conv_2_cast_mul_add_cast_add_cast_reorder_reorder_res4f_conv_0_cast_mul_add_relu_cast_reorder_res4f_conv_1_cast_mul_add_relu_cast_reorder_res4f_conv_2_cast_mul_add_cast_add_cast__683(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4, float* __restrict__ __ins_5, float* __restrict__ __ins_6, int8_t* __restrict__ __ins_7, float* __restrict__ __ins_8, float* __restrict__ __ins_9, int8_t* __restrict__ __ins_10, float* __restrict__ __ins_11, float* __restrict__ __ins_12, int8_t* __restrict__ __ins_13, float* __restrict__ __ins_14, float* __restrict__ __ins_15, int8_t* __restrict__ __ins_16, float* __restrict__ __ins_17, float* __restrict__ __ins_18, int8_t* __restrict__ __ins_19, float* __restrict__ __ins_20, float* __restrict__ __ins_21, int8_t* __restrict__ __ins_22, float* __restrict__ __ins_23, float* __restrict__ __ins_24, int8_t* __restrict__ __ins_25, float* __restrict__ __ins_26, float* __restrict__ __ins_27, int8_t* __restrict__ __ins_28, float* __restrict__ __ins_29, float* __restrict__ __ins_30, int8_t* __restrict__ __ins_31, float* __restrict__ __ins_32, float* __restrict__ __ins_33, int8_t* __restrict__ __ins_34, float* __restrict__ __ins_35, float* __restrict__ __ins_36, int8_t* __restrict__ __ins_37, float* __restrict__ __ins_38, float* __restrict__ __ins_39, int8_t* __restrict__ __ins_40, float* __restrict__ __ins_41, float* __restrict__ __ins_42, int8_t* __restrict__ __ins_43, float* __restrict__ __ins_44, float* __restrict__ __ins_45, int8_t* __restrict__ __ins_46, float* __restrict__ __ins_47, float* __restrict__ __ins_48, int8_t* __restrict__ __ins_49, float* __restrict__ __ins_50, float* __restrict__ __ins_51, int8_t* __restrict__ __ins_52, float* __restrict__ __ins_53, float* __restrict__ __ins_54, int8_t* __restrict__ __ins_55, float* __restrict__ __ins_56, float* __restrict__ __ins_57, int8_t* __restrict__ __ins_58, float* __restrict__ __ins_59, float* __restrict__ __ins_60, int8_t* __restrict__ __ins_61, float* __restrict__ __ins_62, float* __restrict__ __ins_63, int8_t* __restrict__ __ins_64, float* __restrict__ __ins_65, float* __restrict__ __ins_66, int8_t* __restrict__ __ins_67, float* __restrict__ __ins_68, float* __restrict__ __ins_69, int8_t* __restrict__ __ins_70, float* __restrict__ __ins_71, float* __restrict__ __ins_72, int8_t* __restrict__ __ins_73, float* __restrict__ __ins_74, float* __restrict__ __ins_75, int8_t* __restrict__ __ins_76, float* __restrict__ __ins_77, float* __restrict__ __ins_78, int8_t* __restrict__ __ins_79, float* __restrict__ __ins_80, float* __restrict__ __ins_81, int8_t* __restrict__ __ins_82, float* __restrict__ __ins_83, float* __restrict__ __ins_84, int8_t* __restrict__ __ins_85, float* __restrict__ __ins_86, float* __restrict__ __ins_87, int8_t* __restrict__ __ins_88, float* __restrict__ __ins_89, float* __restrict__ __ins_90, int8_t* __restrict__ __ins_91, float* __restrict__ __ins_92, float* __restrict__ __ins_93, int8_t* __restrict__ __ins_94, float* __restrict__ __ins_95, float* __restrict__ __ins_96, int8_t* __restrict__ __ins_97, float* __restrict__ __ins_98, float* __restrict__ __ins_99, int8_t* __restrict__ __ins_100, float* __restrict__ __ins_101, float* __restrict__ __ins_102, int8_t* __restrict__ __ins_103, float* __restrict__ __ins_104, float* __restrict__ __ins_105, int8_t* __restrict__ __ins_106, float* __restrict__ __ins_107, float* __restrict__ __ins_108, int8_t* __restrict__ __ins_109, float* __restrict__ __ins_110, float* __restrict__ __ins_111, int8_t* __restrict__ __ins_112, float* __restrict__ __ins_113, float* __restrict__ __ins_114, int8_t* __restrict__ __ins_115, float* __restrict__ __ins_116, float* __restrict__ __ins_117, int8_t* __restrict__ __ins_118, float* __restrict__ __ins_119, float* __restrict__ __ins_120, int8_t* __restrict__ __ins_121, float* __restrict__ __ins_122, float* __restrict__ __ins_123, int8_t* __restrict__ __ins_124, float* __restrict__ __ins_125, float* __restrict__ __ins_126) noexcept __attribute__((nonnull (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128)));
static bool res5a_conv_0_cast_mul_add_relu_cast_reorder__681(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5a_conv_1_cast_mul_add_relu_cast_reorder__680(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5a_conv_b_cast_mul_add_cast_reorder__682(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5a_conv_2_cast_mul_add_cast_add_cast__679(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res5b_conv_0_cast_mul_add_relu_cast_reorder__678(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5b_conv_1_cast_mul_add_relu_cast_reorder__677(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5b_conv_2_cast_mul_add_cast_add_cast_reorder__676(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res5c_conv_0_cast_mul_add_relu_cast_reorder__675(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5c_conv_1_cast_mul_add_relu_cast_reorder__674(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5c_conv_2_cast_mul_add_cast_add_cast_cast__673(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool reorder__105(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool res2a_conv_0_cast_mul_add_relu_cast__8(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2a_conv_1_cast_mul_add_relu_cast__12(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2a_conv_b_cast_mul_add_cast_reorder__4(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2a_conv_2_cast_mul_add_cast_add_cast_reorder__16(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res2b_conv_0_cast_mul_add_relu_cast__20(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2b_conv_1_cast_mul_add_relu_cast_reorder__24(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2b_conv_2_cast_mul_add_cast_add_cast_reorder__28(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res2c_conv_0_cast_mul_add_relu_cast_reorder__32(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2c_conv_1_cast_mul_add_relu_cast_reorder__36(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2c_conv_2_cast_mul_add_cast_add_cast_reorder__40(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res3a_conv_0_cast_mul_add_relu_cast_reorder__48(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3a_conv_1_cast_mul_add_relu_cast__52(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3a_conv_b_cast_mul_add_cast__44(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3a_conv_2_cast_mul_add_cast_add_cast_reorder__56(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res3b_conv_0_cast_mul_add_relu_cast_reorder__60(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3b_conv_1_cast_mul_add_relu_cast_reorder__64(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3b_conv_2_cast_mul_add_cast_add_cast_reorder__68(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res3c_conv_0_cast_mul_add_relu_cast_reorder__72(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3c_conv_1_cast_mul_add_relu_cast_reorder__76(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3c_conv_2_cast_mul_add_cast_add_cast_reorder__80(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res3d_conv_0_cast_mul_add_relu_cast_reorder__84(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3d_conv_1_cast_mul_add_relu_cast__88(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3d_conv_2_cast_mul_add_cast_add_cast__92(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res4a_conv_0_cast_mul_add_relu_cast__100(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4a_conv_1_cast_mul_add_relu_cast_reorder__104(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4a_conv_b_cast_mul_add_cast_reorder__96(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4a_conv_2_cast_mul_add_cast_add_cast_reorder__108(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res4b_conv_0_cast_mul_add_relu_cast_reorder__112(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4b_conv_1_cast_mul_add_relu_cast_reorder__116(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4b_conv_2_cast_mul_add_cast_add_cast_reorder__120(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res4c_conv_0_cast_mul_add_relu_cast_reorder__124(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4c_conv_1_cast_mul_add_relu_cast_reorder__128(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4c_conv_2_cast_mul_add_cast_add_cast__132(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res4d_conv_0_cast_mul_add_relu_cast__136(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4d_conv_1_cast_mul_add_relu_cast_reorder__140(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4d_conv_2_cast_mul_add_cast_add_cast__144(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res4e_conv_0_cast_mul_add_relu_cast_reorder__148(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4e_conv_1_cast_mul_add_relu_cast_reorder__152(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4e_conv_2_cast_mul_add_cast_add_cast_reorder__156(uint8_t* __restrict__ __outs_0, uint8_t* __restrict__ __outs_1, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6,7)));
static bool reorder__157(uint8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool res4f_conv_0_cast_mul_add_relu_cast_reorder__161(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4f_conv_1_cast_mul_add_relu_cast_reorder__165(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4f_conv_2_cast_mul_add_cast_add_cast__170(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
extern "C" void* memset(void* ptr, int32_t v, uint64_t len) noexcept;
static bool reorder__481(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__488(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__504(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__513(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__520(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__529(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__420(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__424(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__427(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__431(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__434(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__437(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__440(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__443(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__446(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__449(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__452(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__455(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__458(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__462(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__466(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__469(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__472(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__475(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__478(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__485(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__491(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__494(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__498(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__501(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__507(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__510(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__517(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__523(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__526(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__532(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__535(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__538(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__541(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__544(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__547(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__550(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__553(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__556(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__559(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__425(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__432(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__438(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__441(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__450(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__453(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__459(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__467(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__473(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__476(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__421(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__428(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__435(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__444(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__486(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__492(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__495(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__499(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__502(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__508(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__511(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__518(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__524(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__527(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__447(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__456(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__463(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__470(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__479(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__536(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__539(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__545(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__548(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__554(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__557(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__482(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__489(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__505(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__514(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__521(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__530(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__533(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__542(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__551(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__560(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__111(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__112(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__108(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__109(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__117(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__118(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__126(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__127(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__135(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__136(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__120(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__121(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__129(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__130(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__141(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__142(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__114(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__115(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__123(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__124(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__132(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__133(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__147(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__148(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__156(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__157(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__165(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__166(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__174(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__175(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__150(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__151(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__159(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__160(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__168(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__169(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__138(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__139(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__180(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__181(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__144(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__145(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__153(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__154(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__162(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__163(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__171(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__172(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__186(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__187(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__195(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__196(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__204(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__205(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__213(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__214(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__222(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__223(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__231(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__232(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__189(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__190(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__198(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__199(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__207(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__208(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__216(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__217(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__225(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__226(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__177(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__178(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__237(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__238(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__183(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__184(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__192(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__193(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__201(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__202(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__210(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__211(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__219(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__220(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__228(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__229(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__525(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__652(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__651(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__646(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__645(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__640(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__639(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__634(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__633(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__628(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__627(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__622(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__621(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__616(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__615(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__614(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__613(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__608(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__607(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__602(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__601(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__596(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__595(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__590(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__589(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__650(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__649(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__648(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__647(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__644(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__643(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__642(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__641(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__638(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__637(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__636(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__635(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__632(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__631(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__630(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__629(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__626(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__625(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__624(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__623(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__620(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__619(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__618(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__617(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__588(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__587(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__582(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__581(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__576(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__575(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__570(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__569(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__522(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__515(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__506(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__497(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__490(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__528(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__519(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__512(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__503(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__496(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__487(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__422(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__612(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__611(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__610(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__609(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__606(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__605(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__604(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__603(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__600(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__599(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__598(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__597(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__594(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__593(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__592(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__591(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__586(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__585(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__584(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__583(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__580(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__579(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__578(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__577(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__574(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__573(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__572(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__571(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__516(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__509(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__500(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__493(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__484(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__480(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__474(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__465(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__460(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__451(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__483(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__445(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__471(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__464(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__457(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__477(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__468(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__461(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__454(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__439(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__430(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__423(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__448(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__436(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__429(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__442(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__433(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__426(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__419(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__656(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__655(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__534(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__243(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__244(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__252(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__253(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__261(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__262(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__246(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__247(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__255(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__256(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__234(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__235(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__531(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__654(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__653(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__240(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__241(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__537(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__658(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__657(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__660(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__659(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__540(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__662(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__661(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__543(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__249(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__250(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__258(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__259(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__664(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__663(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__546(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__666(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__665(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__549(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__668(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__667(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__552(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__670(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__669(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__555(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__672(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__671(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__558(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));


extern "C" void rn50_backbone_bs2(int8_t* __restrict__ backbone_output, int8_t* __restrict__ backbone_input, float* __restrict__ res2a_weight_b, float* __restrict__ res2a_bias_b, float* __restrict__ res2a_weight_0, float* __restrict__ res2a_bias_0, float* __restrict__ res2a_weight_1, float* __restrict__ res2a_bias_1, float* __restrict__ res2a_weight_2, float* __restrict__ res2a_bias_2, float* __restrict__ res2b_weight_0, float* __restrict__ res2b_bias_0, float* __restrict__ res2b_weight_1, float* __restrict__ res2b_bias_1, float* __restrict__ res2b_weight_2, float* __restrict__ res2b_bias_2, float* __restrict__ res2c_weight_0, float* __restrict__ res2c_bias_0, float* __restrict__ res2c_weight_1, float* __restrict__ res2c_bias_1, float* __restrict__ res2c_weight_2, float* __restrict__ res2c_bias_2, float* __restrict__ res3a_weight_b, float* __restrict__ res3a_bias_b, float* __restrict__ res3a_weight_0, float* __restrict__ res3a_bias_0, float* __restrict__ res3a_weight_1, float* __restrict__ res3a_bias_1, float* __restrict__ res3a_weight_2, float* __restrict__ res3a_bias_2, float* __restrict__ res3b_weight_0, float* __restrict__ res3b_bias_0, float* __restrict__ res3b_weight_1, float* __restrict__ res3b_bias_1, float* __restrict__ res3b_weight_2, float* __restrict__ res3b_bias_2, float* __restrict__ res3c_weight_0, float* __restrict__ res3c_bias_0, float* __restrict__ res3c_weight_1, float* __restrict__ res3c_bias_1, float* __restrict__ res3c_weight_2, float* __restrict__ res3c_bias_2, float* __restrict__ res3d_weight_0, float* __restrict__ res3d_bias_0, float* __restrict__ res3d_weight_1, float* __restrict__ res3d_bias_1, float* __restrict__ res3d_weight_2, float* __restrict__ res3d_bias_2, float* __restrict__ res4a_weight_b, float* __restrict__ res4a_bias_b, float* __restrict__ res4a_weight_0, float* __restrict__ res4a_bias_0, float* __restrict__ res4a_weight_1, float* __restrict__ res4a_bias_1, float* __restrict__ res4a_weight_2, float* __restrict__ res4a_bias_2, float* __restrict__ res4b_weight_0, float* __restrict__ res4b_bias_0, float* __restrict__ res4b_weight_1, float* __restrict__ res4b_bias_1, float* __restrict__ res4b_weight_2, float* __restrict__ res4b_bias_2, float* __restrict__ res4c_weight_0, float* __restrict__ res4c_bias_0, float* __restrict__ res4c_weight_1, float* __restrict__ res4c_bias_1, float* __restrict__ res4c_weight_2, float* __restrict__ res4c_bias_2, float* __restrict__ res4d_weight_0, float* __restrict__ res4d_bias_0, float* __restrict__ res4d_weight_1, float* __restrict__ res4d_bias_1, float* __restrict__ res4d_weight_2, float* __restrict__ res4d_bias_2, float* __restrict__ res4e_weight_0, float* __restrict__ res4e_bias_0, float* __restrict__ res4e_weight_1, float* __restrict__ res4e_bias_1, float* __restrict__ res4e_weight_2, float* __restrict__ res4e_bias_2, float* __restrict__ res4f_weight_0, float* __restrict__ res4f_bias_0, float* __restrict__ res4f_weight_1, float* __restrict__ res4f_bias_1, float* __restrict__ res4f_weight_2, float* __restrict__ res4f_bias_2, float* __restrict__ res5a_weight_b, float* __restrict__ res5a_bias_b, float* __restrict__ res5a_weight_0, float* __restrict__ res5a_bias_0, float* __restrict__ res5a_weight_1, float* __restrict__ res5a_bias_1, float* __restrict__ res5a_weight_2, float* __restrict__ res5a_bias_2, float* __restrict__ res5b_weight_0, float* __restrict__ res5b_bias_0, float* __restrict__ res5b_weight_1, float* __restrict__ res5b_bias_1, float* __restrict__ res5b_weight_2, float* __restrict__ res5b_bias_2, float* __restrict__ res5c_weight_0, float* __restrict__ res5c_bias_0, float* __restrict__ res5c_weight_1, float* __restrict__ res5c_bias_1, float* __restrict__ res5c_weight_2, float* __restrict__ res5c_bias_2) noexcept{
  bool& is_init = *(bool*)(__module_data + 0);
  int8_t* folded_const_281 = (int8_t*)&__uninitialized_data[8608768UL];
  float* folded_const_212 = (float*)&__uninitialized_data[699392UL];
  float* folded_const_211 = (float*)&__uninitialized_data[698368UL];
  int8_t* folded_const_224 = (int8_t*)&__uninitialized_data[3584000UL];
  float* folded_const_252 = (float*)&__uninitialized_data[3599104UL];
  float* folded_const_251 = (float*)&__uninitialized_data[3598848UL];
  int8_t* folded_const_274 = (int8_t*)&__uninitialized_data[8457216UL];
  float* folded_const_250 = (float*)&__uninitialized_data[3598592UL];
  float* folded_const_249 = (float*)&__uninitialized_data[3598336UL];
  int8_t* folded_const_280 = (int8_t*)&__uninitialized_data[8592384UL];
  float* folded_const_210 = (float*)&__uninitialized_data[697344UL];
  float* folded_const_209 = (float*)&__uninitialized_data[696320UL];
  int8_t* folded_const_277 = (int8_t*)&__uninitialized_data[8543232UL];
  float* folded_const_248 = (float*)&__uninitialized_data[3598080UL];
  float* folded_const_247 = (float*)&__uninitialized_data[3597824UL];
  int8_t* folded_const_273 = (int8_t*)&__uninitialized_data[8420352UL];
  float* folded_const_246 = (float*)&__uninitialized_data[3597568UL];
  float* folded_const_245 = (float*)&__uninitialized_data[3597312UL];
  int8_t* folded_const_279 = (int8_t*)&__uninitialized_data[8576000UL];
  float* folded_const_208 = (float*)&__uninitialized_data[695296UL];
  float* folded_const_207 = (float*)&__uninitialized_data[694272UL];
  int8_t* folded_const_276 = (int8_t*)&__uninitialized_data[8526848UL];
  float* folded_const_244 = (float*)&__uninitialized_data[3597056UL];
  float* folded_const_243 = (float*)&__uninitialized_data[3596800UL];
  int8_t* folded_const_272 = (int8_t*)&__uninitialized_data[8383488UL];
  float* folded_const_242 = (float*)&__uninitialized_data[3596544UL];
  float* folded_const_241 = (float*)&__uninitialized_data[3596288UL];
  int8_t* folded_const_278 = (int8_t*)&__uninitialized_data[8559616UL];
  float* folded_const_206 = (float*)&__uninitialized_data[693248UL];
  float* folded_const_205 = (float*)&__uninitialized_data[692224UL];
  int8_t* folded_const_264 = (int8_t*)&__uninitialized_data[7793664UL];
  float* folded_const_180 = (float*)&__uninitialized_data[665600UL];
  float* folded_const_179 = (float*)&__uninitialized_data[663552UL];
  int8_t* folded_const_275 = (int8_t*)&__uninitialized_data[8494080UL];
  float* folded_const_240 = (float*)&__uninitialized_data[3595776UL];
  float* folded_const_239 = (float*)&__uninitialized_data[3595264UL];
  int8_t* folded_const_262 = (int8_t*)&__uninitialized_data[7515136UL];
  float* folded_const_238 = (float*)&__uninitialized_data[3594752UL];
  float* folded_const_237 = (float*)&__uninitialized_data[3594240UL];
  int8_t* folded_const_271 = (int8_t*)&__uninitialized_data[8317952UL];
  float* folded_const_178 = (float*)&__uninitialized_data[661504UL];
  float* folded_const_177 = (float*)&__uninitialized_data[659456UL];
  int8_t* folded_const_267 = (int8_t*)&__uninitialized_data[8055808UL];
  float* folded_const_236 = (float*)&__uninitialized_data[3593728UL];
  float* folded_const_235 = (float*)&__uninitialized_data[3593216UL];
  int8_t* folded_const_261 = (int8_t*)&__uninitialized_data[7367680UL];
  float* folded_const_234 = (float*)&__uninitialized_data[3592704UL];
  float* folded_const_233 = (float*)&__uninitialized_data[3592192UL];
  int8_t* folded_const_270 = (int8_t*)&__uninitialized_data[8252416UL];
  float* folded_const_176 = (float*)&__uninitialized_data[657408UL];
  float* folded_const_175 = (float*)&__uninitialized_data[655360UL];
  int8_t* folded_const_266 = (int8_t*)&__uninitialized_data[7990272UL];
  float* folded_const_232 = (float*)&__uninitialized_data[3591680UL];
  float* folded_const_231 = (float*)&__uninitialized_data[3591168UL];
  int8_t* folded_const_260 = (int8_t*)&__uninitialized_data[7220224UL];
  float* folded_const_230 = (float*)&__uninitialized_data[3590656UL];
  float* folded_const_229 = (float*)&__uninitialized_data[3590144UL];
  int8_t* folded_const_269 = (int8_t*)&__uninitialized_data[8186880UL];
  float* folded_const_174 = (float*)&__uninitialized_data[653312UL];
  float* folded_const_173 = (float*)&__uninitialized_data[651264UL];
  int8_t* folded_const_265 = (int8_t*)&__uninitialized_data[7924736UL];
  float* folded_const_228 = (float*)&__uninitialized_data[3589632UL];
  float* folded_const_227 = (float*)&__uninitialized_data[3589120UL];
  int8_t* folded_const_259 = (int8_t*)&__uninitialized_data[7072768UL];
  float* folded_const_226 = (float*)&__uninitialized_data[3588608UL];
  float* folded_const_225 = (float*)&__uninitialized_data[3588096UL];
  int8_t* folded_const_268 = (int8_t*)&__uninitialized_data[8121344UL];
  float* folded_const_172 = (float*)&__uninitialized_data[649216UL];
  float* folded_const_171 = (float*)&__uninitialized_data[647168UL];
  int8_t* folded_const_258 = (int8_t*)&__uninitialized_data[6548480UL];
  float* folded_const_170 = (float*)&__uninitialized_data[643072UL];
  float* folded_const_169 = (float*)&__uninitialized_data[638976UL];
  int8_t* folded_const_263 = (int8_t*)&__uninitialized_data[7662592UL];
  float* folded_const_204 = (float*)&__uninitialized_data[691200UL];
  float* folded_const_203 = (float*)&__uninitialized_data[690176UL];
  int8_t* folded_const_257 = (int8_t*)&__uninitialized_data[5958656UL];
  float* folded_const_202 = (float*)&__uninitialized_data[689152UL];
  float* folded_const_201 = (float*)&__uninitialized_data[688128UL];
  int8_t* folded_const_223 = (int8_t*)&__uninitialized_data[3321856UL];
  float* folded_const_168 = (float*)&__uninitialized_data[634880UL];
  float* folded_const_167 = (float*)&__uninitialized_data[630784UL];
  int8_t* folded_const_217 = (int8_t*)&__uninitialized_data[1748992UL];
  float* folded_const_200 = (float*)&__uninitialized_data[687104UL];
  float* folded_const_199 = (float*)&__uninitialized_data[686080UL];
  int8_t* folded_const_256 = (int8_t*)&__uninitialized_data[5368832UL];
  float* folded_const_198 = (float*)&__uninitialized_data[685056UL];
  float* folded_const_197 = (float*)&__uninitialized_data[684032UL];
  int8_t* folded_const_222 = (int8_t*)&__uninitialized_data[3059712UL];
  float* folded_const_166 = (float*)&__uninitialized_data[626688UL];
  float* folded_const_165 = (float*)&__uninitialized_data[622592UL];
  int8_t* folded_const_216 = (int8_t*)&__uninitialized_data[1486848UL];
  float* folded_const_196 = (float*)&__uninitialized_data[683008UL];
  float* folded_const_195 = (float*)&__uninitialized_data[681984UL];
  int8_t* folded_const_255 = (int8_t*)&__uninitialized_data[4779008UL];
  float* folded_const_194 = (float*)&__uninitialized_data[680960UL];
  float* folded_const_193 = (float*)&__uninitialized_data[679936UL];
  int8_t* folded_const_221 = (int8_t*)&__uninitialized_data[2797568UL];
  float* folded_const_164 = (float*)&__uninitialized_data[618496UL];
  float* folded_const_163 = (float*)&__uninitialized_data[614400UL];
  int8_t* folded_const_215 = (int8_t*)&__uninitialized_data[1224704UL];
  float* folded_const_192 = (float*)&__uninitialized_data[678912UL];
  float* folded_const_191 = (float*)&__uninitialized_data[677888UL];
  int8_t* folded_const_254 = (int8_t*)&__uninitialized_data[4189184UL];
  float* folded_const_190 = (float*)&__uninitialized_data[676864UL];
  float* folded_const_189 = (float*)&__uninitialized_data[675840UL];
  int8_t* folded_const_220 = (int8_t*)&__uninitialized_data[2535424UL];
  float* folded_const_162 = (float*)&__uninitialized_data[610304UL];
  float* folded_const_161 = (float*)&__uninitialized_data[606208UL];
  int8_t* folded_const_214 = (int8_t*)&__uninitialized_data[962560UL];
  float* folded_const_188 = (float*)&__uninitialized_data[674816UL];
  float* folded_const_187 = (float*)&__uninitialized_data[673792UL];
  int8_t* folded_const_253 = (int8_t*)&__uninitialized_data[3599360UL];
  float* folded_const_186 = (float*)&__uninitialized_data[672768UL];
  float* folded_const_185 = (float*)&__uninitialized_data[671744UL];
  int8_t* folded_const_219 = (int8_t*)&__uninitialized_data[2273280UL];
  float* folded_const_160 = (float*)&__uninitialized_data[602112UL];
  float* folded_const_159 = (float*)&__uninitialized_data[598016UL];
  int8_t* folded_const_213 = (int8_t*)&__uninitialized_data[700416UL];
  float* folded_const_184 = (float*)&__uninitialized_data[670720UL];
  float* folded_const_183 = (float*)&__uninitialized_data[669696UL];
  int8_t* folded_const_156 = (int8_t*)&__uninitialized_data[0UL];
  float* folded_const_182 = (float*)&__uninitialized_data[668672UL];
  float* folded_const_181 = (float*)&__uninitialized_data[667648UL];
  int8_t* folded_const_218 = (int8_t*)&__uninitialized_data[2011136UL];
  float* folded_const_158 = (float*)&__uninitialized_data[593920UL];
  float* folded_const_157 = (float*)&__uninitialized_data[589824UL];
  int8_t* folded_const_284 = (int8_t*)&__uninitialized_data[8629248UL];
  float* folded_const_283 = (float*)&__uninitialized_data[8627200UL];
  float* folded_const_282 = (float*)&__uninitialized_data[8625152UL];
  int8_t* folded_const_288 = (int8_t*)&__uninitialized_data[11267072UL];
  float* folded_const_290 = (float*)&__uninitialized_data[13628416UL];
  float* folded_const_289 = (float*)&__uninitialized_data[13626368UL];
  int8_t* folded_const_285 = (int8_t*)&__uninitialized_data[9153536UL];
  float* folded_const_287 = (float*)&__uninitialized_data[11258880UL];
  float* folded_const_286 = (float*)&__uninitialized_data[11250688UL];
  int8_t* folded_const_293 = (int8_t*)&__uninitialized_data[13646848UL];
  float* folded_const_292 = (float*)&__uninitialized_data[13638656UL];
  float* folded_const_291 = (float*)&__uninitialized_data[13630464UL];
  int8_t* folded_const_296 = (int8_t*)&__uninitialized_data[14699520UL];
  float* folded_const_295 = (float*)&__uninitialized_data[14697472UL];
  float* folded_const_294 = (float*)&__uninitialized_data[14695424UL];
  int8_t* folded_const_299 = (int8_t*)&__uninitialized_data[15752192UL];
  float* folded_const_298 = (float*)&__uninitialized_data[15750144UL];
  float* folded_const_297 = (float*)&__uninitialized_data[15748096UL];
  int8_t* folded_const_302 = (int8_t*)&__uninitialized_data[18127872UL];
  float* folded_const_301 = (float*)&__uninitialized_data[18119680UL];
  float* folded_const_300 = (float*)&__uninitialized_data[18111488UL];
  int8_t* folded_const_305 = (int8_t*)&__uninitialized_data[19180544UL];
  float* folded_const_304 = (float*)&__uninitialized_data[19178496UL];
  float* folded_const_303 = (float*)&__uninitialized_data[19176448UL];
  int8_t* folded_const_308 = (int8_t*)&__uninitialized_data[20233216UL];
  float* folded_const_307 = (float*)&__uninitialized_data[20231168UL];
  float* folded_const_306 = (float*)&__uninitialized_data[20229120UL];
  int8_t* folded_const_311 = (int8_t*)&__uninitialized_data[22608896UL];
  float* folded_const_310 = (float*)&__uninitialized_data[22600704UL];
  float* folded_const_309 = (float*)&__uninitialized_data[22592512UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 713728UL);
  if (!is_init) {
    __init_const_globals(backbone_output, backbone_input, res2a_weight_b, res2a_bias_b, res2a_weight_0, res2a_bias_0, res2a_weight_1, res2a_bias_1, res2a_weight_2, res2a_bias_2, res2b_weight_0, res2b_bias_0, res2b_weight_1, res2b_bias_1, res2b_weight_2, res2b_bias_2, res2c_weight_0, res2c_bias_0, res2c_weight_1, res2c_bias_1, res2c_weight_2, res2c_bias_2, res3a_weight_b, res3a_bias_b, res3a_weight_0, res3a_bias_0, res3a_weight_1, res3a_bias_1, res3a_weight_2, res3a_bias_2, res3b_weight_0, res3b_bias_0, res3b_weight_1, res3b_bias_1, res3b_weight_2, res3b_bias_2, res3c_weight_0, res3c_bias_0, res3c_weight_1, res3c_bias_1, res3c_weight_2, res3c_bias_2, res3d_weight_0, res3d_bias_0, res3d_weight_1, res3d_bias_1, res3d_weight_2, res3d_bias_2, res4a_weight_b, res4a_bias_b, res4a_weight_0, res4a_bias_0, res4a_weight_1, res4a_bias_1, res4a_weight_2, res4a_bias_2, res4b_weight_0, res4b_bias_0, res4b_weight_1, res4b_bias_1, res4b_weight_2, res4b_bias_2, res4c_weight_0, res4c_bias_0, res4c_weight_1, res4c_bias_1, res4c_weight_2, res4c_bias_2, res4d_weight_0, res4d_bias_0, res4d_weight_1, res4d_bias_1, res4d_weight_2, res4d_bias_2, res4e_weight_0, res4e_bias_0, res4e_weight_1, res4e_bias_1, res4e_weight_2, res4e_bias_2, res4f_weight_0, res4f_bias_0, res4f_weight_1, res4f_bias_1, res4f_weight_2, res4f_bias_2, res5a_weight_b, res5a_bias_b, res5a_weight_0, res5a_bias_0, res5a_weight_1, res5a_bias_1, res5a_weight_2, res5a_bias_2, res5b_weight_0, res5b_bias_0, res5b_weight_1, res5b_bias_1, res5b_weight_2, res5b_bias_2, res5c_weight_0, res5c_bias_0, res5c_weight_1, res5c_bias_1, res5c_weight_2, res5c_bias_2);
  }
  // [u8 [2, 4, 14, 14, 256] @ ABCD256b]
  uint8_t* buffer_578 = (uint8_t*)&__rescheduled_0[0UL];
  batchwise_2_fused_res2a_conv_b_cast_mul_add_cast_reorder_res2a_conv_0_cast_mul_add_relu_cast_res2a_conv_1_cast_mul_add_relu_cast_res2a_conv_2_cast_mul_add_cast_add_cast_reorder_res2b_conv_0_cast_mul_add_relu_cast_res2b_conv_1_cast_mul_add_relu_cast_reorder_res2b_conv_2_cast_mul_add_cast_add_cast_reorder_res2c_conv_0_cast_mul_add_relu_cast_reorder_res2c_conv_1_cast_mul_add_relu_cast_reorder_res2c_conv_2_cast_mul_add_cast_add_cast_reorder_res3a_conv_b_cast_mul_add_cast_res3a_conv_0_cast_mul_add_relu_cast_reorder_res3a_conv_1_cast_mul_add_relu_cast_res3a_conv_2_cast_mul_add_cast_add_cast_reorder_res3b_conv_0_cast_mul_add_relu_cast_reorder_res3b_conv_1_cast_mul_add_relu_cast_reorder_res3b_conv_2_cast_mul_add_cast_add_cast_reorder_res3c_conv_0_cast_mul_add_relu_cast_reorder_res3c_conv_1_cast_mul_add_relu_cast_reorder_res3c_conv_2_cast_mul_add_cast_add_cast_reorder_res3d_conv_0_cast_mul_add_relu_cast_reorder_res3d_conv_1_cast_mul_add_relu_cast_res3d_conv_2_cast_mul_add_cast_add_cast_res4a_conv_b_cast_mul_add_cast_reorder_res4a_conv_0_cast_mul_add_relu_cast_res4a_conv_1_cast_mul_add_relu_cast_reorder_res4a_conv_2_cast_mul_add_cast_add_cast_reorder_res4b_conv_0_cast_mul_add_relu_cast_reorder_res4b_conv_1_cast_mul_add_relu_cast_reorder_res4b_conv_2_cast_mul_add_cast_add_cast_reorder_res4c_conv_0_cast_mul_add_relu_cast_reorder_res4c_conv_1_cast_mul_add_relu_cast_reorder_res4c_conv_2_cast_mul_add_cast_add_cast_res4d_conv_0_cast_mul_add_relu_cast_res4d_conv_1_cast_mul_add_relu_cast_reorder_res4d_conv_2_cast_mul_add_cast_add_cast_res4e_conv_0_cast_mul_add_relu_cast_reorder_res4e_conv_1_cast_mul_add_relu_cast_reorder_res4e_conv_2_cast_mul_add_cast_add_cast_reorder_reorder_res4f_conv_0_cast_mul_add_relu_cast_reorder_res4f_conv_1_cast_mul_add_relu_cast_reorder_res4f_conv_2_cast_mul_add_cast_add_cast__683(buffer_578, &backbone_input[0UL], folded_const_281, folded_const_212, folded_const_211, folded_const_224, folded_const_252, folded_const_251, folded_const_274, folded_const_250, folded_const_249, folded_const_280, folded_const_210, folded_const_209, folded_const_277, folded_const_248, folded_const_247, folded_const_273, folded_const_246, folded_const_245, folded_const_279, folded_const_208, folded_const_207, folded_const_276, folded_const_244, folded_const_243, folded_const_272, folded_const_242, folded_const_241, folded_const_278, folded_const_206, folded_const_205, folded_const_264, folded_const_180, folded_const_179, folded_const_275, folded_const_240, folded_const_239, folded_const_262, folded_const_238, folded_const_237, folded_const_271, folded_const_178, folded_const_177, folded_const_267, folded_const_236, folded_const_235, folded_const_261, folded_const_234, folded_const_233, folded_const_270, folded_const_176, folded_const_175, folded_const_266, folded_const_232, folded_const_231, folded_const_260, folded_const_230, folded_const_229, folded_const_269, folded_const_174, folded_const_173, folded_const_265, folded_const_228, folded_const_227, folded_const_259, folded_const_226, folded_const_225, folded_const_268, folded_const_172, folded_const_171, folded_const_258, folded_const_170, folded_const_169, folded_const_263, folded_const_204, folded_const_203, folded_const_257, folded_const_202, folded_const_201, folded_const_223, folded_const_168, folded_const_167, folded_const_217, folded_const_200, folded_const_199, folded_const_256, folded_const_198, folded_const_197, folded_const_222, folded_const_166, folded_const_165, folded_const_216, folded_const_196, folded_const_195, folded_const_255, folded_const_194, folded_const_193, folded_const_221, folded_const_164, folded_const_163, folded_const_215, folded_const_192, folded_const_191, folded_const_254, folded_const_190, folded_const_189, folded_const_220, folded_const_162, folded_const_161, folded_const_214, folded_const_188, folded_const_187, folded_const_253, folded_const_186, folded_const_185, folded_const_219, folded_const_160, folded_const_159, folded_const_213, folded_const_184, folded_const_183, folded_const_156, folded_const_182, folded_const_181, folded_const_218, folded_const_158, folded_const_157);
  // [s8 [2, 2, 16, 16, 256] @ ABCD256b]
  int8_t* buffer_585 = (int8_t*)&__rescheduled_0[401408UL];
  res5a_conv_0_cast_mul_add_relu_cast_reorder__681(buffer_585, buffer_578, folded_const_284, folded_const_283, folded_const_282);
  // [s8 [2, 2, 7, 7, 256] @ ABCD256b]
  int8_t* buffer_588 = (int8_t*)&__rescheduled_0[663552UL];
  res5a_conv_1_cast_mul_add_relu_cast_reorder__680(buffer_588, buffer_585, folded_const_288, folded_const_290, folded_const_289);
  // [s8 [2, 32, 7, 7, 64] @ ABCD64b]
  int8_t* buffer_589 = (int8_t*)&__rescheduled_0[401408UL];
  res5a_conv_b_cast_mul_add_cast_reorder__682(buffer_589, buffer_578, folded_const_285, folded_const_287, folded_const_286);
  // [u8 [2, 32, 7, 7, 64] @ ABCD64b]
  uint8_t* buffer_603 = (uint8_t*)&__rescheduled_0[0UL];
  res5a_conv_2_cast_mul_add_cast_add_cast__679(buffer_603, buffer_588, folded_const_293, folded_const_292, folded_const_291, buffer_589);
  // [s8 [2, 1, 9, 9, 512] @ ABCD512b]
  int8_t* buffer_604 = (int8_t*)&__rescheduled_0[200704UL];
  res5b_conv_0_cast_mul_add_relu_cast_reorder__678(buffer_604, buffer_603, folded_const_296, folded_const_295, folded_const_294);
  // [s8 [2, 2, 7, 7, 256] @ ABCD256b]
  int8_t* buffer_605 = (int8_t*)&__rescheduled_0[283648UL];
  res5b_conv_1_cast_mul_add_relu_cast_reorder__677(buffer_605, buffer_604, folded_const_299, folded_const_298, folded_const_297);
  // [u8 [2, 4, 7, 7, 512] @ ABCD512b]
  uint8_t* buffer_615 = (uint8_t*)&__rescheduled_0[333824UL];
  res5b_conv_2_cast_mul_add_cast_add_cast_reorder__676(buffer_615, buffer_605, folded_const_302, folded_const_301, folded_const_300, buffer_603);
  // [s8 [2, 2, 9, 9, 256] @ ABCD256b]
  int8_t* buffer_616 = (int8_t*)&__rescheduled_0[534528UL];
  res5c_conv_0_cast_mul_add_relu_cast_reorder__675(buffer_616, buffer_615, folded_const_305, folded_const_304, folded_const_303);
  // [s8 [2, 1, 7, 7, 512] @ ABCD512b]
  int8_t* buffer_617 = (int8_t*)&__rescheduled_0[617472UL];
  res5c_conv_1_cast_mul_add_relu_cast_reorder__674(buffer_617, buffer_616, folded_const_308, folded_const_307, folded_const_306);
  // [s8 [2, 4, 7, 7, 512] @ ABCD512b]
  int8_t* buffer_621 = (int8_t*)&__rescheduled_0[0UL];
  res5c_conv_2_cast_mul_add_cast_add_cast_cast__673(buffer_621, buffer_617, folded_const_311, folded_const_310, folded_const_309, buffer_615);
  reorder__105(backbone_output, buffer_621);
  sc_aligned_free(__stream, __rescheduled_0);
}

static bool reorder__481(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_7582___fuseiter_7583_1991 = 0UL; fused_0_fuseiter_7582___fuseiter_7583_1991 < 32UL; fused_0_fuseiter_7582___fuseiter_7583_1991 += 1UL) {
    for (uint64_t _fuseiter_7586 = 0UL; _fuseiter_7586 < 32UL; _fuseiter_7586 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_7582___fuseiter_7583_1991 / 32UL) * 1024UL) + (_fuseiter_7586 + ((fused_0_fuseiter_7582___fuseiter_7583_1991 % 32UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_7582___fuseiter_7583_1991 / 32UL) * 1024UL) + (((fused_0_fuseiter_7582___fuseiter_7583_1991 % 32UL) * 32UL) + _fuseiter_7586))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__488(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7587___fuseiter_7588_1992___fuseiter_7589_1993___fuseiter_7590_1994 = 0UL; fused_0fused_0fused_0_fuseiter_7587___fuseiter_7588_1992___fuseiter_7589_1993___fuseiter_7590_1994 < 8UL; fused_0fused_0fused_0_fuseiter_7587___fuseiter_7588_1992___fuseiter_7589_1993___fuseiter_7590_1994 += 1UL) {
    for (uint64_t _fuseiter_7591 = 0UL; _fuseiter_7591 < 128UL; _fuseiter_7591 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7587___fuseiter_7588_1992___fuseiter_7589_1993___fuseiter_7590_1994 / 8UL) * 1024UL) + (_fuseiter_7591 + ((fused_0fused_0fused_0_fuseiter_7587___fuseiter_7588_1992___fuseiter_7589_1993___fuseiter_7590_1994 % 8UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7587___fuseiter_7588_1992___fuseiter_7589_1993___fuseiter_7590_1994 / 8UL) * 1024UL) + (((fused_0fused_0fused_0_fuseiter_7587___fuseiter_7588_1992___fuseiter_7589_1993___fuseiter_7590_1994 % 8UL) * 128UL) + _fuseiter_7591))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__504(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7592___fuseiter_7593_1995___fuseiter_7594_1996___fuseiter_7595_1997 = 0UL; fused_0fused_0fused_0_fuseiter_7592___fuseiter_7593_1995___fuseiter_7594_1996___fuseiter_7595_1997 < 8UL; fused_0fused_0fused_0_fuseiter_7592___fuseiter_7593_1995___fuseiter_7594_1996___fuseiter_7595_1997 += 1UL) {
    for (uint64_t _fuseiter_7596 = 0UL; _fuseiter_7596 < 128UL; _fuseiter_7596 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7592___fuseiter_7593_1995___fuseiter_7594_1996___fuseiter_7595_1997 / 8UL) * 1024UL) + (_fuseiter_7596 + ((fused_0fused_0fused_0_fuseiter_7592___fuseiter_7593_1995___fuseiter_7594_1996___fuseiter_7595_1997 % 8UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7592___fuseiter_7593_1995___fuseiter_7594_1996___fuseiter_7595_1997 / 8UL) * 1024UL) + (((fused_0fused_0fused_0_fuseiter_7592___fuseiter_7593_1995___fuseiter_7594_1996___fuseiter_7595_1997 % 8UL) * 128UL) + _fuseiter_7596))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__513(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7597___fuseiter_7598_1998___fuseiter_7599_1999___fuseiter_7600_2000 = 0UL; fused_0fused_0fused_0_fuseiter_7597___fuseiter_7598_1998___fuseiter_7599_1999___fuseiter_7600_2000 < 8UL; fused_0fused_0fused_0_fuseiter_7597___fuseiter_7598_1998___fuseiter_7599_1999___fuseiter_7600_2000 += 1UL) {
    for (uint64_t _fuseiter_7601 = 0UL; _fuseiter_7601 < 128UL; _fuseiter_7601 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7597___fuseiter_7598_1998___fuseiter_7599_1999___fuseiter_7600_2000 / 8UL) * 1024UL) + (_fuseiter_7601 + ((fused_0fused_0fused_0_fuseiter_7597___fuseiter_7598_1998___fuseiter_7599_1999___fuseiter_7600_2000 % 8UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7597___fuseiter_7598_1998___fuseiter_7599_1999___fuseiter_7600_2000 / 8UL) * 1024UL) + (((fused_0fused_0fused_0_fuseiter_7597___fuseiter_7598_1998___fuseiter_7599_1999___fuseiter_7600_2000 % 8UL) * 128UL) + _fuseiter_7601))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__520(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7602___fuseiter_7603_2001___fuseiter_7604_2002___fuseiter_7605_2003 = 0UL; fused_0fused_0fused_0_fuseiter_7602___fuseiter_7603_2001___fuseiter_7604_2002___fuseiter_7605_2003 < 8UL; fused_0fused_0fused_0_fuseiter_7602___fuseiter_7603_2001___fuseiter_7604_2002___fuseiter_7605_2003 += 1UL) {
    for (uint64_t _fuseiter_7606 = 0UL; _fuseiter_7606 < 128UL; _fuseiter_7606 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7602___fuseiter_7603_2001___fuseiter_7604_2002___fuseiter_7605_2003 / 8UL) * 1024UL) + (_fuseiter_7606 + ((fused_0fused_0fused_0_fuseiter_7602___fuseiter_7603_2001___fuseiter_7604_2002___fuseiter_7605_2003 % 8UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7602___fuseiter_7603_2001___fuseiter_7604_2002___fuseiter_7605_2003 / 8UL) * 1024UL) + (((fused_0fused_0fused_0_fuseiter_7602___fuseiter_7603_2001___fuseiter_7604_2002___fuseiter_7605_2003 % 8UL) * 128UL) + _fuseiter_7606))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__529(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7607___fuseiter_7608_2004___fuseiter_7609_2005___fuseiter_7610_2006 = 0UL; fused_0fused_0fused_0_fuseiter_7607___fuseiter_7608_2004___fuseiter_7609_2005___fuseiter_7610_2006 < 4UL; fused_0fused_0fused_0_fuseiter_7607___fuseiter_7608_2004___fuseiter_7609_2005___fuseiter_7610_2006 += 1UL) {
    for (uint64_t _fuseiter_7611 = 0UL; _fuseiter_7611 < 256UL; _fuseiter_7611 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7607___fuseiter_7608_2004___fuseiter_7609_2005___fuseiter_7610_2006 / 4UL) * 1024UL) + (_fuseiter_7611 + ((fused_0fused_0fused_0_fuseiter_7607___fuseiter_7608_2004___fuseiter_7609_2005___fuseiter_7610_2006 % 4UL) * 256UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7607___fuseiter_7608_2004___fuseiter_7609_2005___fuseiter_7610_2006 / 4UL) * 1024UL) + (((fused_0fused_0fused_0_fuseiter_7607___fuseiter_7608_2004___fuseiter_7609_2005___fuseiter_7610_2006 % 4UL) * 256UL) + _fuseiter_7611))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__420(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_7612___fuseiter_7613_2007 = 0UL; fused_0_fuseiter_7612___fuseiter_7613_2007 < 16UL; fused_0_fuseiter_7612___fuseiter_7613_2007 += 1UL) {
    for (uint64_t _fuseiter_7616 = 0UL; _fuseiter_7616 < 16UL; _fuseiter_7616 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_7612___fuseiter_7613_2007 / 16UL) * 256UL) + (_fuseiter_7616 + ((fused_0_fuseiter_7612___fuseiter_7613_2007 % 16UL) * 16UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_7612___fuseiter_7613_2007 / 16UL) * 256UL) + (((fused_0_fuseiter_7612___fuseiter_7613_2007 % 16UL) * 16UL) + _fuseiter_7616))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__424(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7617___fuseiter_7618_2008___fuseiter_7619_2009___fuseiter_7620_2010 = 0UL; fused_0fused_0fused_0_fuseiter_7617___fuseiter_7618_2008___fuseiter_7619_2009___fuseiter_7620_2010 < 2UL; fused_0fused_0fused_0_fuseiter_7617___fuseiter_7618_2008___fuseiter_7619_2009___fuseiter_7620_2010 += 1UL) {
    for (uint64_t _fuseiter_7621 = 0UL; _fuseiter_7621 < 32UL; _fuseiter_7621 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7617___fuseiter_7618_2008___fuseiter_7619_2009___fuseiter_7620_2010 / 2UL) * 64UL) + (_fuseiter_7621 + ((fused_0fused_0fused_0_fuseiter_7617___fuseiter_7618_2008___fuseiter_7619_2009___fuseiter_7620_2010 % 2UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7617___fuseiter_7618_2008___fuseiter_7619_2009___fuseiter_7620_2010 / 2UL) * 64UL) + (((fused_0fused_0fused_0_fuseiter_7617___fuseiter_7618_2008___fuseiter_7619_2009___fuseiter_7620_2010 % 2UL) * 32UL) + _fuseiter_7621))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__427(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7622___fuseiter_7623_2011___fuseiter_7624_2012___fuseiter_7625_2013 = 0UL; fused_0fused_0fused_0_fuseiter_7622___fuseiter_7623_2011___fuseiter_7624_2012___fuseiter_7625_2013 < 4UL; fused_0fused_0fused_0_fuseiter_7622___fuseiter_7623_2011___fuseiter_7624_2012___fuseiter_7625_2013 += 1UL) {
    for (uint64_t _fuseiter_7626 = 0UL; _fuseiter_7626 < 64UL; _fuseiter_7626 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7622___fuseiter_7623_2011___fuseiter_7624_2012___fuseiter_7625_2013 / 4UL) * 256UL) + (_fuseiter_7626 + ((fused_0fused_0fused_0_fuseiter_7622___fuseiter_7623_2011___fuseiter_7624_2012___fuseiter_7625_2013 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7622___fuseiter_7623_2011___fuseiter_7624_2012___fuseiter_7625_2013 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7622___fuseiter_7623_2011___fuseiter_7624_2012___fuseiter_7625_2013 % 4UL) * 64UL) + _fuseiter_7626))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__431(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7627___fuseiter_7628_2014___fuseiter_7629_2015___fuseiter_7630_2016 = 0UL; fused_0fused_0fused_0_fuseiter_7627___fuseiter_7628_2014___fuseiter_7629_2015___fuseiter_7630_2016 < 4UL; fused_0fused_0fused_0_fuseiter_7627___fuseiter_7628_2014___fuseiter_7629_2015___fuseiter_7630_2016 += 1UL) {
    for (uint64_t _fuseiter_7631 = 0UL; _fuseiter_7631 < 16UL; _fuseiter_7631 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7627___fuseiter_7628_2014___fuseiter_7629_2015___fuseiter_7630_2016 / 4UL) * 64UL) + (_fuseiter_7631 + ((fused_0fused_0fused_0_fuseiter_7627___fuseiter_7628_2014___fuseiter_7629_2015___fuseiter_7630_2016 % 4UL) * 16UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7627___fuseiter_7628_2014___fuseiter_7629_2015___fuseiter_7630_2016 / 4UL) * 64UL) + (((fused_0fused_0fused_0_fuseiter_7627___fuseiter_7628_2014___fuseiter_7629_2015___fuseiter_7630_2016 % 4UL) * 16UL) + _fuseiter_7631))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__434(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7632___fuseiter_7633_2017___fuseiter_7634_2018___fuseiter_7635_2019 = 0UL; fused_0fused_0fused_0_fuseiter_7632___fuseiter_7633_2017___fuseiter_7634_2018___fuseiter_7635_2019 < 8UL; fused_0fused_0fused_0_fuseiter_7632___fuseiter_7633_2017___fuseiter_7634_2018___fuseiter_7635_2019 += 1UL) {
    for (uint64_t _fuseiter_7636 = 0UL; _fuseiter_7636 < 32UL; _fuseiter_7636 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7632___fuseiter_7633_2017___fuseiter_7634_2018___fuseiter_7635_2019 / 8UL) * 256UL) + (_fuseiter_7636 + ((fused_0fused_0fused_0_fuseiter_7632___fuseiter_7633_2017___fuseiter_7634_2018___fuseiter_7635_2019 % 8UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7632___fuseiter_7633_2017___fuseiter_7634_2018___fuseiter_7635_2019 / 8UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7632___fuseiter_7633_2017___fuseiter_7634_2018___fuseiter_7635_2019 % 8UL) * 32UL) + _fuseiter_7636))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__437(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7637___fuseiter_7638_2020___fuseiter_7639_2021___fuseiter_7640_2022 = 0UL; fused_0fused_0fused_0_fuseiter_7637___fuseiter_7638_2020___fuseiter_7639_2021___fuseiter_7640_2022 < 2UL; fused_0fused_0fused_0_fuseiter_7637___fuseiter_7638_2020___fuseiter_7639_2021___fuseiter_7640_2022 += 1UL) {
    for (uint64_t _fuseiter_7641 = 0UL; _fuseiter_7641 < 32UL; _fuseiter_7641 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7637___fuseiter_7638_2020___fuseiter_7639_2021___fuseiter_7640_2022 / 2UL) * 64UL) + (_fuseiter_7641 + ((fused_0fused_0fused_0_fuseiter_7637___fuseiter_7638_2020___fuseiter_7639_2021___fuseiter_7640_2022 % 2UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7637___fuseiter_7638_2020___fuseiter_7639_2021___fuseiter_7640_2022 / 2UL) * 64UL) + (((fused_0fused_0fused_0_fuseiter_7637___fuseiter_7638_2020___fuseiter_7639_2021___fuseiter_7640_2022 % 2UL) * 32UL) + _fuseiter_7641))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__440(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7642___fuseiter_7643_2023___fuseiter_7644_2024___fuseiter_7645_2025 = 0UL; fused_0fused_0fused_0_fuseiter_7642___fuseiter_7643_2023___fuseiter_7644_2024___fuseiter_7645_2025 < 2UL; fused_0fused_0fused_0_fuseiter_7642___fuseiter_7643_2023___fuseiter_7644_2024___fuseiter_7645_2025 += 1UL) {
    for (uint64_t _fuseiter_7646 = 0UL; _fuseiter_7646 < 32UL; _fuseiter_7646 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7642___fuseiter_7643_2023___fuseiter_7644_2024___fuseiter_7645_2025 / 2UL) * 64UL) + (_fuseiter_7646 + ((fused_0fused_0fused_0_fuseiter_7642___fuseiter_7643_2023___fuseiter_7644_2024___fuseiter_7645_2025 % 2UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7642___fuseiter_7643_2023___fuseiter_7644_2024___fuseiter_7645_2025 / 2UL) * 64UL) + (((fused_0fused_0fused_0_fuseiter_7642___fuseiter_7643_2023___fuseiter_7644_2024___fuseiter_7645_2025 % 2UL) * 32UL) + _fuseiter_7646))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__443(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7647___fuseiter_7648_2026___fuseiter_7649_2027___fuseiter_7650_2028 = 0UL; fused_0fused_0fused_0_fuseiter_7647___fuseiter_7648_2026___fuseiter_7649_2027___fuseiter_7650_2028 < 4UL; fused_0fused_0fused_0_fuseiter_7647___fuseiter_7648_2026___fuseiter_7649_2027___fuseiter_7650_2028 += 1UL) {
    for (uint64_t _fuseiter_7651 = 0UL; _fuseiter_7651 < 64UL; _fuseiter_7651 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7647___fuseiter_7648_2026___fuseiter_7649_2027___fuseiter_7650_2028 / 4UL) * 256UL) + (_fuseiter_7651 + ((fused_0fused_0fused_0_fuseiter_7647___fuseiter_7648_2026___fuseiter_7649_2027___fuseiter_7650_2028 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7647___fuseiter_7648_2026___fuseiter_7649_2027___fuseiter_7650_2028 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7647___fuseiter_7648_2026___fuseiter_7649_2027___fuseiter_7650_2028 % 4UL) * 64UL) + _fuseiter_7651))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__446(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_7652___fuseiter_7653_2029 = 0UL; fused_0_fuseiter_7652___fuseiter_7653_2029 < 16UL; fused_0_fuseiter_7652___fuseiter_7653_2029 += 1UL) {
    for (uint64_t _fuseiter_7656 = 0UL; _fuseiter_7656 < 32UL; _fuseiter_7656 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_7652___fuseiter_7653_2029 / 16UL) * 512UL) + (_fuseiter_7656 + ((fused_0_fuseiter_7652___fuseiter_7653_2029 % 16UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_7652___fuseiter_7653_2029 / 16UL) * 512UL) + (((fused_0_fuseiter_7652___fuseiter_7653_2029 % 16UL) * 32UL) + _fuseiter_7656))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__449(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7657___fuseiter_7658_2030___fuseiter_7659_2031___fuseiter_7660_2032 = 0UL; fused_0fused_0fused_0_fuseiter_7657___fuseiter_7658_2030___fuseiter_7659_2031___fuseiter_7660_2032 < 2UL; fused_0fused_0fused_0_fuseiter_7657___fuseiter_7658_2030___fuseiter_7659_2031___fuseiter_7660_2032 += 1UL) {
    for (uint64_t _fuseiter_7661 = 0UL; _fuseiter_7661 < 64UL; _fuseiter_7661 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7657___fuseiter_7658_2030___fuseiter_7659_2031___fuseiter_7660_2032 / 2UL) * 128UL) + (_fuseiter_7661 + ((fused_0fused_0fused_0_fuseiter_7657___fuseiter_7658_2030___fuseiter_7659_2031___fuseiter_7660_2032 % 2UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7657___fuseiter_7658_2030___fuseiter_7659_2031___fuseiter_7660_2032 / 2UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_7657___fuseiter_7658_2030___fuseiter_7659_2031___fuseiter_7660_2032 % 2UL) * 64UL) + _fuseiter_7661))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__452(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7662___fuseiter_7663_2033___fuseiter_7664_2034___fuseiter_7665_2035 = 0UL; fused_0fused_0fused_0_fuseiter_7662___fuseiter_7663_2033___fuseiter_7664_2034___fuseiter_7665_2035 < 4UL; fused_0fused_0fused_0_fuseiter_7662___fuseiter_7663_2033___fuseiter_7664_2034___fuseiter_7665_2035 += 1UL) {
    for (uint64_t _fuseiter_7666 = 0UL; _fuseiter_7666 < 32UL; _fuseiter_7666 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7662___fuseiter_7663_2033___fuseiter_7664_2034___fuseiter_7665_2035 / 4UL) * 128UL) + (_fuseiter_7666 + ((fused_0fused_0fused_0_fuseiter_7662___fuseiter_7663_2033___fuseiter_7664_2034___fuseiter_7665_2035 % 4UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7662___fuseiter_7663_2033___fuseiter_7664_2034___fuseiter_7665_2035 / 4UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_7662___fuseiter_7663_2033___fuseiter_7664_2034___fuseiter_7665_2035 % 4UL) * 32UL) + _fuseiter_7666))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__455(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_7667___fuseiter_7668_2036 = 0UL; fused_0_fuseiter_7667___fuseiter_7668_2036 < 16UL; fused_0_fuseiter_7667___fuseiter_7668_2036 += 1UL) {
    for (uint64_t _fuseiter_7671 = 0UL; _fuseiter_7671 < 32UL; _fuseiter_7671 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_7667___fuseiter_7668_2036 / 16UL) * 512UL) + (_fuseiter_7671 + ((fused_0_fuseiter_7667___fuseiter_7668_2036 % 16UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_7667___fuseiter_7668_2036 / 16UL) * 512UL) + (((fused_0_fuseiter_7667___fuseiter_7668_2036 % 16UL) * 32UL) + _fuseiter_7671))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__458(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7672___fuseiter_7673_2037___fuseiter_7674_2038___fuseiter_7675_2039 = 0UL; fused_0fused_0fused_0_fuseiter_7672___fuseiter_7673_2037___fuseiter_7674_2038___fuseiter_7675_2039 < 4UL; fused_0fused_0fused_0_fuseiter_7672___fuseiter_7673_2037___fuseiter_7674_2038___fuseiter_7675_2039 += 1UL) {
    for (uint64_t _fuseiter_7676 = 0UL; _fuseiter_7676 < 32UL; _fuseiter_7676 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7672___fuseiter_7673_2037___fuseiter_7674_2038___fuseiter_7675_2039 / 4UL) * 128UL) + (_fuseiter_7676 + ((fused_0fused_0fused_0_fuseiter_7672___fuseiter_7673_2037___fuseiter_7674_2038___fuseiter_7675_2039 % 4UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7672___fuseiter_7673_2037___fuseiter_7674_2038___fuseiter_7675_2039 / 4UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_7672___fuseiter_7673_2037___fuseiter_7674_2038___fuseiter_7675_2039 % 4UL) * 32UL) + _fuseiter_7676))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__462(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7677___fuseiter_7678_2040___fuseiter_7679_2041___fuseiter_7680_2042 = 0UL; fused_0fused_0fused_0_fuseiter_7677___fuseiter_7678_2040___fuseiter_7679_2041___fuseiter_7680_2042 < 4UL; fused_0fused_0fused_0_fuseiter_7677___fuseiter_7678_2040___fuseiter_7679_2041___fuseiter_7680_2042 += 1UL) {
    for (uint64_t _fuseiter_7681 = 0UL; _fuseiter_7681 < 128UL; _fuseiter_7681 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7677___fuseiter_7678_2040___fuseiter_7679_2041___fuseiter_7680_2042 / 4UL) * 512UL) + (_fuseiter_7681 + ((fused_0fused_0fused_0_fuseiter_7677___fuseiter_7678_2040___fuseiter_7679_2041___fuseiter_7680_2042 % 4UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7677___fuseiter_7678_2040___fuseiter_7679_2041___fuseiter_7680_2042 / 4UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_7677___fuseiter_7678_2040___fuseiter_7679_2041___fuseiter_7680_2042 % 4UL) * 128UL) + _fuseiter_7681))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__466(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7682___fuseiter_7683_2043___fuseiter_7684_2044___fuseiter_7685_2045 = 0UL; fused_0fused_0fused_0_fuseiter_7682___fuseiter_7683_2043___fuseiter_7684_2044___fuseiter_7685_2045 < 4UL; fused_0fused_0fused_0_fuseiter_7682___fuseiter_7683_2043___fuseiter_7684_2044___fuseiter_7685_2045 += 1UL) {
    for (uint64_t _fuseiter_7686 = 0UL; _fuseiter_7686 < 32UL; _fuseiter_7686 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7682___fuseiter_7683_2043___fuseiter_7684_2044___fuseiter_7685_2045 / 4UL) * 128UL) + (_fuseiter_7686 + ((fused_0fused_0fused_0_fuseiter_7682___fuseiter_7683_2043___fuseiter_7684_2044___fuseiter_7685_2045 % 4UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7682___fuseiter_7683_2043___fuseiter_7684_2044___fuseiter_7685_2045 / 4UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_7682___fuseiter_7683_2043___fuseiter_7684_2044___fuseiter_7685_2045 % 4UL) * 32UL) + _fuseiter_7686))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__469(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7687___fuseiter_7688_2046___fuseiter_7689_2047___fuseiter_7690_2048 = 0UL; fused_0fused_0fused_0_fuseiter_7687___fuseiter_7688_2046___fuseiter_7689_2047___fuseiter_7690_2048 < 8UL; fused_0fused_0fused_0_fuseiter_7687___fuseiter_7688_2046___fuseiter_7689_2047___fuseiter_7690_2048 += 1UL) {
    for (uint64_t _fuseiter_7691 = 0UL; _fuseiter_7691 < 64UL; _fuseiter_7691 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7687___fuseiter_7688_2046___fuseiter_7689_2047___fuseiter_7690_2048 / 8UL) * 512UL) + (_fuseiter_7691 + ((fused_0fused_0fused_0_fuseiter_7687___fuseiter_7688_2046___fuseiter_7689_2047___fuseiter_7690_2048 % 8UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7687___fuseiter_7688_2046___fuseiter_7689_2047___fuseiter_7690_2048 / 8UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_7687___fuseiter_7688_2046___fuseiter_7689_2047___fuseiter_7690_2048 % 8UL) * 64UL) + _fuseiter_7691))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__472(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7692___fuseiter_7693_2049___fuseiter_7694_2050___fuseiter_7695_2051 = 0UL; fused_0fused_0fused_0_fuseiter_7692___fuseiter_7693_2049___fuseiter_7694_2050___fuseiter_7695_2051 < 2UL; fused_0fused_0fused_0_fuseiter_7692___fuseiter_7693_2049___fuseiter_7694_2050___fuseiter_7695_2051 += 1UL) {
    for (uint64_t _fuseiter_7696 = 0UL; _fuseiter_7696 < 64UL; _fuseiter_7696 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7692___fuseiter_7693_2049___fuseiter_7694_2050___fuseiter_7695_2051 / 2UL) * 128UL) + (_fuseiter_7696 + ((fused_0fused_0fused_0_fuseiter_7692___fuseiter_7693_2049___fuseiter_7694_2050___fuseiter_7695_2051 % 2UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7692___fuseiter_7693_2049___fuseiter_7694_2050___fuseiter_7695_2051 / 2UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_7692___fuseiter_7693_2049___fuseiter_7694_2050___fuseiter_7695_2051 % 2UL) * 64UL) + _fuseiter_7696))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__475(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7697___fuseiter_7698_2052___fuseiter_7699_2053___fuseiter_7700_2054 = 0UL; fused_0fused_0fused_0_fuseiter_7697___fuseiter_7698_2052___fuseiter_7699_2053___fuseiter_7700_2054 < 4UL; fused_0fused_0fused_0_fuseiter_7697___fuseiter_7698_2052___fuseiter_7699_2053___fuseiter_7700_2054 += 1UL) {
    for (uint64_t _fuseiter_7701 = 0UL; _fuseiter_7701 < 32UL; _fuseiter_7701 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7697___fuseiter_7698_2052___fuseiter_7699_2053___fuseiter_7700_2054 / 4UL) * 128UL) + (_fuseiter_7701 + ((fused_0fused_0fused_0_fuseiter_7697___fuseiter_7698_2052___fuseiter_7699_2053___fuseiter_7700_2054 % 4UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7697___fuseiter_7698_2052___fuseiter_7699_2053___fuseiter_7700_2054 / 4UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_7697___fuseiter_7698_2052___fuseiter_7699_2053___fuseiter_7700_2054 % 4UL) * 32UL) + _fuseiter_7701))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__478(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7702___fuseiter_7703_2055___fuseiter_7704_2056___fuseiter_7705_2057 = 0UL; fused_0fused_0fused_0_fuseiter_7702___fuseiter_7703_2055___fuseiter_7704_2056___fuseiter_7705_2057 < 4UL; fused_0fused_0fused_0_fuseiter_7702___fuseiter_7703_2055___fuseiter_7704_2056___fuseiter_7705_2057 += 1UL) {
    for (uint64_t _fuseiter_7706 = 0UL; _fuseiter_7706 < 128UL; _fuseiter_7706 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7702___fuseiter_7703_2055___fuseiter_7704_2056___fuseiter_7705_2057 / 4UL) * 512UL) + (_fuseiter_7706 + ((fused_0fused_0fused_0_fuseiter_7702___fuseiter_7703_2055___fuseiter_7704_2056___fuseiter_7705_2057 % 4UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7702___fuseiter_7703_2055___fuseiter_7704_2056___fuseiter_7705_2057 / 4UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_7702___fuseiter_7703_2055___fuseiter_7704_2056___fuseiter_7705_2057 % 4UL) * 128UL) + _fuseiter_7706))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__485(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7707___fuseiter_7708_2058___fuseiter_7709_2059___fuseiter_7710_2060 = 0UL; fused_0fused_0fused_0_fuseiter_7707___fuseiter_7708_2058___fuseiter_7709_2059___fuseiter_7710_2060 < 2UL; fused_0fused_0fused_0_fuseiter_7707___fuseiter_7708_2058___fuseiter_7709_2059___fuseiter_7710_2060 += 1UL) {
    for (uint64_t _fuseiter_7711 = 0UL; _fuseiter_7711 < 128UL; _fuseiter_7711 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7707___fuseiter_7708_2058___fuseiter_7709_2059___fuseiter_7710_2060 / 2UL) * 256UL) + (_fuseiter_7711 + ((fused_0fused_0fused_0_fuseiter_7707___fuseiter_7708_2058___fuseiter_7709_2059___fuseiter_7710_2060 % 2UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7707___fuseiter_7708_2058___fuseiter_7709_2059___fuseiter_7710_2060 / 2UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7707___fuseiter_7708_2058___fuseiter_7709_2059___fuseiter_7710_2060 % 2UL) * 128UL) + _fuseiter_7711))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__491(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7712___fuseiter_7713_2061___fuseiter_7714_2062___fuseiter_7715_2063 = 0UL; fused_0fused_0fused_0_fuseiter_7712___fuseiter_7713_2061___fuseiter_7714_2062___fuseiter_7715_2063 < 4UL; fused_0fused_0fused_0_fuseiter_7712___fuseiter_7713_2061___fuseiter_7714_2062___fuseiter_7715_2063 += 1UL) {
    for (uint64_t _fuseiter_7716 = 0UL; _fuseiter_7716 < 64UL; _fuseiter_7716 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7712___fuseiter_7713_2061___fuseiter_7714_2062___fuseiter_7715_2063 / 4UL) * 256UL) + (_fuseiter_7716 + ((fused_0fused_0fused_0_fuseiter_7712___fuseiter_7713_2061___fuseiter_7714_2062___fuseiter_7715_2063 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7712___fuseiter_7713_2061___fuseiter_7714_2062___fuseiter_7715_2063 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7712___fuseiter_7713_2061___fuseiter_7714_2062___fuseiter_7715_2063 % 4UL) * 64UL) + _fuseiter_7716))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__494(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_7717___fuseiter_7718_2064 = 0UL; fused_0_fuseiter_7717___fuseiter_7718_2064 < 16UL; fused_0_fuseiter_7717___fuseiter_7718_2064 += 1UL) {
    for (uint64_t _fuseiter_7721 = 0UL; _fuseiter_7721 < 16UL; _fuseiter_7721 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_7717___fuseiter_7718_2064 / 16UL) * 256UL) + (_fuseiter_7721 + ((fused_0_fuseiter_7717___fuseiter_7718_2064 % 16UL) * 16UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_7717___fuseiter_7718_2064 / 16UL) * 256UL) + (((fused_0_fuseiter_7717___fuseiter_7718_2064 % 16UL) * 16UL) + _fuseiter_7721))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__498(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7722___fuseiter_7723_2065___fuseiter_7724_2066___fuseiter_7725_2067 = 0UL; fused_0fused_0fused_0_fuseiter_7722___fuseiter_7723_2065___fuseiter_7724_2066___fuseiter_7725_2067 < 8UL; fused_0fused_0fused_0_fuseiter_7722___fuseiter_7723_2065___fuseiter_7724_2066___fuseiter_7725_2067 += 1UL) {
    for (uint64_t _fuseiter_7726 = 0UL; _fuseiter_7726 < 32UL; _fuseiter_7726 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7722___fuseiter_7723_2065___fuseiter_7724_2066___fuseiter_7725_2067 / 8UL) * 256UL) + (_fuseiter_7726 + ((fused_0fused_0fused_0_fuseiter_7722___fuseiter_7723_2065___fuseiter_7724_2066___fuseiter_7725_2067 % 8UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7722___fuseiter_7723_2065___fuseiter_7724_2066___fuseiter_7725_2067 / 8UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7722___fuseiter_7723_2065___fuseiter_7724_2066___fuseiter_7725_2067 % 8UL) * 32UL) + _fuseiter_7726))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__501(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7727___fuseiter_7728_2068___fuseiter_7729_2069___fuseiter_7730_2070 = 0UL; fused_0fused_0fused_0_fuseiter_7727___fuseiter_7728_2068___fuseiter_7729_2069___fuseiter_7730_2070 < 8UL; fused_0fused_0fused_0_fuseiter_7727___fuseiter_7728_2068___fuseiter_7729_2069___fuseiter_7730_2070 += 1UL) {
    for (uint64_t _fuseiter_7731 = 0UL; _fuseiter_7731 < 32UL; _fuseiter_7731 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7727___fuseiter_7728_2068___fuseiter_7729_2069___fuseiter_7730_2070 / 8UL) * 256UL) + (_fuseiter_7731 + ((fused_0fused_0fused_0_fuseiter_7727___fuseiter_7728_2068___fuseiter_7729_2069___fuseiter_7730_2070 % 8UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7727___fuseiter_7728_2068___fuseiter_7729_2069___fuseiter_7730_2070 / 8UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7727___fuseiter_7728_2068___fuseiter_7729_2069___fuseiter_7730_2070 % 8UL) * 32UL) + _fuseiter_7731))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__507(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7732___fuseiter_7733_2071___fuseiter_7734_2072___fuseiter_7735_2073 = 0UL; fused_0fused_0fused_0_fuseiter_7732___fuseiter_7733_2071___fuseiter_7734_2072___fuseiter_7735_2073 < 4UL; fused_0fused_0fused_0_fuseiter_7732___fuseiter_7733_2071___fuseiter_7734_2072___fuseiter_7735_2073 += 1UL) {
    for (uint64_t _fuseiter_7736 = 0UL; _fuseiter_7736 < 64UL; _fuseiter_7736 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7732___fuseiter_7733_2071___fuseiter_7734_2072___fuseiter_7735_2073 / 4UL) * 256UL) + (_fuseiter_7736 + ((fused_0fused_0fused_0_fuseiter_7732___fuseiter_7733_2071___fuseiter_7734_2072___fuseiter_7735_2073 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7732___fuseiter_7733_2071___fuseiter_7734_2072___fuseiter_7735_2073 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7732___fuseiter_7733_2071___fuseiter_7734_2072___fuseiter_7735_2073 % 4UL) * 64UL) + _fuseiter_7736))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__510(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7737___fuseiter_7738_2074___fuseiter_7739_2075___fuseiter_7740_2076 = 0UL; fused_0fused_0fused_0_fuseiter_7737___fuseiter_7738_2074___fuseiter_7739_2075___fuseiter_7740_2076 < 4UL; fused_0fused_0fused_0_fuseiter_7737___fuseiter_7738_2074___fuseiter_7739_2075___fuseiter_7740_2076 += 1UL) {
    for (uint64_t _fuseiter_7741 = 0UL; _fuseiter_7741 < 64UL; _fuseiter_7741 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7737___fuseiter_7738_2074___fuseiter_7739_2075___fuseiter_7740_2076 / 4UL) * 256UL) + (_fuseiter_7741 + ((fused_0fused_0fused_0_fuseiter_7737___fuseiter_7738_2074___fuseiter_7739_2075___fuseiter_7740_2076 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7737___fuseiter_7738_2074___fuseiter_7739_2075___fuseiter_7740_2076 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7737___fuseiter_7738_2074___fuseiter_7739_2075___fuseiter_7740_2076 % 4UL) * 64UL) + _fuseiter_7741))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__517(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7742___fuseiter_7743_2077___fuseiter_7744_2078___fuseiter_7745_2079 = 0UL; fused_0fused_0fused_0_fuseiter_7742___fuseiter_7743_2077___fuseiter_7744_2078___fuseiter_7745_2079 < 8UL; fused_0fused_0fused_0_fuseiter_7742___fuseiter_7743_2077___fuseiter_7744_2078___fuseiter_7745_2079 += 1UL) {
    for (uint64_t _fuseiter_7746 = 0UL; _fuseiter_7746 < 32UL; _fuseiter_7746 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7742___fuseiter_7743_2077___fuseiter_7744_2078___fuseiter_7745_2079 / 8UL) * 256UL) + (_fuseiter_7746 + ((fused_0fused_0fused_0_fuseiter_7742___fuseiter_7743_2077___fuseiter_7744_2078___fuseiter_7745_2079 % 8UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7742___fuseiter_7743_2077___fuseiter_7744_2078___fuseiter_7745_2079 / 8UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7742___fuseiter_7743_2077___fuseiter_7744_2078___fuseiter_7745_2079 % 8UL) * 32UL) + _fuseiter_7746))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__523(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7747___fuseiter_7748_2080___fuseiter_7749_2081___fuseiter_7750_2082 = 0UL; fused_0fused_0fused_0_fuseiter_7747___fuseiter_7748_2080___fuseiter_7749_2081___fuseiter_7750_2082 < 8UL; fused_0fused_0fused_0_fuseiter_7747___fuseiter_7748_2080___fuseiter_7749_2081___fuseiter_7750_2082 += 1UL) {
    for (uint64_t _fuseiter_7751 = 0UL; _fuseiter_7751 < 32UL; _fuseiter_7751 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7747___fuseiter_7748_2080___fuseiter_7749_2081___fuseiter_7750_2082 / 8UL) * 256UL) + (_fuseiter_7751 + ((fused_0fused_0fused_0_fuseiter_7747___fuseiter_7748_2080___fuseiter_7749_2081___fuseiter_7750_2082 % 8UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7747___fuseiter_7748_2080___fuseiter_7749_2081___fuseiter_7750_2082 / 8UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7747___fuseiter_7748_2080___fuseiter_7749_2081___fuseiter_7750_2082 % 8UL) * 32UL) + _fuseiter_7751))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__526(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7752___fuseiter_7753_2083___fuseiter_7754_2084___fuseiter_7755_2085 = 0UL; fused_0fused_0fused_0_fuseiter_7752___fuseiter_7753_2083___fuseiter_7754_2084___fuseiter_7755_2085 < 4UL; fused_0fused_0fused_0_fuseiter_7752___fuseiter_7753_2083___fuseiter_7754_2084___fuseiter_7755_2085 += 1UL) {
    for (uint64_t _fuseiter_7756 = 0UL; _fuseiter_7756 < 64UL; _fuseiter_7756 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7752___fuseiter_7753_2083___fuseiter_7754_2084___fuseiter_7755_2085 / 4UL) * 256UL) + (_fuseiter_7756 + ((fused_0fused_0fused_0_fuseiter_7752___fuseiter_7753_2083___fuseiter_7754_2084___fuseiter_7755_2085 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7752___fuseiter_7753_2083___fuseiter_7754_2084___fuseiter_7755_2085 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7752___fuseiter_7753_2083___fuseiter_7754_2084___fuseiter_7755_2085 % 4UL) * 64UL) + _fuseiter_7756))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__532(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_7757___fuseiter_7758_2086 = 0UL; fused_0_fuseiter_7757___fuseiter_7758_2086 < 128UL; fused_0_fuseiter_7757___fuseiter_7758_2086 += 1UL) {
    for (uint64_t _fuseiter_7761 = 0UL; _fuseiter_7761 < 16UL; _fuseiter_7761 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_7757___fuseiter_7758_2086 / 128UL) * 2048UL) + (_fuseiter_7761 + ((fused_0_fuseiter_7757___fuseiter_7758_2086 % 128UL) * 16UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_7757___fuseiter_7758_2086 / 128UL) * 2048UL) + (((fused_0_fuseiter_7757___fuseiter_7758_2086 % 128UL) * 16UL) + _fuseiter_7761))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__535(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7762___fuseiter_7763_2087___fuseiter_7764_2088___fuseiter_7765_2089 = 0UL; fused_0fused_0fused_0_fuseiter_7762___fuseiter_7763_2087___fuseiter_7764_2088___fuseiter_7765_2089 < 8UL; fused_0fused_0fused_0_fuseiter_7762___fuseiter_7763_2087___fuseiter_7764_2088___fuseiter_7765_2089 += 1UL) {
    for (uint64_t _fuseiter_7766 = 0UL; _fuseiter_7766 < 64UL; _fuseiter_7766 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7762___fuseiter_7763_2087___fuseiter_7764_2088___fuseiter_7765_2089 / 8UL) * 512UL) + (_fuseiter_7766 + ((fused_0fused_0fused_0_fuseiter_7762___fuseiter_7763_2087___fuseiter_7764_2088___fuseiter_7765_2089 % 8UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7762___fuseiter_7763_2087___fuseiter_7764_2088___fuseiter_7765_2089 / 8UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_7762___fuseiter_7763_2087___fuseiter_7764_2088___fuseiter_7765_2089 % 8UL) * 64UL) + _fuseiter_7766))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__538(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7767___fuseiter_7768_2090___fuseiter_7769_2091___fuseiter_7770_2092 = 0UL; fused_0fused_0fused_0_fuseiter_7767___fuseiter_7768_2090___fuseiter_7769_2091___fuseiter_7770_2092 < 4UL; fused_0fused_0fused_0_fuseiter_7767___fuseiter_7768_2090___fuseiter_7769_2091___fuseiter_7770_2092 += 1UL) {
    for (uint64_t _fuseiter_7771 = 0UL; _fuseiter_7771 < 128UL; _fuseiter_7771 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7767___fuseiter_7768_2090___fuseiter_7769_2091___fuseiter_7770_2092 / 4UL) * 512UL) + (_fuseiter_7771 + ((fused_0fused_0fused_0_fuseiter_7767___fuseiter_7768_2090___fuseiter_7769_2091___fuseiter_7770_2092 % 4UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7767___fuseiter_7768_2090___fuseiter_7769_2091___fuseiter_7770_2092 / 4UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_7767___fuseiter_7768_2090___fuseiter_7769_2091___fuseiter_7770_2092 % 4UL) * 128UL) + _fuseiter_7771))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__541(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_7772___fuseiter_7773_2093 = 0UL; fused_0_fuseiter_7772___fuseiter_7773_2093 < 32UL; fused_0_fuseiter_7772___fuseiter_7773_2093 += 1UL) {
    for (uint64_t _fuseiter_7776 = 0UL; _fuseiter_7776 < 64UL; _fuseiter_7776 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_7772___fuseiter_7773_2093 / 32UL) * 2048UL) + (_fuseiter_7776 + ((fused_0_fuseiter_7772___fuseiter_7773_2093 % 32UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_7772___fuseiter_7773_2093 / 32UL) * 2048UL) + (((fused_0_fuseiter_7772___fuseiter_7773_2093 % 32UL) * 64UL) + _fuseiter_7776))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__544(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7777___fuseiter_7778_2094___fuseiter_7779_2095___fuseiter_7780_2096 = 0UL; fused_0fused_0fused_0_fuseiter_7777___fuseiter_7778_2094___fuseiter_7779_2095___fuseiter_7780_2096 < 4UL; fused_0fused_0fused_0_fuseiter_7777___fuseiter_7778_2094___fuseiter_7779_2095___fuseiter_7780_2096 += 1UL) {
    for (uint64_t _fuseiter_7781 = 0UL; _fuseiter_7781 < 128UL; _fuseiter_7781 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7777___fuseiter_7778_2094___fuseiter_7779_2095___fuseiter_7780_2096 / 4UL) * 512UL) + (_fuseiter_7781 + ((fused_0fused_0fused_0_fuseiter_7777___fuseiter_7778_2094___fuseiter_7779_2095___fuseiter_7780_2096 % 4UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7777___fuseiter_7778_2094___fuseiter_7779_2095___fuseiter_7780_2096 / 4UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_7777___fuseiter_7778_2094___fuseiter_7779_2095___fuseiter_7780_2096 % 4UL) * 128UL) + _fuseiter_7781))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__547(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_7782___fuseiter_7783_2097 = 0UL; fused_0_fuseiter_7782___fuseiter_7783_2097 < 32UL; fused_0_fuseiter_7782___fuseiter_7783_2097 += 1UL) {
    for (uint64_t _fuseiter_7786 = 0UL; _fuseiter_7786 < 16UL; _fuseiter_7786 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_7782___fuseiter_7783_2097 / 32UL) * 512UL) + (_fuseiter_7786 + ((fused_0_fuseiter_7782___fuseiter_7783_2097 % 32UL) * 16UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_7782___fuseiter_7783_2097 / 32UL) * 512UL) + (((fused_0_fuseiter_7782___fuseiter_7783_2097 % 32UL) * 16UL) + _fuseiter_7786))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__550(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_7787___fuseiter_7788_2098 = 0UL; fused_0_fuseiter_7787___fuseiter_7788_2098 < 32UL; fused_0_fuseiter_7787___fuseiter_7788_2098 += 1UL) {
    for (uint64_t _fuseiter_7791 = 0UL; _fuseiter_7791 < 64UL; _fuseiter_7791 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_7787___fuseiter_7788_2098 / 32UL) * 2048UL) + (_fuseiter_7791 + ((fused_0_fuseiter_7787___fuseiter_7788_2098 % 32UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_7787___fuseiter_7788_2098 / 32UL) * 2048UL) + (((fused_0_fuseiter_7787___fuseiter_7788_2098 % 32UL) * 64UL) + _fuseiter_7791))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__553(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_7792___fuseiter_7793_2099 = 0UL; fused_0_fuseiter_7792___fuseiter_7793_2099 < 16UL; fused_0_fuseiter_7792___fuseiter_7793_2099 += 1UL) {
    for (uint64_t _fuseiter_7796 = 0UL; _fuseiter_7796 < 32UL; _fuseiter_7796 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_7792___fuseiter_7793_2099 / 16UL) * 512UL) + (_fuseiter_7796 + ((fused_0_fuseiter_7792___fuseiter_7793_2099 % 16UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_7792___fuseiter_7793_2099 / 16UL) * 512UL) + (((fused_0_fuseiter_7792___fuseiter_7793_2099 % 16UL) * 32UL) + _fuseiter_7796))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__556(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7797___fuseiter_7798_2100___fuseiter_7799_2101___fuseiter_7800_2102 = 0UL; fused_0fused_0fused_0_fuseiter_7797___fuseiter_7798_2100___fuseiter_7799_2101___fuseiter_7800_2102 < 4UL; fused_0fused_0fused_0_fuseiter_7797___fuseiter_7798_2100___fuseiter_7799_2101___fuseiter_7800_2102 += 1UL) {
    for (uint64_t _fuseiter_7801 = 0UL; _fuseiter_7801 < 128UL; _fuseiter_7801 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7797___fuseiter_7798_2100___fuseiter_7799_2101___fuseiter_7800_2102 / 4UL) * 512UL) + (_fuseiter_7801 + ((fused_0fused_0fused_0_fuseiter_7797___fuseiter_7798_2100___fuseiter_7799_2101___fuseiter_7800_2102 % 4UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7797___fuseiter_7798_2100___fuseiter_7799_2101___fuseiter_7800_2102 / 4UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_7797___fuseiter_7798_2100___fuseiter_7799_2101___fuseiter_7800_2102 % 4UL) * 128UL) + _fuseiter_7801))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__559(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7802___fuseiter_7803_2103___fuseiter_7804_2104___fuseiter_7805_2105 = 0UL; fused_0fused_0fused_0_fuseiter_7802___fuseiter_7803_2103___fuseiter_7804_2104___fuseiter_7805_2105 < 4UL; fused_0fused_0fused_0_fuseiter_7802___fuseiter_7803_2103___fuseiter_7804_2104___fuseiter_7805_2105 += 1UL) {
    for (uint64_t _fuseiter_7806 = 0UL; _fuseiter_7806 < 512UL; _fuseiter_7806 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_7802___fuseiter_7803_2103___fuseiter_7804_2104___fuseiter_7805_2105 / 4UL) * 2048UL) + (_fuseiter_7806 + ((fused_0fused_0fused_0_fuseiter_7802___fuseiter_7803_2103___fuseiter_7804_2104___fuseiter_7805_2105 % 4UL) * 512UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_7802___fuseiter_7803_2103___fuseiter_7804_2104___fuseiter_7805_2105 / 4UL) * 2048UL) + (((fused_0fused_0fused_0_fuseiter_7802___fuseiter_7803_2103___fuseiter_7804_2104___fuseiter_7805_2105 % 4UL) * 512UL) + _fuseiter_7806))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__425(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7807___fuseiter_7808_2106___fuseiter_7809_2107___fuseiter_7810_2108 = 0UL; fused_0fused_0fused_0_fuseiter_7807___fuseiter_7808_2106___fuseiter_7809_2107___fuseiter_7810_2108 < 2UL; fused_0fused_0fused_0_fuseiter_7807___fuseiter_7808_2106___fuseiter_7809_2107___fuseiter_7810_2108 += 1UL) {
    for (uint64_t _fuseiter_7811 = 0UL; _fuseiter_7811 < 32UL; _fuseiter_7811 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7807___fuseiter_7808_2106___fuseiter_7809_2107___fuseiter_7810_2108 / 2UL) * 64UL) + (_fuseiter_7811 + ((fused_0fused_0fused_0_fuseiter_7807___fuseiter_7808_2106___fuseiter_7809_2107___fuseiter_7810_2108 % 2UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7807___fuseiter_7808_2106___fuseiter_7809_2107___fuseiter_7810_2108 / 2UL) * 64UL) + (((fused_0fused_0fused_0_fuseiter_7807___fuseiter_7808_2106___fuseiter_7809_2107___fuseiter_7810_2108 % 2UL) * 32UL) + _fuseiter_7811))]);
    }
  }
  return true;
}

static bool reorder__432(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7812___fuseiter_7813_2109___fuseiter_7814_2110___fuseiter_7815_2111 = 0UL; fused_0fused_0fused_0_fuseiter_7812___fuseiter_7813_2109___fuseiter_7814_2110___fuseiter_7815_2111 < 4UL; fused_0fused_0fused_0_fuseiter_7812___fuseiter_7813_2109___fuseiter_7814_2110___fuseiter_7815_2111 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7812___fuseiter_7813_2109___fuseiter_7814_2110___fuseiter_7815_2111 / 4UL) * 64UL) + ((fused_0fused_0fused_0_fuseiter_7812___fuseiter_7813_2109___fuseiter_7814_2110___fuseiter_7815_2111 % 4UL) * 16UL))]);
    vec_f32x16 __cached_1;
    __cached_1 = __cached_0;
    vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7812___fuseiter_7813_2109___fuseiter_7814_2110___fuseiter_7815_2111 / 4UL) * 64UL) + ((fused_0fused_0fused_0_fuseiter_7812___fuseiter_7813_2109___fuseiter_7814_2110___fuseiter_7815_2111 % 4UL) * 16UL))]);
  }
  return true;
}

static bool reorder__438(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7817___fuseiter_7818_2112___fuseiter_7819_2113___fuseiter_7820_2114 = 0UL; fused_0fused_0fused_0_fuseiter_7817___fuseiter_7818_2112___fuseiter_7819_2113___fuseiter_7820_2114 < 2UL; fused_0fused_0fused_0_fuseiter_7817___fuseiter_7818_2112___fuseiter_7819_2113___fuseiter_7820_2114 += 1UL) {
    for (uint64_t _fuseiter_7821 = 0UL; _fuseiter_7821 < 32UL; _fuseiter_7821 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7817___fuseiter_7818_2112___fuseiter_7819_2113___fuseiter_7820_2114 / 2UL) * 64UL) + (_fuseiter_7821 + ((fused_0fused_0fused_0_fuseiter_7817___fuseiter_7818_2112___fuseiter_7819_2113___fuseiter_7820_2114 % 2UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7817___fuseiter_7818_2112___fuseiter_7819_2113___fuseiter_7820_2114 / 2UL) * 64UL) + (((fused_0fused_0fused_0_fuseiter_7817___fuseiter_7818_2112___fuseiter_7819_2113___fuseiter_7820_2114 % 2UL) * 32UL) + _fuseiter_7821))]);
    }
  }
  return true;
}

static bool reorder__441(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7822___fuseiter_7823_2115___fuseiter_7824_2116___fuseiter_7825_2117 = 0UL; fused_0fused_0fused_0_fuseiter_7822___fuseiter_7823_2115___fuseiter_7824_2116___fuseiter_7825_2117 < 2UL; fused_0fused_0fused_0_fuseiter_7822___fuseiter_7823_2115___fuseiter_7824_2116___fuseiter_7825_2117 += 1UL) {
    for (uint64_t _fuseiter_7826 = 0UL; _fuseiter_7826 < 32UL; _fuseiter_7826 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7822___fuseiter_7823_2115___fuseiter_7824_2116___fuseiter_7825_2117 / 2UL) * 64UL) + (_fuseiter_7826 + ((fused_0fused_0fused_0_fuseiter_7822___fuseiter_7823_2115___fuseiter_7824_2116___fuseiter_7825_2117 % 2UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7822___fuseiter_7823_2115___fuseiter_7824_2116___fuseiter_7825_2117 / 2UL) * 64UL) + (((fused_0fused_0fused_0_fuseiter_7822___fuseiter_7823_2115___fuseiter_7824_2116___fuseiter_7825_2117 % 2UL) * 32UL) + _fuseiter_7826))]);
    }
  }
  return true;
}

static bool reorder__450(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7827___fuseiter_7828_2118___fuseiter_7829_2119___fuseiter_7830_2120 = 0UL; fused_0fused_0fused_0_fuseiter_7827___fuseiter_7828_2118___fuseiter_7829_2119___fuseiter_7830_2120 < 2UL; fused_0fused_0fused_0_fuseiter_7827___fuseiter_7828_2118___fuseiter_7829_2119___fuseiter_7830_2120 += 1UL) {
    for (uint64_t _fuseiter_7831 = 0UL; _fuseiter_7831 < 64UL; _fuseiter_7831 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7827___fuseiter_7828_2118___fuseiter_7829_2119___fuseiter_7830_2120 / 2UL) * 128UL) + (_fuseiter_7831 + ((fused_0fused_0fused_0_fuseiter_7827___fuseiter_7828_2118___fuseiter_7829_2119___fuseiter_7830_2120 % 2UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7827___fuseiter_7828_2118___fuseiter_7829_2119___fuseiter_7830_2120 / 2UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_7827___fuseiter_7828_2118___fuseiter_7829_2119___fuseiter_7830_2120 % 2UL) * 64UL) + _fuseiter_7831))]);
    }
  }
  return true;
}

static bool reorder__453(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7832___fuseiter_7833_2121___fuseiter_7834_2122___fuseiter_7835_2123 = 0UL; fused_0fused_0fused_0_fuseiter_7832___fuseiter_7833_2121___fuseiter_7834_2122___fuseiter_7835_2123 < 4UL; fused_0fused_0fused_0_fuseiter_7832___fuseiter_7833_2121___fuseiter_7834_2122___fuseiter_7835_2123 += 1UL) {
    for (uint64_t _fuseiter_7836 = 0UL; _fuseiter_7836 < 32UL; _fuseiter_7836 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7832___fuseiter_7833_2121___fuseiter_7834_2122___fuseiter_7835_2123 / 4UL) * 128UL) + (_fuseiter_7836 + ((fused_0fused_0fused_0_fuseiter_7832___fuseiter_7833_2121___fuseiter_7834_2122___fuseiter_7835_2123 % 4UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7832___fuseiter_7833_2121___fuseiter_7834_2122___fuseiter_7835_2123 / 4UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_7832___fuseiter_7833_2121___fuseiter_7834_2122___fuseiter_7835_2123 % 4UL) * 32UL) + _fuseiter_7836))]);
    }
  }
  return true;
}

static bool reorder__459(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7837___fuseiter_7838_2124___fuseiter_7839_2125___fuseiter_7840_2126 = 0UL; fused_0fused_0fused_0_fuseiter_7837___fuseiter_7838_2124___fuseiter_7839_2125___fuseiter_7840_2126 < 4UL; fused_0fused_0fused_0_fuseiter_7837___fuseiter_7838_2124___fuseiter_7839_2125___fuseiter_7840_2126 += 1UL) {
    for (uint64_t _fuseiter_7841 = 0UL; _fuseiter_7841 < 32UL; _fuseiter_7841 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7837___fuseiter_7838_2124___fuseiter_7839_2125___fuseiter_7840_2126 / 4UL) * 128UL) + (_fuseiter_7841 + ((fused_0fused_0fused_0_fuseiter_7837___fuseiter_7838_2124___fuseiter_7839_2125___fuseiter_7840_2126 % 4UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7837___fuseiter_7838_2124___fuseiter_7839_2125___fuseiter_7840_2126 / 4UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_7837___fuseiter_7838_2124___fuseiter_7839_2125___fuseiter_7840_2126 % 4UL) * 32UL) + _fuseiter_7841))]);
    }
  }
  return true;
}

static bool reorder__467(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7842___fuseiter_7843_2127___fuseiter_7844_2128___fuseiter_7845_2129 = 0UL; fused_0fused_0fused_0_fuseiter_7842___fuseiter_7843_2127___fuseiter_7844_2128___fuseiter_7845_2129 < 4UL; fused_0fused_0fused_0_fuseiter_7842___fuseiter_7843_2127___fuseiter_7844_2128___fuseiter_7845_2129 += 1UL) {
    for (uint64_t _fuseiter_7846 = 0UL; _fuseiter_7846 < 32UL; _fuseiter_7846 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7842___fuseiter_7843_2127___fuseiter_7844_2128___fuseiter_7845_2129 / 4UL) * 128UL) + (_fuseiter_7846 + ((fused_0fused_0fused_0_fuseiter_7842___fuseiter_7843_2127___fuseiter_7844_2128___fuseiter_7845_2129 % 4UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7842___fuseiter_7843_2127___fuseiter_7844_2128___fuseiter_7845_2129 / 4UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_7842___fuseiter_7843_2127___fuseiter_7844_2128___fuseiter_7845_2129 % 4UL) * 32UL) + _fuseiter_7846))]);
    }
  }
  return true;
}

static bool reorder__473(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7847___fuseiter_7848_2130___fuseiter_7849_2131___fuseiter_7850_2132 = 0UL; fused_0fused_0fused_0_fuseiter_7847___fuseiter_7848_2130___fuseiter_7849_2131___fuseiter_7850_2132 < 2UL; fused_0fused_0fused_0_fuseiter_7847___fuseiter_7848_2130___fuseiter_7849_2131___fuseiter_7850_2132 += 1UL) {
    for (uint64_t _fuseiter_7851 = 0UL; _fuseiter_7851 < 64UL; _fuseiter_7851 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7847___fuseiter_7848_2130___fuseiter_7849_2131___fuseiter_7850_2132 / 2UL) * 128UL) + (_fuseiter_7851 + ((fused_0fused_0fused_0_fuseiter_7847___fuseiter_7848_2130___fuseiter_7849_2131___fuseiter_7850_2132 % 2UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7847___fuseiter_7848_2130___fuseiter_7849_2131___fuseiter_7850_2132 / 2UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_7847___fuseiter_7848_2130___fuseiter_7849_2131___fuseiter_7850_2132 % 2UL) * 64UL) + _fuseiter_7851))]);
    }
  }
  return true;
}

static bool reorder__476(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7852___fuseiter_7853_2133___fuseiter_7854_2134___fuseiter_7855_2135 = 0UL; fused_0fused_0fused_0_fuseiter_7852___fuseiter_7853_2133___fuseiter_7854_2134___fuseiter_7855_2135 < 4UL; fused_0fused_0fused_0_fuseiter_7852___fuseiter_7853_2133___fuseiter_7854_2134___fuseiter_7855_2135 += 1UL) {
    for (uint64_t _fuseiter_7856 = 0UL; _fuseiter_7856 < 32UL; _fuseiter_7856 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7852___fuseiter_7853_2133___fuseiter_7854_2134___fuseiter_7855_2135 / 4UL) * 128UL) + (_fuseiter_7856 + ((fused_0fused_0fused_0_fuseiter_7852___fuseiter_7853_2133___fuseiter_7854_2134___fuseiter_7855_2135 % 4UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7852___fuseiter_7853_2133___fuseiter_7854_2134___fuseiter_7855_2135 / 4UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_7852___fuseiter_7853_2133___fuseiter_7854_2134___fuseiter_7855_2135 % 4UL) * 32UL) + _fuseiter_7856))]);
    }
  }
  return true;
}

static bool reorder__421(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_7857___fuseiter_7858_2136 = 0UL; fused_0_fuseiter_7857___fuseiter_7858_2136 < 16UL; fused_0_fuseiter_7857___fuseiter_7858_2136 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0_fuseiter_7857___fuseiter_7858_2136 / 16UL) * 256UL) + ((fused_0_fuseiter_7857___fuseiter_7858_2136 % 16UL) * 16UL))]);
    vec_f32x16 __cached_1;
    __cached_1 = __cached_0;
    vec_f32x16::store(__cached_1, &__outs_0[(((fused_0_fuseiter_7857___fuseiter_7858_2136 / 16UL) * 256UL) + ((fused_0_fuseiter_7857___fuseiter_7858_2136 % 16UL) * 16UL))]);
  }
  return true;
}

static bool reorder__428(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7862___fuseiter_7863_2137___fuseiter_7864_2138___fuseiter_7865_2139 = 0UL; fused_0fused_0fused_0_fuseiter_7862___fuseiter_7863_2137___fuseiter_7864_2138___fuseiter_7865_2139 < 4UL; fused_0fused_0fused_0_fuseiter_7862___fuseiter_7863_2137___fuseiter_7864_2138___fuseiter_7865_2139 += 1UL) {
    for (uint64_t _fuseiter_7866 = 0UL; _fuseiter_7866 < 64UL; _fuseiter_7866 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7862___fuseiter_7863_2137___fuseiter_7864_2138___fuseiter_7865_2139 / 4UL) * 256UL) + (_fuseiter_7866 + ((fused_0fused_0fused_0_fuseiter_7862___fuseiter_7863_2137___fuseiter_7864_2138___fuseiter_7865_2139 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7862___fuseiter_7863_2137___fuseiter_7864_2138___fuseiter_7865_2139 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7862___fuseiter_7863_2137___fuseiter_7864_2138___fuseiter_7865_2139 % 4UL) * 64UL) + _fuseiter_7866))]);
    }
  }
  return true;
}

static bool reorder__435(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7867___fuseiter_7868_2140___fuseiter_7869_2141___fuseiter_7870_2142 = 0UL; fused_0fused_0fused_0_fuseiter_7867___fuseiter_7868_2140___fuseiter_7869_2141___fuseiter_7870_2142 < 8UL; fused_0fused_0fused_0_fuseiter_7867___fuseiter_7868_2140___fuseiter_7869_2141___fuseiter_7870_2142 += 1UL) {
    for (uint64_t _fuseiter_7871 = 0UL; _fuseiter_7871 < 32UL; _fuseiter_7871 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7867___fuseiter_7868_2140___fuseiter_7869_2141___fuseiter_7870_2142 / 8UL) * 256UL) + (_fuseiter_7871 + ((fused_0fused_0fused_0_fuseiter_7867___fuseiter_7868_2140___fuseiter_7869_2141___fuseiter_7870_2142 % 8UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7867___fuseiter_7868_2140___fuseiter_7869_2141___fuseiter_7870_2142 / 8UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7867___fuseiter_7868_2140___fuseiter_7869_2141___fuseiter_7870_2142 % 8UL) * 32UL) + _fuseiter_7871))]);
    }
  }
  return true;
}

static bool reorder__444(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7872___fuseiter_7873_2143___fuseiter_7874_2144___fuseiter_7875_2145 = 0UL; fused_0fused_0fused_0_fuseiter_7872___fuseiter_7873_2143___fuseiter_7874_2144___fuseiter_7875_2145 < 4UL; fused_0fused_0fused_0_fuseiter_7872___fuseiter_7873_2143___fuseiter_7874_2144___fuseiter_7875_2145 += 1UL) {
    for (uint64_t _fuseiter_7876 = 0UL; _fuseiter_7876 < 64UL; _fuseiter_7876 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7872___fuseiter_7873_2143___fuseiter_7874_2144___fuseiter_7875_2145 / 4UL) * 256UL) + (_fuseiter_7876 + ((fused_0fused_0fused_0_fuseiter_7872___fuseiter_7873_2143___fuseiter_7874_2144___fuseiter_7875_2145 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7872___fuseiter_7873_2143___fuseiter_7874_2144___fuseiter_7875_2145 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7872___fuseiter_7873_2143___fuseiter_7874_2144___fuseiter_7875_2145 % 4UL) * 64UL) + _fuseiter_7876))]);
    }
  }
  return true;
}

static bool reorder__486(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7877___fuseiter_7878_2146___fuseiter_7879_2147___fuseiter_7880_2148 = 0UL; fused_0fused_0fused_0_fuseiter_7877___fuseiter_7878_2146___fuseiter_7879_2147___fuseiter_7880_2148 < 2UL; fused_0fused_0fused_0_fuseiter_7877___fuseiter_7878_2146___fuseiter_7879_2147___fuseiter_7880_2148 += 1UL) {
    for (uint64_t _fuseiter_7881 = 0UL; _fuseiter_7881 < 128UL; _fuseiter_7881 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7877___fuseiter_7878_2146___fuseiter_7879_2147___fuseiter_7880_2148 / 2UL) * 256UL) + (_fuseiter_7881 + ((fused_0fused_0fused_0_fuseiter_7877___fuseiter_7878_2146___fuseiter_7879_2147___fuseiter_7880_2148 % 2UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7877___fuseiter_7878_2146___fuseiter_7879_2147___fuseiter_7880_2148 / 2UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7877___fuseiter_7878_2146___fuseiter_7879_2147___fuseiter_7880_2148 % 2UL) * 128UL) + _fuseiter_7881))]);
    }
  }
  return true;
}

static bool reorder__492(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7882___fuseiter_7883_2149___fuseiter_7884_2150___fuseiter_7885_2151 = 0UL; fused_0fused_0fused_0_fuseiter_7882___fuseiter_7883_2149___fuseiter_7884_2150___fuseiter_7885_2151 < 4UL; fused_0fused_0fused_0_fuseiter_7882___fuseiter_7883_2149___fuseiter_7884_2150___fuseiter_7885_2151 += 1UL) {
    for (uint64_t _fuseiter_7886 = 0UL; _fuseiter_7886 < 64UL; _fuseiter_7886 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7882___fuseiter_7883_2149___fuseiter_7884_2150___fuseiter_7885_2151 / 4UL) * 256UL) + (_fuseiter_7886 + ((fused_0fused_0fused_0_fuseiter_7882___fuseiter_7883_2149___fuseiter_7884_2150___fuseiter_7885_2151 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7882___fuseiter_7883_2149___fuseiter_7884_2150___fuseiter_7885_2151 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7882___fuseiter_7883_2149___fuseiter_7884_2150___fuseiter_7885_2151 % 4UL) * 64UL) + _fuseiter_7886))]);
    }
  }
  return true;
}

static bool reorder__495(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_7887___fuseiter_7888_2152 = 0UL; fused_0_fuseiter_7887___fuseiter_7888_2152 < 16UL; fused_0_fuseiter_7887___fuseiter_7888_2152 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0_fuseiter_7887___fuseiter_7888_2152 / 16UL) * 256UL) + ((fused_0_fuseiter_7887___fuseiter_7888_2152 % 16UL) * 16UL))]);
    vec_f32x16 __cached_1;
    __cached_1 = __cached_0;
    vec_f32x16::store(__cached_1, &__outs_0[(((fused_0_fuseiter_7887___fuseiter_7888_2152 / 16UL) * 256UL) + ((fused_0_fuseiter_7887___fuseiter_7888_2152 % 16UL) * 16UL))]);
  }
  return true;
}

static bool reorder__499(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7892___fuseiter_7893_2153___fuseiter_7894_2154___fuseiter_7895_2155 = 0UL; fused_0fused_0fused_0_fuseiter_7892___fuseiter_7893_2153___fuseiter_7894_2154___fuseiter_7895_2155 < 8UL; fused_0fused_0fused_0_fuseiter_7892___fuseiter_7893_2153___fuseiter_7894_2154___fuseiter_7895_2155 += 1UL) {
    for (uint64_t _fuseiter_7896 = 0UL; _fuseiter_7896 < 32UL; _fuseiter_7896 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7892___fuseiter_7893_2153___fuseiter_7894_2154___fuseiter_7895_2155 / 8UL) * 256UL) + (_fuseiter_7896 + ((fused_0fused_0fused_0_fuseiter_7892___fuseiter_7893_2153___fuseiter_7894_2154___fuseiter_7895_2155 % 8UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7892___fuseiter_7893_2153___fuseiter_7894_2154___fuseiter_7895_2155 / 8UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7892___fuseiter_7893_2153___fuseiter_7894_2154___fuseiter_7895_2155 % 8UL) * 32UL) + _fuseiter_7896))]);
    }
  }
  return true;
}

static bool reorder__502(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7897___fuseiter_7898_2156___fuseiter_7899_2157___fuseiter_7900_2158 = 0UL; fused_0fused_0fused_0_fuseiter_7897___fuseiter_7898_2156___fuseiter_7899_2157___fuseiter_7900_2158 < 8UL; fused_0fused_0fused_0_fuseiter_7897___fuseiter_7898_2156___fuseiter_7899_2157___fuseiter_7900_2158 += 1UL) {
    for (uint64_t _fuseiter_7901 = 0UL; _fuseiter_7901 < 32UL; _fuseiter_7901 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7897___fuseiter_7898_2156___fuseiter_7899_2157___fuseiter_7900_2158 / 8UL) * 256UL) + (_fuseiter_7901 + ((fused_0fused_0fused_0_fuseiter_7897___fuseiter_7898_2156___fuseiter_7899_2157___fuseiter_7900_2158 % 8UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7897___fuseiter_7898_2156___fuseiter_7899_2157___fuseiter_7900_2158 / 8UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7897___fuseiter_7898_2156___fuseiter_7899_2157___fuseiter_7900_2158 % 8UL) * 32UL) + _fuseiter_7901))]);
    }
  }
  return true;
}

static bool reorder__508(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7902___fuseiter_7903_2159___fuseiter_7904_2160___fuseiter_7905_2161 = 0UL; fused_0fused_0fused_0_fuseiter_7902___fuseiter_7903_2159___fuseiter_7904_2160___fuseiter_7905_2161 < 4UL; fused_0fused_0fused_0_fuseiter_7902___fuseiter_7903_2159___fuseiter_7904_2160___fuseiter_7905_2161 += 1UL) {
    for (uint64_t _fuseiter_7906 = 0UL; _fuseiter_7906 < 64UL; _fuseiter_7906 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7902___fuseiter_7903_2159___fuseiter_7904_2160___fuseiter_7905_2161 / 4UL) * 256UL) + (_fuseiter_7906 + ((fused_0fused_0fused_0_fuseiter_7902___fuseiter_7903_2159___fuseiter_7904_2160___fuseiter_7905_2161 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7902___fuseiter_7903_2159___fuseiter_7904_2160___fuseiter_7905_2161 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7902___fuseiter_7903_2159___fuseiter_7904_2160___fuseiter_7905_2161 % 4UL) * 64UL) + _fuseiter_7906))]);
    }
  }
  return true;
}

static bool reorder__511(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7907___fuseiter_7908_2162___fuseiter_7909_2163___fuseiter_7910_2164 = 0UL; fused_0fused_0fused_0_fuseiter_7907___fuseiter_7908_2162___fuseiter_7909_2163___fuseiter_7910_2164 < 4UL; fused_0fused_0fused_0_fuseiter_7907___fuseiter_7908_2162___fuseiter_7909_2163___fuseiter_7910_2164 += 1UL) {
    for (uint64_t _fuseiter_7911 = 0UL; _fuseiter_7911 < 64UL; _fuseiter_7911 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7907___fuseiter_7908_2162___fuseiter_7909_2163___fuseiter_7910_2164 / 4UL) * 256UL) + (_fuseiter_7911 + ((fused_0fused_0fused_0_fuseiter_7907___fuseiter_7908_2162___fuseiter_7909_2163___fuseiter_7910_2164 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7907___fuseiter_7908_2162___fuseiter_7909_2163___fuseiter_7910_2164 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7907___fuseiter_7908_2162___fuseiter_7909_2163___fuseiter_7910_2164 % 4UL) * 64UL) + _fuseiter_7911))]);
    }
  }
  return true;
}

static bool reorder__518(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7912___fuseiter_7913_2165___fuseiter_7914_2166___fuseiter_7915_2167 = 0UL; fused_0fused_0fused_0_fuseiter_7912___fuseiter_7913_2165___fuseiter_7914_2166___fuseiter_7915_2167 < 8UL; fused_0fused_0fused_0_fuseiter_7912___fuseiter_7913_2165___fuseiter_7914_2166___fuseiter_7915_2167 += 1UL) {
    for (uint64_t _fuseiter_7916 = 0UL; _fuseiter_7916 < 32UL; _fuseiter_7916 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7912___fuseiter_7913_2165___fuseiter_7914_2166___fuseiter_7915_2167 / 8UL) * 256UL) + (_fuseiter_7916 + ((fused_0fused_0fused_0_fuseiter_7912___fuseiter_7913_2165___fuseiter_7914_2166___fuseiter_7915_2167 % 8UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7912___fuseiter_7913_2165___fuseiter_7914_2166___fuseiter_7915_2167 / 8UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7912___fuseiter_7913_2165___fuseiter_7914_2166___fuseiter_7915_2167 % 8UL) * 32UL) + _fuseiter_7916))]);
    }
  }
  return true;
}

static bool reorder__524(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7917___fuseiter_7918_2168___fuseiter_7919_2169___fuseiter_7920_2170 = 0UL; fused_0fused_0fused_0_fuseiter_7917___fuseiter_7918_2168___fuseiter_7919_2169___fuseiter_7920_2170 < 8UL; fused_0fused_0fused_0_fuseiter_7917___fuseiter_7918_2168___fuseiter_7919_2169___fuseiter_7920_2170 += 1UL) {
    for (uint64_t _fuseiter_7921 = 0UL; _fuseiter_7921 < 32UL; _fuseiter_7921 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7917___fuseiter_7918_2168___fuseiter_7919_2169___fuseiter_7920_2170 / 8UL) * 256UL) + (_fuseiter_7921 + ((fused_0fused_0fused_0_fuseiter_7917___fuseiter_7918_2168___fuseiter_7919_2169___fuseiter_7920_2170 % 8UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7917___fuseiter_7918_2168___fuseiter_7919_2169___fuseiter_7920_2170 / 8UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7917___fuseiter_7918_2168___fuseiter_7919_2169___fuseiter_7920_2170 % 8UL) * 32UL) + _fuseiter_7921))]);
    }
  }
  return true;
}

static bool reorder__527(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7922___fuseiter_7923_2171___fuseiter_7924_2172___fuseiter_7925_2173 = 0UL; fused_0fused_0fused_0_fuseiter_7922___fuseiter_7923_2171___fuseiter_7924_2172___fuseiter_7925_2173 < 4UL; fused_0fused_0fused_0_fuseiter_7922___fuseiter_7923_2171___fuseiter_7924_2172___fuseiter_7925_2173 += 1UL) {
    for (uint64_t _fuseiter_7926 = 0UL; _fuseiter_7926 < 64UL; _fuseiter_7926 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7922___fuseiter_7923_2171___fuseiter_7924_2172___fuseiter_7925_2173 / 4UL) * 256UL) + (_fuseiter_7926 + ((fused_0fused_0fused_0_fuseiter_7922___fuseiter_7923_2171___fuseiter_7924_2172___fuseiter_7925_2173 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7922___fuseiter_7923_2171___fuseiter_7924_2172___fuseiter_7925_2173 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_7922___fuseiter_7923_2171___fuseiter_7924_2172___fuseiter_7925_2173 % 4UL) * 64UL) + _fuseiter_7926))]);
    }
  }
  return true;
}

static bool reorder__447(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_7927___fuseiter_7928_2174 = 0UL; fused_0_fuseiter_7927___fuseiter_7928_2174 < 16UL; fused_0_fuseiter_7927___fuseiter_7928_2174 += 1UL) {
    for (uint64_t _fuseiter_7931 = 0UL; _fuseiter_7931 < 32UL; _fuseiter_7931 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0_fuseiter_7927___fuseiter_7928_2174 / 16UL) * 512UL) + (_fuseiter_7931 + ((fused_0_fuseiter_7927___fuseiter_7928_2174 % 16UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0_fuseiter_7927___fuseiter_7928_2174 / 16UL) * 512UL) + (((fused_0_fuseiter_7927___fuseiter_7928_2174 % 16UL) * 32UL) + _fuseiter_7931))]);
    }
  }
  return true;
}

static bool reorder__456(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_7932___fuseiter_7933_2175 = 0UL; fused_0_fuseiter_7932___fuseiter_7933_2175 < 16UL; fused_0_fuseiter_7932___fuseiter_7933_2175 += 1UL) {
    for (uint64_t _fuseiter_7936 = 0UL; _fuseiter_7936 < 32UL; _fuseiter_7936 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0_fuseiter_7932___fuseiter_7933_2175 / 16UL) * 512UL) + (_fuseiter_7936 + ((fused_0_fuseiter_7932___fuseiter_7933_2175 % 16UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0_fuseiter_7932___fuseiter_7933_2175 / 16UL) * 512UL) + (((fused_0_fuseiter_7932___fuseiter_7933_2175 % 16UL) * 32UL) + _fuseiter_7936))]);
    }
  }
  return true;
}

static bool reorder__463(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7937___fuseiter_7938_2176___fuseiter_7939_2177___fuseiter_7940_2178 = 0UL; fused_0fused_0fused_0_fuseiter_7937___fuseiter_7938_2176___fuseiter_7939_2177___fuseiter_7940_2178 < 4UL; fused_0fused_0fused_0_fuseiter_7937___fuseiter_7938_2176___fuseiter_7939_2177___fuseiter_7940_2178 += 1UL) {
    for (uint64_t _fuseiter_7941 = 0UL; _fuseiter_7941 < 128UL; _fuseiter_7941 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7937___fuseiter_7938_2176___fuseiter_7939_2177___fuseiter_7940_2178 / 4UL) * 512UL) + (_fuseiter_7941 + ((fused_0fused_0fused_0_fuseiter_7937___fuseiter_7938_2176___fuseiter_7939_2177___fuseiter_7940_2178 % 4UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7937___fuseiter_7938_2176___fuseiter_7939_2177___fuseiter_7940_2178 / 4UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_7937___fuseiter_7938_2176___fuseiter_7939_2177___fuseiter_7940_2178 % 4UL) * 128UL) + _fuseiter_7941))]);
    }
  }
  return true;
}

static bool reorder__470(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7942___fuseiter_7943_2179___fuseiter_7944_2180___fuseiter_7945_2181 = 0UL; fused_0fused_0fused_0_fuseiter_7942___fuseiter_7943_2179___fuseiter_7944_2180___fuseiter_7945_2181 < 8UL; fused_0fused_0fused_0_fuseiter_7942___fuseiter_7943_2179___fuseiter_7944_2180___fuseiter_7945_2181 += 1UL) {
    for (uint64_t _fuseiter_7946 = 0UL; _fuseiter_7946 < 64UL; _fuseiter_7946 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7942___fuseiter_7943_2179___fuseiter_7944_2180___fuseiter_7945_2181 / 8UL) * 512UL) + (_fuseiter_7946 + ((fused_0fused_0fused_0_fuseiter_7942___fuseiter_7943_2179___fuseiter_7944_2180___fuseiter_7945_2181 % 8UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7942___fuseiter_7943_2179___fuseiter_7944_2180___fuseiter_7945_2181 / 8UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_7942___fuseiter_7943_2179___fuseiter_7944_2180___fuseiter_7945_2181 % 8UL) * 64UL) + _fuseiter_7946))]);
    }
  }
  return true;
}

static bool reorder__479(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7947___fuseiter_7948_2182___fuseiter_7949_2183___fuseiter_7950_2184 = 0UL; fused_0fused_0fused_0_fuseiter_7947___fuseiter_7948_2182___fuseiter_7949_2183___fuseiter_7950_2184 < 4UL; fused_0fused_0fused_0_fuseiter_7947___fuseiter_7948_2182___fuseiter_7949_2183___fuseiter_7950_2184 += 1UL) {
    for (uint64_t _fuseiter_7951 = 0UL; _fuseiter_7951 < 128UL; _fuseiter_7951 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7947___fuseiter_7948_2182___fuseiter_7949_2183___fuseiter_7950_2184 / 4UL) * 512UL) + (_fuseiter_7951 + ((fused_0fused_0fused_0_fuseiter_7947___fuseiter_7948_2182___fuseiter_7949_2183___fuseiter_7950_2184 % 4UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7947___fuseiter_7948_2182___fuseiter_7949_2183___fuseiter_7950_2184 / 4UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_7947___fuseiter_7948_2182___fuseiter_7949_2183___fuseiter_7950_2184 % 4UL) * 128UL) + _fuseiter_7951))]);
    }
  }
  return true;
}

static bool reorder__536(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7952___fuseiter_7953_2185___fuseiter_7954_2186___fuseiter_7955_2187 = 0UL; fused_0fused_0fused_0_fuseiter_7952___fuseiter_7953_2185___fuseiter_7954_2186___fuseiter_7955_2187 < 8UL; fused_0fused_0fused_0_fuseiter_7952___fuseiter_7953_2185___fuseiter_7954_2186___fuseiter_7955_2187 += 1UL) {
    for (uint64_t _fuseiter_7956 = 0UL; _fuseiter_7956 < 64UL; _fuseiter_7956 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7952___fuseiter_7953_2185___fuseiter_7954_2186___fuseiter_7955_2187 / 8UL) * 512UL) + (_fuseiter_7956 + ((fused_0fused_0fused_0_fuseiter_7952___fuseiter_7953_2185___fuseiter_7954_2186___fuseiter_7955_2187 % 8UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7952___fuseiter_7953_2185___fuseiter_7954_2186___fuseiter_7955_2187 / 8UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_7952___fuseiter_7953_2185___fuseiter_7954_2186___fuseiter_7955_2187 % 8UL) * 64UL) + _fuseiter_7956))]);
    }
  }
  return true;
}

static bool reorder__539(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7957___fuseiter_7958_2188___fuseiter_7959_2189___fuseiter_7960_2190 = 0UL; fused_0fused_0fused_0_fuseiter_7957___fuseiter_7958_2188___fuseiter_7959_2189___fuseiter_7960_2190 < 4UL; fused_0fused_0fused_0_fuseiter_7957___fuseiter_7958_2188___fuseiter_7959_2189___fuseiter_7960_2190 += 1UL) {
    for (uint64_t _fuseiter_7961 = 0UL; _fuseiter_7961 < 128UL; _fuseiter_7961 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7957___fuseiter_7958_2188___fuseiter_7959_2189___fuseiter_7960_2190 / 4UL) * 512UL) + (_fuseiter_7961 + ((fused_0fused_0fused_0_fuseiter_7957___fuseiter_7958_2188___fuseiter_7959_2189___fuseiter_7960_2190 % 4UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7957___fuseiter_7958_2188___fuseiter_7959_2189___fuseiter_7960_2190 / 4UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_7957___fuseiter_7958_2188___fuseiter_7959_2189___fuseiter_7960_2190 % 4UL) * 128UL) + _fuseiter_7961))]);
    }
  }
  return true;
}

static bool reorder__545(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7962___fuseiter_7963_2191___fuseiter_7964_2192___fuseiter_7965_2193 = 0UL; fused_0fused_0fused_0_fuseiter_7962___fuseiter_7963_2191___fuseiter_7964_2192___fuseiter_7965_2193 < 4UL; fused_0fused_0fused_0_fuseiter_7962___fuseiter_7963_2191___fuseiter_7964_2192___fuseiter_7965_2193 += 1UL) {
    for (uint64_t _fuseiter_7966 = 0UL; _fuseiter_7966 < 128UL; _fuseiter_7966 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7962___fuseiter_7963_2191___fuseiter_7964_2192___fuseiter_7965_2193 / 4UL) * 512UL) + (_fuseiter_7966 + ((fused_0fused_0fused_0_fuseiter_7962___fuseiter_7963_2191___fuseiter_7964_2192___fuseiter_7965_2193 % 4UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7962___fuseiter_7963_2191___fuseiter_7964_2192___fuseiter_7965_2193 / 4UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_7962___fuseiter_7963_2191___fuseiter_7964_2192___fuseiter_7965_2193 % 4UL) * 128UL) + _fuseiter_7966))]);
    }
  }
  return true;
}

static bool reorder__548(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_7967___fuseiter_7968_2194 = 0UL; fused_0_fuseiter_7967___fuseiter_7968_2194 < 32UL; fused_0_fuseiter_7967___fuseiter_7968_2194 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0_fuseiter_7967___fuseiter_7968_2194 / 32UL) * 512UL) + ((fused_0_fuseiter_7967___fuseiter_7968_2194 % 32UL) * 16UL))]);
    vec_f32x16 __cached_1;
    __cached_1 = __cached_0;
    vec_f32x16::store(__cached_1, &__outs_0[(((fused_0_fuseiter_7967___fuseiter_7968_2194 / 32UL) * 512UL) + ((fused_0_fuseiter_7967___fuseiter_7968_2194 % 32UL) * 16UL))]);
  }
  return true;
}

static bool reorder__554(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_7972___fuseiter_7973_2195 = 0UL; fused_0_fuseiter_7972___fuseiter_7973_2195 < 16UL; fused_0_fuseiter_7972___fuseiter_7973_2195 += 1UL) {
    for (uint64_t _fuseiter_7976 = 0UL; _fuseiter_7976 < 32UL; _fuseiter_7976 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0_fuseiter_7972___fuseiter_7973_2195 / 16UL) * 512UL) + (_fuseiter_7976 + ((fused_0_fuseiter_7972___fuseiter_7973_2195 % 16UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0_fuseiter_7972___fuseiter_7973_2195 / 16UL) * 512UL) + (((fused_0_fuseiter_7972___fuseiter_7973_2195 % 16UL) * 32UL) + _fuseiter_7976))]);
    }
  }
  return true;
}

static bool reorder__557(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7977___fuseiter_7978_2196___fuseiter_7979_2197___fuseiter_7980_2198 = 0UL; fused_0fused_0fused_0_fuseiter_7977___fuseiter_7978_2196___fuseiter_7979_2197___fuseiter_7980_2198 < 4UL; fused_0fused_0fused_0_fuseiter_7977___fuseiter_7978_2196___fuseiter_7979_2197___fuseiter_7980_2198 += 1UL) {
    for (uint64_t _fuseiter_7981 = 0UL; _fuseiter_7981 < 128UL; _fuseiter_7981 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7977___fuseiter_7978_2196___fuseiter_7979_2197___fuseiter_7980_2198 / 4UL) * 512UL) + (_fuseiter_7981 + ((fused_0fused_0fused_0_fuseiter_7977___fuseiter_7978_2196___fuseiter_7979_2197___fuseiter_7980_2198 % 4UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7977___fuseiter_7978_2196___fuseiter_7979_2197___fuseiter_7980_2198 / 4UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_7977___fuseiter_7978_2196___fuseiter_7979_2197___fuseiter_7980_2198 % 4UL) * 128UL) + _fuseiter_7981))]);
    }
  }
  return true;
}

static bool reorder__482(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_7982___fuseiter_7983_2199 = 0UL; fused_0_fuseiter_7982___fuseiter_7983_2199 < 32UL; fused_0_fuseiter_7982___fuseiter_7983_2199 += 1UL) {
    for (uint64_t _fuseiter_7986 = 0UL; _fuseiter_7986 < 32UL; _fuseiter_7986 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0_fuseiter_7982___fuseiter_7983_2199 / 32UL) * 1024UL) + (_fuseiter_7986 + ((fused_0_fuseiter_7982___fuseiter_7983_2199 % 32UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0_fuseiter_7982___fuseiter_7983_2199 / 32UL) * 1024UL) + (((fused_0_fuseiter_7982___fuseiter_7983_2199 % 32UL) * 32UL) + _fuseiter_7986))]);
    }
  }
  return true;
}

static bool reorder__489(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7987___fuseiter_7988_2200___fuseiter_7989_2201___fuseiter_7990_2202 = 0UL; fused_0fused_0fused_0_fuseiter_7987___fuseiter_7988_2200___fuseiter_7989_2201___fuseiter_7990_2202 < 8UL; fused_0fused_0fused_0_fuseiter_7987___fuseiter_7988_2200___fuseiter_7989_2201___fuseiter_7990_2202 += 1UL) {
    for (uint64_t _fuseiter_7991 = 0UL; _fuseiter_7991 < 128UL; _fuseiter_7991 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7987___fuseiter_7988_2200___fuseiter_7989_2201___fuseiter_7990_2202 / 8UL) * 1024UL) + (_fuseiter_7991 + ((fused_0fused_0fused_0_fuseiter_7987___fuseiter_7988_2200___fuseiter_7989_2201___fuseiter_7990_2202 % 8UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7987___fuseiter_7988_2200___fuseiter_7989_2201___fuseiter_7990_2202 / 8UL) * 1024UL) + (((fused_0fused_0fused_0_fuseiter_7987___fuseiter_7988_2200___fuseiter_7989_2201___fuseiter_7990_2202 % 8UL) * 128UL) + _fuseiter_7991))]);
    }
  }
  return true;
}

static bool reorder__505(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7992___fuseiter_7993_2203___fuseiter_7994_2204___fuseiter_7995_2205 = 0UL; fused_0fused_0fused_0_fuseiter_7992___fuseiter_7993_2203___fuseiter_7994_2204___fuseiter_7995_2205 < 8UL; fused_0fused_0fused_0_fuseiter_7992___fuseiter_7993_2203___fuseiter_7994_2204___fuseiter_7995_2205 += 1UL) {
    for (uint64_t _fuseiter_7996 = 0UL; _fuseiter_7996 < 128UL; _fuseiter_7996 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7992___fuseiter_7993_2203___fuseiter_7994_2204___fuseiter_7995_2205 / 8UL) * 1024UL) + (_fuseiter_7996 + ((fused_0fused_0fused_0_fuseiter_7992___fuseiter_7993_2203___fuseiter_7994_2204___fuseiter_7995_2205 % 8UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7992___fuseiter_7993_2203___fuseiter_7994_2204___fuseiter_7995_2205 / 8UL) * 1024UL) + (((fused_0fused_0fused_0_fuseiter_7992___fuseiter_7993_2203___fuseiter_7994_2204___fuseiter_7995_2205 % 8UL) * 128UL) + _fuseiter_7996))]);
    }
  }
  return true;
}

static bool reorder__514(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_7997___fuseiter_7998_2206___fuseiter_7999_2207___fuseiter_8000_2208 = 0UL; fused_0fused_0fused_0_fuseiter_7997___fuseiter_7998_2206___fuseiter_7999_2207___fuseiter_8000_2208 < 8UL; fused_0fused_0fused_0_fuseiter_7997___fuseiter_7998_2206___fuseiter_7999_2207___fuseiter_8000_2208 += 1UL) {
    for (uint64_t _fuseiter_8001 = 0UL; _fuseiter_8001 < 128UL; _fuseiter_8001 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_7997___fuseiter_7998_2206___fuseiter_7999_2207___fuseiter_8000_2208 / 8UL) * 1024UL) + (_fuseiter_8001 + ((fused_0fused_0fused_0_fuseiter_7997___fuseiter_7998_2206___fuseiter_7999_2207___fuseiter_8000_2208 % 8UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_7997___fuseiter_7998_2206___fuseiter_7999_2207___fuseiter_8000_2208 / 8UL) * 1024UL) + (((fused_0fused_0fused_0_fuseiter_7997___fuseiter_7998_2206___fuseiter_7999_2207___fuseiter_8000_2208 % 8UL) * 128UL) + _fuseiter_8001))]);
    }
  }
  return true;
}

static bool reorder__521(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_8002___fuseiter_8003_2209___fuseiter_8004_2210___fuseiter_8005_2211 = 0UL; fused_0fused_0fused_0_fuseiter_8002___fuseiter_8003_2209___fuseiter_8004_2210___fuseiter_8005_2211 < 8UL; fused_0fused_0fused_0_fuseiter_8002___fuseiter_8003_2209___fuseiter_8004_2210___fuseiter_8005_2211 += 1UL) {
    for (uint64_t _fuseiter_8006 = 0UL; _fuseiter_8006 < 128UL; _fuseiter_8006 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_8002___fuseiter_8003_2209___fuseiter_8004_2210___fuseiter_8005_2211 / 8UL) * 1024UL) + (_fuseiter_8006 + ((fused_0fused_0fused_0_fuseiter_8002___fuseiter_8003_2209___fuseiter_8004_2210___fuseiter_8005_2211 % 8UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_8002___fuseiter_8003_2209___fuseiter_8004_2210___fuseiter_8005_2211 / 8UL) * 1024UL) + (((fused_0fused_0fused_0_fuseiter_8002___fuseiter_8003_2209___fuseiter_8004_2210___fuseiter_8005_2211 % 8UL) * 128UL) + _fuseiter_8006))]);
    }
  }
  return true;
}

static bool reorder__530(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_8007___fuseiter_8008_2212___fuseiter_8009_2213___fuseiter_8010_2214 = 0UL; fused_0fused_0fused_0_fuseiter_8007___fuseiter_8008_2212___fuseiter_8009_2213___fuseiter_8010_2214 < 4UL; fused_0fused_0fused_0_fuseiter_8007___fuseiter_8008_2212___fuseiter_8009_2213___fuseiter_8010_2214 += 1UL) {
    for (uint64_t _fuseiter_8011 = 0UL; _fuseiter_8011 < 256UL; _fuseiter_8011 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_8007___fuseiter_8008_2212___fuseiter_8009_2213___fuseiter_8010_2214 / 4UL) * 1024UL) + (_fuseiter_8011 + ((fused_0fused_0fused_0_fuseiter_8007___fuseiter_8008_2212___fuseiter_8009_2213___fuseiter_8010_2214 % 4UL) * 256UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_8007___fuseiter_8008_2212___fuseiter_8009_2213___fuseiter_8010_2214 / 4UL) * 1024UL) + (((fused_0fused_0fused_0_fuseiter_8007___fuseiter_8008_2212___fuseiter_8009_2213___fuseiter_8010_2214 % 4UL) * 256UL) + _fuseiter_8011))]);
    }
  }
  return true;
}

static bool reorder__533(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_8012___fuseiter_8013_2215 = 0UL; fused_0_fuseiter_8012___fuseiter_8013_2215 < 128UL; fused_0_fuseiter_8012___fuseiter_8013_2215 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0_fuseiter_8012___fuseiter_8013_2215 / 128UL) * 2048UL) + ((fused_0_fuseiter_8012___fuseiter_8013_2215 % 128UL) * 16UL))]);
    vec_f32x16 __cached_1;
    __cached_1 = __cached_0;
    vec_f32x16::store(__cached_1, &__outs_0[(((fused_0_fuseiter_8012___fuseiter_8013_2215 / 128UL) * 2048UL) + ((fused_0_fuseiter_8012___fuseiter_8013_2215 % 128UL) * 16UL))]);
  }
  return true;
}

static bool reorder__542(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_8017___fuseiter_8018_2216 = 0UL; fused_0_fuseiter_8017___fuseiter_8018_2216 < 32UL; fused_0_fuseiter_8017___fuseiter_8018_2216 += 1UL) {
    for (uint64_t _fuseiter_8021 = 0UL; _fuseiter_8021 < 64UL; _fuseiter_8021 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0_fuseiter_8017___fuseiter_8018_2216 / 32UL) * 2048UL) + (_fuseiter_8021 + ((fused_0_fuseiter_8017___fuseiter_8018_2216 % 32UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0_fuseiter_8017___fuseiter_8018_2216 / 32UL) * 2048UL) + (((fused_0_fuseiter_8017___fuseiter_8018_2216 % 32UL) * 64UL) + _fuseiter_8021))]);
    }
  }
  return true;
}

static bool reorder__551(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_8022___fuseiter_8023_2217 = 0UL; fused_0_fuseiter_8022___fuseiter_8023_2217 < 32UL; fused_0_fuseiter_8022___fuseiter_8023_2217 += 1UL) {
    for (uint64_t _fuseiter_8026 = 0UL; _fuseiter_8026 < 64UL; _fuseiter_8026 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0_fuseiter_8022___fuseiter_8023_2217 / 32UL) * 2048UL) + (_fuseiter_8026 + ((fused_0_fuseiter_8022___fuseiter_8023_2217 % 32UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0_fuseiter_8022___fuseiter_8023_2217 / 32UL) * 2048UL) + (((fused_0_fuseiter_8022___fuseiter_8023_2217 % 32UL) * 64UL) + _fuseiter_8026))]);
    }
  }
  return true;
}

static bool reorder__560(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_8027___fuseiter_8028_2218___fuseiter_8029_2219___fuseiter_8030_2220 = 0UL; fused_0fused_0fused_0_fuseiter_8027___fuseiter_8028_2218___fuseiter_8029_2219___fuseiter_8030_2220 < 4UL; fused_0fused_0fused_0_fuseiter_8027___fuseiter_8028_2218___fuseiter_8029_2219___fuseiter_8030_2220 += 1UL) {
    for (uint64_t _fuseiter_8031 = 0UL; _fuseiter_8031 < 512UL; _fuseiter_8031 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_8027___fuseiter_8028_2218___fuseiter_8029_2219___fuseiter_8030_2220 / 4UL) * 2048UL) + (_fuseiter_8031 + ((fused_0fused_0fused_0_fuseiter_8027___fuseiter_8028_2218___fuseiter_8029_2219___fuseiter_8030_2220 % 4UL) * 512UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_8027___fuseiter_8028_2218___fuseiter_8029_2219___fuseiter_8030_2220 / 4UL) * 2048UL) + (((fused_0fused_0fused_0_fuseiter_8027___fuseiter_8028_2218___fuseiter_8029_2219___fuseiter_8030_2220 % 4UL) * 512UL) + _fuseiter_8031))]);
    }
  }
  return true;
}

static bool mul__111(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2221____itr_2_2222 = 0UL; fused_0fused_0__itr_0____itr_1_2221____itr_2_2222 < 4096UL; fused_0fused_0__itr_0____itr_1_2221____itr_2_2222 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2221____itr_2_2222 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_2221____itr_2_2222 % 64UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2221____itr_2_2222 / 64UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2221____itr_2_2222 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_2221____itr_2_2222 % 64UL))] = __cached_2;
  }
  return true;
}

static bool cast__112(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2223____itr_2_2224 = 0UL; fused_0fused_0__itr_0____itr_1_2223____itr_2_2224 < 4096UL; fused_0fused_0__itr_0____itr_1_2223____itr_2_2224 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2223____itr_2_2224 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_2223____itr_2_2224 % 64UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2223____itr_2_2224 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_2223____itr_2_2224 % 64UL))] = __cached_1;
  }
  return true;
}

static bool mul__108(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2225____itr_2_2226 = 0UL; fused_0fused_0__itr_0____itr_1_2225____itr_2_2226 < 16384UL; fused_0fused_0__itr_0____itr_1_2225____itr_2_2226 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2225____itr_2_2226 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_2225____itr_2_2226 % 64UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2225____itr_2_2226 / 64UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2225____itr_2_2226 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_2225____itr_2_2226 % 64UL))] = __cached_2;
  }
  return true;
}

static bool cast__109(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2227____itr_2_2228 = 0UL; fused_0fused_0__itr_0____itr_1_2227____itr_2_2228 < 16384UL; fused_0fused_0__itr_0____itr_1_2227____itr_2_2228 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2227____itr_2_2228 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_2227____itr_2_2228 % 64UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2227____itr_2_2228 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_2227____itr_2_2228 % 64UL))] = __cached_1;
  }
  return true;
}

static bool mul__117(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2229____itr_2_2230 = 0UL; fused_0fused_0__itr_0____itr_1_2229____itr_2_2230 < 16384UL; fused_0fused_0__itr_0____itr_1_2229____itr_2_2230 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2229____itr_2_2230 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_2229____itr_2_2230 % 64UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2229____itr_2_2230 / 64UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2229____itr_2_2230 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_2229____itr_2_2230 % 64UL))] = __cached_2;
  }
  return true;
}

static bool cast__118(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2231____itr_2_2232 = 0UL; fused_0fused_0__itr_0____itr_1_2231____itr_2_2232 < 16384UL; fused_0fused_0__itr_0____itr_1_2231____itr_2_2232 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2231____itr_2_2232 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_2231____itr_2_2232 % 64UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2231____itr_2_2232 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_2231____itr_2_2232 % 64UL))] = __cached_1;
  }
  return true;
}

static bool mul__126(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2233____itr_2_2234 = 0UL; fused_0fused_0__itr_0____itr_1_2233____itr_2_2234 < 16384UL; fused_0fused_0__itr_0____itr_1_2233____itr_2_2234 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2233____itr_2_2234 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_2233____itr_2_2234 % 64UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2233____itr_2_2234 / 64UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2233____itr_2_2234 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_2233____itr_2_2234 % 64UL))] = __cached_2;
  }
  return true;
}

static bool cast__127(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2235____itr_2_2236 = 0UL; fused_0fused_0__itr_0____itr_1_2235____itr_2_2236 < 16384UL; fused_0fused_0__itr_0____itr_1_2235____itr_2_2236 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2235____itr_2_2236 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_2235____itr_2_2236 % 64UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2235____itr_2_2236 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_2235____itr_2_2236 % 64UL))] = __cached_1;
  }
  return true;
}

static bool mul__135(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2237____itr_2_2238 = 0UL; fused_0fused_0__itr_0____itr_1_2237____itr_2_2238 < 16384UL; fused_0fused_0__itr_0____itr_1_2237____itr_2_2238 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2237____itr_2_2238 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_2237____itr_2_2238 % 64UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2237____itr_2_2238 / 64UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2237____itr_2_2238 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_2237____itr_2_2238 % 64UL))] = __cached_2;
  }
  return true;
}

static bool cast__136(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2239____itr_2_2240 = 0UL; fused_0fused_0__itr_0____itr_1_2239____itr_2_2240 < 16384UL; fused_0fused_0__itr_0____itr_1_2239____itr_2_2240 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2239____itr_2_2240 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_2239____itr_2_2240 % 64UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2239____itr_2_2240 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_2239____itr_2_2240 % 64UL))] = __cached_1;
  }
  return true;
}

static bool mul__120(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2241____itr_2_2242 = 0UL; fused_0fused_0__itr_0____itr_1_2241____itr_2_2242 < 16384UL; fused_0fused_0__itr_0____itr_1_2241____itr_2_2242 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2241____itr_2_2242 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2241____itr_2_2242 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2241____itr_2_2242 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2241____itr_2_2242 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2241____itr_2_2242 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__121(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2243____itr_2_2244 = 0UL; fused_0fused_0__itr_0____itr_1_2243____itr_2_2244 < 16384UL; fused_0fused_0__itr_0____itr_1_2243____itr_2_2244 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2243____itr_2_2244 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2243____itr_2_2244 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2243____itr_2_2244 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2243____itr_2_2244 % 256UL))] = __cached_1;
  }
  return true;
}

static bool mul__129(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2245____itr_2_2246 = 0UL; fused_0fused_0__itr_0____itr_1_2245____itr_2_2246 < 16384UL; fused_0fused_0__itr_0____itr_1_2245____itr_2_2246 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2245____itr_2_2246 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2245____itr_2_2246 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2245____itr_2_2246 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2245____itr_2_2246 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2245____itr_2_2246 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__130(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2247____itr_2_2248 = 0UL; fused_0fused_0__itr_0____itr_1_2247____itr_2_2248 < 16384UL; fused_0fused_0__itr_0____itr_1_2247____itr_2_2248 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2247____itr_2_2248 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2247____itr_2_2248 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2247____itr_2_2248 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2247____itr_2_2248 % 256UL))] = __cached_1;
  }
  return true;
}

static bool mul__141(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2249____itr_2_2250 = 0UL; fused_0fused_0__itr_0____itr_1_2249____itr_2_2250 < 32768UL; fused_0fused_0__itr_0____itr_1_2249____itr_2_2250 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2249____itr_2_2250 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2249____itr_2_2250 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2249____itr_2_2250 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2249____itr_2_2250 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2249____itr_2_2250 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__142(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2251____itr_2_2252 = 0UL; fused_0fused_0__itr_0____itr_1_2251____itr_2_2252 < 32768UL; fused_0fused_0__itr_0____itr_1_2251____itr_2_2252 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2251____itr_2_2252 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2251____itr_2_2252 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2251____itr_2_2252 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2251____itr_2_2252 % 256UL))] = __cached_1;
  }
  return true;
}

static bool mul__114(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2253____itr_2_2254 = 0UL; fused_0fused_0__itr_0____itr_1_2253____itr_2_2254 < 12288UL; fused_0fused_0__itr_0____itr_1_2253____itr_2_2254 += 1UL) {
    for (uint64_t _fuseiter_8116 = 0UL; _fuseiter_8116 < 3UL; _fuseiter_8116 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2253____itr_2_2254 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_2253____itr_2_2254 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2253____itr_2_2254 % 3UL) * 3UL))) + _fuseiter_8116)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2253____itr_2_2254 / 192UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2253____itr_2_2254 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_2253____itr_2_2254 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2253____itr_2_2254 % 3UL) * 3UL))) + _fuseiter_8116)] = __cached_2;
    }
  }
  return true;
}

static bool cast__115(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2255____itr_2_2256 = 0UL; fused_0fused_0__itr_0____itr_1_2255____itr_2_2256 < 12288UL; fused_0fused_0__itr_0____itr_1_2255____itr_2_2256 += 1UL) {
    for (uint64_t _fuseiter8121 = 0UL; _fuseiter8121 < 3UL; _fuseiter8121 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2255____itr_2_2256 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_2255____itr_2_2256 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2255____itr_2_2256 % 3UL) * 3UL))) + _fuseiter8121)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2255____itr_2_2256 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_2255____itr_2_2256 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2255____itr_2_2256 % 3UL) * 3UL))) + _fuseiter8121)] = __cached_1;
    }
  }
  return true;
}

static bool mul__123(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2257____itr_2_2258 = 0UL; fused_0fused_0__itr_0____itr_1_2257____itr_2_2258 < 12288UL; fused_0fused_0__itr_0____itr_1_2257____itr_2_2258 += 1UL) {
    for (uint64_t _fuseiter_8126 = 0UL; _fuseiter_8126 < 3UL; _fuseiter_8126 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2257____itr_2_2258 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_2257____itr_2_2258 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2257____itr_2_2258 % 3UL) * 3UL))) + _fuseiter_8126)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2257____itr_2_2258 / 192UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2257____itr_2_2258 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_2257____itr_2_2258 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2257____itr_2_2258 % 3UL) * 3UL))) + _fuseiter_8126)] = __cached_2;
    }
  }
  return true;
}

static bool cast__124(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2259____itr_2_2260 = 0UL; fused_0fused_0__itr_0____itr_1_2259____itr_2_2260 < 12288UL; fused_0fused_0__itr_0____itr_1_2259____itr_2_2260 += 1UL) {
    for (uint64_t _fuseiter8131 = 0UL; _fuseiter8131 < 3UL; _fuseiter8131 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2259____itr_2_2260 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_2259____itr_2_2260 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2259____itr_2_2260 % 3UL) * 3UL))) + _fuseiter8131)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2259____itr_2_2260 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_2259____itr_2_2260 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2259____itr_2_2260 % 3UL) * 3UL))) + _fuseiter8131)] = __cached_1;
    }
  }
  return true;
}

static bool mul__132(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2261____itr_2_2262 = 0UL; fused_0fused_0__itr_0____itr_1_2261____itr_2_2262 < 12288UL; fused_0fused_0__itr_0____itr_1_2261____itr_2_2262 += 1UL) {
    for (uint64_t _fuseiter_8136 = 0UL; _fuseiter_8136 < 3UL; _fuseiter_8136 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2261____itr_2_2262 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_2261____itr_2_2262 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2261____itr_2_2262 % 3UL) * 3UL))) + _fuseiter_8136)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2261____itr_2_2262 / 192UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2261____itr_2_2262 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_2261____itr_2_2262 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2261____itr_2_2262 % 3UL) * 3UL))) + _fuseiter_8136)] = __cached_2;
    }
  }
  return true;
}

static bool cast__133(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2263____itr_2_2264 = 0UL; fused_0fused_0__itr_0____itr_1_2263____itr_2_2264 < 12288UL; fused_0fused_0__itr_0____itr_1_2263____itr_2_2264 += 1UL) {
    for (uint64_t _fuseiter8141 = 0UL; _fuseiter8141 < 3UL; _fuseiter8141 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2263____itr_2_2264 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_2263____itr_2_2264 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2263____itr_2_2264 % 3UL) * 3UL))) + _fuseiter8141)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2263____itr_2_2264 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_2263____itr_2_2264 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2263____itr_2_2264 % 3UL) * 3UL))) + _fuseiter8141)] = __cached_1;
    }
  }
  return true;
}

static bool mul__147(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2265____itr_2_2266 = 0UL; fused_0fused_0__itr_0____itr_1_2265____itr_2_2266 < 65536UL; fused_0fused_0__itr_0____itr_1_2265____itr_2_2266 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2265____itr_2_2266 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2265____itr_2_2266 % 128UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2265____itr_2_2266 / 128UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2265____itr_2_2266 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2265____itr_2_2266 % 128UL))] = __cached_2;
  }
  return true;
}

static bool cast__148(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2267____itr_2_2268 = 0UL; fused_0fused_0__itr_0____itr_1_2267____itr_2_2268 < 65536UL; fused_0fused_0__itr_0____itr_1_2267____itr_2_2268 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2267____itr_2_2268 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2267____itr_2_2268 % 128UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2267____itr_2_2268 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2267____itr_2_2268 % 128UL))] = __cached_1;
  }
  return true;
}

static bool mul__156(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2269____itr_2_2270 = 0UL; fused_0fused_0__itr_0____itr_1_2269____itr_2_2270 < 65536UL; fused_0fused_0__itr_0____itr_1_2269____itr_2_2270 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2269____itr_2_2270 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2269____itr_2_2270 % 128UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2269____itr_2_2270 / 128UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2269____itr_2_2270 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2269____itr_2_2270 % 128UL))] = __cached_2;
  }
  return true;
}

static bool cast__157(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2271____itr_2_2272 = 0UL; fused_0fused_0__itr_0____itr_1_2271____itr_2_2272 < 65536UL; fused_0fused_0__itr_0____itr_1_2271____itr_2_2272 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2271____itr_2_2272 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2271____itr_2_2272 % 128UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2271____itr_2_2272 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2271____itr_2_2272 % 128UL))] = __cached_1;
  }
  return true;
}

static bool mul__165(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2273____itr_2_2274 = 0UL; fused_0fused_0__itr_0____itr_1_2273____itr_2_2274 < 65536UL; fused_0fused_0__itr_0____itr_1_2273____itr_2_2274 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2273____itr_2_2274 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2273____itr_2_2274 % 128UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2273____itr_2_2274 / 128UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2273____itr_2_2274 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2273____itr_2_2274 % 128UL))] = __cached_2;
  }
  return true;
}

static bool cast__166(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2275____itr_2_2276 = 0UL; fused_0fused_0__itr_0____itr_1_2275____itr_2_2276 < 65536UL; fused_0fused_0__itr_0____itr_1_2275____itr_2_2276 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2275____itr_2_2276 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2275____itr_2_2276 % 128UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2275____itr_2_2276 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2275____itr_2_2276 % 128UL))] = __cached_1;
  }
  return true;
}

static bool mul__174(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2277____itr_2_2278 = 0UL; fused_0fused_0__itr_0____itr_1_2277____itr_2_2278 < 65536UL; fused_0fused_0__itr_0____itr_1_2277____itr_2_2278 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2277____itr_2_2278 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2277____itr_2_2278 % 128UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2277____itr_2_2278 / 128UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2277____itr_2_2278 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2277____itr_2_2278 % 128UL))] = __cached_2;
  }
  return true;
}

static bool cast__175(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2279____itr_2_2280 = 0UL; fused_0fused_0__itr_0____itr_1_2279____itr_2_2280 < 65536UL; fused_0fused_0__itr_0____itr_1_2279____itr_2_2280 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2279____itr_2_2280 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2279____itr_2_2280 % 128UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2279____itr_2_2280 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2279____itr_2_2280 % 128UL))] = __cached_1;
  }
  return true;
}

static bool mul__150(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2281____itr_2_2282 = 0UL; fused_0fused_0__itr_0____itr_1_2281____itr_2_2282 < 65536UL; fused_0fused_0__itr_0____itr_1_2281____itr_2_2282 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2281____itr_2_2282 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2281____itr_2_2282 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2281____itr_2_2282 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2281____itr_2_2282 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2281____itr_2_2282 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__151(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2283____itr_2_2284 = 0UL; fused_0fused_0__itr_0____itr_1_2283____itr_2_2284 < 65536UL; fused_0fused_0__itr_0____itr_1_2283____itr_2_2284 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2283____itr_2_2284 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2283____itr_2_2284 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2283____itr_2_2284 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2283____itr_2_2284 % 512UL))] = __cached_1;
  }
  return true;
}

static bool mul__159(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2285____itr_2_2286 = 0UL; fused_0fused_0__itr_0____itr_1_2285____itr_2_2286 < 65536UL; fused_0fused_0__itr_0____itr_1_2285____itr_2_2286 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2285____itr_2_2286 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2285____itr_2_2286 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2285____itr_2_2286 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2285____itr_2_2286 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2285____itr_2_2286 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__160(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2287____itr_2_2288 = 0UL; fused_0fused_0__itr_0____itr_1_2287____itr_2_2288 < 65536UL; fused_0fused_0__itr_0____itr_1_2287____itr_2_2288 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2287____itr_2_2288 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2287____itr_2_2288 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2287____itr_2_2288 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2287____itr_2_2288 % 512UL))] = __cached_1;
  }
  return true;
}

static bool mul__168(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2289____itr_2_2290 = 0UL; fused_0fused_0__itr_0____itr_1_2289____itr_2_2290 < 65536UL; fused_0fused_0__itr_0____itr_1_2289____itr_2_2290 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2289____itr_2_2290 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2289____itr_2_2290 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2289____itr_2_2290 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2289____itr_2_2290 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2289____itr_2_2290 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__169(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2291____itr_2_2292 = 0UL; fused_0fused_0__itr_0____itr_1_2291____itr_2_2292 < 65536UL; fused_0fused_0__itr_0____itr_1_2291____itr_2_2292 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2291____itr_2_2292 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2291____itr_2_2292 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2291____itr_2_2292 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2291____itr_2_2292 % 512UL))] = __cached_1;
  }
  return true;
}

static bool mul__138(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2293____itr_2_2294 = 0UL; fused_0fused_0__itr_0____itr_1_2293____itr_2_2294 < 131072UL; fused_0fused_0__itr_0____itr_1_2293____itr_2_2294 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2293____itr_2_2294 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2293____itr_2_2294 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2293____itr_2_2294 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2293____itr_2_2294 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2293____itr_2_2294 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__139(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2295____itr_2_2296 = 0UL; fused_0fused_0__itr_0____itr_1_2295____itr_2_2296 < 131072UL; fused_0fused_0__itr_0____itr_1_2295____itr_2_2296 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2295____itr_2_2296 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2295____itr_2_2296 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2295____itr_2_2296 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2295____itr_2_2296 % 256UL))] = __cached_1;
  }
  return true;
}

static bool mul__180(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2297____itr_2_2298 = 0UL; fused_0fused_0__itr_0____itr_1_2297____itr_2_2298 < 131072UL; fused_0fused_0__itr_0____itr_1_2297____itr_2_2298 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2297____itr_2_2298 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2297____itr_2_2298 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2297____itr_2_2298 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2297____itr_2_2298 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2297____itr_2_2298 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__181(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2299____itr_2_2300 = 0UL; fused_0fused_0__itr_0____itr_1_2299____itr_2_2300 < 131072UL; fused_0fused_0__itr_0____itr_1_2299____itr_2_2300 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2299____itr_2_2300 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2299____itr_2_2300 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2299____itr_2_2300 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2299____itr_2_2300 % 512UL))] = __cached_1;
  }
  return true;
}

static bool mul__144(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2301____itr_2_2302 = 0UL; fused_0fused_0__itr_0____itr_1_2301____itr_2_2302 < 49152UL; fused_0fused_0__itr_0____itr_1_2301____itr_2_2302 += 1UL) {
    for (uint64_t _fuseiter_8236 = 0UL; _fuseiter_8236 < 3UL; _fuseiter_8236 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2301____itr_2_2302 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2301____itr_2_2302 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2301____itr_2_2302 % 3UL) * 3UL))) + _fuseiter_8236)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2301____itr_2_2302 / 384UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2301____itr_2_2302 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2301____itr_2_2302 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2301____itr_2_2302 % 3UL) * 3UL))) + _fuseiter_8236)] = __cached_2;
    }
  }
  return true;
}

static bool cast__145(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2303____itr_2_2304 = 0UL; fused_0fused_0__itr_0____itr_1_2303____itr_2_2304 < 49152UL; fused_0fused_0__itr_0____itr_1_2303____itr_2_2304 += 1UL) {
    for (uint64_t _fuseiter8241 = 0UL; _fuseiter8241 < 3UL; _fuseiter8241 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2303____itr_2_2304 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2303____itr_2_2304 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2303____itr_2_2304 % 3UL) * 3UL))) + _fuseiter8241)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2303____itr_2_2304 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2303____itr_2_2304 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2303____itr_2_2304 % 3UL) * 3UL))) + _fuseiter8241)] = __cached_1;
    }
  }
  return true;
}

static bool mul__153(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2305____itr_2_2306 = 0UL; fused_0fused_0__itr_0____itr_1_2305____itr_2_2306 < 49152UL; fused_0fused_0__itr_0____itr_1_2305____itr_2_2306 += 1UL) {
    for (uint64_t _fuseiter_8246 = 0UL; _fuseiter_8246 < 3UL; _fuseiter_8246 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2305____itr_2_2306 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2305____itr_2_2306 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2305____itr_2_2306 % 3UL) * 3UL))) + _fuseiter_8246)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2305____itr_2_2306 / 384UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2305____itr_2_2306 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2305____itr_2_2306 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2305____itr_2_2306 % 3UL) * 3UL))) + _fuseiter_8246)] = __cached_2;
    }
  }
  return true;
}

static bool cast__154(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2307____itr_2_2308 = 0UL; fused_0fused_0__itr_0____itr_1_2307____itr_2_2308 < 49152UL; fused_0fused_0__itr_0____itr_1_2307____itr_2_2308 += 1UL) {
    for (uint64_t _fuseiter8251 = 0UL; _fuseiter8251 < 3UL; _fuseiter8251 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2307____itr_2_2308 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2307____itr_2_2308 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2307____itr_2_2308 % 3UL) * 3UL))) + _fuseiter8251)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2307____itr_2_2308 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2307____itr_2_2308 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2307____itr_2_2308 % 3UL) * 3UL))) + _fuseiter8251)] = __cached_1;
    }
  }
  return true;
}

static bool mul__162(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2309____itr_2_2310 = 0UL; fused_0fused_0__itr_0____itr_1_2309____itr_2_2310 < 49152UL; fused_0fused_0__itr_0____itr_1_2309____itr_2_2310 += 1UL) {
    for (uint64_t _fuseiter_8256 = 0UL; _fuseiter_8256 < 3UL; _fuseiter_8256 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2309____itr_2_2310 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2309____itr_2_2310 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2309____itr_2_2310 % 3UL) * 3UL))) + _fuseiter_8256)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2309____itr_2_2310 / 384UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2309____itr_2_2310 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2309____itr_2_2310 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2309____itr_2_2310 % 3UL) * 3UL))) + _fuseiter_8256)] = __cached_2;
    }
  }
  return true;
}

static bool cast__163(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2311____itr_2_2312 = 0UL; fused_0fused_0__itr_0____itr_1_2311____itr_2_2312 < 49152UL; fused_0fused_0__itr_0____itr_1_2311____itr_2_2312 += 1UL) {
    for (uint64_t _fuseiter8261 = 0UL; _fuseiter8261 < 3UL; _fuseiter8261 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2311____itr_2_2312 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2311____itr_2_2312 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2311____itr_2_2312 % 3UL) * 3UL))) + _fuseiter8261)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2311____itr_2_2312 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2311____itr_2_2312 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2311____itr_2_2312 % 3UL) * 3UL))) + _fuseiter8261)] = __cached_1;
    }
  }
  return true;
}

static bool mul__171(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2313____itr_2_2314 = 0UL; fused_0fused_0__itr_0____itr_1_2313____itr_2_2314 < 49152UL; fused_0fused_0__itr_0____itr_1_2313____itr_2_2314 += 1UL) {
    for (uint64_t _fuseiter_8266 = 0UL; _fuseiter_8266 < 3UL; _fuseiter_8266 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2313____itr_2_2314 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2313____itr_2_2314 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2313____itr_2_2314 % 3UL) * 3UL))) + _fuseiter_8266)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2313____itr_2_2314 / 384UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2313____itr_2_2314 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2313____itr_2_2314 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2313____itr_2_2314 % 3UL) * 3UL))) + _fuseiter_8266)] = __cached_2;
    }
  }
  return true;
}

static bool cast__172(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2315____itr_2_2316 = 0UL; fused_0fused_0__itr_0____itr_1_2315____itr_2_2316 < 49152UL; fused_0fused_0__itr_0____itr_1_2315____itr_2_2316 += 1UL) {
    for (uint64_t _fuseiter8271 = 0UL; _fuseiter8271 < 3UL; _fuseiter8271 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2315____itr_2_2316 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2315____itr_2_2316 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2315____itr_2_2316 % 3UL) * 3UL))) + _fuseiter8271)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2315____itr_2_2316 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2315____itr_2_2316 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2315____itr_2_2316 % 3UL) * 3UL))) + _fuseiter8271)] = __cached_1;
    }
  }
  return true;
}

static bool mul__186(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2317____itr_2_2318 = 0UL; fused_0fused_0__itr_0____itr_1_2317____itr_2_2318 < 262144UL; fused_0fused_0__itr_0____itr_1_2317____itr_2_2318 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2317____itr_2_2318 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2317____itr_2_2318 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2317____itr_2_2318 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2317____itr_2_2318 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2317____itr_2_2318 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__187(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2319____itr_2_2320 = 0UL; fused_0fused_0__itr_0____itr_1_2319____itr_2_2320 < 262144UL; fused_0fused_0__itr_0____itr_1_2319____itr_2_2320 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2319____itr_2_2320 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2319____itr_2_2320 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2319____itr_2_2320 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2319____itr_2_2320 % 256UL))] = __cached_1;
  }
  return true;
}

static bool mul__195(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2321____itr_2_2322 = 0UL; fused_0fused_0__itr_0____itr_1_2321____itr_2_2322 < 262144UL; fused_0fused_0__itr_0____itr_1_2321____itr_2_2322 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2321____itr_2_2322 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2321____itr_2_2322 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2321____itr_2_2322 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2321____itr_2_2322 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2321____itr_2_2322 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__196(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2323____itr_2_2324 = 0UL; fused_0fused_0__itr_0____itr_1_2323____itr_2_2324 < 262144UL; fused_0fused_0__itr_0____itr_1_2323____itr_2_2324 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2323____itr_2_2324 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2323____itr_2_2324 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2323____itr_2_2324 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2323____itr_2_2324 % 256UL))] = __cached_1;
  }
  return true;
}

static bool mul__204(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2325____itr_2_2326 = 0UL; fused_0fused_0__itr_0____itr_1_2325____itr_2_2326 < 262144UL; fused_0fused_0__itr_0____itr_1_2325____itr_2_2326 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2325____itr_2_2326 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2325____itr_2_2326 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2325____itr_2_2326 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2325____itr_2_2326 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2325____itr_2_2326 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__205(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2327____itr_2_2328 = 0UL; fused_0fused_0__itr_0____itr_1_2327____itr_2_2328 < 262144UL; fused_0fused_0__itr_0____itr_1_2327____itr_2_2328 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2327____itr_2_2328 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2327____itr_2_2328 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2327____itr_2_2328 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2327____itr_2_2328 % 256UL))] = __cached_1;
  }
  return true;
}

static bool mul__213(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2329____itr_2_2330 = 0UL; fused_0fused_0__itr_0____itr_1_2329____itr_2_2330 < 262144UL; fused_0fused_0__itr_0____itr_1_2329____itr_2_2330 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2329____itr_2_2330 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2329____itr_2_2330 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2329____itr_2_2330 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2329____itr_2_2330 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2329____itr_2_2330 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__214(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2331____itr_2_2332 = 0UL; fused_0fused_0__itr_0____itr_1_2331____itr_2_2332 < 262144UL; fused_0fused_0__itr_0____itr_1_2331____itr_2_2332 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2331____itr_2_2332 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2331____itr_2_2332 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2331____itr_2_2332 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2331____itr_2_2332 % 256UL))] = __cached_1;
  }
  return true;
}

static bool mul__222(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2333____itr_2_2334 = 0UL; fused_0fused_0__itr_0____itr_1_2333____itr_2_2334 < 262144UL; fused_0fused_0__itr_0____itr_1_2333____itr_2_2334 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2333____itr_2_2334 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2333____itr_2_2334 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2333____itr_2_2334 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2333____itr_2_2334 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2333____itr_2_2334 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__223(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2335____itr_2_2336 = 0UL; fused_0fused_0__itr_0____itr_1_2335____itr_2_2336 < 262144UL; fused_0fused_0__itr_0____itr_1_2335____itr_2_2336 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2335____itr_2_2336 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2335____itr_2_2336 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2335____itr_2_2336 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2335____itr_2_2336 % 256UL))] = __cached_1;
  }
  return true;
}

static bool mul__231(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2337____itr_2_2338 = 0UL; fused_0fused_0__itr_0____itr_1_2337____itr_2_2338 < 262144UL; fused_0fused_0__itr_0____itr_1_2337____itr_2_2338 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2337____itr_2_2338 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2337____itr_2_2338 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2337____itr_2_2338 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2337____itr_2_2338 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2337____itr_2_2338 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__232(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2339____itr_2_2340 = 0UL; fused_0fused_0__itr_0____itr_1_2339____itr_2_2340 < 262144UL; fused_0fused_0__itr_0____itr_1_2339____itr_2_2340 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2339____itr_2_2340 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2339____itr_2_2340 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2339____itr_2_2340 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2339____itr_2_2340 % 256UL))] = __cached_1;
  }
  return true;
}

static bool mul__189(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2341____itr_2_2342 = 0UL; fused_0fused_0__itr_0____itr_1_2341____itr_2_2342 < 262144UL; fused_0fused_0__itr_0____itr_1_2341____itr_2_2342 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2341____itr_2_2342 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2341____itr_2_2342 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2341____itr_2_2342 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2341____itr_2_2342 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2341____itr_2_2342 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__190(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2343____itr_2_2344 = 0UL; fused_0fused_0__itr_0____itr_1_2343____itr_2_2344 < 262144UL; fused_0fused_0__itr_0____itr_1_2343____itr_2_2344 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2343____itr_2_2344 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2343____itr_2_2344 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2343____itr_2_2344 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2343____itr_2_2344 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool mul__198(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2345____itr_2_2346 = 0UL; fused_0fused_0__itr_0____itr_1_2345____itr_2_2346 < 262144UL; fused_0fused_0__itr_0____itr_1_2345____itr_2_2346 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2345____itr_2_2346 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2345____itr_2_2346 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2345____itr_2_2346 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2345____itr_2_2346 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2345____itr_2_2346 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__199(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2347____itr_2_2348 = 0UL; fused_0fused_0__itr_0____itr_1_2347____itr_2_2348 < 262144UL; fused_0fused_0__itr_0____itr_1_2347____itr_2_2348 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2347____itr_2_2348 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2347____itr_2_2348 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2347____itr_2_2348 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2347____itr_2_2348 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool mul__207(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2349____itr_2_2350 = 0UL; fused_0fused_0__itr_0____itr_1_2349____itr_2_2350 < 262144UL; fused_0fused_0__itr_0____itr_1_2349____itr_2_2350 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2349____itr_2_2350 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2349____itr_2_2350 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2349____itr_2_2350 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2349____itr_2_2350 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2349____itr_2_2350 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__208(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2351____itr_2_2352 = 0UL; fused_0fused_0__itr_0____itr_1_2351____itr_2_2352 < 262144UL; fused_0fused_0__itr_0____itr_1_2351____itr_2_2352 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2351____itr_2_2352 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2351____itr_2_2352 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2351____itr_2_2352 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2351____itr_2_2352 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool mul__216(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2353____itr_2_2354 = 0UL; fused_0fused_0__itr_0____itr_1_2353____itr_2_2354 < 262144UL; fused_0fused_0__itr_0____itr_1_2353____itr_2_2354 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2353____itr_2_2354 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2353____itr_2_2354 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2353____itr_2_2354 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2353____itr_2_2354 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2353____itr_2_2354 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__217(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2355____itr_2_2356 = 0UL; fused_0fused_0__itr_0____itr_1_2355____itr_2_2356 < 262144UL; fused_0fused_0__itr_0____itr_1_2355____itr_2_2356 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2355____itr_2_2356 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2355____itr_2_2356 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2355____itr_2_2356 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2355____itr_2_2356 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool mul__225(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2357____itr_2_2358 = 0UL; fused_0fused_0__itr_0____itr_1_2357____itr_2_2358 < 262144UL; fused_0fused_0__itr_0____itr_1_2357____itr_2_2358 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2357____itr_2_2358 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2357____itr_2_2358 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2357____itr_2_2358 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2357____itr_2_2358 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2357____itr_2_2358 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__226(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2359____itr_2_2360 = 0UL; fused_0fused_0__itr_0____itr_1_2359____itr_2_2360 < 262144UL; fused_0fused_0__itr_0____itr_1_2359____itr_2_2360 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2359____itr_2_2360 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2359____itr_2_2360 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2359____itr_2_2360 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2359____itr_2_2360 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool mul__177(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2361____itr_2_2362 = 0UL; fused_0fused_0__itr_0____itr_1_2361____itr_2_2362 < 524288UL; fused_0fused_0__itr_0____itr_1_2361____itr_2_2362 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2361____itr_2_2362 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2361____itr_2_2362 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2361____itr_2_2362 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2361____itr_2_2362 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2361____itr_2_2362 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__178(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2363____itr_2_2364 = 0UL; fused_0fused_0__itr_0____itr_1_2363____itr_2_2364 < 524288UL; fused_0fused_0__itr_0____itr_1_2363____itr_2_2364 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2363____itr_2_2364 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2363____itr_2_2364 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2363____itr_2_2364 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2363____itr_2_2364 % 512UL))] = __cached_1;
  }
  return true;
}

static bool mul__237(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2365____itr_2_2366 = 0UL; fused_0fused_0__itr_0____itr_1_2365____itr_2_2366 < 524288UL; fused_0fused_0__itr_0____itr_1_2365____itr_2_2366 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2365____itr_2_2366 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2365____itr_2_2366 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2365____itr_2_2366 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2365____itr_2_2366 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2365____itr_2_2366 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__238(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2367____itr_2_2368 = 0UL; fused_0fused_0__itr_0____itr_1_2367____itr_2_2368 < 524288UL; fused_0fused_0__itr_0____itr_1_2367____itr_2_2368 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2367____itr_2_2368 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2367____itr_2_2368 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2367____itr_2_2368 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2367____itr_2_2368 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool mul__183(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2369____itr_2_2370 = 0UL; fused_0fused_0__itr_0____itr_1_2369____itr_2_2370 < 196608UL; fused_0fused_0__itr_0____itr_1_2369____itr_2_2370 += 1UL) {
    for (uint64_t _fuseiter_8406 = 0UL; _fuseiter_8406 < 3UL; _fuseiter_8406 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2369____itr_2_2370 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2369____itr_2_2370 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2369____itr_2_2370 % 3UL) * 3UL))) + _fuseiter_8406)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2369____itr_2_2370 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2369____itr_2_2370 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2369____itr_2_2370 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2369____itr_2_2370 % 3UL) * 3UL))) + _fuseiter_8406)] = __cached_2;
    }
  }
  return true;
}

static bool cast__184(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2371____itr_2_2372 = 0UL; fused_0fused_0__itr_0____itr_1_2371____itr_2_2372 < 196608UL; fused_0fused_0__itr_0____itr_1_2371____itr_2_2372 += 1UL) {
    for (uint64_t _fuseiter8411 = 0UL; _fuseiter8411 < 3UL; _fuseiter8411 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2371____itr_2_2372 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2371____itr_2_2372 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2371____itr_2_2372 % 3UL) * 3UL))) + _fuseiter8411)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2371____itr_2_2372 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2371____itr_2_2372 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2371____itr_2_2372 % 3UL) * 3UL))) + _fuseiter8411)] = __cached_1;
    }
  }
  return true;
}

static bool mul__192(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2373____itr_2_2374 = 0UL; fused_0fused_0__itr_0____itr_1_2373____itr_2_2374 < 196608UL; fused_0fused_0__itr_0____itr_1_2373____itr_2_2374 += 1UL) {
    for (uint64_t _fuseiter_8416 = 0UL; _fuseiter_8416 < 3UL; _fuseiter_8416 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2373____itr_2_2374 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2373____itr_2_2374 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2373____itr_2_2374 % 3UL) * 3UL))) + _fuseiter_8416)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2373____itr_2_2374 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2373____itr_2_2374 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2373____itr_2_2374 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2373____itr_2_2374 % 3UL) * 3UL))) + _fuseiter_8416)] = __cached_2;
    }
  }
  return true;
}

static bool cast__193(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2375____itr_2_2376 = 0UL; fused_0fused_0__itr_0____itr_1_2375____itr_2_2376 < 196608UL; fused_0fused_0__itr_0____itr_1_2375____itr_2_2376 += 1UL) {
    for (uint64_t _fuseiter8421 = 0UL; _fuseiter8421 < 3UL; _fuseiter8421 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2375____itr_2_2376 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2375____itr_2_2376 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2375____itr_2_2376 % 3UL) * 3UL))) + _fuseiter8421)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2375____itr_2_2376 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2375____itr_2_2376 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2375____itr_2_2376 % 3UL) * 3UL))) + _fuseiter8421)] = __cached_1;
    }
  }
  return true;
}

static bool mul__201(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2377____itr_2_2378 = 0UL; fused_0fused_0__itr_0____itr_1_2377____itr_2_2378 < 196608UL; fused_0fused_0__itr_0____itr_1_2377____itr_2_2378 += 1UL) {
    for (uint64_t _fuseiter_8426 = 0UL; _fuseiter_8426 < 3UL; _fuseiter_8426 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2377____itr_2_2378 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2377____itr_2_2378 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2377____itr_2_2378 % 3UL) * 3UL))) + _fuseiter_8426)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2377____itr_2_2378 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2377____itr_2_2378 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2377____itr_2_2378 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2377____itr_2_2378 % 3UL) * 3UL))) + _fuseiter_8426)] = __cached_2;
    }
  }
  return true;
}

static bool cast__202(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2379____itr_2_2380 = 0UL; fused_0fused_0__itr_0____itr_1_2379____itr_2_2380 < 196608UL; fused_0fused_0__itr_0____itr_1_2379____itr_2_2380 += 1UL) {
    for (uint64_t _fuseiter8431 = 0UL; _fuseiter8431 < 3UL; _fuseiter8431 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2379____itr_2_2380 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2379____itr_2_2380 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2379____itr_2_2380 % 3UL) * 3UL))) + _fuseiter8431)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2379____itr_2_2380 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2379____itr_2_2380 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2379____itr_2_2380 % 3UL) * 3UL))) + _fuseiter8431)] = __cached_1;
    }
  }
  return true;
}

static bool mul__210(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2381____itr_2_2382 = 0UL; fused_0fused_0__itr_0____itr_1_2381____itr_2_2382 < 196608UL; fused_0fused_0__itr_0____itr_1_2381____itr_2_2382 += 1UL) {
    for (uint64_t _fuseiter_8436 = 0UL; _fuseiter_8436 < 3UL; _fuseiter_8436 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2381____itr_2_2382 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2381____itr_2_2382 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2381____itr_2_2382 % 3UL) * 3UL))) + _fuseiter_8436)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2381____itr_2_2382 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2381____itr_2_2382 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2381____itr_2_2382 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2381____itr_2_2382 % 3UL) * 3UL))) + _fuseiter_8436)] = __cached_2;
    }
  }
  return true;
}

static bool cast__211(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2383____itr_2_2384 = 0UL; fused_0fused_0__itr_0____itr_1_2383____itr_2_2384 < 196608UL; fused_0fused_0__itr_0____itr_1_2383____itr_2_2384 += 1UL) {
    for (uint64_t _fuseiter8441 = 0UL; _fuseiter8441 < 3UL; _fuseiter8441 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2383____itr_2_2384 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2383____itr_2_2384 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2383____itr_2_2384 % 3UL) * 3UL))) + _fuseiter8441)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2383____itr_2_2384 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2383____itr_2_2384 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2383____itr_2_2384 % 3UL) * 3UL))) + _fuseiter8441)] = __cached_1;
    }
  }
  return true;
}

static bool mul__219(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2385____itr_2_2386 = 0UL; fused_0fused_0__itr_0____itr_1_2385____itr_2_2386 < 196608UL; fused_0fused_0__itr_0____itr_1_2385____itr_2_2386 += 1UL) {
    for (uint64_t _fuseiter_8446 = 0UL; _fuseiter_8446 < 3UL; _fuseiter_8446 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2385____itr_2_2386 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2385____itr_2_2386 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2385____itr_2_2386 % 3UL) * 3UL))) + _fuseiter_8446)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2385____itr_2_2386 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2385____itr_2_2386 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2385____itr_2_2386 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2385____itr_2_2386 % 3UL) * 3UL))) + _fuseiter_8446)] = __cached_2;
    }
  }
  return true;
}

static bool cast__220(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2387____itr_2_2388 = 0UL; fused_0fused_0__itr_0____itr_1_2387____itr_2_2388 < 196608UL; fused_0fused_0__itr_0____itr_1_2387____itr_2_2388 += 1UL) {
    for (uint64_t _fuseiter8451 = 0UL; _fuseiter8451 < 3UL; _fuseiter8451 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2387____itr_2_2388 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2387____itr_2_2388 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2387____itr_2_2388 % 3UL) * 3UL))) + _fuseiter8451)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2387____itr_2_2388 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2387____itr_2_2388 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2387____itr_2_2388 % 3UL) * 3UL))) + _fuseiter8451)] = __cached_1;
    }
  }
  return true;
}

static bool mul__228(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2389____itr_2_2390 = 0UL; fused_0fused_0__itr_0____itr_1_2389____itr_2_2390 < 196608UL; fused_0fused_0__itr_0____itr_1_2389____itr_2_2390 += 1UL) {
    for (uint64_t _fuseiter_8456 = 0UL; _fuseiter_8456 < 3UL; _fuseiter_8456 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2389____itr_2_2390 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2389____itr_2_2390 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2389____itr_2_2390 % 3UL) * 3UL))) + _fuseiter_8456)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2389____itr_2_2390 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2389____itr_2_2390 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2389____itr_2_2390 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2389____itr_2_2390 % 3UL) * 3UL))) + _fuseiter_8456)] = __cached_2;
    }
  }
  return true;
}

static bool cast__229(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2391____itr_2_2392 = 0UL; fused_0fused_0__itr_0____itr_1_2391____itr_2_2392 < 196608UL; fused_0fused_0__itr_0____itr_1_2391____itr_2_2392 += 1UL) {
    for (uint64_t _fuseiter8461 = 0UL; _fuseiter8461 < 3UL; _fuseiter8461 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2391____itr_2_2392 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2391____itr_2_2392 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2391____itr_2_2392 % 3UL) * 3UL))) + _fuseiter8461)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2391____itr_2_2392 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2391____itr_2_2392 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2391____itr_2_2392 % 3UL) * 3UL))) + _fuseiter8461)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__525(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_8462___fuseiter_8463_2393 = 0UL; fused_0_fuseiter_8462___fuseiter_8463_2393 < 16UL; fused_0_fuseiter_8462___fuseiter_8463_2393 += 1UL) {
    for (uint64_t _fuseiter_8464 = 0UL; _fuseiter_8464 < 3UL; _fuseiter_8464 += 1UL) {
      for (uint64_t _fuseiter_8465 = 0UL; _fuseiter_8465 < 3UL; _fuseiter_8465 += 1UL) {
        for (uint64_t _fuseiter_8466 = 0UL; _fuseiter_8466 < 16UL; _fuseiter_8466 += 1UL) {
          for (uint64_t _fuseiter_8467 = 0UL; _fuseiter_8467 < 64UL; _fuseiter_8467 += 1UL) {
            for (uint64_t _fuseiter_8468 = 0UL; _fuseiter_8468 < 4UL; _fuseiter_8468 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_8467 + ((fused_0_fuseiter_8462___fuseiter_8463_2393 / 4UL) * 64UL)) * 2304UL) + ((((_fuseiter_8468 + (_fuseiter_8466 * 4UL)) + ((fused_0_fuseiter_8462___fuseiter_8463_2393 % 4UL) * 64UL)) * 9UL) + ((_fuseiter_8464 * 3UL) + _fuseiter_8465)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_8462___fuseiter_8463_2393 / 4UL) * 147456UL) + (((fused_0_fuseiter_8462___fuseiter_8463_2393 % 4UL) * 36864UL) + ((_fuseiter_8464 * 12288UL) + ((_fuseiter_8465 * 4096UL) + ((_fuseiter_8466 * 256UL) + ((_fuseiter_8467 * 4UL) + _fuseiter_8468))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool mul__652(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2394____itr_2_2395____itr_3_2396 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2394____itr_2_2395____itr_3_2396 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2394____itr_2_2395____itr_3_2396 += 1UL) {
    for (uint64_t _fuseiter_8473 = 0UL; _fuseiter_8473 < 256UL; _fuseiter_8473 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2394____itr_2_2395____itr_3_2396 / 4UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2394____itr_2_2395____itr_3_2396 % 4UL) * 256UL)) + _fuseiter_8473)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2394____itr_2_2395____itr_3_2396 / 4UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2394____itr_2_2395____itr_3_2396 % 4UL) * 256UL)) + _fuseiter_8473)]);
    }
  }
  return true;
}

static bool mul__651(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2397____itr_2_2398____itr_3_2399 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2397____itr_2_2398____itr_3_2399 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2397____itr_2_2398____itr_3_2399 += 1UL) {
    for (uint64_t _fuseiter_8479 = 0UL; _fuseiter_8479 < 256UL; _fuseiter_8479 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2397____itr_2_2398____itr_3_2399 / 4UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2397____itr_2_2398____itr_3_2399 % 4UL) * 256UL)) + _fuseiter_8479)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2397____itr_2_2398____itr_3_2399 / 4UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2397____itr_2_2398____itr_3_2399 % 4UL) * 256UL)) + _fuseiter_8479)]);
    }
  }
  return true;
}

static bool mul__646(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2400____itr_2_2401____itr_3_2402 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2400____itr_2_2401____itr_3_2402 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2400____itr_2_2401____itr_3_2402 += 1UL) {
    for (uint64_t _fuseiter_8485 = 0UL; _fuseiter_8485 < 128UL; _fuseiter_8485 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2400____itr_2_2401____itr_3_2402 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2400____itr_2_2401____itr_3_2402 % 8UL) * 128UL)) + _fuseiter_8485)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2400____itr_2_2401____itr_3_2402 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2400____itr_2_2401____itr_3_2402 % 8UL) * 128UL)) + _fuseiter_8485)]);
    }
  }
  return true;
}

static bool mul__645(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2403____itr_2_2404____itr_3_2405 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2403____itr_2_2404____itr_3_2405 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2403____itr_2_2404____itr_3_2405 += 1UL) {
    for (uint64_t _fuseiter_8491 = 0UL; _fuseiter_8491 < 128UL; _fuseiter_8491 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2403____itr_2_2404____itr_3_2405 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2403____itr_2_2404____itr_3_2405 % 8UL) * 128UL)) + _fuseiter_8491)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2403____itr_2_2404____itr_3_2405 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2403____itr_2_2404____itr_3_2405 % 8UL) * 128UL)) + _fuseiter_8491)]);
    }
  }
  return true;
}

static bool mul__640(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2406____itr_2_2407____itr_3_2408 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2406____itr_2_2407____itr_3_2408 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2406____itr_2_2407____itr_3_2408 += 1UL) {
    for (uint64_t _fuseiter_8497 = 0UL; _fuseiter_8497 < 128UL; _fuseiter_8497 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2406____itr_2_2407____itr_3_2408 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2406____itr_2_2407____itr_3_2408 % 8UL) * 128UL)) + _fuseiter_8497)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2406____itr_2_2407____itr_3_2408 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2406____itr_2_2407____itr_3_2408 % 8UL) * 128UL)) + _fuseiter_8497)]);
    }
  }
  return true;
}

static bool mul__639(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2409____itr_2_2410____itr_3_2411 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2409____itr_2_2410____itr_3_2411 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2409____itr_2_2410____itr_3_2411 += 1UL) {
    for (uint64_t _fuseiter_8503 = 0UL; _fuseiter_8503 < 128UL; _fuseiter_8503 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2409____itr_2_2410____itr_3_2411 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2409____itr_2_2410____itr_3_2411 % 8UL) * 128UL)) + _fuseiter_8503)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2409____itr_2_2410____itr_3_2411 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2409____itr_2_2410____itr_3_2411 % 8UL) * 128UL)) + _fuseiter_8503)]);
    }
  }
  return true;
}

static bool mul__634(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2412____itr_2_2413____itr_3_2414 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2412____itr_2_2413____itr_3_2414 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2412____itr_2_2413____itr_3_2414 += 1UL) {
    for (uint64_t _fuseiter_8509 = 0UL; _fuseiter_8509 < 128UL; _fuseiter_8509 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2412____itr_2_2413____itr_3_2414 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2412____itr_2_2413____itr_3_2414 % 8UL) * 128UL)) + _fuseiter_8509)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2412____itr_2_2413____itr_3_2414 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2412____itr_2_2413____itr_3_2414 % 8UL) * 128UL)) + _fuseiter_8509)]);
    }
  }
  return true;
}

static bool mul__633(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2415____itr_2_2416____itr_3_2417 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2415____itr_2_2416____itr_3_2417 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2415____itr_2_2416____itr_3_2417 += 1UL) {
    for (uint64_t _fuseiter_8515 = 0UL; _fuseiter_8515 < 128UL; _fuseiter_8515 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2415____itr_2_2416____itr_3_2417 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2415____itr_2_2416____itr_3_2417 % 8UL) * 128UL)) + _fuseiter_8515)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2415____itr_2_2416____itr_3_2417 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2415____itr_2_2416____itr_3_2417 % 8UL) * 128UL)) + _fuseiter_8515)]);
    }
  }
  return true;
}

static bool mul__628(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_8521 = 0UL; _fuseiter_8521 < 1024UL; _fuseiter_8521 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_8521]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_8521]);
  }
  return true;
}

static bool mul__627(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_8527 = 0UL; _fuseiter_8527 < 1024UL; _fuseiter_8527 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_8527]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_8527]);
  }
  return true;
}

static bool mul__622(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2424____itr_2_2425____itr_3_2426 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2424____itr_2_2425____itr_3_2426 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2424____itr_2_2425____itr_3_2426 += 1UL) {
    for (uint64_t _fuseiter_8533 = 0UL; _fuseiter_8533 < 128UL; _fuseiter_8533 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2424____itr_2_2425____itr_3_2426 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2424____itr_2_2425____itr_3_2426 % 8UL) * 128UL)) + _fuseiter_8533)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2424____itr_2_2425____itr_3_2426 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2424____itr_2_2425____itr_3_2426 % 8UL) * 128UL)) + _fuseiter_8533)]);
    }
  }
  return true;
}

static bool mul__621(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2427____itr_2_2428____itr_3_2429 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2427____itr_2_2428____itr_3_2429 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2427____itr_2_2428____itr_3_2429 += 1UL) {
    for (uint64_t _fuseiter_8539 = 0UL; _fuseiter_8539 < 128UL; _fuseiter_8539 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2427____itr_2_2428____itr_3_2429 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2427____itr_2_2428____itr_3_2429 % 8UL) * 128UL)) + _fuseiter_8539)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2427____itr_2_2428____itr_3_2429 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2427____itr_2_2428____itr_3_2429 % 8UL) * 128UL)) + _fuseiter_8539)]);
    }
  }
  return true;
}

static bool mul__616(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2430____itr_2_2431____itr_3_2432 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2430____itr_2_2431____itr_3_2432 < 32UL; fused_0fused_0fused_0__itr_0____itr_1_2430____itr_2_2431____itr_3_2432 += 1UL) {
    for (uint64_t _fuseiter_8545 = 0UL; _fuseiter_8545 < 32UL; _fuseiter_8545 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2430____itr_2_2431____itr_3_2432 / 32UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2430____itr_2_2431____itr_3_2432 % 32UL) * 32UL)) + _fuseiter_8545)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2430____itr_2_2431____itr_3_2432 / 32UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2430____itr_2_2431____itr_3_2432 % 32UL) * 32UL)) + _fuseiter_8545)]);
    }
  }
  return true;
}

static bool mul__615(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2433____itr_2_2434____itr_3_2435 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2433____itr_2_2434____itr_3_2435 < 32UL; fused_0fused_0fused_0__itr_0____itr_1_2433____itr_2_2434____itr_3_2435 += 1UL) {
    for (uint64_t _fuseiter_8551 = 0UL; _fuseiter_8551 < 32UL; _fuseiter_8551 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2433____itr_2_2434____itr_3_2435 / 32UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2433____itr_2_2434____itr_3_2435 % 32UL) * 32UL)) + _fuseiter_8551)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2433____itr_2_2434____itr_3_2435 / 32UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2433____itr_2_2434____itr_3_2435 % 32UL) * 32UL)) + _fuseiter_8551)]);
    }
  }
  return true;
}

static bool mul__614(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2436____itr_2_2437____itr_3_2438 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2436____itr_2_2437____itr_3_2438 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2436____itr_2_2437____itr_3_2438 += 1UL) {
    for (uint64_t _fuseiter_8557 = 0UL; _fuseiter_8557 < 128UL; _fuseiter_8557 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2436____itr_2_2437____itr_3_2438 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2436____itr_2_2437____itr_3_2438 % 4UL) * 128UL)) + _fuseiter_8557)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2436____itr_2_2437____itr_3_2438 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2436____itr_2_2437____itr_3_2438 % 4UL) * 128UL)) + _fuseiter_8557)]);
    }
  }
  return true;
}

static bool mul__613(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2439____itr_2_2440____itr_3_2441 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2439____itr_2_2440____itr_3_2441 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2439____itr_2_2440____itr_3_2441 += 1UL) {
    for (uint64_t _fuseiter_8563 = 0UL; _fuseiter_8563 < 128UL; _fuseiter_8563 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2439____itr_2_2440____itr_3_2441 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2439____itr_2_2440____itr_3_2441 % 4UL) * 128UL)) + _fuseiter_8563)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2439____itr_2_2440____itr_3_2441 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2439____itr_2_2440____itr_3_2441 % 4UL) * 128UL)) + _fuseiter_8563)]);
    }
  }
  return true;
}

static bool mul__608(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2442____itr_2_2443____itr_3_2444 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2442____itr_2_2443____itr_3_2444 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2442____itr_2_2443____itr_3_2444 += 1UL) {
    for (uint64_t _fuseiter_8569 = 0UL; _fuseiter_8569 < 64UL; _fuseiter_8569 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2442____itr_2_2443____itr_3_2444 / 8UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2442____itr_2_2443____itr_3_2444 % 8UL) * 64UL)) + _fuseiter_8569)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2442____itr_2_2443____itr_3_2444 / 8UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2442____itr_2_2443____itr_3_2444 % 8UL) * 64UL)) + _fuseiter_8569)]);
    }
  }
  return true;
}

static bool mul__607(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2445____itr_2_2446____itr_3_2447 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2445____itr_2_2446____itr_3_2447 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2445____itr_2_2446____itr_3_2447 += 1UL) {
    for (uint64_t _fuseiter_8575 = 0UL; _fuseiter_8575 < 64UL; _fuseiter_8575 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2445____itr_2_2446____itr_3_2447 / 8UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2445____itr_2_2446____itr_3_2447 % 8UL) * 64UL)) + _fuseiter_8575)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2445____itr_2_2446____itr_3_2447 / 8UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2445____itr_2_2446____itr_3_2447 % 8UL) * 64UL)) + _fuseiter_8575)]);
    }
  }
  return true;
}

static bool mul__602(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2448____itr_2_2449____itr_3_2450 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2448____itr_2_2449____itr_3_2450 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2448____itr_2_2449____itr_3_2450 += 1UL) {
    for (uint64_t _fuseiter_8581 = 0UL; _fuseiter_8581 < 128UL; _fuseiter_8581 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2448____itr_2_2449____itr_3_2450 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2448____itr_2_2449____itr_3_2450 % 4UL) * 128UL)) + _fuseiter_8581)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2448____itr_2_2449____itr_3_2450 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2448____itr_2_2449____itr_3_2450 % 4UL) * 128UL)) + _fuseiter_8581)]);
    }
  }
  return true;
}

static bool mul__601(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2451____itr_2_2452____itr_3_2453 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2451____itr_2_2452____itr_3_2453 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2451____itr_2_2452____itr_3_2453 += 1UL) {
    for (uint64_t _fuseiter_8587 = 0UL; _fuseiter_8587 < 128UL; _fuseiter_8587 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2451____itr_2_2452____itr_3_2453 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2451____itr_2_2452____itr_3_2453 % 4UL) * 128UL)) + _fuseiter_8587)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2451____itr_2_2452____itr_3_2453 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2451____itr_2_2452____itr_3_2453 % 4UL) * 128UL)) + _fuseiter_8587)]);
    }
  }
  return true;
}

static bool mul__596(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2454____itr_2_2455____itr_3_2456 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2454____itr_2_2455____itr_3_2456 < 16UL; fused_0fused_0fused_0__itr_0____itr_1_2454____itr_2_2455____itr_3_2456 += 1UL) {
    for (uint64_t _fuseiter_8593 = 0UL; _fuseiter_8593 < 32UL; _fuseiter_8593 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2454____itr_2_2455____itr_3_2456 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2454____itr_2_2455____itr_3_2456 % 16UL) * 32UL)) + _fuseiter_8593)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2454____itr_2_2455____itr_3_2456 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2454____itr_2_2455____itr_3_2456 % 16UL) * 32UL)) + _fuseiter_8593)]);
    }
  }
  return true;
}

static bool mul__595(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2457____itr_2_2458____itr_3_2459 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2457____itr_2_2458____itr_3_2459 < 16UL; fused_0fused_0fused_0__itr_0____itr_1_2457____itr_2_2458____itr_3_2459 += 1UL) {
    for (uint64_t _fuseiter_8599 = 0UL; _fuseiter_8599 < 32UL; _fuseiter_8599 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2457____itr_2_2458____itr_3_2459 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2457____itr_2_2458____itr_3_2459 % 16UL) * 32UL)) + _fuseiter_8599)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2457____itr_2_2458____itr_3_2459 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2457____itr_2_2458____itr_3_2459 % 16UL) * 32UL)) + _fuseiter_8599)]);
    }
  }
  return true;
}

static bool mul__590(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2460____itr_2_2461____itr_3_2462 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2460____itr_2_2461____itr_3_2462 < 16UL; fused_0fused_0fused_0__itr_0____itr_1_2460____itr_2_2461____itr_3_2462 += 1UL) {
    for (uint64_t _fuseiter_8605 = 0UL; _fuseiter_8605 < 32UL; _fuseiter_8605 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2460____itr_2_2461____itr_3_2462 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2460____itr_2_2461____itr_3_2462 % 16UL) * 32UL)) + _fuseiter_8605)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2460____itr_2_2461____itr_3_2462 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2460____itr_2_2461____itr_3_2462 % 16UL) * 32UL)) + _fuseiter_8605)]);
    }
  }
  return true;
}

static bool mul__589(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2463____itr_2_2464____itr_3_2465 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2463____itr_2_2464____itr_3_2465 < 16UL; fused_0fused_0fused_0__itr_0____itr_1_2463____itr_2_2464____itr_3_2465 += 1UL) {
    for (uint64_t _fuseiter_8611 = 0UL; _fuseiter_8611 < 32UL; _fuseiter_8611 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2463____itr_2_2464____itr_3_2465 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2463____itr_2_2464____itr_3_2465 % 16UL) * 32UL)) + _fuseiter_8611)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2463____itr_2_2464____itr_3_2465 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2463____itr_2_2464____itr_3_2465 % 16UL) * 32UL)) + _fuseiter_8611)]);
    }
  }
  return true;
}

static bool mul__650(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2466____itr_2_2467____itr_3_2468 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2466____itr_2_2467____itr_3_2468 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2466____itr_2_2467____itr_3_2468 += 1UL) {
    for (uint64_t _fuseiter_8617 = 0UL; _fuseiter_8617 < 64UL; _fuseiter_8617 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2466____itr_2_2467____itr_3_2468 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2466____itr_2_2467____itr_3_2468 % 4UL) * 64UL)) + _fuseiter_8617)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2466____itr_2_2467____itr_3_2468 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2466____itr_2_2467____itr_3_2468 % 4UL) * 64UL)) + _fuseiter_8617)]);
    }
  }
  return true;
}

static bool mul__649(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2469____itr_2_2470____itr_3_2471 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2469____itr_2_2470____itr_3_2471 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2469____itr_2_2470____itr_3_2471 += 1UL) {
    for (uint64_t _fuseiter_8623 = 0UL; _fuseiter_8623 < 64UL; _fuseiter_8623 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2469____itr_2_2470____itr_3_2471 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2469____itr_2_2470____itr_3_2471 % 4UL) * 64UL)) + _fuseiter_8623)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2469____itr_2_2470____itr_3_2471 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2469____itr_2_2470____itr_3_2471 % 4UL) * 64UL)) + _fuseiter_8623)]);
    }
  }
  return true;
}

static bool mul__648(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2472____itr_2_2473____itr_3_2474 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2472____itr_2_2473____itr_3_2474 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2472____itr_2_2473____itr_3_2474 += 1UL) {
    for (uint64_t _fuseiter_8629 = 0UL; _fuseiter_8629 < 32UL; _fuseiter_8629 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2472____itr_2_2473____itr_3_2474 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2472____itr_2_2473____itr_3_2474 % 8UL) * 32UL)) + _fuseiter_8629)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2472____itr_2_2473____itr_3_2474 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2472____itr_2_2473____itr_3_2474 % 8UL) * 32UL)) + _fuseiter_8629)]);
    }
  }
  return true;
}

static bool mul__647(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2475____itr_2_2476____itr_3_2477 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2475____itr_2_2476____itr_3_2477 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2475____itr_2_2476____itr_3_2477 += 1UL) {
    for (uint64_t _fuseiter_8635 = 0UL; _fuseiter_8635 < 32UL; _fuseiter_8635 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2475____itr_2_2476____itr_3_2477 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2475____itr_2_2476____itr_3_2477 % 8UL) * 32UL)) + _fuseiter_8635)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2475____itr_2_2476____itr_3_2477 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2475____itr_2_2476____itr_3_2477 % 8UL) * 32UL)) + _fuseiter_8635)]);
    }
  }
  return true;
}

static bool mul__644(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2478____itr_2_2479____itr_3_2480 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2478____itr_2_2479____itr_3_2480 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2478____itr_2_2479____itr_3_2480 += 1UL) {
    for (uint64_t _fuseiter_8641 = 0UL; _fuseiter_8641 < 32UL; _fuseiter_8641 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2478____itr_2_2479____itr_3_2480 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2478____itr_2_2479____itr_3_2480 % 8UL) * 32UL)) + _fuseiter_8641)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2478____itr_2_2479____itr_3_2480 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2478____itr_2_2479____itr_3_2480 % 8UL) * 32UL)) + _fuseiter_8641)]);
    }
  }
  return true;
}

static bool mul__643(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2481____itr_2_2482____itr_3_2483 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2481____itr_2_2482____itr_3_2483 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2481____itr_2_2482____itr_3_2483 += 1UL) {
    for (uint64_t _fuseiter_8647 = 0UL; _fuseiter_8647 < 32UL; _fuseiter_8647 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2481____itr_2_2482____itr_3_2483 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2481____itr_2_2482____itr_3_2483 % 8UL) * 32UL)) + _fuseiter_8647)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2481____itr_2_2482____itr_3_2483 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2481____itr_2_2482____itr_3_2483 % 8UL) * 32UL)) + _fuseiter_8647)]);
    }
  }
  return true;
}

static bool mul__642(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_8653 = 0UL; _fuseiter_8653 < 256UL; _fuseiter_8653 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_8653]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_8653]);
  }
  return true;
}

static bool mul__641(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_8659 = 0UL; _fuseiter_8659 < 256UL; _fuseiter_8659 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_8659]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_8659]);
  }
  return true;
}

static bool mul__638(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2490____itr_2_2491____itr_3_2492 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2490____itr_2_2491____itr_3_2492 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2490____itr_2_2491____itr_3_2492 += 1UL) {
    for (uint64_t _fuseiter_8665 = 0UL; _fuseiter_8665 < 64UL; _fuseiter_8665 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2490____itr_2_2491____itr_3_2492 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2490____itr_2_2491____itr_3_2492 % 4UL) * 64UL)) + _fuseiter_8665)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2490____itr_2_2491____itr_3_2492 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2490____itr_2_2491____itr_3_2492 % 4UL) * 64UL)) + _fuseiter_8665)]);
    }
  }
  return true;
}

static bool mul__637(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2493____itr_2_2494____itr_3_2495 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2493____itr_2_2494____itr_3_2495 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2493____itr_2_2494____itr_3_2495 += 1UL) {
    for (uint64_t _fuseiter_8671 = 0UL; _fuseiter_8671 < 64UL; _fuseiter_8671 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2493____itr_2_2494____itr_3_2495 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2493____itr_2_2494____itr_3_2495 % 4UL) * 64UL)) + _fuseiter_8671)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2493____itr_2_2494____itr_3_2495 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2493____itr_2_2494____itr_3_2495 % 4UL) * 64UL)) + _fuseiter_8671)]);
    }
  }
  return true;
}

static bool mul__636(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2496____itr_2_2497____itr_3_2498 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2496____itr_2_2497____itr_3_2498 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2496____itr_2_2497____itr_3_2498 += 1UL) {
    for (uint64_t _fuseiter_8677 = 0UL; _fuseiter_8677 < 64UL; _fuseiter_8677 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2496____itr_2_2497____itr_3_2498 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2496____itr_2_2497____itr_3_2498 % 4UL) * 64UL)) + _fuseiter_8677)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2496____itr_2_2497____itr_3_2498 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2496____itr_2_2497____itr_3_2498 % 4UL) * 64UL)) + _fuseiter_8677)]);
    }
  }
  return true;
}

static bool mul__635(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2499____itr_2_2500____itr_3_2501 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2499____itr_2_2500____itr_3_2501 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2499____itr_2_2500____itr_3_2501 += 1UL) {
    for (uint64_t _fuseiter_8683 = 0UL; _fuseiter_8683 < 64UL; _fuseiter_8683 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2499____itr_2_2500____itr_3_2501 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2499____itr_2_2500____itr_3_2501 % 4UL) * 64UL)) + _fuseiter_8683)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2499____itr_2_2500____itr_3_2501 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2499____itr_2_2500____itr_3_2501 % 4UL) * 64UL)) + _fuseiter_8683)]);
    }
  }
  return true;
}

static bool mul__632(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2502____itr_2_2503____itr_3_2504 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2502____itr_2_2503____itr_3_2504 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2502____itr_2_2503____itr_3_2504 += 1UL) {
    for (uint64_t _fuseiter_8689 = 0UL; _fuseiter_8689 < 32UL; _fuseiter_8689 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2502____itr_2_2503____itr_3_2504 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2502____itr_2_2503____itr_3_2504 % 8UL) * 32UL)) + _fuseiter_8689)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2502____itr_2_2503____itr_3_2504 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2502____itr_2_2503____itr_3_2504 % 8UL) * 32UL)) + _fuseiter_8689)]);
    }
  }
  return true;
}

static bool mul__631(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2505____itr_2_2506____itr_3_2507 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2505____itr_2_2506____itr_3_2507 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2505____itr_2_2506____itr_3_2507 += 1UL) {
    for (uint64_t _fuseiter_8695 = 0UL; _fuseiter_8695 < 32UL; _fuseiter_8695 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2505____itr_2_2506____itr_3_2507 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2505____itr_2_2506____itr_3_2507 % 8UL) * 32UL)) + _fuseiter_8695)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2505____itr_2_2506____itr_3_2507 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2505____itr_2_2506____itr_3_2507 % 8UL) * 32UL)) + _fuseiter_8695)]);
    }
  }
  return true;
}

static bool mul__630(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2508____itr_2_2509____itr_3_2510 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2508____itr_2_2509____itr_3_2510 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2508____itr_2_2509____itr_3_2510 += 1UL) {
    for (uint64_t _fuseiter_8701 = 0UL; _fuseiter_8701 < 32UL; _fuseiter_8701 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2508____itr_2_2509____itr_3_2510 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2508____itr_2_2509____itr_3_2510 % 8UL) * 32UL)) + _fuseiter_8701)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2508____itr_2_2509____itr_3_2510 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2508____itr_2_2509____itr_3_2510 % 8UL) * 32UL)) + _fuseiter_8701)]);
    }
  }
  return true;
}

static bool mul__629(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2511____itr_2_2512____itr_3_2513 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2511____itr_2_2512____itr_3_2513 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2511____itr_2_2512____itr_3_2513 += 1UL) {
    for (uint64_t _fuseiter_8707 = 0UL; _fuseiter_8707 < 32UL; _fuseiter_8707 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2511____itr_2_2512____itr_3_2513 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2511____itr_2_2512____itr_3_2513 % 8UL) * 32UL)) + _fuseiter_8707)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2511____itr_2_2512____itr_3_2513 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2511____itr_2_2512____itr_3_2513 % 8UL) * 32UL)) + _fuseiter_8707)]);
    }
  }
  return true;
}

static bool mul__626(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2514____itr_2_2515____itr_3_2516 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2514____itr_2_2515____itr_3_2516 < 16UL; fused_0fused_0fused_0__itr_0____itr_1_2514____itr_2_2515____itr_3_2516 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0__itr_0____itr_1_2514____itr_2_2515____itr_3_2516 / 16UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2514____itr_2_2515____itr_3_2516 % 16UL) * 16UL))]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[(((fused_0fused_0fused_0__itr_0____itr_1_2514____itr_2_2515____itr_3_2516 / 16UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2514____itr_2_2515____itr_3_2516 % 16UL) * 16UL))]);
  }
  return true;
}

static bool mul__625(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2517____itr_2_2518____itr_3_2519 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2517____itr_2_2518____itr_3_2519 < 16UL; fused_0fused_0fused_0__itr_0____itr_1_2517____itr_2_2518____itr_3_2519 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0__itr_0____itr_1_2517____itr_2_2518____itr_3_2519 / 16UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2517____itr_2_2518____itr_3_2519 % 16UL) * 16UL))]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[(((fused_0fused_0fused_0__itr_0____itr_1_2517____itr_2_2518____itr_3_2519 / 16UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2517____itr_2_2518____itr_3_2519 % 16UL) * 16UL))]);
  }
  return true;
}

static bool mul__624(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2520____itr_2_2521____itr_3_2522 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2520____itr_2_2521____itr_3_2522 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2520____itr_2_2521____itr_3_2522 += 1UL) {
    for (uint64_t _fuseiter_8725 = 0UL; _fuseiter_8725 < 64UL; _fuseiter_8725 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2520____itr_2_2521____itr_3_2522 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2520____itr_2_2521____itr_3_2522 % 4UL) * 64UL)) + _fuseiter_8725)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2520____itr_2_2521____itr_3_2522 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2520____itr_2_2521____itr_3_2522 % 4UL) * 64UL)) + _fuseiter_8725)]);
    }
  }
  return true;
}

static bool mul__623(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2523____itr_2_2524____itr_3_2525 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2523____itr_2_2524____itr_3_2525 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2523____itr_2_2524____itr_3_2525 += 1UL) {
    for (uint64_t _fuseiter_8731 = 0UL; _fuseiter_8731 < 64UL; _fuseiter_8731 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2523____itr_2_2524____itr_3_2525 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2523____itr_2_2524____itr_3_2525 % 4UL) * 64UL)) + _fuseiter_8731)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2523____itr_2_2524____itr_3_2525 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2523____itr_2_2524____itr_3_2525 % 4UL) * 64UL)) + _fuseiter_8731)]);
    }
  }
  return true;
}

static bool mul__620(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2526____itr_2_2527____itr_3_2528 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2526____itr_2_2527____itr_3_2528 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_2526____itr_2_2527____itr_3_2528 += 1UL) {
    for (uint64_t _fuseiter_8737 = 0UL; _fuseiter_8737 < 128UL; _fuseiter_8737 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2526____itr_2_2527____itr_3_2528 / 2UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2526____itr_2_2527____itr_3_2528 % 2UL) * 128UL)) + _fuseiter_8737)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2526____itr_2_2527____itr_3_2528 / 2UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2526____itr_2_2527____itr_3_2528 % 2UL) * 128UL)) + _fuseiter_8737)]);
    }
  }
  return true;
}

static bool mul__619(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2529____itr_2_2530____itr_3_2531 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2529____itr_2_2530____itr_3_2531 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_2529____itr_2_2530____itr_3_2531 += 1UL) {
    for (uint64_t _fuseiter_8743 = 0UL; _fuseiter_8743 < 128UL; _fuseiter_8743 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2529____itr_2_2530____itr_3_2531 / 2UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2529____itr_2_2530____itr_3_2531 % 2UL) * 128UL)) + _fuseiter_8743)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2529____itr_2_2530____itr_3_2531 / 2UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2529____itr_2_2530____itr_3_2531 % 2UL) * 128UL)) + _fuseiter_8743)]);
    }
  }
  return true;
}

static bool mul__618(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_8749 = 0UL; _fuseiter_8749 < 256UL; _fuseiter_8749 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_8749]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_8749]);
  }
  return true;
}

static bool mul__617(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_8755 = 0UL; _fuseiter_8755 < 256UL; _fuseiter_8755 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_8755]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_8755]);
  }
  return true;
}

static bool mul__588(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2538____itr_2_2539____itr_3_2540 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2538____itr_2_2539____itr_3_2540 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2538____itr_2_2539____itr_3_2540 += 1UL) {
    for (uint64_t _fuseiter_8761 = 0UL; _fuseiter_8761 < 64UL; _fuseiter_8761 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2538____itr_2_2539____itr_3_2540 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2538____itr_2_2539____itr_3_2540 % 4UL) * 64UL)) + _fuseiter_8761)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2538____itr_2_2539____itr_3_2540 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2538____itr_2_2539____itr_3_2540 % 4UL) * 64UL)) + _fuseiter_8761)]);
    }
  }
  return true;
}

static bool mul__587(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2541____itr_2_2542____itr_3_2543 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2541____itr_2_2542____itr_3_2543 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2541____itr_2_2542____itr_3_2543 += 1UL) {
    for (uint64_t _fuseiter_8767 = 0UL; _fuseiter_8767 < 64UL; _fuseiter_8767 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2541____itr_2_2542____itr_3_2543 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2541____itr_2_2542____itr_3_2543 % 4UL) * 64UL)) + _fuseiter_8767)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2541____itr_2_2542____itr_3_2543 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2541____itr_2_2542____itr_3_2543 % 4UL) * 64UL)) + _fuseiter_8767)]);
    }
  }
  return true;
}

static bool mul__582(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2544____itr_2_2545____itr_3_2546 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2544____itr_2_2545____itr_3_2546 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2544____itr_2_2545____itr_3_2546 += 1UL) {
    for (uint64_t _fuseiter_8773 = 0UL; _fuseiter_8773 < 32UL; _fuseiter_8773 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2544____itr_2_2545____itr_3_2546 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2544____itr_2_2545____itr_3_2546 % 8UL) * 32UL)) + _fuseiter_8773)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2544____itr_2_2545____itr_3_2546 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2544____itr_2_2545____itr_3_2546 % 8UL) * 32UL)) + _fuseiter_8773)]);
    }
  }
  return true;
}

static bool mul__581(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2547____itr_2_2548____itr_3_2549 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2547____itr_2_2548____itr_3_2549 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2547____itr_2_2548____itr_3_2549 += 1UL) {
    for (uint64_t _fuseiter_8779 = 0UL; _fuseiter_8779 < 32UL; _fuseiter_8779 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2547____itr_2_2548____itr_3_2549 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2547____itr_2_2548____itr_3_2549 % 8UL) * 32UL)) + _fuseiter_8779)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2547____itr_2_2548____itr_3_2549 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2547____itr_2_2548____itr_3_2549 % 8UL) * 32UL)) + _fuseiter_8779)]);
    }
  }
  return true;
}

static bool mul__576(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2550____itr_2_2551____itr_3_2552 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2550____itr_2_2551____itr_3_2552 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2550____itr_2_2551____itr_3_2552 += 1UL) {
    for (uint64_t _fuseiter_8785 = 0UL; _fuseiter_8785 < 64UL; _fuseiter_8785 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2550____itr_2_2551____itr_3_2552 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2550____itr_2_2551____itr_3_2552 % 4UL) * 64UL)) + _fuseiter_8785)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2550____itr_2_2551____itr_3_2552 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2550____itr_2_2551____itr_3_2552 % 4UL) * 64UL)) + _fuseiter_8785)]);
    }
  }
  return true;
}

static bool mul__575(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2553____itr_2_2554____itr_3_2555 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2553____itr_2_2554____itr_3_2555 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2553____itr_2_2554____itr_3_2555 += 1UL) {
    for (uint64_t _fuseiter_8791 = 0UL; _fuseiter_8791 < 64UL; _fuseiter_8791 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2553____itr_2_2554____itr_3_2555 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2553____itr_2_2554____itr_3_2555 % 4UL) * 64UL)) + _fuseiter_8791)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2553____itr_2_2554____itr_3_2555 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2553____itr_2_2554____itr_3_2555 % 4UL) * 64UL)) + _fuseiter_8791)]);
    }
  }
  return true;
}

static bool mul__570(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2556____itr_2_2557____itr_3_2558 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2556____itr_2_2557____itr_3_2558 < 16UL; fused_0fused_0fused_0__itr_0____itr_1_2556____itr_2_2557____itr_3_2558 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0__itr_0____itr_1_2556____itr_2_2557____itr_3_2558 / 16UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2556____itr_2_2557____itr_3_2558 % 16UL) * 16UL))]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[(((fused_0fused_0fused_0__itr_0____itr_1_2556____itr_2_2557____itr_3_2558 / 16UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2556____itr_2_2557____itr_3_2558 % 16UL) * 16UL))]);
  }
  return true;
}

static bool mul__569(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2559____itr_2_2560____itr_3_2561 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2559____itr_2_2560____itr_3_2561 < 16UL; fused_0fused_0fused_0__itr_0____itr_1_2559____itr_2_2560____itr_3_2561 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0__itr_0____itr_1_2559____itr_2_2560____itr_3_2561 / 16UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2559____itr_2_2560____itr_3_2561 % 16UL) * 16UL))]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[(((fused_0fused_0fused_0__itr_0____itr_1_2559____itr_2_2560____itr_3_2561 / 16UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2559____itr_2_2560____itr_3_2561 % 16UL) * 16UL))]);
  }
  return true;
}

static bool reorder__522(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_8805___fuseiter_8806_2562 = 0UL; fused_0_fuseiter_8805___fuseiter_8806_2562 < 16UL; fused_0_fuseiter_8805___fuseiter_8806_2562 += 1UL) {
    for (uint64_t _fuseiter_8809 = 0UL; _fuseiter_8809 < 128UL; _fuseiter_8809 += 1UL) {
      for (uint64_t _fuseiter_8810 = 0UL; _fuseiter_8810 < 32UL; _fuseiter_8810 += 1UL) {
        for (uint64_t _fuseiter_8811 = 0UL; _fuseiter_8811 < 4UL; _fuseiter_8811 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_8810 + ((fused_0_fuseiter_8805___fuseiter_8806_2562 / 2UL) * 32UL)) * 1024UL) + ((_fuseiter_8811 + (_fuseiter_8809 * 4UL)) + ((fused_0_fuseiter_8805___fuseiter_8806_2562 % 2UL) * 512UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_8805___fuseiter_8806_2562 / 2UL) * 32768UL) + (((fused_0_fuseiter_8805___fuseiter_8806_2562 % 2UL) * 16384UL) + ((_fuseiter_8809 * 128UL) + ((_fuseiter_8810 * 4UL) + _fuseiter_8811))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool reorder__515(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_8812___fuseiter_8813_2563___fuseiter_8814_2564___fuseiter_8815_2565___fuseiter_8816_2566 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_8812___fuseiter_8813_2563___fuseiter_8814_2564___fuseiter_8815_2565___fuseiter_8816_2566 < 256UL; fused_0fused_0fused_0fused_0_fuseiter_8812___fuseiter_8813_2563___fuseiter_8814_2564___fuseiter_8815_2565___fuseiter_8816_2566 += 1UL) {
    for (uint64_t _fuseiter_8817 = 0UL; _fuseiter_8817 < 256UL; _fuseiter_8817 += 1UL) {
      for (uint64_t _fuseiter_8818 = 0UL; _fuseiter_8818 < 4UL; _fuseiter_8818 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_8817 + ((fused_0fused_0fused_0fused_0_fuseiter_8812___fuseiter_8813_2563___fuseiter_8814_2564___fuseiter_8815_2565___fuseiter_8816_2566 / 256UL) * 256UL)) * 1024UL) + ((_fuseiter_8818 + ((fused_0fused_0fused_0fused_0_fuseiter_8812___fuseiter_8813_2563___fuseiter_8814_2564___fuseiter_8815_2565___fuseiter_8816_2566 % 32UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_8812___fuseiter_8813_2563___fuseiter_8814_2564___fuseiter_8815_2565___fuseiter_8816_2566 / 32UL) % 8UL) * 128UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_8812___fuseiter_8813_2563___fuseiter_8814_2564___fuseiter_8815_2565___fuseiter_8816_2566 / 256UL) * 262144UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_8812___fuseiter_8813_2563___fuseiter_8814_2564___fuseiter_8815_2565___fuseiter_8816_2566 / 32UL) % 8UL) * 32768UL) + (((fused_0fused_0fused_0fused_0_fuseiter_8812___fuseiter_8813_2563___fuseiter_8814_2564___fuseiter_8815_2565___fuseiter_8816_2566 % 32UL) * 1024UL) + ((_fuseiter_8817 * 4UL) + _fuseiter_8818))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__506(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_8819___fuseiter_8820_2567 = 0UL; fused_0_fuseiter_8819___fuseiter_8820_2567 < 32UL; fused_0_fuseiter_8819___fuseiter_8820_2567 += 1UL) {
    for (uint64_t _fuseiter_8823 = 0UL; _fuseiter_8823 < 32UL; _fuseiter_8823 += 1UL) {
      for (uint64_t _fuseiter_8824 = 0UL; _fuseiter_8824 < 64UL; _fuseiter_8824 += 1UL) {
        for (uint64_t _fuseiter_8825 = 0UL; _fuseiter_8825 < 4UL; _fuseiter_8825 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_8824 + ((fused_0_fuseiter_8819___fuseiter_8820_2567 / 8UL) * 64UL)) * 1024UL) + ((_fuseiter_8825 + (_fuseiter_8823 * 4UL)) + ((fused_0_fuseiter_8819___fuseiter_8820_2567 % 8UL) * 128UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_8819___fuseiter_8820_2567 / 8UL) * 65536UL) + (((fused_0_fuseiter_8819___fuseiter_8820_2567 % 8UL) * 8192UL) + ((_fuseiter_8823 * 256UL) + ((_fuseiter_8824 * 4UL) + _fuseiter_8825))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool reorder__497(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_8826___fuseiter_8827_2568 = 0UL; fused_0_fuseiter_8826___fuseiter_8827_2568 < 64UL; fused_0_fuseiter_8826___fuseiter_8827_2568 += 1UL) {
    for (uint64_t _fuseiter_8830 = 0UL; _fuseiter_8830 < 32UL; _fuseiter_8830 += 1UL) {
      for (uint64_t _fuseiter_8831 = 0UL; _fuseiter_8831 < 32UL; _fuseiter_8831 += 1UL) {
        for (uint64_t _fuseiter_8832 = 0UL; _fuseiter_8832 < 4UL; _fuseiter_8832 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_8831 + ((fused_0_fuseiter_8826___fuseiter_8827_2568 / 8UL) * 32UL)) * 1024UL) + ((_fuseiter_8832 + (_fuseiter_8830 * 4UL)) + ((fused_0_fuseiter_8826___fuseiter_8827_2568 % 8UL) * 128UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_8826___fuseiter_8827_2568 / 8UL) * 32768UL) + (((fused_0_fuseiter_8826___fuseiter_8827_2568 % 8UL) * 4096UL) + ((_fuseiter_8830 * 128UL) + ((_fuseiter_8831 * 4UL) + _fuseiter_8832))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool reorder__490(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_8833___fuseiter_8834_2569___fuseiter_8835_2570___fuseiter_8836_2571___fuseiter_8837_2572 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_8833___fuseiter_8834_2569___fuseiter_8835_2570___fuseiter_8836_2571___fuseiter_8837_2572 < 1024UL; fused_0fused_0fused_0fused_0_fuseiter_8833___fuseiter_8834_2569___fuseiter_8835_2570___fuseiter_8836_2571___fuseiter_8837_2572 += 1UL) {
    for (uint64_t _fuseiter_8838 = 0UL; _fuseiter_8838 < 64UL; _fuseiter_8838 += 1UL) {
      for (uint64_t _fuseiter_8839 = 0UL; _fuseiter_8839 < 4UL; _fuseiter_8839 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_8838 + ((fused_0fused_0fused_0fused_0_fuseiter_8833___fuseiter_8834_2569___fuseiter_8835_2570___fuseiter_8836_2571___fuseiter_8837_2572 / 256UL) * 64UL)) * 1024UL) + (_fuseiter_8839 + ((fused_0fused_0fused_0fused_0_fuseiter_8833___fuseiter_8834_2569___fuseiter_8835_2570___fuseiter_8836_2571___fuseiter_8837_2572 % 256UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_8833___fuseiter_8834_2569___fuseiter_8835_2570___fuseiter_8836_2571___fuseiter_8837_2572 / 256UL) * 65536UL) + (((fused_0fused_0fused_0fused_0_fuseiter_8833___fuseiter_8834_2569___fuseiter_8835_2570___fuseiter_8836_2571___fuseiter_8837_2572 % 256UL) * 256UL) + ((_fuseiter_8838 * 4UL) + _fuseiter_8839)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__528(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_8840___fuseiter_8841_2573___fuseiter_8842_2574___fuseiter_8843_2575___fuseiter_8844_2576 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_8840___fuseiter_8841_2573___fuseiter_8842_2574___fuseiter_8843_2575___fuseiter_8844_2576 < 256UL; fused_0fused_0fused_0fused_0_fuseiter_8840___fuseiter_8841_2573___fuseiter_8842_2574___fuseiter_8843_2575___fuseiter_8844_2576 += 1UL) {
    for (uint64_t _fuseiter_8845 = 0UL; _fuseiter_8845 < 256UL; _fuseiter_8845 += 1UL) {
      for (uint64_t _fuseiter_8846 = 0UL; _fuseiter_8846 < 4UL; _fuseiter_8846 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_8845 + ((fused_0fused_0fused_0fused_0_fuseiter_8840___fuseiter_8841_2573___fuseiter_8842_2574___fuseiter_8843_2575___fuseiter_8844_2576 / 64UL) * 256UL)) * 256UL) + ((_fuseiter_8846 + ((fused_0fused_0fused_0fused_0_fuseiter_8840___fuseiter_8841_2573___fuseiter_8842_2574___fuseiter_8843_2575___fuseiter_8844_2576 % 32UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_8840___fuseiter_8841_2573___fuseiter_8842_2574___fuseiter_8843_2575___fuseiter_8844_2576 / 32UL) % 2UL) * 128UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_8840___fuseiter_8841_2573___fuseiter_8842_2574___fuseiter_8843_2575___fuseiter_8844_2576 / 64UL) * 65536UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_8840___fuseiter_8841_2573___fuseiter_8842_2574___fuseiter_8843_2575___fuseiter_8844_2576 / 32UL) % 2UL) * 32768UL) + (((fused_0fused_0fused_0fused_0_fuseiter_8840___fuseiter_8841_2573___fuseiter_8842_2574___fuseiter_8843_2575___fuseiter_8844_2576 % 32UL) * 1024UL) + ((_fuseiter_8845 * 4UL) + _fuseiter_8846))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__519(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_8847___fuseiter_8848_2577___fuseiter_8849_2578___fuseiter_8850_2579___fuseiter_8851_2580 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_8847___fuseiter_8848_2577___fuseiter_8849_2578___fuseiter_8850_2579___fuseiter_8851_2580 < 512UL; fused_0fused_0fused_0fused_0_fuseiter_8847___fuseiter_8848_2577___fuseiter_8849_2578___fuseiter_8850_2579___fuseiter_8851_2580 += 1UL) {
    for (uint64_t _fuseiter_8852 = 0UL; _fuseiter_8852 < 128UL; _fuseiter_8852 += 1UL) {
      for (uint64_t _fuseiter_8853 = 0UL; _fuseiter_8853 < 4UL; _fuseiter_8853 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_8852 + ((fused_0fused_0fused_0fused_0_fuseiter_8847___fuseiter_8848_2577___fuseiter_8849_2578___fuseiter_8850_2579___fuseiter_8851_2580 / 64UL) * 128UL)) * 256UL) + (_fuseiter_8853 + ((fused_0fused_0fused_0fused_0_fuseiter_8847___fuseiter_8848_2577___fuseiter_8849_2578___fuseiter_8850_2579___fuseiter_8851_2580 % 64UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_8847___fuseiter_8848_2577___fuseiter_8849_2578___fuseiter_8850_2579___fuseiter_8851_2580 / 64UL) * 32768UL) + (((fused_0fused_0fused_0fused_0_fuseiter_8847___fuseiter_8848_2577___fuseiter_8849_2578___fuseiter_8850_2579___fuseiter_8851_2580 % 64UL) * 512UL) + ((_fuseiter_8852 * 4UL) + _fuseiter_8853)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__512(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_8854___fuseiter_8855_2581 = 0UL; fused_0_fuseiter_8854___fuseiter_8855_2581 < 16UL; fused_0_fuseiter_8854___fuseiter_8855_2581 += 1UL) {
    for (uint64_t _fuseiter_8858 = 0UL; _fuseiter_8858 < 32UL; _fuseiter_8858 += 1UL) {
      for (uint64_t _fuseiter_8859 = 0UL; _fuseiter_8859 < 128UL; _fuseiter_8859 += 1UL) {
        for (uint64_t _fuseiter_8860 = 0UL; _fuseiter_8860 < 4UL; _fuseiter_8860 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_8859 + ((fused_0_fuseiter_8854___fuseiter_8855_2581 / 2UL) * 128UL)) * 256UL) + ((_fuseiter_8860 + (_fuseiter_8858 * 4UL)) + ((fused_0_fuseiter_8854___fuseiter_8855_2581 % 2UL) * 128UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_8854___fuseiter_8855_2581 / 2UL) * 32768UL) + (((fused_0_fuseiter_8854___fuseiter_8855_2581 % 2UL) * 16384UL) + ((_fuseiter_8858 * 512UL) + ((_fuseiter_8859 * 4UL) + _fuseiter_8860))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool reorder__503(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_8861___fuseiter_8862_2582___fuseiter_8863_2583___fuseiter_8864_2584___fuseiter_8865_2585 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_8861___fuseiter_8862_2582___fuseiter_8863_2583___fuseiter_8864_2584___fuseiter_8865_2585 < 512UL; fused_0fused_0fused_0fused_0_fuseiter_8861___fuseiter_8862_2582___fuseiter_8863_2583___fuseiter_8864_2584___fuseiter_8865_2585 += 1UL) {
    for (uint64_t _fuseiter_8866 = 0UL; _fuseiter_8866 < 128UL; _fuseiter_8866 += 1UL) {
      for (uint64_t _fuseiter_8867 = 0UL; _fuseiter_8867 < 4UL; _fuseiter_8867 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_8866 + ((fused_0fused_0fused_0fused_0_fuseiter_8861___fuseiter_8862_2582___fuseiter_8863_2583___fuseiter_8864_2584___fuseiter_8865_2585 / 64UL) * 128UL)) * 256UL) + (_fuseiter_8867 + ((fused_0fused_0fused_0fused_0_fuseiter_8861___fuseiter_8862_2582___fuseiter_8863_2583___fuseiter_8864_2584___fuseiter_8865_2585 % 64UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_8861___fuseiter_8862_2582___fuseiter_8863_2583___fuseiter_8864_2584___fuseiter_8865_2585 / 64UL) * 32768UL) + (((fused_0fused_0fused_0fused_0_fuseiter_8861___fuseiter_8862_2582___fuseiter_8863_2583___fuseiter_8864_2584___fuseiter_8865_2585 % 64UL) * 512UL) + ((_fuseiter_8866 * 4UL) + _fuseiter_8867)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__496(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_8868___fuseiter_8869_2586___fuseiter_8870_2587___fuseiter_8871_2588___fuseiter_8872_2589 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_8868___fuseiter_8869_2586___fuseiter_8870_2587___fuseiter_8871_2588___fuseiter_8872_2589 < 64UL; fused_0fused_0fused_0fused_0_fuseiter_8868___fuseiter_8869_2586___fuseiter_8870_2587___fuseiter_8871_2588___fuseiter_8872_2589 += 1UL) {
    for (uint64_t _fuseiter_8873 = 0UL; _fuseiter_8873 < 1024UL; _fuseiter_8873 += 1UL) {
      for (uint64_t _fuseiter_8874 = 0UL; _fuseiter_8874 < 4UL; _fuseiter_8874 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_8873 + ((fused_0fused_0fused_0fused_0_fuseiter_8868___fuseiter_8869_2586___fuseiter_8870_2587___fuseiter_8871_2588___fuseiter_8872_2589 / 64UL) * 1024UL)) * 256UL) + ((_fuseiter_8874 + ((fused_0fused_0fused_0fused_0_fuseiter_8868___fuseiter_8869_2586___fuseiter_8870_2587___fuseiter_8871_2588___fuseiter_8872_2589 % 16UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_8868___fuseiter_8869_2586___fuseiter_8870_2587___fuseiter_8871_2588___fuseiter_8872_2589 / 16UL) % 4UL) * 64UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_8868___fuseiter_8869_2586___fuseiter_8870_2587___fuseiter_8871_2588___fuseiter_8872_2589 / 64UL) * 262144UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_8868___fuseiter_8869_2586___fuseiter_8870_2587___fuseiter_8871_2588___fuseiter_8872_2589 / 16UL) % 4UL) * 65536UL) + (((fused_0fused_0fused_0fused_0_fuseiter_8868___fuseiter_8869_2586___fuseiter_8870_2587___fuseiter_8871_2588___fuseiter_8872_2589 % 16UL) * 4096UL) + ((_fuseiter_8873 * 4UL) + _fuseiter_8874))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__487(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_8875___fuseiter_8876_2590___fuseiter_8877_2591___fuseiter_8878_2592___fuseiter_8879_2593 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_8875___fuseiter_8876_2590___fuseiter_8877_2591___fuseiter_8878_2592___fuseiter_8879_2593 < 512UL; fused_0fused_0fused_0fused_0_fuseiter_8875___fuseiter_8876_2590___fuseiter_8877_2591___fuseiter_8878_2592___fuseiter_8879_2593 += 1UL) {
    for (uint64_t _fuseiter_8880 = 0UL; _fuseiter_8880 < 128UL; _fuseiter_8880 += 1UL) {
      for (uint64_t _fuseiter_8881 = 0UL; _fuseiter_8881 < 4UL; _fuseiter_8881 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_8880 + ((fused_0fused_0fused_0fused_0_fuseiter_8875___fuseiter_8876_2590___fuseiter_8877_2591___fuseiter_8878_2592___fuseiter_8879_2593 / 64UL) * 128UL)) * 256UL) + (_fuseiter_8881 + ((fused_0fused_0fused_0fused_0_fuseiter_8875___fuseiter_8876_2590___fuseiter_8877_2591___fuseiter_8878_2592___fuseiter_8879_2593 % 64UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_8875___fuseiter_8876_2590___fuseiter_8877_2591___fuseiter_8878_2592___fuseiter_8879_2593 / 64UL) * 32768UL) + (((fused_0fused_0fused_0fused_0_fuseiter_8875___fuseiter_8876_2590___fuseiter_8877_2591___fuseiter_8878_2592___fuseiter_8879_2593 % 64UL) * 512UL) + ((_fuseiter_8880 * 4UL) + _fuseiter_8881)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__422(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_8882___fuseiter_8883_2594___fuseiter_8884_2595___fuseiter_8885_2596___fuseiter_8886_2597 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_8882___fuseiter_8883_2594___fuseiter_8884_2595___fuseiter_8885_2596___fuseiter_8886_2597 < 16UL; fused_0fused_0fused_0fused_0_fuseiter_8882___fuseiter_8883_2594___fuseiter_8884_2595___fuseiter_8885_2596___fuseiter_8886_2597 += 1UL) {
    for (uint64_t _fuseiter_8887 = 0UL; _fuseiter_8887 < 64UL; _fuseiter_8887 += 1UL) {
      for (uint64_t _fuseiter_8888 = 0UL; _fuseiter_8888 < 4UL; _fuseiter_8888 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_8887 + ((fused_0fused_0fused_0fused_0_fuseiter_8882___fuseiter_8883_2594___fuseiter_8884_2595___fuseiter_8885_2596___fuseiter_8886_2597 / 16UL) * 64UL)) * 64UL) + (_fuseiter_8888 + ((fused_0fused_0fused_0fused_0_fuseiter_8882___fuseiter_8883_2594___fuseiter_8884_2595___fuseiter_8885_2596___fuseiter_8886_2597 % 16UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_8882___fuseiter_8883_2594___fuseiter_8884_2595___fuseiter_8885_2596___fuseiter_8886_2597 / 16UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_8882___fuseiter_8883_2594___fuseiter_8884_2595___fuseiter_8885_2596___fuseiter_8886_2597 % 16UL) * 256UL) + ((_fuseiter_8887 * 4UL) + _fuseiter_8888)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__612(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2598____itr_2_2599____itr_3_2600 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2598____itr_2_2599____itr_3_2600 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2598____itr_2_2599____itr_3_2600 += 1UL) {
    for (uint64_t _fuseiter_8893 = 0UL; _fuseiter_8893 < 32UL; _fuseiter_8893 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2598____itr_2_2599____itr_3_2600 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2598____itr_2_2599____itr_3_2600 % 4UL) * 32UL)) + _fuseiter_8893)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2598____itr_2_2599____itr_3_2600 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2598____itr_2_2599____itr_3_2600 % 4UL) * 32UL)) + _fuseiter_8893)]);
    }
  }
  return true;
}

static bool mul__611(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2601____itr_2_2602____itr_3_2603 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2601____itr_2_2602____itr_3_2603 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2601____itr_2_2602____itr_3_2603 += 1UL) {
    for (uint64_t _fuseiter_8899 = 0UL; _fuseiter_8899 < 32UL; _fuseiter_8899 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2601____itr_2_2602____itr_3_2603 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2601____itr_2_2602____itr_3_2603 % 4UL) * 32UL)) + _fuseiter_8899)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2601____itr_2_2602____itr_3_2603 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2601____itr_2_2602____itr_3_2603 % 4UL) * 32UL)) + _fuseiter_8899)]);
    }
  }
  return true;
}

static bool mul__610(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2604____itr_2_2605____itr_3_2606 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2604____itr_2_2605____itr_3_2606 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_2604____itr_2_2605____itr_3_2606 += 1UL) {
    for (uint64_t _fuseiter_8905 = 0UL; _fuseiter_8905 < 64UL; _fuseiter_8905 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2604____itr_2_2605____itr_3_2606 / 2UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2604____itr_2_2605____itr_3_2606 % 2UL) * 64UL)) + _fuseiter_8905)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2604____itr_2_2605____itr_3_2606 / 2UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2604____itr_2_2605____itr_3_2606 % 2UL) * 64UL)) + _fuseiter_8905)]);
    }
  }
  return true;
}

static bool mul__609(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2607____itr_2_2608____itr_3_2609 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2607____itr_2_2608____itr_3_2609 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_2607____itr_2_2608____itr_3_2609 += 1UL) {
    for (uint64_t _fuseiter_8911 = 0UL; _fuseiter_8911 < 64UL; _fuseiter_8911 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2607____itr_2_2608____itr_3_2609 / 2UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2607____itr_2_2608____itr_3_2609 % 2UL) * 64UL)) + _fuseiter_8911)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2607____itr_2_2608____itr_3_2609 / 2UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2607____itr_2_2608____itr_3_2609 % 2UL) * 64UL)) + _fuseiter_8911)]);
    }
  }
  return true;
}

static bool mul__606(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2610____itr_2_2611____itr_3_2612 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2610____itr_2_2611____itr_3_2612 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2610____itr_2_2611____itr_3_2612 += 1UL) {
    for (uint64_t _fuseiter_8917 = 0UL; _fuseiter_8917 < 32UL; _fuseiter_8917 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2610____itr_2_2611____itr_3_2612 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2610____itr_2_2611____itr_3_2612 % 4UL) * 32UL)) + _fuseiter_8917)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2610____itr_2_2611____itr_3_2612 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2610____itr_2_2611____itr_3_2612 % 4UL) * 32UL)) + _fuseiter_8917)]);
    }
  }
  return true;
}

static bool mul__605(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2613____itr_2_2614____itr_3_2615 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2613____itr_2_2614____itr_3_2615 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2613____itr_2_2614____itr_3_2615 += 1UL) {
    for (uint64_t _fuseiter_8923 = 0UL; _fuseiter_8923 < 32UL; _fuseiter_8923 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2613____itr_2_2614____itr_3_2615 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2613____itr_2_2614____itr_3_2615 % 4UL) * 32UL)) + _fuseiter_8923)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2613____itr_2_2614____itr_3_2615 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2613____itr_2_2614____itr_3_2615 % 4UL) * 32UL)) + _fuseiter_8923)]);
    }
  }
  return true;
}

static bool mul__604(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_8929 = 0UL; _fuseiter_8929 < 128UL; _fuseiter_8929 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_8929]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_8929]);
  }
  return true;
}

static bool mul__603(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_8935 = 0UL; _fuseiter_8935 < 128UL; _fuseiter_8935 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_8935]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_8935]);
  }
  return true;
}

static bool mul__600(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_8941 = 0UL; _fuseiter_8941 < 128UL; _fuseiter_8941 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_8941]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_8941]);
  }
  return true;
}

static bool mul__599(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_8947 = 0UL; _fuseiter_8947 < 128UL; _fuseiter_8947 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_8947]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_8947]);
  }
  return true;
}

static bool mul__598(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2628____itr_2_2629____itr_3_2630 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2628____itr_2_2629____itr_3_2630 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2628____itr_2_2629____itr_3_2630 += 1UL) {
    for (uint64_t _fuseiter_8953 = 0UL; _fuseiter_8953 < 32UL; _fuseiter_8953 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2628____itr_2_2629____itr_3_2630 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2628____itr_2_2629____itr_3_2630 % 4UL) * 32UL)) + _fuseiter_8953)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2628____itr_2_2629____itr_3_2630 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2628____itr_2_2629____itr_3_2630 % 4UL) * 32UL)) + _fuseiter_8953)]);
    }
  }
  return true;
}

static bool mul__597(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2631____itr_2_2632____itr_3_2633 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2631____itr_2_2632____itr_3_2633 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2631____itr_2_2632____itr_3_2633 += 1UL) {
    for (uint64_t _fuseiter_8959 = 0UL; _fuseiter_8959 < 32UL; _fuseiter_8959 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2631____itr_2_2632____itr_3_2633 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2631____itr_2_2632____itr_3_2633 % 4UL) * 32UL)) + _fuseiter_8959)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2631____itr_2_2632____itr_3_2633 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2631____itr_2_2632____itr_3_2633 % 4UL) * 32UL)) + _fuseiter_8959)]);
    }
  }
  return true;
}

static bool mul__594(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2634____itr_2_2635____itr_3_2636 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2634____itr_2_2635____itr_3_2636 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2634____itr_2_2635____itr_3_2636 += 1UL) {
    for (uint64_t _fuseiter_8965 = 0UL; _fuseiter_8965 < 32UL; _fuseiter_8965 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2634____itr_2_2635____itr_3_2636 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2634____itr_2_2635____itr_3_2636 % 4UL) * 32UL)) + _fuseiter_8965)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2634____itr_2_2635____itr_3_2636 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2634____itr_2_2635____itr_3_2636 % 4UL) * 32UL)) + _fuseiter_8965)]);
    }
  }
  return true;
}

static bool mul__593(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2637____itr_2_2638____itr_3_2639 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2637____itr_2_2638____itr_3_2639 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2637____itr_2_2638____itr_3_2639 += 1UL) {
    for (uint64_t _fuseiter_8971 = 0UL; _fuseiter_8971 < 32UL; _fuseiter_8971 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2637____itr_2_2638____itr_3_2639 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2637____itr_2_2638____itr_3_2639 % 4UL) * 32UL)) + _fuseiter_8971)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2637____itr_2_2638____itr_3_2639 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2637____itr_2_2638____itr_3_2639 % 4UL) * 32UL)) + _fuseiter_8971)]);
    }
  }
  return true;
}

static bool mul__592(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2640____itr_2_2641____itr_3_2642 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2640____itr_2_2641____itr_3_2642 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_2640____itr_2_2641____itr_3_2642 += 1UL) {
    for (uint64_t _fuseiter_8977 = 0UL; _fuseiter_8977 < 64UL; _fuseiter_8977 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2640____itr_2_2641____itr_3_2642 / 2UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2640____itr_2_2641____itr_3_2642 % 2UL) * 64UL)) + _fuseiter_8977)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2640____itr_2_2641____itr_3_2642 / 2UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2640____itr_2_2641____itr_3_2642 % 2UL) * 64UL)) + _fuseiter_8977)]);
    }
  }
  return true;
}

static bool mul__591(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2643____itr_2_2644____itr_3_2645 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2643____itr_2_2644____itr_3_2645 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_2643____itr_2_2644____itr_3_2645 += 1UL) {
    for (uint64_t _fuseiter_8983 = 0UL; _fuseiter_8983 < 64UL; _fuseiter_8983 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2643____itr_2_2644____itr_3_2645 / 2UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2643____itr_2_2644____itr_3_2645 % 2UL) * 64UL)) + _fuseiter_8983)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2643____itr_2_2644____itr_3_2645 / 2UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2643____itr_2_2644____itr_3_2645 % 2UL) * 64UL)) + _fuseiter_8983)]);
    }
  }
  return true;
}

static bool mul__586(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2646____itr_2_2647____itr_3_2648 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2646____itr_2_2647____itr_3_2648 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_2646____itr_2_2647____itr_3_2648 += 1UL) {
    for (uint64_t _fuseiter_8989 = 0UL; _fuseiter_8989 < 32UL; _fuseiter_8989 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2646____itr_2_2647____itr_3_2648 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2646____itr_2_2647____itr_3_2648 % 2UL) * 32UL)) + _fuseiter_8989)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2646____itr_2_2647____itr_3_2648 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2646____itr_2_2647____itr_3_2648 % 2UL) * 32UL)) + _fuseiter_8989)]);
    }
  }
  return true;
}

static bool mul__585(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2649____itr_2_2650____itr_3_2651 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2649____itr_2_2650____itr_3_2651 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_2649____itr_2_2650____itr_3_2651 += 1UL) {
    for (uint64_t _fuseiter_8995 = 0UL; _fuseiter_8995 < 32UL; _fuseiter_8995 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2649____itr_2_2650____itr_3_2651 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2649____itr_2_2650____itr_3_2651 % 2UL) * 32UL)) + _fuseiter_8995)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2649____itr_2_2650____itr_3_2651 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2649____itr_2_2650____itr_3_2651 % 2UL) * 32UL)) + _fuseiter_8995)]);
    }
  }
  return true;
}

static bool mul__584(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2652____itr_2_2653____itr_3_2654 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2652____itr_2_2653____itr_3_2654 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_2652____itr_2_2653____itr_3_2654 += 1UL) {
    for (uint64_t _fuseiter_9001 = 0UL; _fuseiter_9001 < 32UL; _fuseiter_9001 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2652____itr_2_2653____itr_3_2654 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2652____itr_2_2653____itr_3_2654 % 2UL) * 32UL)) + _fuseiter_9001)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2652____itr_2_2653____itr_3_2654 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2652____itr_2_2653____itr_3_2654 % 2UL) * 32UL)) + _fuseiter_9001)]);
    }
  }
  return true;
}

static bool mul__583(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2655____itr_2_2656____itr_3_2657 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2655____itr_2_2656____itr_3_2657 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_2655____itr_2_2656____itr_3_2657 += 1UL) {
    for (uint64_t _fuseiter_9007 = 0UL; _fuseiter_9007 < 32UL; _fuseiter_9007 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2655____itr_2_2656____itr_3_2657 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2655____itr_2_2656____itr_3_2657 % 2UL) * 32UL)) + _fuseiter_9007)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2655____itr_2_2656____itr_3_2657 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2655____itr_2_2656____itr_3_2657 % 2UL) * 32UL)) + _fuseiter_9007)]);
    }
  }
  return true;
}

static bool mul__580(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2658____itr_2_2659____itr_3_2660 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2658____itr_2_2659____itr_3_2660 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2658____itr_2_2659____itr_3_2660 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0__itr_0____itr_1_2658____itr_2_2659____itr_3_2660 / 4UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2658____itr_2_2659____itr_3_2660 % 4UL) * 16UL))]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[(((fused_0fused_0fused_0__itr_0____itr_1_2658____itr_2_2659____itr_3_2660 / 4UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2658____itr_2_2659____itr_3_2660 % 4UL) * 16UL))]);
  }
  return true;
}

static bool mul__579(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2661____itr_2_2662____itr_3_2663 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2661____itr_2_2662____itr_3_2663 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2661____itr_2_2662____itr_3_2663 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0__itr_0____itr_1_2661____itr_2_2662____itr_3_2663 / 4UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2661____itr_2_2662____itr_3_2663 % 4UL) * 16UL))]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[(((fused_0fused_0fused_0__itr_0____itr_1_2661____itr_2_2662____itr_3_2663 / 4UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2661____itr_2_2662____itr_3_2663 % 4UL) * 16UL))]);
  }
  return true;
}

static bool mul__578(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_9025 = 0UL; _fuseiter_9025 < 64UL; _fuseiter_9025 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_9025]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_9025]);
  }
  return true;
}

static bool mul__577(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_9031 = 0UL; _fuseiter_9031 < 64UL; _fuseiter_9031 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_9031]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_9031]);
  }
  return true;
}

static bool mul__574(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2670____itr_2_2671____itr_3_2672 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2670____itr_2_2671____itr_3_2672 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_2670____itr_2_2671____itr_3_2672 += 1UL) {
    for (uint64_t _fuseiter_9037 = 0UL; _fuseiter_9037 < 32UL; _fuseiter_9037 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2670____itr_2_2671____itr_3_2672 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2670____itr_2_2671____itr_3_2672 % 2UL) * 32UL)) + _fuseiter_9037)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2670____itr_2_2671____itr_3_2672 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2670____itr_2_2671____itr_3_2672 % 2UL) * 32UL)) + _fuseiter_9037)]);
    }
  }
  return true;
}

static bool mul__573(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2673____itr_2_2674____itr_3_2675 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2673____itr_2_2674____itr_3_2675 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_2673____itr_2_2674____itr_3_2675 += 1UL) {
    for (uint64_t _fuseiter_9043 = 0UL; _fuseiter_9043 < 32UL; _fuseiter_9043 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2673____itr_2_2674____itr_3_2675 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2673____itr_2_2674____itr_3_2675 % 2UL) * 32UL)) + _fuseiter_9043)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2673____itr_2_2674____itr_3_2675 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2673____itr_2_2674____itr_3_2675 % 2UL) * 32UL)) + _fuseiter_9043)]);
    }
  }
  return true;
}

static bool mul__572(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_9049 = 0UL; _fuseiter_9049 < 64UL; _fuseiter_9049 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_9049]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_9049]);
  }
  return true;
}

static bool mul__571(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_9055 = 0UL; _fuseiter_9055 < 64UL; _fuseiter_9055 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_9055]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_9055]);
  }
  return true;
}

static bool reorder__516(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_9057___fuseiter_9058_2682 = 0UL; fused_0_fuseiter_9057___fuseiter_9058_2682 < 16UL; fused_0_fuseiter_9057___fuseiter_9058_2682 += 1UL) {
    for (uint64_t _fuseiter_9059 = 0UL; _fuseiter_9059 < 3UL; _fuseiter_9059 += 1UL) {
      for (uint64_t _fuseiter_9060 = 0UL; _fuseiter_9060 < 3UL; _fuseiter_9060 += 1UL) {
        for (uint64_t _fuseiter_9061 = 0UL; _fuseiter_9061 < 32UL; _fuseiter_9061 += 1UL) {
          for (uint64_t _fuseiter_9062 = 0UL; _fuseiter_9062 < 32UL; _fuseiter_9062 += 1UL) {
            for (uint64_t _fuseiter_9063 = 0UL; _fuseiter_9063 < 4UL; _fuseiter_9063 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_9062 + ((fused_0_fuseiter_9057___fuseiter_9058_2682 / 2UL) * 32UL)) * 2304UL) + ((((_fuseiter_9063 + (_fuseiter_9061 * 4UL)) + ((fused_0_fuseiter_9057___fuseiter_9058_2682 % 2UL) * 128UL)) * 9UL) + ((_fuseiter_9059 * 3UL) + _fuseiter_9060)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_9057___fuseiter_9058_2682 / 2UL) * 73728UL) + (((fused_0_fuseiter_9057___fuseiter_9058_2682 % 2UL) * 36864UL) + ((_fuseiter_9059 * 12288UL) + ((_fuseiter_9060 * 4096UL) + ((_fuseiter_9061 * 128UL) + ((_fuseiter_9062 * 4UL) + _fuseiter_9063))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__509(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_9064___fuseiter_9065_2683 = 0UL; fused_0_fuseiter_9064___fuseiter_9065_2683 < 16UL; fused_0_fuseiter_9064___fuseiter_9065_2683 += 1UL) {
    for (uint64_t _fuseiter_9066 = 0UL; _fuseiter_9066 < 3UL; _fuseiter_9066 += 1UL) {
      for (uint64_t _fuseiter_9067 = 0UL; _fuseiter_9067 < 3UL; _fuseiter_9067 += 1UL) {
        for (uint64_t _fuseiter_9068 = 0UL; _fuseiter_9068 < 16UL; _fuseiter_9068 += 1UL) {
          for (uint64_t _fuseiter_9069 = 0UL; _fuseiter_9069 < 64UL; _fuseiter_9069 += 1UL) {
            for (uint64_t _fuseiter_9070 = 0UL; _fuseiter_9070 < 4UL; _fuseiter_9070 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_9069 + ((fused_0_fuseiter_9064___fuseiter_9065_2683 / 4UL) * 64UL)) * 2304UL) + ((((_fuseiter_9070 + (_fuseiter_9068 * 4UL)) + ((fused_0_fuseiter_9064___fuseiter_9065_2683 % 4UL) * 64UL)) * 9UL) + ((_fuseiter_9066 * 3UL) + _fuseiter_9067)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_9064___fuseiter_9065_2683 / 4UL) * 147456UL) + (((fused_0_fuseiter_9064___fuseiter_9065_2683 % 4UL) * 36864UL) + ((_fuseiter_9066 * 12288UL) + ((_fuseiter_9067 * 4096UL) + ((_fuseiter_9068 * 256UL) + ((_fuseiter_9069 * 4UL) + _fuseiter_9070))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__500(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_9071___fuseiter_9072_2684 = 0UL; fused_0_fuseiter_9071___fuseiter_9072_2684 < 16UL; fused_0_fuseiter_9071___fuseiter_9072_2684 += 1UL) {
    for (uint64_t _fuseiter_9073 = 0UL; _fuseiter_9073 < 3UL; _fuseiter_9073 += 1UL) {
      for (uint64_t _fuseiter_9074 = 0UL; _fuseiter_9074 < 3UL; _fuseiter_9074 += 1UL) {
        for (uint64_t _fuseiter_9075 = 0UL; _fuseiter_9075 < 32UL; _fuseiter_9075 += 1UL) {
          for (uint64_t _fuseiter_9076 = 0UL; _fuseiter_9076 < 32UL; _fuseiter_9076 += 1UL) {
            for (uint64_t _fuseiter_9077 = 0UL; _fuseiter_9077 < 4UL; _fuseiter_9077 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_9076 + ((fused_0_fuseiter_9071___fuseiter_9072_2684 / 2UL) * 32UL)) * 2304UL) + ((((_fuseiter_9077 + (_fuseiter_9075 * 4UL)) + ((fused_0_fuseiter_9071___fuseiter_9072_2684 % 2UL) * 128UL)) * 9UL) + ((_fuseiter_9073 * 3UL) + _fuseiter_9074)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_9071___fuseiter_9072_2684 / 2UL) * 73728UL) + (((fused_0_fuseiter_9071___fuseiter_9072_2684 % 2UL) * 36864UL) + ((_fuseiter_9073 * 12288UL) + ((_fuseiter_9074 * 4096UL) + ((_fuseiter_9075 * 128UL) + ((_fuseiter_9076 * 4UL) + _fuseiter_9077))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__493(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_9078 = 0UL; _fuseiter_9078 < 16UL; _fuseiter_9078 += 1UL) {
    for (uint64_t _fuseiter_9080 = 0UL; _fuseiter_9080 < 3UL; _fuseiter_9080 += 1UL) {
      for (uint64_t _fuseiter_9081 = 0UL; _fuseiter_9081 < 3UL; _fuseiter_9081 += 1UL) {
        for (uint64_t _fuseiter_9082 = 0UL; _fuseiter_9082 < 64UL; _fuseiter_9082 += 1UL) {
          for (uint64_t _fuseiter_9083 = 0UL; _fuseiter_9083 < 16UL; _fuseiter_9083 += 1UL) {
            for (uint64_t _fuseiter_9084 = 0UL; _fuseiter_9084 < 4UL; _fuseiter_9084 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_9083 + (_fuseiter_9078 * 16UL)) * 2304UL) + (((_fuseiter_9084 + (_fuseiter_9082 * 4UL)) * 9UL) + ((_fuseiter_9080 * 3UL) + _fuseiter_9081)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[((_fuseiter_9078 * 36864UL) + ((_fuseiter_9080 * 12288UL) + ((_fuseiter_9081 * 4096UL) + ((_fuseiter_9082 * 64UL) + ((_fuseiter_9083 * 4UL) + _fuseiter_9084)))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__484(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_9085___fuseiter_9086_2685___fuseiter_9087_2686___fuseiter_9088_2687 = 0UL; fused_0fused_0fused_0_fuseiter_9085___fuseiter_9086_2685___fuseiter_9087_2686___fuseiter_9088_2687 < 18UL; fused_0fused_0fused_0_fuseiter_9085___fuseiter_9086_2685___fuseiter_9087_2686___fuseiter_9088_2687 += 1UL) {
    for (uint64_t _fuseiter_9089 = 0UL; _fuseiter_9089 < 64UL; _fuseiter_9089 += 1UL) {
      for (uint64_t _fuseiter_9090 = 0UL; _fuseiter_9090 < 128UL; _fuseiter_9090 += 1UL) {
        for (uint64_t _fuseiter_9091 = 0UL; _fuseiter_9091 < 4UL; _fuseiter_9091 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_9090 + ((fused_0fused_0fused_0_fuseiter_9085___fuseiter_9086_2685___fuseiter_9087_2686___fuseiter_9088_2687 / 9UL) * 128UL)) * 2304UL) + (((_fuseiter_9091 + (_fuseiter_9089 * 4UL)) * 9UL) + ((((fused_0fused_0fused_0_fuseiter_9085___fuseiter_9086_2685___fuseiter_9087_2686___fuseiter_9088_2687 / 3UL) % 3UL) * 3UL) + (fused_0fused_0fused_0_fuseiter_9085___fuseiter_9086_2685___fuseiter_9087_2686___fuseiter_9088_2687 % 3UL))))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0fused_0fused_0_fuseiter_9085___fuseiter_9086_2685___fuseiter_9087_2686___fuseiter_9088_2687 / 9UL) * 294912UL) + ((((fused_0fused_0fused_0_fuseiter_9085___fuseiter_9086_2685___fuseiter_9087_2686___fuseiter_9088_2687 / 3UL) % 3UL) * 98304UL) + (((fused_0fused_0fused_0_fuseiter_9085___fuseiter_9086_2685___fuseiter_9087_2686___fuseiter_9088_2687 % 3UL) * 32768UL) + ((_fuseiter_9089 * 512UL) + ((_fuseiter_9090 * 4UL) + _fuseiter_9091)))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool reorder__480(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_9092 = 0UL; _fuseiter_9092 < 32UL; _fuseiter_9092 += 1UL) {
    for (uint64_t _fuseiter_9093 = 0UL; _fuseiter_9093 < 4UL; _fuseiter_9093 += 1UL) {
      for (uint64_t _fuseiter_9096 = 0UL; _fuseiter_9096 < 32UL; _fuseiter_9096 += 1UL) {
        for (uint64_t _fuseiter_9097 = 0UL; _fuseiter_9097 < 32UL; _fuseiter_9097 += 1UL) {
          for (uint64_t _fuseiter_9098 = 0UL; _fuseiter_9098 < 4UL; _fuseiter_9098 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_9097 + (_fuseiter_9092 * 32UL)) * 512UL) + ((_fuseiter_9098 + (_fuseiter_9096 * 4UL)) + (_fuseiter_9093 * 128UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_9092 * 16384UL) + ((_fuseiter_9093 * 4096UL) + ((_fuseiter_9096 * 128UL) + ((_fuseiter_9097 * 4UL) + _fuseiter_9098))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__474(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_9099___fuseiter_9100_2688___fuseiter_9101_2689 = 0UL; fused_0fused_0_fuseiter_9099___fuseiter_9100_2688___fuseiter_9101_2689 < 12UL; fused_0fused_0_fuseiter_9099___fuseiter_9100_2688___fuseiter_9101_2689 += 1UL) {
    for (uint64_t _fuseiter_9102 = 0UL; _fuseiter_9102 < 3UL; _fuseiter_9102 += 1UL) {
      for (uint64_t _fuseiter_9103 = 0UL; _fuseiter_9103 < 32UL; _fuseiter_9103 += 1UL) {
        for (uint64_t _fuseiter_9104 = 0UL; _fuseiter_9104 < 32UL; _fuseiter_9104 += 1UL) {
          for (uint64_t _fuseiter_9105 = 0UL; _fuseiter_9105 < 4UL; _fuseiter_9105 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_9104 + ((fused_0fused_0_fuseiter_9099___fuseiter_9100_2688___fuseiter_9101_2689 / 3UL) * 32UL)) * 1152UL) + (((_fuseiter_9105 + (_fuseiter_9103 * 4UL)) * 9UL) + (((fused_0fused_0_fuseiter_9099___fuseiter_9100_2688___fuseiter_9101_2689 % 3UL) * 3UL) + _fuseiter_9102)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_9099___fuseiter_9100_2688___fuseiter_9101_2689 / 3UL) * 36864UL) + (((fused_0fused_0_fuseiter_9099___fuseiter_9100_2688___fuseiter_9101_2689 % 3UL) * 12288UL) + ((_fuseiter_9102 * 4096UL) + ((_fuseiter_9103 * 128UL) + ((_fuseiter_9104 * 4UL) + _fuseiter_9105)))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__465(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_9106___fuseiter_9107_2690___fuseiter_9108_2691 = 0UL; fused_0fused_0_fuseiter_9106___fuseiter_9107_2690___fuseiter_9108_2691 < 24UL; fused_0fused_0_fuseiter_9106___fuseiter_9107_2690___fuseiter_9108_2691 += 1UL) {
    for (uint64_t _fuseiter_9109 = 0UL; _fuseiter_9109 < 3UL; _fuseiter_9109 += 1UL) {
      for (uint64_t _fuseiter_9110 = 0UL; _fuseiter_9110 < 16UL; _fuseiter_9110 += 1UL) {
        for (uint64_t _fuseiter_9111 = 0UL; _fuseiter_9111 < 32UL; _fuseiter_9111 += 1UL) {
          for (uint64_t _fuseiter_9112 = 0UL; _fuseiter_9112 < 4UL; _fuseiter_9112 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_9111 + ((fused_0fused_0_fuseiter_9106___fuseiter_9107_2690___fuseiter_9108_2691 / 6UL) * 32UL)) * 1152UL) + ((((_fuseiter_9112 + (_fuseiter_9110 * 4UL)) + (((fused_0fused_0_fuseiter_9106___fuseiter_9107_2690___fuseiter_9108_2691 / 3UL) % 2UL) * 64UL)) * 9UL) + (((fused_0fused_0_fuseiter_9106___fuseiter_9107_2690___fuseiter_9108_2691 % 3UL) * 3UL) + _fuseiter_9109)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_9106___fuseiter_9107_2690___fuseiter_9108_2691 / 6UL) * 36864UL) + ((((fused_0fused_0_fuseiter_9106___fuseiter_9107_2690___fuseiter_9108_2691 / 3UL) % 2UL) * 18432UL) + (((fused_0fused_0_fuseiter_9106___fuseiter_9107_2690___fuseiter_9108_2691 % 3UL) * 6144UL) + ((_fuseiter_9109 * 2048UL) + ((_fuseiter_9110 * 128UL) + ((_fuseiter_9111 * 4UL) + _fuseiter_9112))))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__460(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_9113___fuseiter_9114_2692___fuseiter_9115_2693___fuseiter_9116_2694 = 0UL; fused_0fused_0fused_0_fuseiter_9113___fuseiter_9114_2692___fuseiter_9115_2693___fuseiter_9116_2694 < 18UL; fused_0fused_0fused_0_fuseiter_9113___fuseiter_9114_2692___fuseiter_9115_2693___fuseiter_9116_2694 += 1UL) {
    for (uint64_t _fuseiter_9117 = 0UL; _fuseiter_9117 < 16UL; _fuseiter_9117 += 1UL) {
      for (uint64_t _fuseiter_9118 = 0UL; _fuseiter_9118 < 128UL; _fuseiter_9118 += 1UL) {
        for (uint64_t _fuseiter_9119 = 0UL; _fuseiter_9119 < 4UL; _fuseiter_9119 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_9118 + ((fused_0fused_0fused_0_fuseiter_9113___fuseiter_9114_2692___fuseiter_9115_2693___fuseiter_9116_2694 / 18UL) * 128UL)) * 1152UL) + ((((_fuseiter_9119 + (_fuseiter_9117 * 4UL)) + (((fused_0fused_0fused_0_fuseiter_9113___fuseiter_9114_2692___fuseiter_9115_2693___fuseiter_9116_2694 / 9UL) % 2UL) * 64UL)) * 9UL) + ((((fused_0fused_0fused_0_fuseiter_9113___fuseiter_9114_2692___fuseiter_9115_2693___fuseiter_9116_2694 / 3UL) % 3UL) * 3UL) + (fused_0fused_0fused_0_fuseiter_9113___fuseiter_9114_2692___fuseiter_9115_2693___fuseiter_9116_2694 % 3UL))))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0fused_0fused_0_fuseiter_9113___fuseiter_9114_2692___fuseiter_9115_2693___fuseiter_9116_2694 / 18UL) * 147456UL) + ((((fused_0fused_0fused_0_fuseiter_9113___fuseiter_9114_2692___fuseiter_9115_2693___fuseiter_9116_2694 / 9UL) % 2UL) * 73728UL) + ((((fused_0fused_0fused_0_fuseiter_9113___fuseiter_9114_2692___fuseiter_9115_2693___fuseiter_9116_2694 / 3UL) % 3UL) * 24576UL) + (((fused_0fused_0fused_0_fuseiter_9113___fuseiter_9114_2692___fuseiter_9115_2693___fuseiter_9116_2694 % 3UL) * 8192UL) + ((_fuseiter_9117 * 512UL) + ((_fuseiter_9118 * 4UL) + _fuseiter_9119))))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool reorder__451(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_9120___fuseiter_9121_2695___fuseiter_9122_2696 = 0UL; fused_0fused_0_fuseiter_9120___fuseiter_9121_2695___fuseiter_9122_2696 < 12UL; fused_0fused_0_fuseiter_9120___fuseiter_9121_2695___fuseiter_9122_2696 += 1UL) {
    for (uint64_t _fuseiter_9123 = 0UL; _fuseiter_9123 < 3UL; _fuseiter_9123 += 1UL) {
      for (uint64_t _fuseiter_9124 = 0UL; _fuseiter_9124 < 32UL; _fuseiter_9124 += 1UL) {
        for (uint64_t _fuseiter_9125 = 0UL; _fuseiter_9125 < 32UL; _fuseiter_9125 += 1UL) {
          for (uint64_t _fuseiter_9126 = 0UL; _fuseiter_9126 < 4UL; _fuseiter_9126 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_9125 + ((fused_0fused_0_fuseiter_9120___fuseiter_9121_2695___fuseiter_9122_2696 / 3UL) * 32UL)) * 1152UL) + (((_fuseiter_9126 + (_fuseiter_9124 * 4UL)) * 9UL) + (((fused_0fused_0_fuseiter_9120___fuseiter_9121_2695___fuseiter_9122_2696 % 3UL) * 3UL) + _fuseiter_9123)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_9120___fuseiter_9121_2695___fuseiter_9122_2696 / 3UL) * 36864UL) + (((fused_0fused_0_fuseiter_9120___fuseiter_9121_2695___fuseiter_9122_2696 % 3UL) * 12288UL) + ((_fuseiter_9123 * 4096UL) + ((_fuseiter_9124 * 128UL) + ((_fuseiter_9125 * 4UL) + _fuseiter_9126)))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__483(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_9127___fuseiter_9128_2697___fuseiter_9129_2698___fuseiter_9130_2699___fuseiter_9131_2700 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_9127___fuseiter_9128_2697___fuseiter_9129_2698___fuseiter_9130_2699___fuseiter_9131_2700 < 128UL; fused_0fused_0fused_0fused_0_fuseiter_9127___fuseiter_9128_2697___fuseiter_9129_2698___fuseiter_9130_2699___fuseiter_9131_2700 += 1UL) {
    for (uint64_t _fuseiter_9132 = 0UL; _fuseiter_9132 < 256UL; _fuseiter_9132 += 1UL) {
      for (uint64_t _fuseiter_9133 = 0UL; _fuseiter_9133 < 4UL; _fuseiter_9133 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_9132 + ((fused_0fused_0fused_0fused_0_fuseiter_9127___fuseiter_9128_2697___fuseiter_9129_2698___fuseiter_9130_2699___fuseiter_9131_2700 / 128UL) * 256UL)) * 512UL) + ((_fuseiter_9133 + ((fused_0fused_0fused_0fused_0_fuseiter_9127___fuseiter_9128_2697___fuseiter_9129_2698___fuseiter_9130_2699___fuseiter_9131_2700 % 32UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_9127___fuseiter_9128_2697___fuseiter_9129_2698___fuseiter_9130_2699___fuseiter_9131_2700 / 32UL) % 4UL) * 128UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_9127___fuseiter_9128_2697___fuseiter_9129_2698___fuseiter_9130_2699___fuseiter_9131_2700 / 128UL) * 131072UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_9127___fuseiter_9128_2697___fuseiter_9129_2698___fuseiter_9130_2699___fuseiter_9131_2700 / 32UL) % 4UL) * 32768UL) + (((fused_0fused_0fused_0fused_0_fuseiter_9127___fuseiter_9128_2697___fuseiter_9129_2698___fuseiter_9130_2699___fuseiter_9131_2700 % 32UL) * 1024UL) + ((_fuseiter_9132 * 4UL) + _fuseiter_9133))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__445(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_9134 = 0UL; _fuseiter_9134 < 16UL; _fuseiter_9134 += 1UL) {
    for (uint64_t _fuseiter_9135 = 0UL; _fuseiter_9135 < 2UL; _fuseiter_9135 += 1UL) {
      for (uint64_t _fuseiter_9138 = 0UL; _fuseiter_9138 < 32UL; _fuseiter_9138 += 1UL) {
        for (uint64_t _fuseiter_9139 = 0UL; _fuseiter_9139 < 32UL; _fuseiter_9139 += 1UL) {
          for (uint64_t _fuseiter_9140 = 0UL; _fuseiter_9140 < 4UL; _fuseiter_9140 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_9139 + (_fuseiter_9134 * 32UL)) * 256UL) + ((_fuseiter_9140 + (_fuseiter_9138 * 4UL)) + (_fuseiter_9135 * 128UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_9134 * 8192UL) + ((_fuseiter_9135 * 4096UL) + ((_fuseiter_9138 * 128UL) + ((_fuseiter_9139 * 4UL) + _fuseiter_9140))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__471(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_9141___fuseiter_9142_2701___fuseiter_9143_2702___fuseiter_9144_2703___fuseiter_9145_2704 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_9141___fuseiter_9142_2701___fuseiter_9143_2702___fuseiter_9144_2703___fuseiter_9145_2704 < 256UL; fused_0fused_0fused_0fused_0_fuseiter_9141___fuseiter_9142_2701___fuseiter_9143_2702___fuseiter_9144_2703___fuseiter_9145_2704 += 1UL) {
    for (uint64_t _fuseiter_9146 = 0UL; _fuseiter_9146 < 64UL; _fuseiter_9146 += 1UL) {
      for (uint64_t _fuseiter_9147 = 0UL; _fuseiter_9147 < 4UL; _fuseiter_9147 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_9146 + ((fused_0fused_0fused_0fused_0_fuseiter_9141___fuseiter_9142_2701___fuseiter_9143_2702___fuseiter_9144_2703___fuseiter_9145_2704 / 128UL) * 64UL)) * 512UL) + ((_fuseiter_9147 + ((fused_0fused_0fused_0fused_0_fuseiter_9141___fuseiter_9142_2701___fuseiter_9143_2702___fuseiter_9144_2703___fuseiter_9145_2704 % 32UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_9141___fuseiter_9142_2701___fuseiter_9143_2702___fuseiter_9144_2703___fuseiter_9145_2704 / 32UL) % 4UL) * 128UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_9141___fuseiter_9142_2701___fuseiter_9143_2702___fuseiter_9144_2703___fuseiter_9145_2704 / 128UL) * 32768UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_9141___fuseiter_9142_2701___fuseiter_9143_2702___fuseiter_9144_2703___fuseiter_9145_2704 / 32UL) % 4UL) * 8192UL) + (((fused_0fused_0fused_0fused_0_fuseiter_9141___fuseiter_9142_2701___fuseiter_9143_2702___fuseiter_9144_2703___fuseiter_9145_2704 % 32UL) * 256UL) + ((_fuseiter_9146 * 4UL) + _fuseiter_9147))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__464(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_9148___fuseiter_9149_2705___fuseiter_9150_2706___fuseiter_9151_2707___fuseiter_9152_2708 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_9148___fuseiter_9149_2705___fuseiter_9150_2706___fuseiter_9151_2707___fuseiter_9152_2708 < 128UL; fused_0fused_0fused_0fused_0_fuseiter_9148___fuseiter_9149_2705___fuseiter_9150_2706___fuseiter_9151_2707___fuseiter_9152_2708 += 1UL) {
    for (uint64_t _fuseiter_9153 = 0UL; _fuseiter_9153 < 128UL; _fuseiter_9153 += 1UL) {
      for (uint64_t _fuseiter_9154 = 0UL; _fuseiter_9154 < 4UL; _fuseiter_9154 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_9153 + ((fused_0fused_0fused_0fused_0_fuseiter_9148___fuseiter_9149_2705___fuseiter_9150_2706___fuseiter_9151_2707___fuseiter_9152_2708 / 128UL) * 128UL)) * 512UL) + ((_fuseiter_9154 + ((fused_0fused_0fused_0fused_0_fuseiter_9148___fuseiter_9149_2705___fuseiter_9150_2706___fuseiter_9151_2707___fuseiter_9152_2708 % 16UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_9148___fuseiter_9149_2705___fuseiter_9150_2706___fuseiter_9151_2707___fuseiter_9152_2708 / 16UL) % 8UL) * 64UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_9148___fuseiter_9149_2705___fuseiter_9150_2706___fuseiter_9151_2707___fuseiter_9152_2708 / 128UL) * 65536UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_9148___fuseiter_9149_2705___fuseiter_9150_2706___fuseiter_9151_2707___fuseiter_9152_2708 / 16UL) % 8UL) * 8192UL) + (((fused_0fused_0fused_0fused_0_fuseiter_9148___fuseiter_9149_2705___fuseiter_9150_2706___fuseiter_9151_2707___fuseiter_9152_2708 % 16UL) * 512UL) + ((_fuseiter_9153 * 4UL) + _fuseiter_9154))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__457(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_9155___fuseiter_9156_2709 = 0UL; fused_0_fuseiter_9155___fuseiter_9156_2709 < 16UL; fused_0_fuseiter_9155___fuseiter_9156_2709 += 1UL) {
    for (uint64_t _fuseiter_9159 = 0UL; _fuseiter_9159 < 32UL; _fuseiter_9159 += 1UL) {
      for (uint64_t _fuseiter_9160 = 0UL; _fuseiter_9160 < 32UL; _fuseiter_9160 += 1UL) {
        for (uint64_t _fuseiter_9161 = 0UL; _fuseiter_9161 < 4UL; _fuseiter_9161 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_9160 + ((fused_0_fuseiter_9155___fuseiter_9156_2709 / 4UL) * 32UL)) * 512UL) + ((_fuseiter_9161 + (_fuseiter_9159 * 4UL)) + ((fused_0_fuseiter_9155___fuseiter_9156_2709 % 4UL) * 128UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_9155___fuseiter_9156_2709 / 4UL) * 16384UL) + (((fused_0_fuseiter_9155___fuseiter_9156_2709 % 4UL) * 4096UL) + ((_fuseiter_9159 * 128UL) + ((_fuseiter_9160 * 4UL) + _fuseiter_9161))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool reorder__477(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_9162___fuseiter_9163_2710 = 0UL; fused_0_fuseiter_9162___fuseiter_9163_2710 < 16UL; fused_0_fuseiter_9162___fuseiter_9163_2710 += 1UL) {
    for (uint64_t _fuseiter_9166 = 0UL; _fuseiter_9166 < 8UL; _fuseiter_9166 += 1UL) {
      for (uint64_t _fuseiter_9167 = 0UL; _fuseiter_9167 < 128UL; _fuseiter_9167 += 1UL) {
        for (uint64_t _fuseiter_9168 = 0UL; _fuseiter_9168 < 4UL; _fuseiter_9168 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_9167 + ((fused_0_fuseiter_9162___fuseiter_9163_2710 / 4UL) * 128UL)) * 128UL) + ((_fuseiter_9168 + (_fuseiter_9166 * 4UL)) + ((fused_0_fuseiter_9162___fuseiter_9163_2710 % 4UL) * 32UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_9162___fuseiter_9163_2710 / 4UL) * 16384UL) + (((fused_0_fuseiter_9162___fuseiter_9163_2710 % 4UL) * 4096UL) + ((_fuseiter_9166 * 512UL) + ((_fuseiter_9167 * 4UL) + _fuseiter_9168))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool reorder__468(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_9169___fuseiter_9170_2711___fuseiter_9171_2712___fuseiter_9172_2713___fuseiter_9173_2714 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_9169___fuseiter_9170_2711___fuseiter_9171_2712___fuseiter_9172_2713___fuseiter_9173_2714 < 256UL; fused_0fused_0fused_0fused_0_fuseiter_9169___fuseiter_9170_2711___fuseiter_9171_2712___fuseiter_9172_2713___fuseiter_9173_2714 += 1UL) {
    for (uint64_t _fuseiter_9174 = 0UL; _fuseiter_9174 < 64UL; _fuseiter_9174 += 1UL) {
      for (uint64_t _fuseiter_9175 = 0UL; _fuseiter_9175 < 4UL; _fuseiter_9175 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_9174 + ((fused_0fused_0fused_0fused_0_fuseiter_9169___fuseiter_9170_2711___fuseiter_9171_2712___fuseiter_9172_2713___fuseiter_9173_2714 / 32UL) * 64UL)) * 128UL) + (_fuseiter_9175 + ((fused_0fused_0fused_0fused_0_fuseiter_9169___fuseiter_9170_2711___fuseiter_9171_2712___fuseiter_9172_2713___fuseiter_9173_2714 % 32UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_9169___fuseiter_9170_2711___fuseiter_9171_2712___fuseiter_9172_2713___fuseiter_9173_2714 / 32UL) * 8192UL) + (((fused_0fused_0fused_0fused_0_fuseiter_9169___fuseiter_9170_2711___fuseiter_9171_2712___fuseiter_9172_2713___fuseiter_9173_2714 % 32UL) * 256UL) + ((_fuseiter_9174 * 4UL) + _fuseiter_9175)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__461(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_9176___fuseiter_9177_2715___fuseiter_9178_2716___fuseiter_9179_2717___fuseiter_9180_2718 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_9176___fuseiter_9177_2715___fuseiter_9178_2716___fuseiter_9179_2717___fuseiter_9180_2718 < 128UL; fused_0fused_0fused_0fused_0_fuseiter_9176___fuseiter_9177_2715___fuseiter_9178_2716___fuseiter_9179_2717___fuseiter_9180_2718 += 1UL) {
    for (uint64_t _fuseiter_9181 = 0UL; _fuseiter_9181 < 128UL; _fuseiter_9181 += 1UL) {
      for (uint64_t _fuseiter_9182 = 0UL; _fuseiter_9182 < 4UL; _fuseiter_9182 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_9181 + ((fused_0fused_0fused_0fused_0_fuseiter_9176___fuseiter_9177_2715___fuseiter_9178_2716___fuseiter_9179_2717___fuseiter_9180_2718 / 32UL) * 128UL)) * 128UL) + ((_fuseiter_9182 + ((fused_0fused_0fused_0fused_0_fuseiter_9176___fuseiter_9177_2715___fuseiter_9178_2716___fuseiter_9179_2717___fuseiter_9180_2718 % 16UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_9176___fuseiter_9177_2715___fuseiter_9178_2716___fuseiter_9179_2717___fuseiter_9180_2718 / 16UL) % 2UL) * 64UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_9176___fuseiter_9177_2715___fuseiter_9178_2716___fuseiter_9179_2717___fuseiter_9180_2718 / 32UL) * 16384UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_9176___fuseiter_9177_2715___fuseiter_9178_2716___fuseiter_9179_2717___fuseiter_9180_2718 / 16UL) % 2UL) * 8192UL) + (((fused_0fused_0fused_0fused_0_fuseiter_9176___fuseiter_9177_2715___fuseiter_9178_2716___fuseiter_9179_2717___fuseiter_9180_2718 % 16UL) * 512UL) + ((_fuseiter_9181 * 4UL) + _fuseiter_9182))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__454(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_9183 = 0UL; _fuseiter_9183 < 16UL; _fuseiter_9183 += 1UL) {
    for (uint64_t _fuseiter_9184 = 0UL; _fuseiter_9184 < 4UL; _fuseiter_9184 += 1UL) {
      for (uint64_t _fuseiter_9187 = 0UL; _fuseiter_9187 < 8UL; _fuseiter_9187 += 1UL) {
        for (uint64_t _fuseiter_9188 = 0UL; _fuseiter_9188 < 32UL; _fuseiter_9188 += 1UL) {
          for (uint64_t _fuseiter_9189 = 0UL; _fuseiter_9189 < 4UL; _fuseiter_9189 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_9188 + (_fuseiter_9183 * 32UL)) * 128UL) + ((_fuseiter_9189 + (_fuseiter_9187 * 4UL)) + (_fuseiter_9184 * 32UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_9183 * 4096UL) + ((_fuseiter_9184 * 1024UL) + ((_fuseiter_9187 * 128UL) + ((_fuseiter_9188 * 4UL) + _fuseiter_9189))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__439(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_9190___fuseiter_9191_2719___fuseiter_9192_2720___fuseiter_9193_2721 = 0UL; fused_0fused_0fused_0_fuseiter_9190___fuseiter_9191_2719___fuseiter_9192_2720___fuseiter_9193_2721 < 18UL; fused_0fused_0fused_0_fuseiter_9190___fuseiter_9191_2719___fuseiter_9192_2720___fuseiter_9193_2721 += 1UL) {
    for (uint64_t _fuseiter_9194 = 0UL; _fuseiter_9194 < 16UL; _fuseiter_9194 += 1UL) {
      for (uint64_t _fuseiter_9195 = 0UL; _fuseiter_9195 < 32UL; _fuseiter_9195 += 1UL) {
        for (uint64_t _fuseiter_9196 = 0UL; _fuseiter_9196 < 4UL; _fuseiter_9196 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_9195 + ((fused_0fused_0fused_0_fuseiter_9190___fuseiter_9191_2719___fuseiter_9192_2720___fuseiter_9193_2721 / 9UL) * 32UL)) * 576UL) + (((_fuseiter_9196 + (_fuseiter_9194 * 4UL)) * 9UL) + ((((fused_0fused_0fused_0_fuseiter_9190___fuseiter_9191_2719___fuseiter_9192_2720___fuseiter_9193_2721 / 3UL) % 3UL) * 3UL) + (fused_0fused_0fused_0_fuseiter_9190___fuseiter_9191_2719___fuseiter_9192_2720___fuseiter_9193_2721 % 3UL))))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0fused_0fused_0_fuseiter_9190___fuseiter_9191_2719___fuseiter_9192_2720___fuseiter_9193_2721 / 9UL) * 18432UL) + ((((fused_0fused_0fused_0_fuseiter_9190___fuseiter_9191_2719___fuseiter_9192_2720___fuseiter_9193_2721 / 3UL) % 3UL) * 6144UL) + (((fused_0fused_0fused_0_fuseiter_9190___fuseiter_9191_2719___fuseiter_9192_2720___fuseiter_9193_2721 % 3UL) * 2048UL) + ((_fuseiter_9194 * 128UL) + ((_fuseiter_9195 * 4UL) + _fuseiter_9196)))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool reorder__430(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_9197___fuseiter_9198_2722___fuseiter_9199_2723 = 0UL; fused_0fused_0_fuseiter_9197___fuseiter_9198_2722___fuseiter_9199_2723 < 12UL; fused_0fused_0_fuseiter_9197___fuseiter_9198_2722___fuseiter_9199_2723 += 1UL) {
    for (uint64_t _fuseiter_9200 = 0UL; _fuseiter_9200 < 3UL; _fuseiter_9200 += 1UL) {
      for (uint64_t _fuseiter_9201 = 0UL; _fuseiter_9201 < 16UL; _fuseiter_9201 += 1UL) {
        for (uint64_t _fuseiter_9202 = 0UL; _fuseiter_9202 < 16UL; _fuseiter_9202 += 1UL) {
          for (uint64_t _fuseiter_9203 = 0UL; _fuseiter_9203 < 4UL; _fuseiter_9203 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_9202 + ((fused_0fused_0_fuseiter_9197___fuseiter_9198_2722___fuseiter_9199_2723 / 3UL) * 16UL)) * 576UL) + (((_fuseiter_9203 + (_fuseiter_9201 * 4UL)) * 9UL) + (((fused_0fused_0_fuseiter_9197___fuseiter_9198_2722___fuseiter_9199_2723 % 3UL) * 3UL) + _fuseiter_9200)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_9197___fuseiter_9198_2722___fuseiter_9199_2723 / 3UL) * 9216UL) + (((fused_0fused_0_fuseiter_9197___fuseiter_9198_2722___fuseiter_9199_2723 % 3UL) * 3072UL) + ((_fuseiter_9200 * 1024UL) + ((_fuseiter_9201 * 64UL) + ((_fuseiter_9202 * 4UL) + _fuseiter_9203)))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__423(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_9204___fuseiter_9205_2724___fuseiter_9206_2725___fuseiter_9207_2726 = 0UL; fused_0fused_0fused_0_fuseiter_9204___fuseiter_9205_2724___fuseiter_9206_2725___fuseiter_9207_2726 < 18UL; fused_0fused_0fused_0_fuseiter_9204___fuseiter_9205_2724___fuseiter_9206_2725___fuseiter_9207_2726 += 1UL) {
    for (uint64_t _fuseiter_9208 = 0UL; _fuseiter_9208 < 16UL; _fuseiter_9208 += 1UL) {
      for (uint64_t _fuseiter_9209 = 0UL; _fuseiter_9209 < 32UL; _fuseiter_9209 += 1UL) {
        for (uint64_t _fuseiter_9210 = 0UL; _fuseiter_9210 < 4UL; _fuseiter_9210 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_9209 + ((fused_0fused_0fused_0_fuseiter_9204___fuseiter_9205_2724___fuseiter_9206_2725___fuseiter_9207_2726 / 9UL) * 32UL)) * 576UL) + (((_fuseiter_9210 + (_fuseiter_9208 * 4UL)) * 9UL) + ((((fused_0fused_0fused_0_fuseiter_9204___fuseiter_9205_2724___fuseiter_9206_2725___fuseiter_9207_2726 / 3UL) % 3UL) * 3UL) + (fused_0fused_0fused_0_fuseiter_9204___fuseiter_9205_2724___fuseiter_9206_2725___fuseiter_9207_2726 % 3UL))))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0fused_0fused_0_fuseiter_9204___fuseiter_9205_2724___fuseiter_9206_2725___fuseiter_9207_2726 / 9UL) * 18432UL) + ((((fused_0fused_0fused_0_fuseiter_9204___fuseiter_9205_2724___fuseiter_9206_2725___fuseiter_9207_2726 / 3UL) % 3UL) * 6144UL) + (((fused_0fused_0fused_0_fuseiter_9204___fuseiter_9205_2724___fuseiter_9206_2725___fuseiter_9207_2726 % 3UL) * 2048UL) + ((_fuseiter_9208 * 128UL) + ((_fuseiter_9209 * 4UL) + _fuseiter_9210)))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool reorder__448(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_9211___fuseiter_9212_2727___fuseiter_9213_2728___fuseiter_9214_2729___fuseiter_9215_2730 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_9211___fuseiter_9212_2727___fuseiter_9213_2728___fuseiter_9214_2729___fuseiter_9215_2730 < 128UL; fused_0fused_0fused_0fused_0_fuseiter_9211___fuseiter_9212_2727___fuseiter_9213_2728___fuseiter_9214_2729___fuseiter_9215_2730 += 1UL) {
    for (uint64_t _fuseiter_9216 = 0UL; _fuseiter_9216 < 64UL; _fuseiter_9216 += 1UL) {
      for (uint64_t _fuseiter_9217 = 0UL; _fuseiter_9217 < 4UL; _fuseiter_9217 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_9216 + ((fused_0fused_0fused_0fused_0_fuseiter_9211___fuseiter_9212_2727___fuseiter_9213_2728___fuseiter_9214_2729___fuseiter_9215_2730 / 64UL) * 64UL)) * 256UL) + ((_fuseiter_9217 + ((fused_0fused_0fused_0fused_0_fuseiter_9211___fuseiter_9212_2727___fuseiter_9213_2728___fuseiter_9214_2729___fuseiter_9215_2730 % 32UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_9211___fuseiter_9212_2727___fuseiter_9213_2728___fuseiter_9214_2729___fuseiter_9215_2730 / 32UL) % 2UL) * 128UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_9211___fuseiter_9212_2727___fuseiter_9213_2728___fuseiter_9214_2729___fuseiter_9215_2730 / 64UL) * 16384UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_9211___fuseiter_9212_2727___fuseiter_9213_2728___fuseiter_9214_2729___fuseiter_9215_2730 / 32UL) % 2UL) * 8192UL) + (((fused_0fused_0fused_0fused_0_fuseiter_9211___fuseiter_9212_2727___fuseiter_9213_2728___fuseiter_9214_2729___fuseiter_9215_2730 % 32UL) * 256UL) + ((_fuseiter_9216 * 4UL) + _fuseiter_9217))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__436(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_9218___fuseiter_9219_2731___fuseiter_9220_2732___fuseiter_9221_2733___fuseiter_9222_2734 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_9218___fuseiter_9219_2731___fuseiter_9220_2732___fuseiter_9221_2733___fuseiter_9222_2734 < 128UL; fused_0fused_0fused_0fused_0_fuseiter_9218___fuseiter_9219_2731___fuseiter_9220_2732___fuseiter_9221_2733___fuseiter_9222_2734 += 1UL) {
    for (uint64_t _fuseiter_9223 = 0UL; _fuseiter_9223 < 32UL; _fuseiter_9223 += 1UL) {
      for (uint64_t _fuseiter_9224 = 0UL; _fuseiter_9224 < 4UL; _fuseiter_9224 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_9223 + ((fused_0fused_0fused_0fused_0_fuseiter_9218___fuseiter_9219_2731___fuseiter_9220_2732___fuseiter_9221_2733___fuseiter_9222_2734 / 64UL) * 32UL)) * 256UL) + ((_fuseiter_9224 + ((fused_0fused_0fused_0fused_0_fuseiter_9218___fuseiter_9219_2731___fuseiter_9220_2732___fuseiter_9221_2733___fuseiter_9222_2734 % 16UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_9218___fuseiter_9219_2731___fuseiter_9220_2732___fuseiter_9221_2733___fuseiter_9222_2734 / 16UL) % 4UL) * 64UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_9218___fuseiter_9219_2731___fuseiter_9220_2732___fuseiter_9221_2733___fuseiter_9222_2734 / 64UL) * 8192UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_9218___fuseiter_9219_2731___fuseiter_9220_2732___fuseiter_9221_2733___fuseiter_9222_2734 / 16UL) % 4UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_9218___fuseiter_9219_2731___fuseiter_9220_2732___fuseiter_9221_2733___fuseiter_9222_2734 % 16UL) * 128UL) + ((_fuseiter_9223 * 4UL) + _fuseiter_9224))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__429(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_9225___fuseiter_9226_2735___fuseiter_9227_2736___fuseiter_9228_2737___fuseiter_9229_2738 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_9225___fuseiter_9226_2735___fuseiter_9227_2736___fuseiter_9228_2737___fuseiter_9229_2738 < 64UL; fused_0fused_0fused_0fused_0_fuseiter_9225___fuseiter_9226_2735___fuseiter_9227_2736___fuseiter_9228_2737___fuseiter_9229_2738 += 1UL) {
    for (uint64_t _fuseiter_9230 = 0UL; _fuseiter_9230 < 64UL; _fuseiter_9230 += 1UL) {
      for (uint64_t _fuseiter_9231 = 0UL; _fuseiter_9231 < 4UL; _fuseiter_9231 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_9230 + ((fused_0fused_0fused_0fused_0_fuseiter_9225___fuseiter_9226_2735___fuseiter_9227_2736___fuseiter_9228_2737___fuseiter_9229_2738 / 64UL) * 64UL)) * 256UL) + ((_fuseiter_9231 + ((fused_0fused_0fused_0fused_0_fuseiter_9225___fuseiter_9226_2735___fuseiter_9227_2736___fuseiter_9228_2737___fuseiter_9229_2738 % 8UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_9225___fuseiter_9226_2735___fuseiter_9227_2736___fuseiter_9228_2737___fuseiter_9229_2738 / 8UL) % 8UL) * 32UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_9225___fuseiter_9226_2735___fuseiter_9227_2736___fuseiter_9228_2737___fuseiter_9229_2738 / 64UL) * 16384UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_9225___fuseiter_9226_2735___fuseiter_9227_2736___fuseiter_9228_2737___fuseiter_9229_2738 / 8UL) % 8UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_9225___fuseiter_9226_2735___fuseiter_9227_2736___fuseiter_9228_2737___fuseiter_9229_2738 % 8UL) * 256UL) + ((_fuseiter_9230 * 4UL) + _fuseiter_9231))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__442(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_9232___fuseiter_9233_2739___fuseiter_9234_2740___fuseiter_9235_2741___fuseiter_9236_2742 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_9232___fuseiter_9233_2739___fuseiter_9234_2740___fuseiter_9235_2741___fuseiter_9236_2742 < 64UL; fused_0fused_0fused_0fused_0_fuseiter_9232___fuseiter_9233_2739___fuseiter_9234_2740___fuseiter_9235_2741___fuseiter_9236_2742 += 1UL) {
    for (uint64_t _fuseiter_9237 = 0UL; _fuseiter_9237 < 64UL; _fuseiter_9237 += 1UL) {
      for (uint64_t _fuseiter_9238 = 0UL; _fuseiter_9238 < 4UL; _fuseiter_9238 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_9237 + ((fused_0fused_0fused_0fused_0_fuseiter_9232___fuseiter_9233_2739___fuseiter_9234_2740___fuseiter_9235_2741___fuseiter_9236_2742 / 16UL) * 64UL)) * 64UL) + (_fuseiter_9238 + ((fused_0fused_0fused_0fused_0_fuseiter_9232___fuseiter_9233_2739___fuseiter_9234_2740___fuseiter_9235_2741___fuseiter_9236_2742 % 16UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_9232___fuseiter_9233_2739___fuseiter_9234_2740___fuseiter_9235_2741___fuseiter_9236_2742 / 16UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_9232___fuseiter_9233_2739___fuseiter_9234_2740___fuseiter_9235_2741___fuseiter_9236_2742 % 16UL) * 256UL) + ((_fuseiter_9237 * 4UL) + _fuseiter_9238)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__433(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_9239___fuseiter_9240_2743___fuseiter_9241_2744___fuseiter_9242_2745___fuseiter_9243_2746 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_9239___fuseiter_9240_2743___fuseiter_9241_2744___fuseiter_9242_2745___fuseiter_9243_2746 < 128UL; fused_0fused_0fused_0fused_0_fuseiter_9239___fuseiter_9240_2743___fuseiter_9241_2744___fuseiter_9242_2745___fuseiter_9243_2746 += 1UL) {
    for (uint64_t _fuseiter_9244 = 0UL; _fuseiter_9244 < 32UL; _fuseiter_9244 += 1UL) {
      for (uint64_t _fuseiter_9245 = 0UL; _fuseiter_9245 < 4UL; _fuseiter_9245 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_9244 + ((fused_0fused_0fused_0fused_0_fuseiter_9239___fuseiter_9240_2743___fuseiter_9241_2744___fuseiter_9242_2745___fuseiter_9243_2746 / 16UL) * 32UL)) * 64UL) + (_fuseiter_9245 + ((fused_0fused_0fused_0fused_0_fuseiter_9239___fuseiter_9240_2743___fuseiter_9241_2744___fuseiter_9242_2745___fuseiter_9243_2746 % 16UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_9239___fuseiter_9240_2743___fuseiter_9241_2744___fuseiter_9242_2745___fuseiter_9243_2746 / 16UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_9239___fuseiter_9240_2743___fuseiter_9241_2744___fuseiter_9242_2745___fuseiter_9243_2746 % 16UL) * 128UL) + ((_fuseiter_9244 * 4UL) + _fuseiter_9245)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__426(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_9246___fuseiter_9247_2747___fuseiter_9248_2748___fuseiter_9249_2749___fuseiter_9250_2750 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_9246___fuseiter_9247_2747___fuseiter_9248_2748___fuseiter_9249_2749___fuseiter_9250_2750 < 64UL; fused_0fused_0fused_0fused_0_fuseiter_9246___fuseiter_9247_2747___fuseiter_9248_2748___fuseiter_9249_2749___fuseiter_9250_2750 += 1UL) {
    for (uint64_t _fuseiter_9251 = 0UL; _fuseiter_9251 < 64UL; _fuseiter_9251 += 1UL) {
      for (uint64_t _fuseiter_9252 = 0UL; _fuseiter_9252 < 4UL; _fuseiter_9252 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_9251 + ((fused_0fused_0fused_0fused_0_fuseiter_9246___fuseiter_9247_2747___fuseiter_9248_2748___fuseiter_9249_2749___fuseiter_9250_2750 / 16UL) * 64UL)) * 64UL) + ((_fuseiter_9252 + ((fused_0fused_0fused_0fused_0_fuseiter_9246___fuseiter_9247_2747___fuseiter_9248_2748___fuseiter_9249_2749___fuseiter_9250_2750 % 8UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_9246___fuseiter_9247_2747___fuseiter_9248_2748___fuseiter_9249_2749___fuseiter_9250_2750 / 8UL) % 2UL) * 32UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_9246___fuseiter_9247_2747___fuseiter_9248_2748___fuseiter_9249_2749___fuseiter_9250_2750 / 16UL) * 4096UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_9246___fuseiter_9247_2747___fuseiter_9248_2748___fuseiter_9249_2749___fuseiter_9250_2750 / 8UL) % 2UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_9246___fuseiter_9247_2747___fuseiter_9248_2748___fuseiter_9249_2749___fuseiter_9250_2750 % 8UL) * 256UL) + ((_fuseiter_9251 * 4UL) + _fuseiter_9252))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__419(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_9253 = 0UL; _fuseiter_9253 < 16UL; _fuseiter_9253 += 1UL) {
    for (uint64_t _fuseiter_9257 = 0UL; _fuseiter_9257 < 16UL; _fuseiter_9257 += 1UL) {
      for (uint64_t _fuseiter_9258 = 0UL; _fuseiter_9258 < 16UL; _fuseiter_9258 += 1UL) {
        for (uint64_t _fuseiter_9259 = 0UL; _fuseiter_9259 < 4UL; _fuseiter_9259 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_9258 + (_fuseiter_9253 * 16UL)) * 64UL) + (_fuseiter_9259 + (_fuseiter_9257 * 4UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[((_fuseiter_9253 * 1024UL) + ((_fuseiter_9257 * 64UL) + ((_fuseiter_9258 * 4UL) + _fuseiter_9259)))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__656(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2751____itr_2_2752____itr_3_2753 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2751____itr_2_2752____itr_3_2753 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2751____itr_2_2752____itr_3_2753 += 1UL) {
    for (uint64_t _fuseiter_9264 = 0UL; _fuseiter_9264 < 64UL; _fuseiter_9264 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2751____itr_2_2752____itr_3_2753 / 8UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2751____itr_2_2752____itr_3_2753 % 8UL) * 64UL)) + _fuseiter_9264)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2751____itr_2_2752____itr_3_2753 / 8UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2751____itr_2_2752____itr_3_2753 % 8UL) * 64UL)) + _fuseiter_9264)]);
    }
  }
  return true;
}

static bool mul__655(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2754____itr_2_2755____itr_3_2756 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2754____itr_2_2755____itr_3_2756 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_2754____itr_2_2755____itr_3_2756 += 1UL) {
    for (uint64_t _fuseiter_9270 = 0UL; _fuseiter_9270 < 64UL; _fuseiter_9270 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2754____itr_2_2755____itr_3_2756 / 8UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2754____itr_2_2755____itr_3_2756 % 8UL) * 64UL)) + _fuseiter_9270)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2754____itr_2_2755____itr_3_2756 / 8UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2754____itr_2_2755____itr_3_2756 % 8UL) * 64UL)) + _fuseiter_9270)]);
    }
  }
  return true;
}

static bool reorder__534(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_9272___fuseiter_9273_2757 = 0UL; fused_0_fuseiter_9272___fuseiter_9273_2757 < 32UL; fused_0_fuseiter_9272___fuseiter_9273_2757 += 1UL) {
    for (uint64_t _fuseiter_9276 = 0UL; _fuseiter_9276 < 64UL; _fuseiter_9276 += 1UL) {
      for (uint64_t _fuseiter_9277 = 0UL; _fuseiter_9277 < 64UL; _fuseiter_9277 += 1UL) {
        for (uint64_t _fuseiter_9278 = 0UL; _fuseiter_9278 < 4UL; _fuseiter_9278 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_9277 + ((fused_0_fuseiter_9272___fuseiter_9273_2757 / 4UL) * 64UL)) * 1024UL) + ((_fuseiter_9278 + (_fuseiter_9276 * 4UL)) + ((fused_0_fuseiter_9272___fuseiter_9273_2757 % 4UL) * 256UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_9272___fuseiter_9273_2757 / 4UL) * 65536UL) + (((fused_0_fuseiter_9272___fuseiter_9273_2757 % 4UL) * 16384UL) + ((_fuseiter_9276 * 256UL) + ((_fuseiter_9277 * 4UL) + _fuseiter_9278))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__243(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2758____itr_2_2759 = 0UL; fused_0fused_0__itr_0____itr_1_2758____itr_2_2759 < 1048576UL; fused_0fused_0__itr_0____itr_1_2758____itr_2_2759 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2758____itr_2_2759 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2758____itr_2_2759 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2758____itr_2_2759 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2758____itr_2_2759 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2758____itr_2_2759 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__244(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2760____itr_2_2761 = 0UL; fused_0fused_0__itr_0____itr_1_2760____itr_2_2761 < 1048576UL; fused_0fused_0__itr_0____itr_1_2760____itr_2_2761 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2760____itr_2_2761 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2760____itr_2_2761 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2760____itr_2_2761 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2760____itr_2_2761 % 512UL))] = __cached_1;
  }
  return true;
}

static bool mul__252(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2762____itr_2_2763 = 0UL; fused_0fused_0__itr_0____itr_1_2762____itr_2_2763 < 1048576UL; fused_0fused_0__itr_0____itr_1_2762____itr_2_2763 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2762____itr_2_2763 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2762____itr_2_2763 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2762____itr_2_2763 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2762____itr_2_2763 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2762____itr_2_2763 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__253(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2764____itr_2_2765 = 0UL; fused_0fused_0__itr_0____itr_1_2764____itr_2_2765 < 1048576UL; fused_0fused_0__itr_0____itr_1_2764____itr_2_2765 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2764____itr_2_2765 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2764____itr_2_2765 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2764____itr_2_2765 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2764____itr_2_2765 % 512UL))] = __cached_1;
  }
  return true;
}

static bool mul__261(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2766____itr_2_2767 = 0UL; fused_0fused_0__itr_0____itr_1_2766____itr_2_2767 < 1048576UL; fused_0fused_0__itr_0____itr_1_2766____itr_2_2767 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2766____itr_2_2767 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2766____itr_2_2767 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2766____itr_2_2767 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2766____itr_2_2767 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2766____itr_2_2767 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__262(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2768____itr_2_2769 = 0UL; fused_0fused_0__itr_0____itr_1_2768____itr_2_2769 < 1048576UL; fused_0fused_0__itr_0____itr_1_2768____itr_2_2769 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2768____itr_2_2769 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2768____itr_2_2769 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2768____itr_2_2769 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2768____itr_2_2769 % 512UL))] = __cached_1;
  }
  return true;
}

static bool mul__246(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2770____itr_2_2771 = 0UL; fused_0fused_0__itr_0____itr_1_2770____itr_2_2771 < 1048576UL; fused_0fused_0__itr_0____itr_1_2770____itr_2_2771 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2770____itr_2_2771 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_2770____itr_2_2771 % 2048UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2770____itr_2_2771 / 2048UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2770____itr_2_2771 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_2770____itr_2_2771 % 2048UL))] = __cached_2;
  }
  return true;
}

static bool cast__247(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2772____itr_2_2773 = 0UL; fused_0fused_0__itr_0____itr_1_2772____itr_2_2773 < 1048576UL; fused_0fused_0__itr_0____itr_1_2772____itr_2_2773 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2772____itr_2_2773 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_2772____itr_2_2773 % 2048UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2772____itr_2_2773 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_2772____itr_2_2773 % 2048UL))] = __cached_1;
  }
  return true;
}

static bool mul__255(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2774____itr_2_2775 = 0UL; fused_0fused_0__itr_0____itr_1_2774____itr_2_2775 < 1048576UL; fused_0fused_0__itr_0____itr_1_2774____itr_2_2775 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2774____itr_2_2775 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_2774____itr_2_2775 % 2048UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2774____itr_2_2775 / 2048UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2774____itr_2_2775 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_2774____itr_2_2775 % 2048UL))] = __cached_2;
  }
  return true;
}

static bool cast__256(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2776____itr_2_2777 = 0UL; fused_0fused_0__itr_0____itr_1_2776____itr_2_2777 < 1048576UL; fused_0fused_0__itr_0____itr_1_2776____itr_2_2777 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2776____itr_2_2777 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_2776____itr_2_2777 % 2048UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2776____itr_2_2777 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_2776____itr_2_2777 % 2048UL))] = __cached_1;
  }
  return true;
}

static bool mul__234(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2778____itr_2_2779 = 0UL; fused_0fused_0__itr_0____itr_1_2778____itr_2_2779 < 2097152UL; fused_0fused_0__itr_0____itr_1_2778____itr_2_2779 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2778____itr_2_2779 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2778____itr_2_2779 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2778____itr_2_2779 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2778____itr_2_2779 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2778____itr_2_2779 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__235(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2780____itr_2_2781 = 0UL; fused_0fused_0__itr_0____itr_1_2780____itr_2_2781 < 2097152UL; fused_0fused_0__itr_0____itr_1_2780____itr_2_2781 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2780____itr_2_2781 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2780____itr_2_2781 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2780____itr_2_2781 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2780____itr_2_2781 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool batchwise_2_fused_res2a_conv_b_cast_mul_add_cast_reorder_res2a_conv_0_cast_mul_add_relu_cast_res2a_conv_1_cast_mul_add_relu_cast_res2a_conv_2_cast_mul_add_cast_add_cast_reorder_res2b_conv_0_cast_mul_add_relu_cast_res2b_conv_1_cast_mul_add_relu_cast_reorder_res2b_conv_2_cast_mul_add_cast_add_cast_reorder_res2c_conv_0_cast_mul_add_relu_cast_reorder_res2c_conv_1_cast_mul_add_relu_cast_reorder_res2c_conv_2_cast_mul_add_cast_add_cast_reorder_res3a_conv_b_cast_mul_add_cast_res3a_conv_0_cast_mul_add_relu_cast_reorder_res3a_conv_1_cast_mul_add_relu_cast_res3a_conv_2_cast_mul_add_cast_add_cast_reorder_res3b_conv_0_cast_mul_add_relu_cast_reorder_res3b_conv_1_cast_mul_add_relu_cast_reorder_res3b_conv_2_cast_mul_add_cast_add_cast_reorder_res3c_conv_0_cast_mul_add_relu_cast_reorder_res3c_conv_1_cast_mul_add_relu_cast_reorder_res3c_conv_2_cast_mul_add_cast_add_cast_reorder_res3d_conv_0_cast_mul_add_relu_cast_reorder_res3d_conv_1_cast_mul_add_relu_cast_res3d_conv_2_cast_mul_add_cast_add_cast_res4a_conv_b_cast_mul_add_cast_reorder_res4a_conv_0_cast_mul_add_relu_cast_res4a_conv_1_cast_mul_add_relu_cast_reorder_res4a_conv_2_cast_mul_add_cast_add_cast_reorder_res4b_conv_0_cast_mul_add_relu_cast_reorder_res4b_conv_1_cast_mul_add_relu_cast_reorder_res4b_conv_2_cast_mul_add_cast_add_cast_reorder_res4c_conv_0_cast_mul_add_relu_cast_reorder_res4c_conv_1_cast_mul_add_relu_cast_reorder_res4c_conv_2_cast_mul_add_cast_add_cast_res4d_conv_0_cast_mul_add_relu_cast_res4d_conv_1_cast_mul_add_relu_cast_reorder_res4d_conv_2_cast_mul_add_cast_add_cast_res4e_conv_0_cast_mul_add_relu_cast_reorder_res4e_conv_1_cast_mul_add_relu_cast_reorder_res4e_conv_2_cast_mul_add_cast_add_cast_reorder_reorder_res4f_conv_0_cast_mul_add_relu_cast_reorder_res4f_conv_1_cast_mul_add_relu_cast_reorder_res4f_conv_2_cast_mul_add_cast_add_cast__683(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4, float* __restrict__ __ins_5, float* __restrict__ __ins_6, int8_t* __restrict__ __ins_7, float* __restrict__ __ins_8, float* __restrict__ __ins_9, int8_t* __restrict__ __ins_10, float* __restrict__ __ins_11, float* __restrict__ __ins_12, int8_t* __restrict__ __ins_13, float* __restrict__ __ins_14, float* __restrict__ __ins_15, int8_t* __restrict__ __ins_16, float* __restrict__ __ins_17, float* __restrict__ __ins_18, int8_t* __restrict__ __ins_19, float* __restrict__ __ins_20, float* __restrict__ __ins_21, int8_t* __restrict__ __ins_22, float* __restrict__ __ins_23, float* __restrict__ __ins_24, int8_t* __restrict__ __ins_25, float* __restrict__ __ins_26, float* __restrict__ __ins_27, int8_t* __restrict__ __ins_28, float* __restrict__ __ins_29, float* __restrict__ __ins_30, int8_t* __restrict__ __ins_31, float* __restrict__ __ins_32, float* __restrict__ __ins_33, int8_t* __restrict__ __ins_34, float* __restrict__ __ins_35, float* __restrict__ __ins_36, int8_t* __restrict__ __ins_37, float* __restrict__ __ins_38, float* __restrict__ __ins_39, int8_t* __restrict__ __ins_40, float* __restrict__ __ins_41, float* __restrict__ __ins_42, int8_t* __restrict__ __ins_43, float* __restrict__ __ins_44, float* __restrict__ __ins_45, int8_t* __restrict__ __ins_46, float* __restrict__ __ins_47, float* __restrict__ __ins_48, int8_t* __restrict__ __ins_49, float* __restrict__ __ins_50, float* __restrict__ __ins_51, int8_t* __restrict__ __ins_52, float* __restrict__ __ins_53, float* __restrict__ __ins_54, int8_t* __restrict__ __ins_55, float* __restrict__ __ins_56, float* __restrict__ __ins_57, int8_t* __restrict__ __ins_58, float* __restrict__ __ins_59, float* __restrict__ __ins_60, int8_t* __restrict__ __ins_61, float* __restrict__ __ins_62, float* __restrict__ __ins_63, int8_t* __restrict__ __ins_64, float* __restrict__ __ins_65, float* __restrict__ __ins_66, int8_t* __restrict__ __ins_67, float* __restrict__ __ins_68, float* __restrict__ __ins_69, int8_t* __restrict__ __ins_70, float* __restrict__ __ins_71, float* __restrict__ __ins_72, int8_t* __restrict__ __ins_73, float* __restrict__ __ins_74, float* __restrict__ __ins_75, int8_t* __restrict__ __ins_76, float* __restrict__ __ins_77, float* __restrict__ __ins_78, int8_t* __restrict__ __ins_79, float* __restrict__ __ins_80, float* __restrict__ __ins_81, int8_t* __restrict__ __ins_82, float* __restrict__ __ins_83, float* __restrict__ __ins_84, int8_t* __restrict__ __ins_85, float* __restrict__ __ins_86, float* __restrict__ __ins_87, int8_t* __restrict__ __ins_88, float* __restrict__ __ins_89, float* __restrict__ __ins_90, int8_t* __restrict__ __ins_91, float* __restrict__ __ins_92, float* __restrict__ __ins_93, int8_t* __restrict__ __ins_94, float* __restrict__ __ins_95, float* __restrict__ __ins_96, int8_t* __restrict__ __ins_97, float* __restrict__ __ins_98, float* __restrict__ __ins_99, int8_t* __restrict__ __ins_100, float* __restrict__ __ins_101, float* __restrict__ __ins_102, int8_t* __restrict__ __ins_103, float* __restrict__ __ins_104, float* __restrict__ __ins_105, int8_t* __restrict__ __ins_106, float* __restrict__ __ins_107, float* __restrict__ __ins_108, int8_t* __restrict__ __ins_109, float* __restrict__ __ins_110, float* __restrict__ __ins_111, int8_t* __restrict__ __ins_112, float* __restrict__ __ins_113, float* __restrict__ __ins_114, int8_t* __restrict__ __ins_115, float* __restrict__ __ins_116, float* __restrict__ __ins_117, int8_t* __restrict__ __ins_118, float* __restrict__ __ins_119, float* __restrict__ __ins_120, int8_t* __restrict__ __ins_121, float* __restrict__ __ins_122, float* __restrict__ __ins_123, int8_t* __restrict__ __ins_124, float* __restrict__ __ins_125, float* __restrict__ __ins_126) noexcept{
  for (uint64_t __batchwise_iter_0 = 0UL; __batchwise_iter_0 < 2UL; __batchwise_iter_0 += 1UL) {
    int8_t* __rescheduled_1 = (int8_t*)sc_aligned_malloc(__stream, 2021632UL);
    // [s8 [1, 1, 58, 58, 64] @ ABCD64b]
    int8_t* buffer_127 = (int8_t*)&__rescheduled_1[0UL];
    res2a_conv_0_cast_mul_add_relu_cast__8(buffer_127, &__ins_0[(__batchwise_iter_0 * 200704UL)], &__ins_4[0UL], &__ins_5[0UL], &__ins_6[0UL]);
    // [s8 [1, 2, 56, 56, 32] @ ABCD32b]
    int8_t* buffer_128 = (int8_t*)&__rescheduled_1[802816UL];
    res2a_conv_1_cast_mul_add_relu_cast__12(buffer_128, buffer_127, &__ins_7[0UL], &__ins_8[0UL], &__ins_9[0UL]);
    // [s8 [1, 4, 56, 56, 64] @ ABCD64b]
    int8_t* buffer_129 = (int8_t*)&__rescheduled_1[0UL];
    res2a_conv_b_cast_mul_add_cast_reorder__4(buffer_129, &__ins_0[(__batchwise_iter_0 * 200704UL)], &__ins_1[0UL], &__ins_2[0UL], &__ins_3[0UL]);
    // [u8 [1, 8, 56, 56, 32] @ ABCD32b]
    uint8_t* buffer_130 = (uint8_t*)&__rescheduled_1[1218816UL];
    res2a_conv_2_cast_mul_add_cast_add_cast_reorder__16(buffer_130, buffer_128, &__ins_10[0UL], &__ins_11[0UL], &__ins_12[0UL], buffer_129);
    // [s8 [1, 1, 58, 58, 64] @ ABCD64b]
    int8_t* buffer_131 = (int8_t*)&__rescheduled_1[0UL];
    res2b_conv_0_cast_mul_add_relu_cast__20(buffer_131, buffer_130, &__ins_13[0UL], &__ins_14[0UL], &__ins_15[0UL]);
    // [s8 [1, 1, 56, 56, 64] @ ABCD64b]
    int8_t* buffer_132 = (int8_t*)&__rescheduled_1[215296UL];
    res2b_conv_1_cast_mul_add_relu_cast_reorder__24(buffer_132, buffer_131, &__ins_16[0UL], &__ins_17[0UL], &__ins_18[0UL]);
    // [u8 [1, 4, 56, 56, 64] @ ABCD64b]
    uint8_t* buffer_133 = (uint8_t*)&__rescheduled_1[416000UL];
    res2b_conv_2_cast_mul_add_cast_add_cast_reorder__28(buffer_133, buffer_132, &__ins_19[0UL], &__ins_20[0UL], &__ins_21[0UL], buffer_130);
    // [s8 [1, 1, 58, 58, 64] @ ABCD64b]
    int8_t* buffer_134 = (int8_t*)&__rescheduled_1[0UL];
    res2c_conv_0_cast_mul_add_relu_cast_reorder__32(buffer_134, buffer_133, &__ins_22[0UL], &__ins_23[0UL], &__ins_24[0UL]);
    // [s8 [1, 1, 56, 56, 64] @ ABCD64b]
    int8_t* buffer_135 = (int8_t*)&__rescheduled_1[215296UL];
    res2c_conv_1_cast_mul_add_relu_cast_reorder__36(buffer_135, buffer_134, &__ins_25[0UL], &__ins_26[0UL], &__ins_27[0UL]);
    // [u8 [1, 2, 56, 56, 128] @ ABCD128b]
    uint8_t* buffer_136 = (uint8_t*)&__rescheduled_1[1218816UL];
    res2c_conv_2_cast_mul_add_cast_add_cast_reorder__40(buffer_136, buffer_135, &__ins_28[0UL], &__ins_29[0UL], &__ins_30[0UL], buffer_133);
    // [s8 [1, 1, 58, 58, 128] @ ABCD128b]
    int8_t* buffer_137 = (int8_t*)&__rescheduled_1[0UL];
    res3a_conv_0_cast_mul_add_relu_cast_reorder__48(buffer_137, buffer_136, &__ins_34[0UL], &__ins_35[0UL], &__ins_36[0UL]);
    // [s8 [1, 4, 28, 28, 32] @ ABCD32b]
    int8_t* buffer_138 = (int8_t*)&__rescheduled_1[430592UL];
    res3a_conv_1_cast_mul_add_relu_cast__52(buffer_138, buffer_137, &__ins_37[0UL], &__ins_38[0UL], &__ins_39[0UL]);
    // [s8 [1, 16, 28, 28, 32] @ ABCD32b]
    int8_t* buffer_139 = (int8_t*)&__rescheduled_1[0UL];
    res3a_conv_b_cast_mul_add_cast__44(buffer_139, buffer_136, &__ins_31[0UL], &__ins_32[0UL], &__ins_33[0UL]);
    // [u8 [1, 4, 28, 28, 128] @ ABCD128b]
    uint8_t* buffer_140 = (uint8_t*)&__rescheduled_1[530944UL];
    res3a_conv_2_cast_mul_add_cast_add_cast_reorder__56(buffer_140, buffer_138, &__ins_40[0UL], &__ins_41[0UL], &__ins_42[0UL], buffer_139);
    // [s8 [1, 2, 30, 30, 64] @ ABCD64b]
    int8_t* buffer_141 = (int8_t*)&__rescheduled_1[0UL];
    res3b_conv_0_cast_mul_add_relu_cast_reorder__60(buffer_141, buffer_140, &__ins_43[0UL], &__ins_44[0UL], &__ins_45[0UL]);
    // [s8 [1, 2, 28, 28, 64] @ ABCD64b]
    int8_t* buffer_142 = (int8_t*)&__rescheduled_1[115200UL];
    res3b_conv_1_cast_mul_add_relu_cast_reorder__64(buffer_142, buffer_141, &__ins_46[0UL], &__ins_47[0UL], &__ins_48[0UL]);
    // [u8 [1, 8, 28, 28, 64] @ ABCD64b]
    uint8_t* buffer_143 = (uint8_t*)&__rescheduled_1[932352UL];
    res3b_conv_2_cast_mul_add_cast_add_cast_reorder__68(buffer_143, buffer_142, &__ins_49[0UL], &__ins_50[0UL], &__ins_51[0UL], buffer_140);
    // [s8 [1, 2, 30, 30, 64] @ ABCD64b]
    int8_t* buffer_144 = (int8_t*)&__rescheduled_1[1333760UL];
    res3c_conv_0_cast_mul_add_relu_cast_reorder__72(buffer_144, buffer_143, &__ins_52[0UL], &__ins_53[0UL], &__ins_54[0UL]);
    // [s8 [1, 1, 28, 28, 128] @ ABCD128b]
    int8_t* buffer_145 = (int8_t*)&__rescheduled_1[1448960UL];
    res3c_conv_1_cast_mul_add_relu_cast_reorder__76(buffer_145, buffer_144, &__ins_55[0UL], &__ins_56[0UL], &__ins_57[0UL]);
    // [u8 [1, 4, 28, 28, 128] @ ABCD128b]
    uint8_t* buffer_146 = (uint8_t*)&__rescheduled_1[1549312UL];
    res3c_conv_2_cast_mul_add_cast_add_cast_reorder__80(buffer_146, buffer_145, &__ins_58[0UL], &__ins_59[0UL], &__ins_60[0UL], buffer_143);
    // [s8 [1, 1, 30, 30, 128] @ ABCD128b]
    int8_t* buffer_147 = (int8_t*)&__rescheduled_1[0UL];
    res3d_conv_0_cast_mul_add_relu_cast_reorder__84(buffer_147, buffer_146, &__ins_61[0UL], &__ins_62[0UL], &__ins_63[0UL]);
    // [s8 [1, 4, 28, 28, 32] @ ABCD32b]
    int8_t* buffer_148 = (int8_t*)&__rescheduled_1[115200UL];
    res3d_conv_1_cast_mul_add_relu_cast__88(buffer_148, buffer_147, &__ins_64[0UL], &__ins_65[0UL], &__ins_66[0UL]);
    // [u8 [1, 4, 28, 28, 128] @ ABCD128b]
    uint8_t* buffer_149 = (uint8_t*)&__rescheduled_1[215552UL];
    res3d_conv_2_cast_mul_add_cast_add_cast__92(buffer_149, buffer_148, &__ins_67[0UL], &__ins_68[0UL], &__ins_69[0UL], buffer_146);
    // [s8 [1, 1, 30, 30, 256] @ ABCD256b]
    int8_t* buffer_150 = (int8_t*)&__rescheduled_1[616960UL];
    res4a_conv_0_cast_mul_add_relu_cast__100(buffer_150, buffer_149, &__ins_73[0UL], &__ins_74[0UL], &__ins_75[0UL]);
    // [s8 [1, 1, 14, 14, 256] @ ABCD256b]
    int8_t* buffer_151 = (int8_t*)&__rescheduled_1[0UL];
    res4a_conv_1_cast_mul_add_relu_cast_reorder__104(buffer_151, buffer_150, &__ins_76[0UL], &__ins_77[0UL], &__ins_78[0UL]);
    // [s8 [1, 8, 14, 14, 128] @ ABCD128b]
    int8_t* buffer_152 = (int8_t*)&__rescheduled_1[616960UL];
    res4a_conv_b_cast_mul_add_cast_reorder__96(buffer_152, buffer_149, &__ins_70[0UL], &__ins_71[0UL], &__ins_72[0UL]);
    // [u8 [1, 1, 14, 14, 1024] @ ABCD1024b]
    uint8_t* buffer_153 = (uint8_t*)&__rescheduled_1[50176UL];
    res4a_conv_2_cast_mul_add_cast_add_cast_reorder__108(buffer_153, buffer_151, &__ins_79[0UL], &__ins_80[0UL], &__ins_81[0UL], buffer_152);
    // [s8 [1, 1, 16, 16, 256] @ ABCD256b]
    int8_t* buffer_154 = (int8_t*)&__rescheduled_1[250880UL];
    res4b_conv_0_cast_mul_add_relu_cast_reorder__112(buffer_154, buffer_153, &__ins_82[0UL], &__ins_83[0UL], &__ins_84[0UL]);
    // [s8 [1, 4, 14, 14, 64] @ ABCD64b]
    int8_t* buffer_155 = (int8_t*)&__rescheduled_1[0UL];
    res4b_conv_1_cast_mul_add_relu_cast_reorder__116(buffer_155, buffer_154, &__ins_85[0UL], &__ins_86[0UL], &__ins_87[0UL]);
    // [u8 [1, 8, 14, 14, 128] @ ABCD128b]
    uint8_t* buffer_156 = (uint8_t*)&__rescheduled_1[250880UL];
    res4b_conv_2_cast_mul_add_cast_add_cast_reorder__120(buffer_156, buffer_155, &__ins_88[0UL], &__ins_89[0UL], &__ins_90[0UL], buffer_153);
    // [s8 [1, 2, 16, 16, 128] @ ABCD128b]
    int8_t* buffer_157 = (int8_t*)&__rescheduled_1[0UL];
    res4c_conv_0_cast_mul_add_relu_cast_reorder__124(buffer_157, buffer_156, &__ins_91[0UL], &__ins_92[0UL], &__ins_93[0UL]);
    // [s8 [1, 1, 14, 14, 256] @ ABCD256b]
    int8_t* buffer_158 = (int8_t*)&__rescheduled_1[65536UL];
    res4c_conv_1_cast_mul_add_relu_cast_reorder__128(buffer_158, buffer_157, &__ins_94[0UL], &__ins_95[0UL], &__ins_96[0UL]);
    // [u8 [1, 8, 14, 14, 128] @ ABCD128b]
    uint8_t* buffer_159 = (uint8_t*)&__rescheduled_1[451584UL];
    res4c_conv_2_cast_mul_add_cast_add_cast__132(buffer_159, buffer_158, &__ins_97[0UL], &__ins_98[0UL], &__ins_99[0UL], buffer_156);
    // [s8 [1, 4, 16, 16, 64] @ ABCD64b]
    int8_t* buffer_160 = (int8_t*)&__rescheduled_1[0UL];
    res4d_conv_0_cast_mul_add_relu_cast__136(buffer_160, buffer_159, &__ins_100[0UL], &__ins_101[0UL], &__ins_102[0UL]);
    // [s8 [1, 2, 14, 14, 128] @ ABCD128b]
    int8_t* buffer_161 = (int8_t*)&__rescheduled_1[65536UL];
    res4d_conv_1_cast_mul_add_relu_cast_reorder__140(buffer_161, buffer_160, &__ins_103[0UL], &__ins_104[0UL], &__ins_105[0UL]);
    // [u8 [1, 8, 14, 14, 128] @ ABCD128b]
    uint8_t* buffer_162 = (uint8_t*)&__rescheduled_1[115712UL];
    res4d_conv_2_cast_mul_add_cast_add_cast__144(buffer_162, buffer_161, &__ins_106[0UL], &__ins_107[0UL], &__ins_108[0UL], buffer_159);
    // [s8 [1, 2, 16, 16, 128] @ ABCD128b]
    int8_t* buffer_163 = (int8_t*)&__rescheduled_1[0UL];
    res4e_conv_0_cast_mul_add_relu_cast_reorder__148(buffer_163, buffer_162, &__ins_109[0UL], &__ins_110[0UL], &__ins_111[0UL]);
    // [s8 [1, 1, 14, 14, 256] @ ABCD256b]
    int8_t* buffer_164 = (int8_t*)&__rescheduled_1[65536UL];
    res4e_conv_1_cast_mul_add_relu_cast_reorder__152(buffer_164, buffer_163, &__ins_112[0UL], &__ins_113[0UL], &__ins_114[0UL]);
    // [u8 [1, 8, 14, 14, 128] @ ABCD128b]
    uint8_t* buffer_165 = (uint8_t*)&__rescheduled_1[316416UL];
    // [u8 [1, 4, 14, 14, 256] @ ABCD256b]
    uint8_t* buffer_166 = (uint8_t*)&__rescheduled_1[517120UL];
    res4e_conv_2_cast_mul_add_cast_add_cast_reorder__156(buffer_165, buffer_166, buffer_164, &__ins_115[0UL], &__ins_116[0UL], &__ins_117[0UL], buffer_162);
    // [u8 [1, 2, 14, 14, 512] @ ABCD512b]
    uint8_t* buffer_167 = (uint8_t*)&__rescheduled_1[0UL];
    reorder__157(buffer_167, buffer_165);
    // [s8 [1, 4, 16, 16, 64] @ ABCD64b]
    int8_t* buffer_168 = (int8_t*)&__rescheduled_1[200704UL];
    res4f_conv_0_cast_mul_add_relu_cast_reorder__161(buffer_168, buffer_167, &__ins_118[0UL], &__ins_119[0UL], &__ins_120[0UL]);
    // [s8 [1, 2, 14, 14, 128] @ ABCD128b]
    int8_t* buffer_169 = (int8_t*)&__rescheduled_1[0UL];
    res4f_conv_1_cast_mul_add_relu_cast_reorder__165(buffer_169, buffer_168, &__ins_121[0UL], &__ins_122[0UL], &__ins_123[0UL]);
    res4f_conv_2_cast_mul_add_cast_add_cast__170(&__outs_0[(__batchwise_iter_0 * 200704UL)], buffer_169, &__ins_124[0UL], &__ins_125[0UL], &__ins_126[0UL], buffer_166);
    sc_aligned_free(__stream, __rescheduled_1);
  }
  return true;
}


static bool res2a_conv_0_cast_mul_add_relu_cast__8(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache = *(void**)(__module_data + 8);
  alignas(64) int8_t __rescheduled_0[128UL];
  memset(&__outs_0[0UL], 0, 3712UL);
  for (uint64_t p1 = 0UL; p1 < 56UL; p1 += 1UL) {
    memset(&__outs_0[((p1 + 1UL) * 3712UL)], 0, 64UL);
    memset(&__outs_0[(((p1 + 1UL) * 3712UL) + 3648UL)], 0, 64UL);
  }
  memset(&__outs_0[211584UL], 0, 3712UL);
  for (uint64_t p_o = 0UL; p_o < 28UL; p_o += 1UL) {
    int32_t* __origouts_2600_shr = (int32_t*)sc_aligned_malloc(__stream, 28672UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    void* __cached_0;
    __cached_0 = &__ins_0[(p_o * 7168UL)];
    A_list[0UL] = __cached_0;
    void* __cached_1;
    __cached_1 = &__ins_1[0UL];
    B_list[0UL] = __cached_1;
    void* _arg_cache_0 = &__origouts_2600_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache, A_list, B_list, &__origouts_2600_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
    for (uint64_t _fuseiter9341 = 0UL; _fuseiter9341 < 2UL; _fuseiter9341 += 1UL) {
      for (uint64_t _fuseiter9342 = 0UL; _fuseiter9342 < 56UL; _fuseiter9342 += 1UL) {
        for (uint64_t _fuseiter9343 = 0UL; _fuseiter9343 < 64UL; _fuseiter9343 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2600_shr[((_fuseiter9341 * 3584UL) + ((_fuseiter9342 * 64UL) + _fuseiter9343))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter9343]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter9343]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16::store(__cached_6, &__outs_0[(((((p_o * 2UL) + 1UL) * 3712UL) + 64UL) + ((_fuseiter9341 * 3712UL) + ((_fuseiter9342 * 64UL) + _fuseiter9343)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2600_shr);
  }
  return true;
}

static bool res2a_conv_1_cast_mul_add_relu_cast__12(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_206 = *(void**)(__module_data + 16);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 320UL);
  for (uint64_t fused_0k_o__n_2784 = 0UL; fused_0k_o__n_2784 < 2UL; fused_0k_o__n_2784 += 1UL) {
    int32_t* __origouts_2610_shr = (int32_t*)sc_aligned_malloc(__stream, 401408UL);
    for (uint64_t o_o = 0UL; o_o < 224UL; o_o += 1UL) {
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[128UL];
      memset(&__origouts_2610_shr[((((o_o * 14UL) / 56UL) * 1792UL) + (((o_o * 14UL) % 56UL) * 32UL))], 0, 1792UL);
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_0;
          __cached_0 = &__ins_0[(((((o_o * 14UL) / 56UL) + r) * 3712UL) + ((((o_o * 14UL) % 56UL) + s) * 64UL))];
          A_list[((r * 3UL) + s)] = __cached_0;
          void* __cached_1;
          __cached_1 = &__ins_1[((fused_0k_o__n_2784 * 18432UL) + ((r * 6144UL) + (s * 2048UL)))];
          B_list[((r * 3UL) + s)] = __cached_1;
        }
      }
      void* _arg_cache_1 = &__origouts_2610_shr[((((o_o * 14UL) / 56UL) * 1792UL) + (((o_o * 14UL) % 56UL) * 32UL))];
      dnnl_brgemm_list_call(__sc_kernel_cache_206, A_list, B_list, &__origouts_2610_shr[((((o_o * 14UL) / 56UL) * 1792UL) + (((o_o * 14UL) % 56UL) * 32UL))], 1, 64, 2048, 9, 7, 7, __stream);
    }
    float* _cast_buf_0_shr = (float*)&__rescheduled_0[256UL];
    for (uint64_t _fuseiter9371 = 0UL; _fuseiter9371 < 56UL; _fuseiter9371 += 1UL) {
      for (uint64_t _fuseiter9372 = 0UL; _fuseiter9372 < 56UL; _fuseiter9372 += 1UL) {
        for (uint64_t _fuseiter9373 = 0UL; _fuseiter9373 < 32UL; _fuseiter9373 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2610_shr[((_fuseiter9371 * 1792UL) + ((_fuseiter9372 * 32UL) + _fuseiter9373))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16::store(__cached_3, &_cast_buf_0_shr[((((0UL - _fuseiter9371) * 16UL) + (((0UL - _fuseiter9372) * 16UL) + (0UL - _fuseiter9373))) + ((_fuseiter9371 * 16UL) + ((_fuseiter9372 * 16UL) + _fuseiter9373)))]);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&_cast_buf_0_shr[0UL]);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_2[((fused_0k_o__n_2784 * 32UL) + _fuseiter9373)]);
          __cached_4 = (__cached_4 * __cached_5);
          vec_f32x16 __cached_6;
          __cached_6 = vec_f32x16::load(&__ins_3[((fused_0k_o__n_2784 * 32UL) + _fuseiter9373)]);
          __cached_4 = (__cached_4 + __cached_6);
          __cached_4 = sc_max(__cached_4, vec_f32x16(0.f));
          vec_f32x16::store(__cached_4, &_cast_buf_0_shr[0UL]);
          vec_s8x16 __cached_7;
          __cached_7 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_4));
          vec_s8x16::store(__cached_7, &__outs_0[((fused_0k_o__n_2784 * 100352UL) + ((_fuseiter9371 * 1792UL) + ((_fuseiter9372 * 32UL) + _fuseiter9373)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2610_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res2a_conv_b_cast_mul_add_cast_reorder__4(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_208 = *(void**)(__module_data + 24);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_2785 = 0UL; fused_0n__k_2785 < 16UL; fused_0n__k_2785 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 28UL; p_o += 1UL) {
      int32_t* __origouts_2620_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0n__k_2785 / 16UL) * 200704UL) + (p_o * 7168UL))];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[((fused_0n__k_2785 % 16UL) * 1024UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_2 = &__origouts_2620_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_208, A_list, B_list, &__origouts_2620_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
      for (uint64_t _fuseiter9401 = 0UL; _fuseiter9401 < 2UL; _fuseiter9401 += 1UL) {
        for (uint64_t _fuseiter9402 = 0UL; _fuseiter9402 < 56UL; _fuseiter9402 += 1UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2620_shr[((_fuseiter9401 * 896UL) + (_fuseiter9402 * 16UL))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0n__k_2785 / 16UL) * 256UL) + ((fused_0n__k_2785 % 16UL) * 16UL))]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0n__k_2785 % 16UL) * 16UL)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[(((fused_0n__k_2785 / 16UL) * 802816UL) + (((((fused_0n__k_2785 % 16UL) * 16UL) / 64UL) * 200704UL) + (((_fuseiter9401 + (p_o * 2UL)) * 3584UL) + ((_fuseiter9402 * 64UL) + (((fused_0n__k_2785 % 16UL) * 16UL) % 64UL)))))]);
        }
      }
      sc_aligned_free(__stream, __origouts_2620_shr);
    }
  }
  return true;
}

static bool res2a_conv_2_cast_mul_add_cast_add_cast_reorder__16(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_210 = *(void**)(__module_data + 32);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_2786 = 0UL; fused_0n__k_2786 < 4UL; fused_0n__k_2786 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
      int32_t* __origouts_2630_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0n__k_2786 / 4UL) * 200704UL) + ((c * 100352UL) + (p_o * 1792UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0n__k_2786 % 4UL) * 4096UL) + (c * 2048UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_3 = &__origouts_2630_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_210, A_list, B_list, &__origouts_2630_shr[0UL], 1, 1, 1, 2, 7, 7, __stream);
      for (uint64_t _fuseiter9431 = 0UL; _fuseiter9431 < 56UL; _fuseiter9431 += 1UL) {
        for (uint64_t _fuseiter9432 = 0UL; _fuseiter9432 < 64UL; _fuseiter9432 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2630_shr[((_fuseiter9431 * 64UL) + _fuseiter9432)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0n__k_2786 / 4UL) * 256UL) + ((fused_0n__k_2786 % 4UL) * 64UL)) + _fuseiter9432)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_2786 % 4UL) * 64UL) + _fuseiter9432)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0n__k_2786 / 4UL) * 802816UL) + (((fused_0n__k_2786 % 4UL) * 200704UL) + (p_o * 3584UL))) + ((_fuseiter9431 * 64UL) + _fuseiter9432))]);
          __cached_6 = (__cached_6 + __cached_7);
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16 __cached_9;
          __cached_9 = __cached_8;
          vec_u8x16::store(__cached_9, &__outs_0[(((fused_0n__k_2786 / 4UL) * 802816UL) + ((((_fuseiter9432 + ((fused_0n__k_2786 % 4UL) * 64UL)) / 32UL) * 100352UL) + ((p_o * 1792UL) + ((_fuseiter9431 * 32UL) + ((_fuseiter9432 + ((fused_0n__k_2786 % 4UL) * 64UL)) % 32UL)))))]);
        }
      }
      sc_aligned_free(__stream, __origouts_2630_shr);
    }
  }
  return true;
}

static bool res2b_conv_0_cast_mul_add_relu_cast__20(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_212 = *(void**)(__module_data + 40);
  alignas(64) int8_t __rescheduled_0[128UL];
  memset(&__outs_0[0UL], 0, 3712UL);
  for (uint64_t p1 = 0UL; p1 < 56UL; p1 += 1UL) {
    memset(&__outs_0[((p1 + 1UL) * 3712UL)], 0, 64UL);
    memset(&__outs_0[(((p1 + 1UL) * 3712UL) + 3648UL)], 0, 64UL);
  }
  memset(&__outs_0[211584UL], 0, 3712UL);
  for (uint64_t p_o = 0UL; p_o < 8UL; p_o += 1UL) {
    int32_t* __origouts_2640_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[((c * 100352UL) + (p_o * 12544UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(c * 2048UL)];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_4 = &__origouts_2640_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_212, A_list, B_list, &__origouts_2640_shr[0UL], 1, 1, 1, 8, 8, 7, __stream);
    for (uint64_t _fuseiter9471 = 0UL; _fuseiter9471 < 7UL; _fuseiter9471 += 1UL) {
      for (uint64_t _fuseiter9472 = 0UL; _fuseiter9472 < 56UL; _fuseiter9472 += 1UL) {
        for (uint64_t _fuseiter9473 = 0UL; _fuseiter9473 < 64UL; _fuseiter9473 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2640_shr[((_fuseiter9471 * 3584UL) + ((_fuseiter9472 * 64UL) + _fuseiter9473))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter9473]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter9473]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16::store(__cached_6, &__outs_0[(((((p_o * 7UL) + 1UL) * 3712UL) + 64UL) + ((_fuseiter9471 * 3712UL) + ((_fuseiter9472 * 64UL) + _fuseiter9473)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2640_shr);
  }
  return true;
}

static bool res2b_conv_1_cast_mul_add_relu_cast_reorder__24(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_214 = *(void**)(__module_data + 48);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
  for (uint64_t fused_0n__k_o_2789 = 0UL; fused_0n__k_o_2789 < 4UL; fused_0n__k_o_2789 += 1UL) {
    int32_t* __origouts_2650_shr = (int32_t*)sc_aligned_malloc(__stream, 200704UL);
    for (uint64_t o_o = 0UL; o_o < 112UL; o_o += 1UL) {
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[128UL];
      memset(&__origouts_2650_shr[((((o_o * 28UL) / 56UL) * 896UL) + (((o_o * 28UL) % 56UL) * 16UL))], 0, 1792UL);
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_0;
          __cached_0 = &__ins_0[(((fused_0n__k_o_2789 / 4UL) * 215296UL) + (((((o_o * 28UL) / 56UL) + r) * 3712UL) + ((((o_o * 28UL) % 56UL) + s) * 64UL)))];
          A_list[((r * 3UL) + s)] = __cached_0;
          void* __cached_1;
          __cached_1 = &__ins_1[(((fused_0n__k_o_2789 % 4UL) * 9216UL) + ((r * 3072UL) + (s * 1024UL)))];
          B_list[((r * 3UL) + s)] = __cached_1;
        }
      }
      void* _arg_cache_5 = &__origouts_2650_shr[((((o_o * 28UL) / 56UL) * 896UL) + (((o_o * 28UL) % 56UL) * 16UL))];
      dnnl_brgemm_list_call(__sc_kernel_cache_214, A_list, B_list, &__origouts_2650_shr[((((o_o * 28UL) / 56UL) * 896UL) + (((o_o * 28UL) % 56UL) * 16UL))], 1, 64, 1024, 9, 7, 7, __stream);
    }
    for (uint64_t _fuseiter9501 = 0UL; _fuseiter9501 < 56UL; _fuseiter9501 += 1UL) {
      for (uint64_t _fuseiter9502 = 0UL; _fuseiter9502 < 56UL; _fuseiter9502 += 1UL) {
        vec_s32x16 __cached_2;
        __cached_2 = vec_s32x16::load(&__origouts_2650_shr[((_fuseiter9501 * 896UL) + (_fuseiter9502 * 16UL))]);
        vec_f32x16 __cached_3;
        __cached_3 = (vec_f32x16)(__cached_2);
        vec_f32x16 __cached_4;
        __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0n__k_o_2789 / 4UL) * 64UL) + ((fused_0n__k_o_2789 % 4UL) * 16UL))]);
        __cached_3 = (__cached_3 * __cached_4);
        vec_f32x16 __cached_5;
        __cached_5 = vec_f32x16::load(&__ins_3[((fused_0n__k_o_2789 % 4UL) * 16UL)]);
        __cached_3 = (__cached_3 + __cached_5);
        __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
        vec_s8x16 __cached_6;
        __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
        vec_s8x16 __cached_7;
        __cached_7 = __cached_6;
        vec_s8x16::store(__cached_7, &__outs_0[(((fused_0n__k_o_2789 / 4UL) * 200704UL) + (((((fused_0n__k_o_2789 % 4UL) * 16UL) / 64UL) * 200704UL) + ((_fuseiter9501 * 3584UL) + ((_fuseiter9502 * 64UL) + (((fused_0n__k_o_2789 % 4UL) * 16UL) % 64UL)))))]);
      }
    }
    sc_aligned_free(__stream, __origouts_2650_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res2b_conv_2_cast_mul_add_cast_add_cast_reorder__28(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_216 = *(void**)(__module_data + 56);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0k__n_2790 = 0UL; fused_0k__n_2790 < 8UL; fused_0k__n_2790 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
      int32_t* __origouts_2660_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(p_o * 3584UL)];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(fused_0k__n_2790 * 2048UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_6 = &__origouts_2660_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_216, A_list, B_list, &__origouts_2660_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
      for (uint64_t _fuseiter9537 = 0UL; _fuseiter9537 < 56UL; _fuseiter9537 += 1UL) {
        for (uint64_t _fuseiter9538 = 0UL; _fuseiter9538 < 32UL; _fuseiter9538 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2660_shr[((_fuseiter9537 * 32UL) + _fuseiter9538)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_2790 * 32UL) + _fuseiter9538)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_2790 * 32UL) + _fuseiter9538)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_u8x16 __cached_7;
          __cached_7 = vec_u8x16::load(&__ins_4[(((fused_0k__n_2790 * 100352UL) + (p_o * 1792UL)) + ((_fuseiter9537 * 32UL) + _fuseiter9538))]);
          __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16 __cached_9;
          __cached_9 = __cached_8;
          vec_u8x16::store(__cached_9, &__outs_0[((((_fuseiter9538 + (fused_0k__n_2790 * 32UL)) / 64UL) * 200704UL) + ((p_o * 3584UL) + ((_fuseiter9537 * 64UL) + ((_fuseiter9538 + (fused_0k__n_2790 * 32UL)) % 64UL))))]);
        }
      }
      sc_aligned_free(__stream, __origouts_2660_shr);
    }
  }
  return true;
}

static bool res2c_conv_0_cast_mul_add_relu_cast_reorder__32(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_218 = *(void**)(__module_data + 64);
  alignas(64) int8_t __rescheduled_0[128UL];
  memset(&__outs_0[0UL], 0, 3712UL);
  for (uint64_t p1 = 0UL; p1 < 56UL; p1 += 1UL) {
    memset(&__outs_0[((p1 + 1UL) * 3712UL)], 0, 64UL);
    memset(&__outs_0[(((p1 + 1UL) * 3712UL) + 3648UL)], 0, 64UL);
  }
  memset(&__outs_0[211584UL], 0, 3712UL);
  for (uint64_t fused_0n__k_2792 = 0UL; fused_0n__k_2792 < 2UL; fused_0n__k_2792 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
      int32_t* __origouts_2670_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0n__k_2792 / 2UL) * 802816UL) + ((c * 200704UL) + (p_o * 3584UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0n__k_2792 % 2UL) * 8192UL) + (c * 2048UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_7 = &__origouts_2670_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_218, A_list, B_list, &__origouts_2670_shr[0UL], 1, 1, 1, 4, 8, 7, __stream);
      for (uint64_t _fuseiter9578 = 0UL; _fuseiter9578 < 56UL; _fuseiter9578 += 1UL) {
        for (uint64_t _fuseiter9579 = 0UL; _fuseiter9579 < 32UL; _fuseiter9579 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2670_shr[((_fuseiter9578 * 32UL) + _fuseiter9579)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0n__k_2792 / 2UL) * 64UL) + ((fused_0n__k_2792 % 2UL) * 32UL)) + _fuseiter9579)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_2792 % 2UL) * 32UL) + _fuseiter9579)]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[(((fused_0n__k_2792 / 2UL) * 215296UL) + ((((_fuseiter9579 + ((fused_0n__k_2792 % 2UL) * 32UL)) / 64UL) * 215296UL) + (((p_o + 1UL) * 3712UL) + (((_fuseiter9578 + 1UL) * 64UL) + ((_fuseiter9579 + ((fused_0n__k_2792 % 2UL) * 32UL)) % 64UL)))))]);
        }
      }
      sc_aligned_free(__stream, __origouts_2670_shr);
    }
  }
  return true;
}

static bool res2c_conv_1_cast_mul_add_relu_cast_reorder__36(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_220 = *(void**)(__module_data + 72);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
  for (uint64_t fused_0k_o__n_2793 = 0UL; fused_0k_o__n_2793 < 2UL; fused_0k_o__n_2793 += 1UL) {
    int32_t* __origouts_2680_shr = (int32_t*)sc_aligned_malloc(__stream, 401408UL);
    for (uint64_t o_o = 0UL; o_o < 112UL; o_o += 1UL) {
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[128UL];
      memset(&__origouts_2680_shr[((((o_o * 28UL) / 56UL) * 1792UL) + (((o_o * 28UL) % 56UL) * 32UL))], 0, 3584UL);
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_0;
          __cached_0 = &__ins_0[(((((o_o * 28UL) / 56UL) + r) * 3712UL) + ((((o_o * 28UL) % 56UL) + s) * 64UL))];
          A_list[((r * 3UL) + s)] = __cached_0;
          void* __cached_1;
          __cached_1 = &__ins_1[((fused_0k_o__n_2793 * 18432UL) + ((r * 6144UL) + (s * 2048UL)))];
          B_list[((r * 3UL) + s)] = __cached_1;
        }
      }
      void* _arg_cache_8 = &__origouts_2680_shr[((((o_o * 28UL) / 56UL) * 1792UL) + (((o_o * 28UL) % 56UL) * 32UL))];
      dnnl_brgemm_list_call(__sc_kernel_cache_220, A_list, B_list, &__origouts_2680_shr[((((o_o * 28UL) / 56UL) * 1792UL) + (((o_o * 28UL) % 56UL) * 32UL))], 1, 64, 2048, 9, 7, 7, __stream);
    }
    for (uint64_t _fuseiter9612 = 0UL; _fuseiter9612 < 56UL; _fuseiter9612 += 1UL) {
      for (uint64_t _fuseiter9613 = 0UL; _fuseiter9613 < 56UL; _fuseiter9613 += 1UL) {
        for (uint64_t _fuseiter9614 = 0UL; _fuseiter9614 < 32UL; _fuseiter9614 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2680_shr[((_fuseiter9612 * 1792UL) + ((_fuseiter9613 * 32UL) + _fuseiter9614))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k_o__n_2793 * 32UL) + _fuseiter9614)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k_o__n_2793 * 32UL) + _fuseiter9614)]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[((((_fuseiter9614 + (fused_0k_o__n_2793 * 32UL)) / 64UL) * 200704UL) + ((_fuseiter9612 * 3584UL) + ((_fuseiter9613 * 64UL) + ((_fuseiter9614 + (fused_0k_o__n_2793 * 32UL)) % 64UL))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2680_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res2c_conv_2_cast_mul_add_cast_add_cast_reorder__40(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_222 = *(void**)(__module_data + 80);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_2794 = 0UL; fused_0n__k_2794 < 4UL; fused_0n__k_2794 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
      int32_t* __origouts_2690_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0n__k_2794 / 4UL) * 200704UL) + (p_o * 3584UL))];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[((fused_0n__k_2794 % 4UL) * 4096UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_9 = &__origouts_2690_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_222, A_list, B_list, &__origouts_2690_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
      for (uint64_t _fuseiter9648 = 0UL; _fuseiter9648 < 56UL; _fuseiter9648 += 1UL) {
        for (uint64_t _fuseiter9649 = 0UL; _fuseiter9649 < 64UL; _fuseiter9649 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2690_shr[((_fuseiter9648 * 64UL) + _fuseiter9649)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0n__k_2794 / 4UL) * 256UL) + ((fused_0n__k_2794 % 4UL) * 64UL)) + _fuseiter9649)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_2794 % 4UL) * 64UL) + _fuseiter9649)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_u8x16 __cached_7;
          __cached_7 = vec_u8x16::load(&__ins_4[((((fused_0n__k_2794 / 4UL) * 802816UL) + (((fused_0n__k_2794 % 4UL) * 200704UL) + (p_o * 3584UL))) + ((_fuseiter9648 * 64UL) + _fuseiter9649))]);
          __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16 __cached_9;
          __cached_9 = __cached_8;
          vec_u8x16::store(__cached_9, &__outs_0[(((fused_0n__k_2794 / 4UL) * 802816UL) + ((((_fuseiter9649 + ((fused_0n__k_2794 % 4UL) * 64UL)) / 128UL) * 401408UL) + ((p_o * 7168UL) + ((_fuseiter9648 * 128UL) + ((_fuseiter9649 + ((fused_0n__k_2794 % 4UL) * 64UL)) % 128UL)))))]);
        }
      }
      sc_aligned_free(__stream, __origouts_2690_shr);
    }
  }
  return true;
}

static bool res3a_conv_0_cast_mul_add_relu_cast_reorder__48(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_224 = *(void**)(__module_data + 88);
  alignas(64) int8_t __rescheduled_0[128UL];
  memset(&__outs_0[0UL], 0, 7424UL);
  for (uint64_t p1 = 0UL; p1 < 56UL; p1 += 1UL) {
    memset(&__outs_0[((p1 + 1UL) * 7424UL)], 0, 128UL);
    memset(&__outs_0[(((p1 + 1UL) * 7424UL) + 7296UL)], 0, 128UL);
  }
  memset(&__outs_0[423168UL], 0, 7424UL);
  for (uint64_t fused_0k__n_2796 = 0UL; fused_0k__n_2796 < 2UL; fused_0k__n_2796 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 28UL; p_o += 1UL) {
      int32_t* __origouts_2700_shr = (int32_t*)sc_aligned_malloc(__stream, 28672UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[((c * 401408UL) + (p_o * 14336UL))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[((fused_0k__n_2796 * 16384UL) + (c * 8192UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_10 = &__origouts_2700_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_224, A_list, B_list, &__origouts_2700_shr[0UL], 1, 1, 1, 2, 8, 7, __stream);
      for (uint64_t _fuseiter9688 = 0UL; _fuseiter9688 < 2UL; _fuseiter9688 += 1UL) {
        for (uint64_t _fuseiter9689 = 0UL; _fuseiter9689 < 56UL; _fuseiter9689 += 1UL) {
          for (uint64_t _fuseiter9690 = 0UL; _fuseiter9690 < 64UL; _fuseiter9690 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_2700_shr[((_fuseiter9688 * 3584UL) + ((_fuseiter9689 * 64UL) + _fuseiter9690))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_2796 * 64UL) + _fuseiter9690)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_2796 * 64UL) + _fuseiter9690)]);
            __cached_3 = (__cached_3 + __cached_5);
            __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = __cached_6;
            vec_s8x16::store(__cached_7, &__outs_0[((((_fuseiter9690 + (fused_0k__n_2796 * 64UL)) / 128UL) * 430592UL) + ((((_fuseiter9688 + (p_o * 2UL)) + 1UL) * 7424UL) + (((_fuseiter9689 + 1UL) * 128UL) + ((_fuseiter9690 + (fused_0k__n_2796 * 64UL)) % 128UL))))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_2700_shr);
    }
  }
  return true;
}

static bool res3a_conv_1_cast_mul_add_relu_cast__52(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_226 = *(void**)(__module_data + 96);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 320UL);
  for (uint64_t fused_0n__k_o_2797 = 0UL; fused_0n__k_o_2797 < 4UL; fused_0n__k_o_2797 += 1UL) {
    int32_t* __origouts_2710_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    for (uint64_t o_o = 0UL; o_o < 28UL; o_o += 1UL) {
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[128UL];
      memset(&__origouts_2710_shr[(((o_o * 28UL) / 28UL) * 896UL)], 0, 3584UL);
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_0;
          __cached_0 = &__ins_0[(((fused_0n__k_o_2797 / 4UL) * 430592UL) + ((((((o_o * 28UL) / 28UL) * 2UL) + r) * 7424UL) + (s * 128UL)))];
          A_list[((r * 3UL) + s)] = __cached_0;
          void* __cached_1;
          __cached_1 = &__ins_1[(((fused_0n__k_o_2797 % 4UL) * 36864UL) + ((r * 12288UL) + (s * 4096UL)))];
          B_list[((r * 3UL) + s)] = __cached_1;
        }
      }
      void* _arg_cache_11 = &__origouts_2710_shr[(((o_o * 28UL) / 28UL) * 896UL)];
      dnnl_brgemm_list_call(__sc_kernel_cache_226, A_list, B_list, &__origouts_2710_shr[(((o_o * 28UL) / 28UL) * 896UL)], 1, 128, 4096, 9, 7, 7, __stream);
    }
    float* _cast_buf_0_shr = (float*)&__rescheduled_0[256UL];
    for (uint64_t _fuseiter9723 = 0UL; _fuseiter9723 < 28UL; _fuseiter9723 += 1UL) {
      for (uint64_t _fuseiter9724 = 0UL; _fuseiter9724 < 28UL; _fuseiter9724 += 1UL) {
        for (uint64_t _fuseiter9725 = 0UL; _fuseiter9725 < 32UL; _fuseiter9725 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2710_shr[((_fuseiter9723 * 896UL) + ((_fuseiter9724 * 32UL) + _fuseiter9725))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16::store(__cached_3, &_cast_buf_0_shr[((((0UL - _fuseiter9723) * 16UL) + (((0UL - _fuseiter9724) * 16UL) + (0UL - _fuseiter9725))) + ((_fuseiter9723 * 16UL) + ((_fuseiter9724 * 16UL) + _fuseiter9725)))]);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&_cast_buf_0_shr[0UL]);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_2[(((fused_0n__k_o_2797 / 4UL) * 128UL) + (((fused_0n__k_o_2797 % 4UL) * 32UL) + _fuseiter9725))]);
          __cached_4 = (__cached_4 * __cached_5);
          vec_f32x16 __cached_6;
          __cached_6 = vec_f32x16::load(&__ins_3[(((fused_0n__k_o_2797 % 4UL) * 32UL) + _fuseiter9725)]);
          __cached_4 = (__cached_4 + __cached_6);
          __cached_4 = sc_max(__cached_4, vec_f32x16(0.f));
          vec_f32x16::store(__cached_4, &_cast_buf_0_shr[0UL]);
          vec_s8x16 __cached_7;
          __cached_7 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_4));
          vec_s8x16::store(__cached_7, &__outs_0[(((fused_0n__k_o_2797 / 4UL) * 100352UL) + (((fused_0n__k_o_2797 % 4UL) * 25088UL) + ((_fuseiter9723 * 896UL) + ((_fuseiter9724 * 32UL) + _fuseiter9725))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2710_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res3a_conv_b_cast_mul_add_cast__44(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_228 = *(void**)(__module_data + 104);
  alignas(64) int8_t __rescheduled_0[128UL];
  uint8_t* input_tmp = (uint8_t*)sc_aligned_malloc(__stream, 200704UL);
  for (uint64_t fused_0n__c_o_2798 = 0UL; fused_0n__c_o_2798 < 2UL; fused_0n__c_o_2798 += 1UL) {
    for (uint64_t p = 0UL; p < 28UL; p += 1UL) {
      for (uint64_t q = 0UL; q < 28UL; q += 1UL) {
        for (uint64_t c_i = 0UL; c_i < 128UL; c_i += 64UL) {
          vec_u8x64 __cached_0;
          __cached_0 = vec_u8x64::load(&__ins_0[(((fused_0n__c_o_2798 / 2UL) * 802816UL) + (((fused_0n__c_o_2798 % 2UL) * 401408UL) + ((p * 14336UL) + ((q * 256UL) + c_i))))]);
          vec_u8x64 __cached_1;
          __cached_1 = __cached_0;
          vec_u8x64::store(__cached_1, &input_tmp[(((fused_0n__c_o_2798 / 2UL) * 200704UL) + (((fused_0n__c_o_2798 % 2UL) * 100352UL) + ((p * 3584UL) + ((q * 128UL) + c_i))))]);
        }
      }
    }
  }
  for (uint64_t fused_0k__n_2799 = 0UL; fused_0k__n_2799 < 16UL; fused_0k__n_2799 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 4UL; p_o += 1UL) {
      int32_t* __origouts_2720_shr = (int32_t*)sc_aligned_malloc(__stream, 25088UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
        void* __cached_2;
        __cached_2 = &input_tmp[((c * 100352UL) + (p_o * 25088UL))];
        A_list[c] = __cached_2;
        void* __cached_3;
        __cached_3 = &__ins_1[((fused_0k__n_2799 * 8192UL) + (c * 4096UL))];
        B_list[c] = __cached_3;
      }
      void* _arg_cache_12 = &__origouts_2720_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_228, A_list, B_list, &__origouts_2720_shr[0UL], 1, 1, 1, 2, 8, 7, __stream);
      for (uint64_t _fuseiter9753 = 0UL; _fuseiter9753 < 7UL; _fuseiter9753 += 1UL) {
        for (uint64_t _fuseiter9754 = 0UL; _fuseiter9754 < 28UL; _fuseiter9754 += 1UL) {
          for (uint64_t _fuseiter9755 = 0UL; _fuseiter9755 < 32UL; _fuseiter9755 += 16UL) {
            vec_s32x16 __cached_4;
            __cached_4 = vec_s32x16::load(&__origouts_2720_shr[((_fuseiter9753 * 896UL) + ((_fuseiter9754 * 32UL) + _fuseiter9755))]);
            vec_f32x16 __cached_5;
            __cached_5 = (vec_f32x16)(__cached_4);
            vec_f32x16 __cached_6;
            __cached_6 = vec_f32x16::load(&__ins_2[((fused_0k__n_2799 * 32UL) + _fuseiter9755)]);
            __cached_5 = (__cached_5 * __cached_6);
            vec_f32x16 __cached_7;
            __cached_7 = vec_f32x16::load(&__ins_3[((fused_0k__n_2799 * 32UL) + _fuseiter9755)]);
            __cached_5 = (__cached_5 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_5));
            vec_s8x16::store(__cached_8, &__outs_0[(((fused_0k__n_2799 * 25088UL) + (p_o * 6272UL)) + ((_fuseiter9753 * 896UL) + ((_fuseiter9754 * 32UL) + _fuseiter9755)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_2720_shr);
    }
  }
  sc_aligned_free(__stream, input_tmp);
  return true;
}

static bool res3a_conv_2_cast_mul_add_cast_add_cast_reorder__56(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_230 = *(void**)(__module_data + 112);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0k__n_2800 = 0UL; fused_0k__n_2800 < 16UL; fused_0k__n_2800 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 2UL; p_o += 1UL) {
      int32_t* __origouts_2730_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[((c * 25088UL) + (p_o * 12544UL))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[((fused_0k__n_2800 * 4096UL) + (c * 1024UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_13 = &__origouts_2730_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_230, A_list, B_list, &__origouts_2730_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
      for (uint64_t _fuseiter9777 = 0UL; _fuseiter9777 < 14UL; _fuseiter9777 += 1UL) {
        for (uint64_t _fuseiter9778 = 0UL; _fuseiter9778 < 28UL; _fuseiter9778 += 1UL) {
          for (uint64_t _fuseiter9779 = 0UL; _fuseiter9779 < 32UL; _fuseiter9779 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_2730_shr[((_fuseiter9777 * 896UL) + ((_fuseiter9778 * 32UL) + _fuseiter9779))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_2800 * 32UL) + _fuseiter9779)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_2800 * 32UL) + _fuseiter9779)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = vec_s8x16::load(&__ins_4[(((fused_0k__n_2800 * 25088UL) + (p_o * 12544UL)) + ((_fuseiter9777 * 896UL) + ((_fuseiter9778 * 32UL) + _fuseiter9779)))]);
            __cached_6 = (__cached_6 + __cached_7);
            vec_u8x16 __cached_8;
            __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
            vec_u8x16 __cached_9;
            __cached_9 = __cached_8;
            vec_u8x16::store(__cached_9, &__outs_0[((((_fuseiter9779 + (fused_0k__n_2800 * 32UL)) / 128UL) * 100352UL) + (((_fuseiter9777 + (p_o * 14UL)) * 3584UL) + ((_fuseiter9778 * 128UL) + ((_fuseiter9779 + (fused_0k__n_2800 * 32UL)) % 128UL))))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_2730_shr);
    }
  }
  return true;
}

static bool res3b_conv_0_cast_mul_add_relu_cast_reorder__60(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_232 = *(void**)(__module_data + 120);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_2801 = 0UL; fused_0n__k_2801 < 2UL; fused_0n__k_2801 += 1UL) {
    memset(&__outs_0[(((fused_0n__k_2801 / 2UL) * 115200UL) + ((fused_0n__k_2801 % 2UL) * 57600UL))], 0, 1920UL);
    for (uint64_t p1 = 0UL; p1 < 28UL; p1 += 1UL) {
      memset(&__outs_0[(((fused_0n__k_2801 / 2UL) * 115200UL) + (((fused_0n__k_2801 % 2UL) * 57600UL) + ((p1 + 1UL) * 1920UL)))], 0, 64UL);
      memset(&__outs_0[((((fused_0n__k_2801 / 2UL) * 115200UL) + (((fused_0n__k_2801 % 2UL) * 57600UL) + ((p1 + 1UL) * 1920UL))) + 1856UL)], 0, 64UL);
    }
    memset(&__outs_0[((((fused_0n__k_2801 / 2UL) * 115200UL) + ((fused_0n__k_2801 % 2UL) * 57600UL)) + 55680UL)], 0, 1920UL);
  }
  for (uint64_t fused_0k__n_2802 = 0UL; fused_0k__n_2802 < 4UL; fused_0k__n_2802 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_2740_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[((c * 100352UL) + (p_o * 14336UL))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[((fused_0k__n_2802 * 16384UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_14 = &__origouts_2740_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_232, A_list, B_list, &__origouts_2740_shr[0UL], 1, 1, 1, 4, 8, 7, __stream);
      for (uint64_t _fuseiter9818 = 0UL; _fuseiter9818 < 4UL; _fuseiter9818 += 1UL) {
        for (uint64_t _fuseiter9819 = 0UL; _fuseiter9819 < 28UL; _fuseiter9819 += 1UL) {
          for (uint64_t _fuseiter9820 = 0UL; _fuseiter9820 < 32UL; _fuseiter9820 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_2740_shr[((_fuseiter9818 * 896UL) + ((_fuseiter9819 * 32UL) + _fuseiter9820))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_2802 * 32UL) + _fuseiter9820)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_2802 * 32UL) + _fuseiter9820)]);
            __cached_3 = (__cached_3 + __cached_5);
            __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = __cached_6;
            vec_s8x16::store(__cached_7, &__outs_0[((((_fuseiter9820 + (fused_0k__n_2802 * 32UL)) / 64UL) * 57600UL) + ((((_fuseiter9818 + (p_o * 4UL)) + 1UL) * 1920UL) + (((_fuseiter9819 + 1UL) * 64UL) + ((_fuseiter9820 + (fused_0k__n_2802 * 32UL)) % 64UL))))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_2740_shr);
    }
  }
  return true;
}

static bool res3b_conv_1_cast_mul_add_relu_cast_reorder__64(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr = (void**)&__uninitialized_data[23657488UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 512UL);
  int32_t* conv_os_blk_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[64UL];
  int32_t __cached_0;
  __cached_0 = 392;
  conv_os_blk_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 0;
  conv_os_acc_size[0] = __cached_1;
  int32_t __cached_2;
  __cached_2 = 392;
  conv_os_blk_size[1] = __cached_2;
  int32_t __cached_3;
  __cached_3 = 392;
  conv_os_acc_size[1] = __cached_3;
  int32_t* __origouts_2750_shr = (int32_t*)sc_aligned_malloc(__stream, 401408UL);
  for (uint64_t o_o = 0UL; o_o < 2UL; o_o += 1UL) {
    void** A_list = (void**)&__rescheduled_0[128UL];
    void** B_list = (void**)&__rescheduled_0[320UL];
    int32_t __cached_4;
    __cached_4 = conv_os_acc_size[o_o];
    int32_t __cached_5;
    __cached_5 = conv_os_blk_size[o_o];
    memset(&__origouts_2750_shr[(uint64_t)(((__cached_4 / 28) * 3584) + ((__cached_4 % 28) * 128))], 0, ((uint64_t)(__cached_5 * 128) * 4UL));
    for (uint64_t c_o = 0UL; c_o < 2UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_6;
          __cached_6 = &__ins_0[((c_o * 57600UL) + (((((o_o * 419UL) / 30UL) + r) * 1920UL) + ((((o_o * 419UL) % 30UL) + s) * 64UL)))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_6;
          void* __cached_7;
          __cached_7 = &__ins_1[((c_o * 73728UL) + ((r * 24576UL) + (s * 8192UL)))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_7;
        }
      }
    }
    void* _arg_cache_15 = &__origouts_2750_shr[(uint64_t)(((__cached_4 / 28) * 3584) + ((__cached_4 % 28) * 128))];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr[o_o], A_list, B_list, &__origouts_2750_shr[(uint64_t)(((__cached_4 / 28) * 3584) + ((__cached_4 % 28) * 128))], 1, 64, 8192, 18, 7, 7, __stream);
  }
  for (uint64_t _fuseiter9853 = 0UL; _fuseiter9853 < 28UL; _fuseiter9853 += 1UL) {
    for (uint64_t _fuseiter9854 = 0UL; _fuseiter9854 < 28UL; _fuseiter9854 += 1UL) {
      for (uint64_t _fuseiter9855 = 0UL; _fuseiter9855 < 128UL; _fuseiter9855 += 16UL) {
        vec_s32x16 __cached_8;
        __cached_8 = vec_s32x16::load(&__origouts_2750_shr[((_fuseiter9853 * 3584UL) + ((_fuseiter9854 * 128UL) + _fuseiter9855))]);
        vec_f32x16 __cached_9;
        __cached_9 = (vec_f32x16)(__cached_8);
        vec_f32x16 __cached_10;
        __cached_10 = vec_f32x16::load(&__ins_2[_fuseiter9855]);
        __cached_9 = (__cached_9 * __cached_10);
        vec_f32x16 __cached_11;
        __cached_11 = vec_f32x16::load(&__ins_3[_fuseiter9855]);
        __cached_9 = (__cached_9 + __cached_11);
        __cached_9 = sc_max(__cached_9, vec_f32x16(0.f));
        vec_s8x16 __cached_12;
        __cached_12 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_9));
        vec_s8x16 __cached_13;
        __cached_13 = __cached_12;
        vec_s8x16::store(__cached_13, &__outs_0[(((_fuseiter9855 / 64UL) * 50176UL) + ((_fuseiter9853 * 1792UL) + ((_fuseiter9854 * 64UL) + (_fuseiter9855 % 64UL))))]);
      }
    }
  }
  sc_aligned_free(__stream, __origouts_2750_shr);
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res3b_conv_2_cast_mul_add_cast_add_cast_reorder__68(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_235 = *(void**)(__module_data + 128);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_2804 = 0UL; fused_0n__k_2804 < 4UL; fused_0n__k_2804 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 28UL; p_o += 1UL) {
      int32_t* __origouts_2760_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0n__k_2804 / 4UL) * 100352UL) + ((c * 50176UL) + (p_o * 1792UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0n__k_2804 % 4UL) * 16384UL) + (c * 8192UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_16 = &__origouts_2760_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_235, A_list, B_list, &__origouts_2760_shr[0UL], 1, 1, 1, 2, 7, 7, __stream);
      for (uint64_t _fuseiter9889 = 0UL; _fuseiter9889 < 28UL; _fuseiter9889 += 1UL) {
        for (uint64_t _fuseiter9890 = 0UL; _fuseiter9890 < 128UL; _fuseiter9890 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2760_shr[((_fuseiter9889 * 128UL) + _fuseiter9890)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0n__k_2804 / 4UL) * 512UL) + ((fused_0n__k_2804 % 4UL) * 128UL)) + _fuseiter9890)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_2804 % 4UL) * 128UL) + _fuseiter9890)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_u8x16 __cached_7;
          __cached_7 = vec_u8x16::load(&__ins_4[((((fused_0n__k_2804 / 4UL) * 401408UL) + (((fused_0n__k_2804 % 4UL) * 100352UL) + (p_o * 3584UL))) + ((_fuseiter9889 * 128UL) + _fuseiter9890))]);
          __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16 __cached_9;
          __cached_9 = __cached_8;
          vec_u8x16::store(__cached_9, &__outs_0[(((fused_0n__k_2804 / 4UL) * 401408UL) + ((((_fuseiter9890 + ((fused_0n__k_2804 % 4UL) * 128UL)) / 64UL) * 50176UL) + ((p_o * 1792UL) + ((_fuseiter9889 * 64UL) + ((_fuseiter9890 + ((fused_0n__k_2804 % 4UL) * 128UL)) % 64UL)))))]);
        }
      }
      sc_aligned_free(__stream, __origouts_2760_shr);
    }
  }
  return true;
}

static bool res3c_conv_0_cast_mul_add_relu_cast_reorder__72(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_237 = *(void**)(__module_data + 136);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_2805 = 0UL; fused_0n__k_2805 < 2UL; fused_0n__k_2805 += 1UL) {
    memset(&__outs_0[(((fused_0n__k_2805 / 2UL) * 115200UL) + ((fused_0n__k_2805 % 2UL) * 57600UL))], 0, 1920UL);
    for (uint64_t p1 = 0UL; p1 < 28UL; p1 += 1UL) {
      memset(&__outs_0[(((fused_0n__k_2805 / 2UL) * 115200UL) + (((fused_0n__k_2805 % 2UL) * 57600UL) + ((p1 + 1UL) * 1920UL)))], 0, 64UL);
      memset(&__outs_0[((((fused_0n__k_2805 / 2UL) * 115200UL) + (((fused_0n__k_2805 % 2UL) * 57600UL) + ((p1 + 1UL) * 1920UL))) + 1856UL)], 0, 64UL);
    }
    memset(&__outs_0[((((fused_0n__k_2805 / 2UL) * 115200UL) + ((fused_0n__k_2805 % 2UL) * 57600UL)) + 55680UL)], 0, 1920UL);
  }
  int32_t* __origouts_2770_shr = (int32_t*)sc_aligned_malloc(__stream, 401408UL);
  void** A_list = (void**)&__rescheduled_0[0UL];
  void** B_list = (void**)&__rescheduled_0[64UL];
  for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
    void* __cached_0;
    __cached_0 = &__ins_0[(c * 50176UL)];
    A_list[c] = __cached_0;
    void* __cached_1;
    __cached_1 = &__ins_1[(c * 8192UL)];
    B_list[c] = __cached_1;
  }
  void* _arg_cache_17 = &__origouts_2770_shr[0UL];
  dnnl_brgemm_list_call(__sc_kernel_cache_237, A_list, B_list, &__origouts_2770_shr[0UL], 1, 1, 1, 8, 8, 7, __stream);
  for (uint64_t _fuseiter9929 = 0UL; _fuseiter9929 < 28UL; _fuseiter9929 += 1UL) {
    for (uint64_t _fuseiter9930 = 0UL; _fuseiter9930 < 28UL; _fuseiter9930 += 1UL) {
      for (uint64_t _fuseiter9931 = 0UL; _fuseiter9931 < 128UL; _fuseiter9931 += 16UL) {
        vec_s32x16 __cached_2;
        __cached_2 = vec_s32x16::load(&__origouts_2770_shr[((_fuseiter9929 * 3584UL) + ((_fuseiter9930 * 128UL) + _fuseiter9931))]);
        vec_f32x16 __cached_3;
        __cached_3 = (vec_f32x16)(__cached_2);
        vec_f32x16 __cached_4;
        __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter9931]);
        __cached_3 = (__cached_3 * __cached_4);
        vec_f32x16 __cached_5;
        __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter9931]);
        __cached_3 = (__cached_3 + __cached_5);
        __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
        vec_s8x16 __cached_6;
        __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
        vec_s8x16 __cached_7;
        __cached_7 = __cached_6;
        vec_s8x16::store(__cached_7, &__outs_0[(((_fuseiter9931 / 64UL) * 57600UL) + (((_fuseiter9929 + 1UL) * 1920UL) + (((_fuseiter9930 + 1UL) * 64UL) + (_fuseiter9931 % 64UL))))]);
      }
    }
  }
  sc_aligned_free(__stream, __origouts_2770_shr);
  return true;
}

static bool res3c_conv_1_cast_mul_add_relu_cast_reorder__76(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_239 = *(void**)(__module_data + 144);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 384UL);
  for (uint64_t fused_0k_o__n_2807 = 0UL; fused_0k_o__n_2807 < 4UL; fused_0k_o__n_2807 += 1UL) {
    int32_t* __origouts_2780_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    for (uint64_t o_o = 0UL; o_o < 28UL; o_o += 1UL) {
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[192UL];
      memset(&__origouts_2780_shr[(((o_o * 28UL) / 28UL) * 896UL)], 0, 3584UL);
      for (uint64_t c_o = 0UL; c_o < 2UL; c_o += 1UL) {
        for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
          for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
            void* __cached_0;
            __cached_0 = &__ins_0[((c_o * 57600UL) + (((((o_o * 28UL) / 28UL) + r) * 1920UL) + (s * 64UL)))];
            A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_0;
            void* __cached_1;
            __cached_1 = &__ins_1[((fused_0k_o__n_2807 * 36864UL) + ((c_o * 18432UL) + ((r * 6144UL) + (s * 2048UL))))];
            B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_1;
          }
        }
      }
      void* _arg_cache_18 = &__origouts_2780_shr[(((o_o * 28UL) / 28UL) * 896UL)];
      dnnl_brgemm_list_call(__sc_kernel_cache_239, A_list, B_list, &__origouts_2780_shr[(((o_o * 28UL) / 28UL) * 896UL)], 1, 64, 2048, 18, 7, 7, __stream);
    }
    for (uint64_t _fuseiter9964 = 0UL; _fuseiter9964 < 28UL; _fuseiter9964 += 1UL) {
      for (uint64_t _fuseiter9965 = 0UL; _fuseiter9965 < 28UL; _fuseiter9965 += 1UL) {
        for (uint64_t _fuseiter9966 = 0UL; _fuseiter9966 < 32UL; _fuseiter9966 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2780_shr[((_fuseiter9964 * 896UL) + ((_fuseiter9965 * 32UL) + _fuseiter9966))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k_o__n_2807 * 32UL) + _fuseiter9966)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k_o__n_2807 * 32UL) + _fuseiter9966)]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[((((_fuseiter9966 + (fused_0k_o__n_2807 * 32UL)) / 128UL) * 100352UL) + ((_fuseiter9964 * 3584UL) + ((_fuseiter9965 * 128UL) + ((_fuseiter9966 + (fused_0k_o__n_2807 * 32UL)) % 128UL))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2780_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res3c_conv_2_cast_mul_add_cast_add_cast_reorder__80(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_241 = *(void**)(__module_data + 152);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0k__n_2808 = 0UL; fused_0k__n_2808 < 8UL; fused_0k__n_2808 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 14UL; p_o += 1UL) {
      int32_t* __origouts_2790_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(p_o * 7168UL)];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(fused_0k__n_2808 * 8192UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_19 = &__origouts_2790_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_241, A_list, B_list, &__origouts_2790_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
      for (uint64_t _fuseiter9999 = 0UL; _fuseiter9999 < 2UL; _fuseiter9999 += 1UL) {
        for (uint64_t _fuseiter10000 = 0UL; _fuseiter10000 < 28UL; _fuseiter10000 += 1UL) {
          for (uint64_t _fuseiter10001 = 0UL; _fuseiter10001 < 64UL; _fuseiter10001 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_2790_shr[((_fuseiter9999 * 1792UL) + ((_fuseiter10000 * 64UL) + _fuseiter10001))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_2808 * 64UL) + _fuseiter10001)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_2808 * 64UL) + _fuseiter10001)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_u8x16 __cached_7;
            __cached_7 = vec_u8x16::load(&__ins_4[(((fused_0k__n_2808 * 50176UL) + (p_o * 3584UL)) + ((_fuseiter9999 * 1792UL) + ((_fuseiter10000 * 64UL) + _fuseiter10001)))]);
            __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
            vec_u8x16 __cached_8;
            __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
            vec_u8x16 __cached_9;
            __cached_9 = __cached_8;
            vec_u8x16::store(__cached_9, &__outs_0[((((_fuseiter10001 + (fused_0k__n_2808 * 64UL)) / 128UL) * 100352UL) + (((_fuseiter9999 + (p_o * 2UL)) * 3584UL) + ((_fuseiter10000 * 128UL) + ((_fuseiter10001 + (fused_0k__n_2808 * 64UL)) % 128UL))))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_2790_shr);
    }
  }
  return true;
}

static bool res3d_conv_0_cast_mul_add_relu_cast_reorder__84(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_243 = *(void**)(__module_data + 160);
  alignas(64) int8_t __rescheduled_0[128UL];
  memset(&__outs_0[0UL], 0, 3840UL);
  for (uint64_t p1 = 0UL; p1 < 28UL; p1 += 1UL) {
    memset(&__outs_0[((p1 + 1UL) * 3840UL)], 0, 128UL);
    memset(&__outs_0[(((p1 + 1UL) * 3840UL) + 3712UL)], 0, 128UL);
  }
  memset(&__outs_0[111360UL], 0, 3840UL);
  for (uint64_t fused_0k__n_2810 = 0UL; fused_0k__n_2810 < 2UL; fused_0k__n_2810 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 14UL; p_o += 1UL) {
      int32_t* __origouts_2800_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[((c * 100352UL) + (p_o * 7168UL))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[((fused_0k__n_2810 * 32768UL) + (c * 8192UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_20 = &__origouts_2800_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_243, A_list, B_list, &__origouts_2800_shr[0UL], 1, 1, 1, 4, 8, 7, __stream);
      for (uint64_t _fuseiter10040 = 0UL; _fuseiter10040 < 2UL; _fuseiter10040 += 1UL) {
        for (uint64_t _fuseiter10041 = 0UL; _fuseiter10041 < 28UL; _fuseiter10041 += 1UL) {
          for (uint64_t _fuseiter10042 = 0UL; _fuseiter10042 < 64UL; _fuseiter10042 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_2800_shr[((_fuseiter10040 * 1792UL) + ((_fuseiter10041 * 64UL) + _fuseiter10042))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_2810 * 64UL) + _fuseiter10042)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_2810 * 64UL) + _fuseiter10042)]);
            __cached_3 = (__cached_3 + __cached_5);
            __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = __cached_6;
            vec_s8x16::store(__cached_7, &__outs_0[((((_fuseiter10042 + (fused_0k__n_2810 * 64UL)) / 128UL) * 115200UL) + ((((_fuseiter10040 + (p_o * 2UL)) + 1UL) * 3840UL) + (((_fuseiter10041 + 1UL) * 128UL) + ((_fuseiter10042 + (fused_0k__n_2810 * 64UL)) % 128UL))))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_2800_shr);
    }
  }
  return true;
}

static bool res3d_conv_1_cast_mul_add_relu_cast__88(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_244 = *(void**)(__module_data + 168);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 320UL);
  for (uint64_t fused_0k_o__n_2811 = 0UL; fused_0k_o__n_2811 < 4UL; fused_0k_o__n_2811 += 1UL) {
    int32_t* __origouts_2810_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    for (uint64_t o_o = 0UL; o_o < 28UL; o_o += 1UL) {
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[128UL];
      memset(&__origouts_2810_shr[(((o_o * 28UL) / 28UL) * 896UL)], 0, 3584UL);
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_0;
          __cached_0 = &__ins_0[(((((o_o * 28UL) / 28UL) + r) * 3840UL) + (s * 128UL))];
          A_list[((r * 3UL) + s)] = __cached_0;
          void* __cached_1;
          __cached_1 = &__ins_1[((fused_0k_o__n_2811 * 36864UL) + ((r * 12288UL) + (s * 4096UL)))];
          B_list[((r * 3UL) + s)] = __cached_1;
        }
      }
      void* _arg_cache_21 = &__origouts_2810_shr[(((o_o * 28UL) / 28UL) * 896UL)];
      dnnl_brgemm_list_call(__sc_kernel_cache_244, A_list, B_list, &__origouts_2810_shr[(((o_o * 28UL) / 28UL) * 896UL)], 1, 128, 4096, 9, 7, 7, __stream);
    }
    float* _cast_buf_0_shr = (float*)&__rescheduled_0[256UL];
    for (uint64_t _fuseiter10075 = 0UL; _fuseiter10075 < 28UL; _fuseiter10075 += 1UL) {
      for (uint64_t _fuseiter10076 = 0UL; _fuseiter10076 < 28UL; _fuseiter10076 += 1UL) {
        for (uint64_t _fuseiter10077 = 0UL; _fuseiter10077 < 32UL; _fuseiter10077 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2810_shr[((_fuseiter10075 * 896UL) + ((_fuseiter10076 * 32UL) + _fuseiter10077))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16::store(__cached_3, &_cast_buf_0_shr[((((0UL - _fuseiter10075) * 16UL) + (((0UL - _fuseiter10076) * 16UL) + (0UL - _fuseiter10077))) + ((_fuseiter10075 * 16UL) + ((_fuseiter10076 * 16UL) + _fuseiter10077)))]);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&_cast_buf_0_shr[0UL]);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_2[((fused_0k_o__n_2811 * 32UL) + _fuseiter10077)]);
          __cached_4 = (__cached_4 * __cached_5);
          vec_f32x16 __cached_6;
          __cached_6 = vec_f32x16::load(&__ins_3[((fused_0k_o__n_2811 * 32UL) + _fuseiter10077)]);
          __cached_4 = (__cached_4 + __cached_6);
          __cached_4 = sc_max(__cached_4, vec_f32x16(0.f));
          vec_f32x16::store(__cached_4, &_cast_buf_0_shr[0UL]);
          vec_s8x16 __cached_7;
          __cached_7 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_4));
          vec_s8x16::store(__cached_7, &__outs_0[((fused_0k_o__n_2811 * 25088UL) + ((_fuseiter10075 * 896UL) + ((_fuseiter10076 * 32UL) + _fuseiter10077)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2810_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res3d_conv_2_cast_mul_add_cast_add_cast__92(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_246 = *(void**)(__module_data + 176);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0k__n_2812 = 0UL; fused_0k__n_2812 < 4UL; fused_0k__n_2812 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 4UL; p_o += 1UL) {
      int32_t* __origouts_2820_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[((c * 25088UL) + (p_o * 6272UL))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[((fused_0k__n_2812 * 16384UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_22 = &__origouts_2820_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_246, A_list, B_list, &__origouts_2820_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
      for (uint64_t _fuseiter10105 = 0UL; _fuseiter10105 < 7UL; _fuseiter10105 += 1UL) {
        for (uint64_t _fuseiter10106 = 0UL; _fuseiter10106 < 28UL; _fuseiter10106 += 1UL) {
          for (uint64_t _fuseiter10107 = 0UL; _fuseiter10107 < 128UL; _fuseiter10107 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_2820_shr[((_fuseiter10105 * 3584UL) + ((_fuseiter10106 * 128UL) + _fuseiter10107))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_2812 * 128UL) + _fuseiter10107)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_2812 * 128UL) + _fuseiter10107)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_u8x16 __cached_7;
            __cached_7 = vec_u8x16::load(&__ins_4[(((fused_0k__n_2812 * 100352UL) + (p_o * 25088UL)) + ((_fuseiter10105 * 3584UL) + ((_fuseiter10106 * 128UL) + _fuseiter10107)))]);
            __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
            vec_u8x16 __cached_8;
            __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
            vec_u8x16::store(__cached_8, &__outs_0[(((fused_0k__n_2812 * 100352UL) + (p_o * 25088UL)) + ((_fuseiter10105 * 3584UL) + ((_fuseiter10106 * 128UL) + _fuseiter10107)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_2820_shr);
    }
  }
  return true;
}

static bool res4a_conv_0_cast_mul_add_relu_cast__100(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_248 = *(void**)(__module_data + 184);
  alignas(64) int8_t __rescheduled_0[128UL];
  memset(&__outs_0[0UL], 0, 7680UL);
  for (uint64_t p1 = 0UL; p1 < 28UL; p1 += 1UL) {
    memset(&__outs_0[((p1 + 1UL) * 7680UL)], 0, 256UL);
    memset(&__outs_0[(((p1 + 1UL) * 7680UL) + 7424UL)], 0, 256UL);
  }
  memset(&__outs_0[222720UL], 0, 7680UL);
  for (uint64_t p_o = 0UL; p_o < 14UL; p_o += 1UL) {
    int32_t* __origouts_2830_shr = (int32_t*)sc_aligned_malloc(__stream, 57344UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[((c * 100352UL) + (p_o * 7168UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(c * 32768UL)];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_23 = &__origouts_2830_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_248, A_list, B_list, &__origouts_2830_shr[0UL], 1, 1, 1, 4, 8, 7, __stream);
    for (uint64_t _fuseiter10141 = 0UL; _fuseiter10141 < 2UL; _fuseiter10141 += 1UL) {
      for (uint64_t _fuseiter10142 = 0UL; _fuseiter10142 < 28UL; _fuseiter10142 += 1UL) {
        for (uint64_t _fuseiter10143 = 0UL; _fuseiter10143 < 256UL; _fuseiter10143 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2830_shr[((_fuseiter10141 * 7168UL) + ((_fuseiter10142 * 256UL) + _fuseiter10143))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter10143]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter10143]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16::store(__cached_6, &__outs_0[(((((p_o * 2UL) + 1UL) * 7680UL) + 256UL) + ((_fuseiter10141 * 7680UL) + ((_fuseiter10142 * 256UL) + _fuseiter10143)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2830_shr);
  }
  return true;
}

static bool res4a_conv_1_cast_mul_add_relu_cast_reorder__104(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_252 = (void**)&__uninitialized_data[23657512UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 384UL);
  int32_t* conv_os_blk_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[64UL];
  int32_t __cached_0;
  __cached_0 = 196;
  conv_os_blk_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 0;
  conv_os_acc_size[0] = __cached_1;
  for (uint64_t fused_0k_o__n_2815 = 0UL; fused_0k_o__n_2815 < 2UL; fused_0k_o__n_2815 += 1UL) {
    int32_t* __origouts_2840_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    void** A_list = (void**)&__rescheduled_0[128UL];
    void** B_list = (void**)&__rescheduled_0[256UL];
    int32_t __cached_2;
    __cached_2 = conv_os_acc_size[0UL];
    int32_t __cached_3;
    __cached_3 = conv_os_blk_size[0UL];
    memset(&__origouts_2840_shr[(uint64_t)(((__cached_2 / 14) * 1792) + ((__cached_2 % 14) * 128))], 0, ((uint64_t)(__cached_3 * 128) * 4UL));
    for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
      for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
        void* __cached_4;
        __cached_4 = &__ins_0[((r * 7680UL) + (s * 256UL))];
        A_list[((r * 3UL) + s)] = __cached_4;
        void* __cached_5;
        __cached_5 = &__ins_1[((fused_0k_o__n_2815 * 294912UL) + ((r * 98304UL) + (s * 32768UL)))];
        B_list[((r * 3UL) + s)] = __cached_5;
      }
    }
    void* _arg_cache_24 = &__origouts_2840_shr[(uint64_t)(((__cached_2 / 14) * 1792) + ((__cached_2 % 14) * 128))];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_252[0UL], A_list, B_list, &__origouts_2840_shr[(uint64_t)(((__cached_2 / 14) * 1792) + ((__cached_2 % 14) * 128))], 1, 256, 32768, 9, 7, 7, __stream);
    for (uint64_t _fuseiter10171 = 0UL; _fuseiter10171 < 14UL; _fuseiter10171 += 1UL) {
      for (uint64_t _fuseiter10172 = 0UL; _fuseiter10172 < 14UL; _fuseiter10172 += 1UL) {
        for (uint64_t _fuseiter10173 = 0UL; _fuseiter10173 < 128UL; _fuseiter10173 += 16UL) {
          vec_s32x16 __cached_6;
          __cached_6 = vec_s32x16::load(&__origouts_2840_shr[((_fuseiter10171 * 1792UL) + ((_fuseiter10172 * 128UL) + _fuseiter10173))]);
          vec_f32x16 __cached_7;
          __cached_7 = (vec_f32x16)(__cached_6);
          vec_f32x16 __cached_8;
          __cached_8 = vec_f32x16::load(&__ins_2[((fused_0k_o__n_2815 * 128UL) + _fuseiter10173)]);
          __cached_7 = (__cached_7 * __cached_8);
          vec_f32x16 __cached_9;
          __cached_9 = vec_f32x16::load(&__ins_3[((fused_0k_o__n_2815 * 128UL) + _fuseiter10173)]);
          __cached_7 = (__cached_7 + __cached_9);
          __cached_7 = sc_max(__cached_7, vec_f32x16(0.f));
          vec_s8x16 __cached_10;
          __cached_10 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_7));
          vec_s8x16 __cached_11;
          __cached_11 = __cached_10;
          vec_s8x16::store(__cached_11, &__outs_0[((((_fuseiter10173 + (fused_0k_o__n_2815 * 128UL)) / 256UL) * 50176UL) + ((_fuseiter10171 * 3584UL) + ((_fuseiter10172 * 256UL) + ((_fuseiter10173 + (fused_0k_o__n_2815 * 128UL)) % 256UL))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2840_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4a_conv_b_cast_mul_add_cast_reorder__96(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_254 = *(void**)(__module_data + 192);
  alignas(64) int8_t __rescheduled_0[128UL];
  uint8_t* input_tmp = (uint8_t*)sc_aligned_malloc(__stream, 100352UL);
  for (uint64_t fused_0n__c_o_2816 = 0UL; fused_0n__c_o_2816 < 4UL; fused_0n__c_o_2816 += 1UL) {
    for (uint64_t p = 0UL; p < 14UL; p += 1UL) {
      for (uint64_t q = 0UL; q < 14UL; q += 1UL) {
        for (uint64_t c_i = 0UL; c_i < 128UL; c_i += 64UL) {
          vec_u8x64 __cached_0;
          __cached_0 = vec_u8x64::load(&__ins_0[(((fused_0n__c_o_2816 / 4UL) * 401408UL) + (((fused_0n__c_o_2816 % 4UL) * 100352UL) + ((p * 7168UL) + ((q * 256UL) + c_i))))]);
          vec_u8x64 __cached_1;
          __cached_1 = __cached_0;
          vec_u8x64::store(__cached_1, &input_tmp[(((fused_0n__c_o_2816 / 4UL) * 100352UL) + (((fused_0n__c_o_2816 % 4UL) * 25088UL) + ((p * 1792UL) + ((q * 128UL) + c_i))))]);
        }
      }
    }
  }
  for (uint64_t fused_0k__n_2817 = 0UL; fused_0k__n_2817 < 32UL; fused_0k__n_2817 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 2UL; p_o += 1UL) {
      int32_t* __origouts_2850_shr = (int32_t*)sc_aligned_malloc(__stream, 12544UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_2;
        __cached_2 = &input_tmp[((c * 25088UL) + (p_o * 12544UL))];
        A_list[c] = __cached_2;
        void* __cached_3;
        __cached_3 = &__ins_1[((fused_0k__n_2817 * 16384UL) + (c * 4096UL))];
        B_list[c] = __cached_3;
      }
      void* _arg_cache_25 = &__origouts_2850_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_254, A_list, B_list, &__origouts_2850_shr[0UL], 1, 1, 1, 4, 8, 7, __stream);
      for (uint64_t _fuseiter10206 = 0UL; _fuseiter10206 < 7UL; _fuseiter10206 += 1UL) {
        for (uint64_t _fuseiter10207 = 0UL; _fuseiter10207 < 14UL; _fuseiter10207 += 1UL) {
          for (uint64_t _fuseiter10208 = 0UL; _fuseiter10208 < 32UL; _fuseiter10208 += 16UL) {
            vec_s32x16 __cached_4;
            __cached_4 = vec_s32x16::load(&__origouts_2850_shr[((_fuseiter10206 * 448UL) + ((_fuseiter10207 * 32UL) + _fuseiter10208))]);
            vec_f32x16 __cached_5;
            __cached_5 = (vec_f32x16)(__cached_4);
            vec_f32x16 __cached_6;
            __cached_6 = vec_f32x16::load(&__ins_2[((fused_0k__n_2817 * 32UL) + _fuseiter10208)]);
            __cached_5 = (__cached_5 * __cached_6);
            vec_f32x16 __cached_7;
            __cached_7 = vec_f32x16::load(&__ins_3[((fused_0k__n_2817 * 32UL) + _fuseiter10208)]);
            __cached_5 = (__cached_5 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_5));
            vec_s8x16 __cached_9;
            __cached_9 = __cached_8;
            vec_s8x16::store(__cached_9, &__outs_0[((((_fuseiter10208 + (fused_0k__n_2817 * 32UL)) / 128UL) * 25088UL) + (((_fuseiter10206 + (p_o * 7UL)) * 1792UL) + ((_fuseiter10207 * 128UL) + ((_fuseiter10208 + (fused_0k__n_2817 * 32UL)) % 128UL))))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_2850_shr);
    }
  }
  sc_aligned_free(__stream, input_tmp);
  return true;
}

static bool res4a_conv_2_cast_mul_add_cast_add_cast_reorder__108(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_256 = *(void**)(__module_data + 200);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0k__n_2818 = 0UL; fused_0k__n_2818 < 8UL; fused_0k__n_2818 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 14UL; p_o += 1UL) {
      int32_t* __origouts_2860_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(p_o * 3584UL)];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(fused_0k__n_2818 * 32768UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_26 = &__origouts_2860_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_256, A_list, B_list, &__origouts_2860_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
      for (uint64_t _fuseiter10236 = 0UL; _fuseiter10236 < 14UL; _fuseiter10236 += 1UL) {
        for (uint64_t _fuseiter10237 = 0UL; _fuseiter10237 < 128UL; _fuseiter10237 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2860_shr[((_fuseiter10236 * 128UL) + _fuseiter10237)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_2818 * 128UL) + _fuseiter10237)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_2818 * 128UL) + _fuseiter10237)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = vec_s8x16::load(&__ins_4[(((fused_0k__n_2818 * 25088UL) + (p_o * 1792UL)) + ((_fuseiter10236 * 128UL) + _fuseiter10237))]);
          __cached_6 = (__cached_6 + __cached_7);
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16 __cached_9;
          __cached_9 = __cached_8;
          vec_u8x16::store(__cached_9, &__outs_0[((((_fuseiter10237 + (fused_0k__n_2818 * 128UL)) / 1024UL) * 200704UL) + ((p_o * 14336UL) + ((_fuseiter10236 * 1024UL) + ((_fuseiter10237 + (fused_0k__n_2818 * 128UL)) % 1024UL))))]);
        }
      }
      sc_aligned_free(__stream, __origouts_2860_shr);
    }
  }
  return true;
}

static bool res4b_conv_0_cast_mul_add_relu_cast_reorder__112(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_258 = *(void**)(__module_data + 208);
  alignas(64) int8_t __rescheduled_0[128UL];
  memset(&__outs_0[0UL], 0, 4096UL);
  for (uint64_t p1 = 0UL; p1 < 14UL; p1 += 1UL) {
    memset(&__outs_0[((p1 + 1UL) * 4096UL)], 0, 256UL);
    memset(&__outs_0[(((p1 + 1UL) * 4096UL) + 3840UL)], 0, 256UL);
  }
  memset(&__outs_0[61440UL], 0, 4096UL);
  for (uint64_t fused_0n__k_2820 = 0UL; fused_0n__k_2820 < 4UL; fused_0n__k_2820 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_2870_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0n__k_2820 / 4UL) * 200704UL) + (p_o * 28672UL))];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[((fused_0n__k_2820 % 4UL) * 65536UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_27 = &__origouts_2870_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_258, A_list, B_list, &__origouts_2870_shr[0UL], 1, 1, 1, 1, 8, 7, __stream);
      for (uint64_t _fuseiter10276 = 0UL; _fuseiter10276 < 2UL; _fuseiter10276 += 1UL) {
        for (uint64_t _fuseiter10277 = 0UL; _fuseiter10277 < 14UL; _fuseiter10277 += 1UL) {
          for (uint64_t _fuseiter10278 = 0UL; _fuseiter10278 < 64UL; _fuseiter10278 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_2870_shr[((_fuseiter10276 * 896UL) + ((_fuseiter10277 * 64UL) + _fuseiter10278))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0n__k_2820 / 4UL) * 256UL) + ((fused_0n__k_2820 % 4UL) * 64UL)) + _fuseiter10278)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_2820 % 4UL) * 64UL) + _fuseiter10278)]);
            __cached_3 = (__cached_3 + __cached_5);
            __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = __cached_6;
            vec_s8x16::store(__cached_7, &__outs_0[(((fused_0n__k_2820 / 4UL) * 65536UL) + ((((_fuseiter10278 + ((fused_0n__k_2820 % 4UL) * 64UL)) / 256UL) * 65536UL) + ((((_fuseiter10276 + (p_o * 2UL)) + 1UL) * 4096UL) + (((_fuseiter10277 + 1UL) * 256UL) + ((_fuseiter10278 + ((fused_0n__k_2820 % 4UL) * 64UL)) % 256UL)))))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_2870_shr);
    }
  }
  return true;
}

static bool res4b_conv_1_cast_mul_add_relu_cast_reorder__116(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_262 = (void**)&__uninitialized_data[23657528UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 384UL);
  int32_t* conv_os_blk_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[64UL];
  int32_t __cached_0;
  __cached_0 = 196;
  conv_os_blk_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 0;
  conv_os_acc_size[0] = __cached_1;
  for (uint64_t fused_0n__k_o_2821 = 0UL; fused_0n__k_o_2821 < 16UL; fused_0n__k_o_2821 += 1UL) {
    int32_t* __origouts_2880_shr = (int32_t*)sc_aligned_malloc(__stream, 12544UL);
    void** A_list = (void**)&__rescheduled_0[128UL];
    void** B_list = (void**)&__rescheduled_0[256UL];
    int32_t __cached_2;
    __cached_2 = conv_os_acc_size[0UL];
    int32_t __cached_3;
    __cached_3 = conv_os_blk_size[0UL];
    memset(&__origouts_2880_shr[(uint64_t)(((__cached_2 / 14) * 224) + ((__cached_2 % 14) * 16))], 0, ((uint64_t)(__cached_3 * 16) * 4UL));
    for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
      for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
        void* __cached_4;
        __cached_4 = &__ins_0[(((fused_0n__k_o_2821 / 16UL) * 65536UL) + ((r * 4096UL) + (s * 256UL)))];
        A_list[((r * 3UL) + s)] = __cached_4;
        void* __cached_5;
        __cached_5 = &__ins_1[(((fused_0n__k_o_2821 % 16UL) * 36864UL) + ((r * 12288UL) + (s * 4096UL)))];
        B_list[((r * 3UL) + s)] = __cached_5;
      }
    }
    void* _arg_cache_28 = &__origouts_2880_shr[(uint64_t)(((__cached_2 / 14) * 224) + ((__cached_2 % 14) * 16))];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_262[0UL], A_list, B_list, &__origouts_2880_shr[(uint64_t)(((__cached_2 / 14) * 224) + ((__cached_2 % 14) * 16))], 1, 256, 4096, 9, 7, 7, __stream);
    for (uint64_t _fuseiter10311 = 0UL; _fuseiter10311 < 14UL; _fuseiter10311 += 1UL) {
      for (uint64_t _fuseiter10312 = 0UL; _fuseiter10312 < 14UL; _fuseiter10312 += 1UL) {
        vec_s32x16 __cached_6;
        __cached_6 = vec_s32x16::load(&__origouts_2880_shr[((_fuseiter10311 * 224UL) + (_fuseiter10312 * 16UL))]);
        vec_f32x16 __cached_7;
        __cached_7 = (vec_f32x16)(__cached_6);
        vec_f32x16 __cached_8;
        __cached_8 = vec_f32x16::load(&__ins_2[(((fused_0n__k_o_2821 / 16UL) * 256UL) + ((fused_0n__k_o_2821 % 16UL) * 16UL))]);
        __cached_7 = (__cached_7 * __cached_8);
        vec_f32x16 __cached_9;
        __cached_9 = vec_f32x16::load(&__ins_3[((fused_0n__k_o_2821 % 16UL) * 16UL)]);
        __cached_7 = (__cached_7 + __cached_9);
        __cached_7 = sc_max(__cached_7, vec_f32x16(0.f));
        vec_s8x16 __cached_10;
        __cached_10 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_7));
        vec_s8x16 __cached_11;
        __cached_11 = __cached_10;
        vec_s8x16::store(__cached_11, &__outs_0[(((fused_0n__k_o_2821 / 16UL) * 50176UL) + (((((fused_0n__k_o_2821 % 16UL) * 16UL) / 64UL) * 12544UL) + ((_fuseiter10311 * 896UL) + ((_fuseiter10312 * 64UL) + (((fused_0n__k_o_2821 % 16UL) * 16UL) % 64UL)))))]);
      }
    }
    sc_aligned_free(__stream, __origouts_2880_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4b_conv_2_cast_mul_add_cast_add_cast_reorder__120(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_264 = *(void**)(__module_data + 216);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t p_o = 0UL; p_o < 2UL; p_o += 1UL) {
    int32_t* __origouts_2890_shr = (int32_t*)sc_aligned_malloc(__stream, 401408UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[((c * 12544UL) + (p_o * 6272UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(c * 65536UL)];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_29 = &__origouts_2890_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_264, A_list, B_list, &__origouts_2890_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
    for (uint64_t _fuseiter10346 = 0UL; _fuseiter10346 < 7UL; _fuseiter10346 += 1UL) {
      for (uint64_t _fuseiter10347 = 0UL; _fuseiter10347 < 14UL; _fuseiter10347 += 1UL) {
        for (uint64_t _fuseiter10348 = 0UL; _fuseiter10348 < 1024UL; _fuseiter10348 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2890_shr[((_fuseiter10346 * 14336UL) + ((_fuseiter10347 * 1024UL) + _fuseiter10348))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter10348]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter10348]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_u8x16 __cached_7;
          __cached_7 = vec_u8x16::load(&__ins_4[((p_o * 100352UL) + ((_fuseiter10346 * 14336UL) + ((_fuseiter10347 * 1024UL) + _fuseiter10348)))]);
          __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16 __cached_9;
          __cached_9 = __cached_8;
          vec_u8x16::store(__cached_9, &__outs_0[(((_fuseiter10348 / 128UL) * 25088UL) + (((_fuseiter10346 + (p_o * 7UL)) * 1792UL) + ((_fuseiter10347 * 128UL) + (_fuseiter10348 % 128UL))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2890_shr);
  }
  return true;
}

static bool res4c_conv_0_cast_mul_add_relu_cast_reorder__124(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_266 = *(void**)(__module_data + 224);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_2823 = 0UL; fused_0n__k_2823 < 2UL; fused_0n__k_2823 += 1UL) {
    memset(&__outs_0[(((fused_0n__k_2823 / 2UL) * 65536UL) + ((fused_0n__k_2823 % 2UL) * 32768UL))], 0, 2048UL);
    for (uint64_t p1 = 0UL; p1 < 14UL; p1 += 1UL) {
      memset(&__outs_0[(((fused_0n__k_2823 / 2UL) * 65536UL) + (((fused_0n__k_2823 % 2UL) * 32768UL) + ((p1 + 1UL) * 2048UL)))], 0, 128UL);
      memset(&__outs_0[((((fused_0n__k_2823 / 2UL) * 65536UL) + (((fused_0n__k_2823 % 2UL) * 32768UL) + ((p1 + 1UL) * 2048UL))) + 1920UL)], 0, 128UL);
    }
    memset(&__outs_0[((((fused_0n__k_2823 / 2UL) * 65536UL) + ((fused_0n__k_2823 % 2UL) * 32768UL)) + 30720UL)], 0, 2048UL);
  }
  for (uint64_t fused_0k__n_2824 = 0UL; fused_0k__n_2824 < 8UL; fused_0k__n_2824 += 1UL) {
    int32_t* __origouts_2900_shr = (int32_t*)sc_aligned_malloc(__stream, 25088UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(c * 25088UL)];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[((fused_0k__n_2824 * 32768UL) + (c * 4096UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_30 = &__origouts_2900_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_266, A_list, B_list, &__origouts_2900_shr[0UL], 1, 1, 1, 8, 8, 7, __stream);
    for (uint64_t _fuseiter10387 = 0UL; _fuseiter10387 < 14UL; _fuseiter10387 += 1UL) {
      for (uint64_t _fuseiter10388 = 0UL; _fuseiter10388 < 14UL; _fuseiter10388 += 1UL) {
        for (uint64_t _fuseiter10389 = 0UL; _fuseiter10389 < 32UL; _fuseiter10389 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2900_shr[((_fuseiter10387 * 448UL) + ((_fuseiter10388 * 32UL) + _fuseiter10389))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_2824 * 32UL) + _fuseiter10389)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_2824 * 32UL) + _fuseiter10389)]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[((((_fuseiter10389 + (fused_0k__n_2824 * 32UL)) / 128UL) * 32768UL) + (((_fuseiter10387 + 1UL) * 2048UL) + (((_fuseiter10388 + 1UL) * 128UL) + ((_fuseiter10389 + (fused_0k__n_2824 * 32UL)) % 128UL))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2900_shr);
  }
  return true;
}

static bool res4c_conv_1_cast_mul_add_relu_cast_reorder__128(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_268 = (void**)&__uninitialized_data[23657536UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 512UL);
  int32_t* conv_os_blk_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[64UL];
  int32_t __cached_0;
  __cached_0 = 196;
  conv_os_blk_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 0;
  conv_os_acc_size[0] = __cached_1;
  for (uint64_t fused_0k_o__n_2825 = 0UL; fused_0k_o__n_2825 < 8UL; fused_0k_o__n_2825 += 1UL) {
    int32_t* __origouts_2910_shr = (int32_t*)sc_aligned_malloc(__stream, 25088UL);
    void** A_list = (void**)&__rescheduled_0[128UL];
    void** B_list = (void**)&__rescheduled_0[320UL];
    int32_t __cached_2;
    __cached_2 = conv_os_acc_size[0UL];
    int32_t __cached_3;
    __cached_3 = conv_os_blk_size[0UL];
    memset(&__origouts_2910_shr[(uint64_t)(((__cached_2 / 14) * 448) + ((__cached_2 % 14) * 32))], 0, ((uint64_t)(__cached_3 * 32) * 4UL));
    for (uint64_t c_o = 0UL; c_o < 2UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_4;
          __cached_4 = &__ins_0[((c_o * 32768UL) + ((r * 2048UL) + (s * 128UL)))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_4;
          void* __cached_5;
          __cached_5 = &__ins_1[((fused_0k_o__n_2825 * 73728UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_5;
        }
      }
    }
    void* _arg_cache_31 = &__origouts_2910_shr[(uint64_t)(((__cached_2 / 14) * 448) + ((__cached_2 % 14) * 32))];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_268[0UL], A_list, B_list, &__origouts_2910_shr[(uint64_t)(((__cached_2 / 14) * 448) + ((__cached_2 % 14) * 32))], 1, 128, 4096, 18, 7, 7, __stream);
    for (uint64_t _fuseiter10422 = 0UL; _fuseiter10422 < 14UL; _fuseiter10422 += 1UL) {
      for (uint64_t _fuseiter10423 = 0UL; _fuseiter10423 < 14UL; _fuseiter10423 += 1UL) {
        for (uint64_t _fuseiter10424 = 0UL; _fuseiter10424 < 32UL; _fuseiter10424 += 16UL) {
          vec_s32x16 __cached_6;
          __cached_6 = vec_s32x16::load(&__origouts_2910_shr[((_fuseiter10422 * 448UL) + ((_fuseiter10423 * 32UL) + _fuseiter10424))]);
          vec_f32x16 __cached_7;
          __cached_7 = (vec_f32x16)(__cached_6);
          vec_f32x16 __cached_8;
          __cached_8 = vec_f32x16::load(&__ins_2[((fused_0k_o__n_2825 * 32UL) + _fuseiter10424)]);
          __cached_7 = (__cached_7 * __cached_8);
          vec_f32x16 __cached_9;
          __cached_9 = vec_f32x16::load(&__ins_3[((fused_0k_o__n_2825 * 32UL) + _fuseiter10424)]);
          __cached_7 = (__cached_7 + __cached_9);
          __cached_7 = sc_max(__cached_7, vec_f32x16(0.f));
          vec_s8x16 __cached_10;
          __cached_10 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_7));
          vec_s8x16 __cached_11;
          __cached_11 = __cached_10;
          vec_s8x16::store(__cached_11, &__outs_0[((((_fuseiter10424 + (fused_0k_o__n_2825 * 32UL)) / 256UL) * 50176UL) + ((_fuseiter10422 * 3584UL) + ((_fuseiter10423 * 256UL) + ((_fuseiter10424 + (fused_0k_o__n_2825 * 32UL)) % 256UL))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2910_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4c_conv_2_cast_mul_add_cast_add_cast__132(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_270 = *(void**)(__module_data + 232);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_2826 = 0UL; fused_0n__k_2826 < 8UL; fused_0n__k_2826 += 1UL) {
    int32_t* __origouts_2920_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    void* __cached_0;
    __cached_0 = &__ins_0[((fused_0n__k_2826 / 8UL) * 50176UL)];
    A_list[0UL] = __cached_0;
    void* __cached_1;
    __cached_1 = &__ins_1[((fused_0n__k_2826 % 8UL) * 32768UL)];
    B_list[0UL] = __cached_1;
    void* _arg_cache_32 = &__origouts_2920_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_270, A_list, B_list, &__origouts_2920_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
    for (uint64_t _fuseiter10457 = 0UL; _fuseiter10457 < 14UL; _fuseiter10457 += 1UL) {
      for (uint64_t _fuseiter10458 = 0UL; _fuseiter10458 < 14UL; _fuseiter10458 += 1UL) {
        for (uint64_t _fuseiter10459 = 0UL; _fuseiter10459 < 128UL; _fuseiter10459 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2920_shr[((_fuseiter10457 * 1792UL) + ((_fuseiter10458 * 128UL) + _fuseiter10459))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0n__k_2826 / 8UL) * 1024UL) + ((fused_0n__k_2826 % 8UL) * 128UL)) + _fuseiter10459)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_2826 % 8UL) * 128UL) + _fuseiter10459)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_u8x16 __cached_7;
          __cached_7 = vec_u8x16::load(&__ins_4[((((fused_0n__k_2826 / 8UL) * 200704UL) + ((fused_0n__k_2826 % 8UL) * 25088UL)) + ((_fuseiter10457 * 1792UL) + ((_fuseiter10458 * 128UL) + _fuseiter10459)))]);
          __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16::store(__cached_8, &__outs_0[((((fused_0n__k_2826 / 8UL) * 200704UL) + ((fused_0n__k_2826 % 8UL) * 25088UL)) + ((_fuseiter10457 * 1792UL) + ((_fuseiter10458 * 128UL) + _fuseiter10459)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2920_shr);
  }
  return true;
}

static bool res4d_conv_0_cast_mul_add_relu_cast__136(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_272 = *(void**)(__module_data + 240);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_2827 = 0UL; fused_0n__k_2827 < 4UL; fused_0n__k_2827 += 1UL) {
    memset(&__outs_0[(((fused_0n__k_2827 / 4UL) * 65536UL) + ((fused_0n__k_2827 % 4UL) * 16384UL))], 0, 1024UL);
    for (uint64_t p1 = 0UL; p1 < 14UL; p1 += 1UL) {
      memset(&__outs_0[(((fused_0n__k_2827 / 4UL) * 65536UL) + (((fused_0n__k_2827 % 4UL) * 16384UL) + ((p1 + 1UL) * 1024UL)))], 0, 64UL);
      memset(&__outs_0[((((fused_0n__k_2827 / 4UL) * 65536UL) + (((fused_0n__k_2827 % 4UL) * 16384UL) + ((p1 + 1UL) * 1024UL))) + 960UL)], 0, 64UL);
    }
    memset(&__outs_0[((((fused_0n__k_2827 / 4UL) * 65536UL) + ((fused_0n__k_2827 % 4UL) * 16384UL)) + 15360UL)], 0, 1024UL);
  }
  for (uint64_t fused_0k__n_2828 = 0UL; fused_0k__n_2828 < 4UL; fused_0k__n_2828 += 1UL) {
    int32_t* __origouts_2930_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(c * 25088UL)];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[((fused_0k__n_2828 * 65536UL) + (c * 8192UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_33 = &__origouts_2930_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_272, A_list, B_list, &__origouts_2930_shr[0UL], 1, 1, 1, 8, 8, 7, __stream);
    for (uint64_t _fuseiter10493 = 0UL; _fuseiter10493 < 14UL; _fuseiter10493 += 1UL) {
      for (uint64_t _fuseiter10494 = 0UL; _fuseiter10494 < 14UL; _fuseiter10494 += 1UL) {
        for (uint64_t _fuseiter10495 = 0UL; _fuseiter10495 < 64UL; _fuseiter10495 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2930_shr[((_fuseiter10493 * 896UL) + ((_fuseiter10494 * 64UL) + _fuseiter10495))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_2828 * 64UL) + _fuseiter10495)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_2828 * 64UL) + _fuseiter10495)]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16::store(__cached_6, &__outs_0[(((fused_0k__n_2828 * 16384UL) + 1088UL) + ((_fuseiter10493 * 1024UL) + ((_fuseiter10494 * 64UL) + _fuseiter10495)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2930_shr);
  }
  return true;
}

static bool res4d_conv_1_cast_mul_add_relu_cast_reorder__140(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_274 = (void**)&__uninitialized_data[23657544UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 768UL);
  int32_t* conv_os_blk_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[64UL];
  int32_t __cached_0;
  __cached_0 = 196;
  conv_os_blk_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 0;
  conv_os_acc_size[0] = __cached_1;
  for (uint64_t fused_0k_o__n_2829 = 0UL; fused_0k_o__n_2829 < 4UL; fused_0k_o__n_2829 += 1UL) {
    int32_t* __origouts_2940_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
    void** A_list = (void**)&__rescheduled_0[128UL];
    void** B_list = (void**)&__rescheduled_0[448UL];
    int32_t __cached_2;
    __cached_2 = conv_os_acc_size[0UL];
    int32_t __cached_3;
    __cached_3 = conv_os_blk_size[0UL];
    memset(&__origouts_2940_shr[(uint64_t)(((__cached_2 / 14) * 896) + ((__cached_2 % 14) * 64))], 0, ((uint64_t)(__cached_3 * 64) * 4UL));
    for (uint64_t c_o = 0UL; c_o < 4UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_4;
          __cached_4 = &__ins_0[((c_o * 16384UL) + ((r * 1024UL) + (s * 64UL)))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_4;
          void* __cached_5;
          __cached_5 = &__ins_1[((fused_0k_o__n_2829 * 147456UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_5;
        }
      }
    }
    void* _arg_cache_34 = &__origouts_2940_shr[(uint64_t)(((__cached_2 / 14) * 896) + ((__cached_2 % 14) * 64))];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_274[0UL], A_list, B_list, &__origouts_2940_shr[(uint64_t)(((__cached_2 / 14) * 896) + ((__cached_2 % 14) * 64))], 1, 64, 4096, 36, 7, 7, __stream);
    for (uint64_t _fuseiter10523 = 0UL; _fuseiter10523 < 14UL; _fuseiter10523 += 1UL) {
      for (uint64_t _fuseiter10524 = 0UL; _fuseiter10524 < 14UL; _fuseiter10524 += 1UL) {
        for (uint64_t _fuseiter10525 = 0UL; _fuseiter10525 < 64UL; _fuseiter10525 += 16UL) {
          vec_s32x16 __cached_6;
          __cached_6 = vec_s32x16::load(&__origouts_2940_shr[((_fuseiter10523 * 896UL) + ((_fuseiter10524 * 64UL) + _fuseiter10525))]);
          vec_f32x16 __cached_7;
          __cached_7 = (vec_f32x16)(__cached_6);
          vec_f32x16 __cached_8;
          __cached_8 = vec_f32x16::load(&__ins_2[((fused_0k_o__n_2829 * 64UL) + _fuseiter10525)]);
          __cached_7 = (__cached_7 * __cached_8);
          vec_f32x16 __cached_9;
          __cached_9 = vec_f32x16::load(&__ins_3[((fused_0k_o__n_2829 * 64UL) + _fuseiter10525)]);
          __cached_7 = (__cached_7 + __cached_9);
          __cached_7 = sc_max(__cached_7, vec_f32x16(0.f));
          vec_s8x16 __cached_10;
          __cached_10 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_7));
          vec_s8x16 __cached_11;
          __cached_11 = __cached_10;
          vec_s8x16::store(__cached_11, &__outs_0[((((_fuseiter10525 + (fused_0k_o__n_2829 * 64UL)) / 128UL) * 25088UL) + ((_fuseiter10523 * 1792UL) + ((_fuseiter10524 * 128UL) + ((_fuseiter10525 + (fused_0k_o__n_2829 * 64UL)) % 128UL))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2940_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4d_conv_2_cast_mul_add_cast_add_cast__144(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_276 = *(void**)(__module_data + 248);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0k__n_2830 = 0UL; fused_0k__n_2830 < 8UL; fused_0k__n_2830 += 1UL) {
    int32_t* __origouts_2950_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(c * 25088UL)];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[((fused_0k__n_2830 * 32768UL) + (c * 16384UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_35 = &__origouts_2950_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_276, A_list, B_list, &__origouts_2950_shr[0UL], 1, 1, 1, 2, 7, 7, __stream);
    for (uint64_t _fuseiter10558 = 0UL; _fuseiter10558 < 14UL; _fuseiter10558 += 1UL) {
      for (uint64_t _fuseiter10559 = 0UL; _fuseiter10559 < 14UL; _fuseiter10559 += 1UL) {
        for (uint64_t _fuseiter10560 = 0UL; _fuseiter10560 < 128UL; _fuseiter10560 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2950_shr[((_fuseiter10558 * 1792UL) + ((_fuseiter10559 * 128UL) + _fuseiter10560))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_2830 * 128UL) + _fuseiter10560)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_2830 * 128UL) + _fuseiter10560)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_u8x16 __cached_7;
          __cached_7 = vec_u8x16::load(&__ins_4[((fused_0k__n_2830 * 25088UL) + ((_fuseiter10558 * 1792UL) + ((_fuseiter10559 * 128UL) + _fuseiter10560)))]);
          __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16::store(__cached_8, &__outs_0[((fused_0k__n_2830 * 25088UL) + ((_fuseiter10558 * 1792UL) + ((_fuseiter10559 * 128UL) + _fuseiter10560)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2950_shr);
  }
  return true;
}

static bool res4e_conv_0_cast_mul_add_relu_cast_reorder__148(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_278 = *(void**)(__module_data + 256);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_2831 = 0UL; fused_0n__k_2831 < 2UL; fused_0n__k_2831 += 1UL) {
    memset(&__outs_0[(((fused_0n__k_2831 / 2UL) * 65536UL) + ((fused_0n__k_2831 % 2UL) * 32768UL))], 0, 2048UL);
    for (uint64_t p1 = 0UL; p1 < 14UL; p1 += 1UL) {
      memset(&__outs_0[(((fused_0n__k_2831 / 2UL) * 65536UL) + (((fused_0n__k_2831 % 2UL) * 32768UL) + ((p1 + 1UL) * 2048UL)))], 0, 128UL);
      memset(&__outs_0[((((fused_0n__k_2831 / 2UL) * 65536UL) + (((fused_0n__k_2831 % 2UL) * 32768UL) + ((p1 + 1UL) * 2048UL))) + 1920UL)], 0, 128UL);
    }
    memset(&__outs_0[((((fused_0n__k_2831 / 2UL) * 65536UL) + ((fused_0n__k_2831 % 2UL) * 32768UL)) + 30720UL)], 0, 2048UL);
  }
  for (uint64_t p_o = 0UL; p_o < 2UL; p_o += 1UL) {
    int32_t* __origouts_2960_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[((c * 25088UL) + (p_o * 12544UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(c * 32768UL)];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_36 = &__origouts_2960_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_278, A_list, B_list, &__origouts_2960_shr[0UL], 1, 1, 1, 8, 8, 7, __stream);
    for (uint64_t _fuseiter10594 = 0UL; _fuseiter10594 < 7UL; _fuseiter10594 += 1UL) {
      for (uint64_t _fuseiter10595 = 0UL; _fuseiter10595 < 14UL; _fuseiter10595 += 1UL) {
        for (uint64_t _fuseiter10596 = 0UL; _fuseiter10596 < 256UL; _fuseiter10596 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2960_shr[((_fuseiter10594 * 3584UL) + ((_fuseiter10595 * 256UL) + _fuseiter10596))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter10596]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter10596]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[(((_fuseiter10596 / 128UL) * 32768UL) + ((((_fuseiter10594 + (p_o * 7UL)) + 1UL) * 2048UL) + (((_fuseiter10595 + 1UL) * 128UL) + (_fuseiter10596 % 128UL))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2960_shr);
  }
  return true;
}

static bool res4e_conv_1_cast_mul_add_relu_cast_reorder__152(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_268 = (void**)&__uninitialized_data[23657536UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 512UL);
  int32_t* conv_os_blk_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[64UL];
  int32_t __cached_0;
  __cached_0 = 196;
  conv_os_blk_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 0;
  conv_os_acc_size[0] = __cached_1;
  for (uint64_t fused_0n__k_o_2833 = 0UL; fused_0n__k_o_2833 < 8UL; fused_0n__k_o_2833 += 1UL) {
    int32_t* __origouts_2970_shr = (int32_t*)sc_aligned_malloc(__stream, 25088UL);
    void** A_list = (void**)&__rescheduled_0[128UL];
    void** B_list = (void**)&__rescheduled_0[320UL];
    int32_t __cached_2;
    __cached_2 = conv_os_acc_size[0UL];
    int32_t __cached_3;
    __cached_3 = conv_os_blk_size[0UL];
    memset(&__origouts_2970_shr[(uint64_t)(((__cached_2 / 14) * 448) + ((__cached_2 % 14) * 32))], 0, ((uint64_t)(__cached_3 * 32) * 4UL));
    for (uint64_t c_o = 0UL; c_o < 2UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_4;
          __cached_4 = &__ins_0[(((fused_0n__k_o_2833 / 8UL) * 65536UL) + ((c_o * 32768UL) + ((r * 2048UL) + (s * 128UL))))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_4;
          void* __cached_5;
          __cached_5 = &__ins_1[(((fused_0n__k_o_2833 % 8UL) * 73728UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_5;
        }
      }
    }
    void* _arg_cache_37 = &__origouts_2970_shr[(uint64_t)(((__cached_2 / 14) * 448) + ((__cached_2 % 14) * 32))];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_268[0UL], A_list, B_list, &__origouts_2970_shr[(uint64_t)(((__cached_2 / 14) * 448) + ((__cached_2 % 14) * 32))], 1, 128, 4096, 18, 7, 7, __stream);
    for (uint64_t _fuseiter10629 = 0UL; _fuseiter10629 < 14UL; _fuseiter10629 += 1UL) {
      for (uint64_t _fuseiter10630 = 0UL; _fuseiter10630 < 14UL; _fuseiter10630 += 1UL) {
        for (uint64_t _fuseiter10631 = 0UL; _fuseiter10631 < 32UL; _fuseiter10631 += 16UL) {
          vec_s32x16 __cached_6;
          __cached_6 = vec_s32x16::load(&__origouts_2970_shr[((_fuseiter10629 * 448UL) + ((_fuseiter10630 * 32UL) + _fuseiter10631))]);
          vec_f32x16 __cached_7;
          __cached_7 = (vec_f32x16)(__cached_6);
          vec_f32x16 __cached_8;
          __cached_8 = vec_f32x16::load(&__ins_2[((((fused_0n__k_o_2833 / 8UL) * 256UL) + ((fused_0n__k_o_2833 % 8UL) * 32UL)) + _fuseiter10631)]);
          __cached_7 = (__cached_7 * __cached_8);
          vec_f32x16 __cached_9;
          __cached_9 = vec_f32x16::load(&__ins_3[(((fused_0n__k_o_2833 % 8UL) * 32UL) + _fuseiter10631)]);
          __cached_7 = (__cached_7 + __cached_9);
          __cached_7 = sc_max(__cached_7, vec_f32x16(0.f));
          vec_s8x16 __cached_10;
          __cached_10 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_7));
          vec_s8x16 __cached_11;
          __cached_11 = __cached_10;
          vec_s8x16::store(__cached_11, &__outs_0[(((fused_0n__k_o_2833 / 8UL) * 50176UL) + ((((_fuseiter10631 + ((fused_0n__k_o_2833 % 8UL) * 32UL)) / 256UL) * 50176UL) + ((_fuseiter10629 * 3584UL) + ((_fuseiter10630 * 256UL) + ((_fuseiter10631 + ((fused_0n__k_o_2833 % 8UL) * 32UL)) % 256UL)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2970_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4e_conv_2_cast_mul_add_cast_add_cast_reorder__156(uint8_t* __restrict__ __outs_0, uint8_t* __restrict__ __outs_1, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_256 = *(void**)(__module_data + 200);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_2834 = 0UL; fused_0n__k_2834 < 8UL; fused_0n__k_2834 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 14UL; p_o += 1UL) {
      int32_t* __origouts_2980_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0n__k_2834 / 8UL) * 50176UL) + (p_o * 3584UL))];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[((fused_0n__k_2834 % 8UL) * 32768UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_38 = &__origouts_2980_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_256, A_list, B_list, &__origouts_2980_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
      for (uint64_t _fuseiter10665 = 0UL; _fuseiter10665 < 14UL; _fuseiter10665 += 1UL) {
        for (uint64_t _fuseiter10666 = 0UL; _fuseiter10666 < 128UL; _fuseiter10666 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2980_shr[((_fuseiter10665 * 128UL) + _fuseiter10666)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0n__k_2834 / 8UL) * 1024UL) + ((fused_0n__k_2834 % 8UL) * 128UL)) + _fuseiter10666)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_2834 % 8UL) * 128UL) + _fuseiter10666)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_u8x16 __cached_7;
          __cached_7 = vec_u8x16::load(&__ins_4[((((fused_0n__k_2834 / 8UL) * 200704UL) + (((fused_0n__k_2834 % 8UL) * 25088UL) + (p_o * 1792UL))) + ((_fuseiter10665 * 128UL) + _fuseiter10666))]);
          __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16::store(__cached_8, &__outs_0[((((fused_0n__k_2834 / 8UL) * 200704UL) + (((fused_0n__k_2834 % 8UL) * 25088UL) + (p_o * 1792UL))) + ((_fuseiter10665 * 128UL) + _fuseiter10666))]);
          vec_u8x16 __cached_9;
          __cached_9 = __cached_8;
          vec_u8x16::store(__cached_9, &__outs_1[(((fused_0n__k_2834 / 8UL) * 200704UL) + ((((_fuseiter10666 + ((fused_0n__k_2834 % 8UL) * 128UL)) / 256UL) * 50176UL) + ((p_o * 3584UL) + ((_fuseiter10665 * 256UL) + ((_fuseiter10666 + ((fused_0n__k_2834 % 8UL) * 128UL)) % 256UL)))))]);
        }
      }
      sc_aligned_free(__stream, __origouts_2980_shr);
    }
  }
  return true;
}

static bool reorder__157(uint8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_10703___fuseiter_10704_2835___fuseiter_10705_2836 = 0UL; fused_0fused_0_fuseiter_10703___fuseiter_10704_2835___fuseiter_10705_2836 < 28UL; fused_0fused_0_fuseiter_10703___fuseiter_10704_2835___fuseiter_10705_2836 += 1UL) {
    for (uint64_t _fuseiter_10706 = 0UL; _fuseiter_10706 < 14UL; _fuseiter_10706 += 1UL) {
      for (uint64_t _fuseiter_10707 = 0UL; _fuseiter_10707 < 512UL; _fuseiter_10707 += 16UL) {
        vec_u8x16 __cached_0;
        __cached_0 = vec_u8x16::load(&__ins_0[(((fused_0fused_0_fuseiter_10703___fuseiter_10704_2835___fuseiter_10705_2836 / 28UL) * 200704UL) + ((((_fuseiter_10707 + (((fused_0fused_0_fuseiter_10703___fuseiter_10704_2835___fuseiter_10705_2836 / 14UL) % 2UL) * 512UL)) / 128UL) * 25088UL) + (((fused_0fused_0_fuseiter_10703___fuseiter_10704_2835___fuseiter_10705_2836 % 14UL) * 1792UL) + ((_fuseiter_10706 * 128UL) + ((_fuseiter_10707 + (((fused_0fused_0_fuseiter_10703___fuseiter_10704_2835___fuseiter_10705_2836 / 14UL) % 2UL) * 512UL)) % 128UL)))))]);
        vec_u8x16 __cached_1;
        __cached_1 = __cached_0;
        vec_u8x16::store(__cached_1, &__outs_0[(((fused_0fused_0_fuseiter_10703___fuseiter_10704_2835___fuseiter_10705_2836 / 28UL) * 200704UL) + ((((fused_0fused_0_fuseiter_10703___fuseiter_10704_2835___fuseiter_10705_2836 / 14UL) % 2UL) * 100352UL) + (((fused_0fused_0_fuseiter_10703___fuseiter_10704_2835___fuseiter_10705_2836 % 14UL) * 7168UL) + ((_fuseiter_10706 * 512UL) + _fuseiter_10707))))]);
      }
    }
  }
  return true;
}

static bool res4f_conv_0_cast_mul_add_relu_cast_reorder__161(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_280 = *(void**)(__module_data + 264);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_2837 = 0UL; fused_0n__k_2837 < 4UL; fused_0n__k_2837 += 1UL) {
    memset(&__outs_0[(((fused_0n__k_2837 / 4UL) * 65536UL) + ((fused_0n__k_2837 % 4UL) * 16384UL))], 0, 1024UL);
    for (uint64_t p1 = 0UL; p1 < 14UL; p1 += 1UL) {
      memset(&__outs_0[(((fused_0n__k_2837 / 4UL) * 65536UL) + (((fused_0n__k_2837 % 4UL) * 16384UL) + ((p1 + 1UL) * 1024UL)))], 0, 64UL);
      memset(&__outs_0[((((fused_0n__k_2837 / 4UL) * 65536UL) + (((fused_0n__k_2837 % 4UL) * 16384UL) + ((p1 + 1UL) * 1024UL))) + 960UL)], 0, 64UL);
    }
    memset(&__outs_0[((((fused_0n__k_2837 / 4UL) * 65536UL) + ((fused_0n__k_2837 % 4UL) * 16384UL)) + 15360UL)], 0, 1024UL);
  }
  for (uint64_t fused_0k__n_2838 = 0UL; fused_0k__n_2838 < 8UL; fused_0k__n_2838 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_2990_shr = (int32_t*)sc_aligned_malloc(__stream, 3584UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[((c * 100352UL) + (p_o * 14336UL))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[((fused_0k__n_2838 * 32768UL) + (c * 16384UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_39 = &__origouts_2990_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_280, A_list, B_list, &__origouts_2990_shr[0UL], 1, 1, 1, 2, 8, 7, __stream);
      for (uint64_t _fuseiter10710 = 0UL; _fuseiter10710 < 2UL; _fuseiter10710 += 1UL) {
        for (uint64_t _fuseiter10711 = 0UL; _fuseiter10711 < 14UL; _fuseiter10711 += 1UL) {
          for (uint64_t _fuseiter10712 = 0UL; _fuseiter10712 < 32UL; _fuseiter10712 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_2990_shr[((_fuseiter10710 * 448UL) + ((_fuseiter10711 * 32UL) + _fuseiter10712))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_2838 * 32UL) + _fuseiter10712)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_2838 * 32UL) + _fuseiter10712)]);
            __cached_3 = (__cached_3 + __cached_5);
            __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = __cached_6;
            vec_s8x16::store(__cached_7, &__outs_0[((((_fuseiter10712 + (fused_0k__n_2838 * 32UL)) / 64UL) * 16384UL) + ((((_fuseiter10710 + (p_o * 2UL)) + 1UL) * 1024UL) + (((_fuseiter10711 + 1UL) * 64UL) + ((_fuseiter10712 + (fused_0k__n_2838 * 32UL)) % 64UL))))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_2990_shr);
    }
  }
  return true;
}

static bool res4f_conv_1_cast_mul_add_relu_cast_reorder__165(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_274 = (void**)&__uninitialized_data[23657544UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 768UL);
  int32_t* conv_os_blk_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[64UL];
  int32_t __cached_0;
  __cached_0 = 196;
  conv_os_blk_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 0;
  conv_os_acc_size[0] = __cached_1;
  for (uint64_t fused_0k_o__n_2839 = 0UL; fused_0k_o__n_2839 < 4UL; fused_0k_o__n_2839 += 1UL) {
    int32_t* __origouts_3000_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
    void** A_list = (void**)&__rescheduled_0[128UL];
    void** B_list = (void**)&__rescheduled_0[448UL];
    int32_t __cached_2;
    __cached_2 = conv_os_acc_size[0UL];
    int32_t __cached_3;
    __cached_3 = conv_os_blk_size[0UL];
    memset(&__origouts_3000_shr[(uint64_t)(((__cached_2 / 14) * 896) + ((__cached_2 % 14) * 64))], 0, ((uint64_t)(__cached_3 * 64) * 4UL));
    for (uint64_t c_o = 0UL; c_o < 4UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_4;
          __cached_4 = &__ins_0[((c_o * 16384UL) + ((r * 1024UL) + (s * 64UL)))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_4;
          void* __cached_5;
          __cached_5 = &__ins_1[((fused_0k_o__n_2839 * 147456UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_5;
        }
      }
    }
    void* _arg_cache_40 = &__origouts_3000_shr[(uint64_t)(((__cached_2 / 14) * 896) + ((__cached_2 % 14) * 64))];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_274[0UL], A_list, B_list, &__origouts_3000_shr[(uint64_t)(((__cached_2 / 14) * 896) + ((__cached_2 % 14) * 64))], 1, 64, 4096, 36, 7, 7, __stream);
    for (uint64_t _fuseiter10745 = 0UL; _fuseiter10745 < 14UL; _fuseiter10745 += 1UL) {
      for (uint64_t _fuseiter10746 = 0UL; _fuseiter10746 < 14UL; _fuseiter10746 += 1UL) {
        for (uint64_t _fuseiter10747 = 0UL; _fuseiter10747 < 64UL; _fuseiter10747 += 16UL) {
          vec_s32x16 __cached_6;
          __cached_6 = vec_s32x16::load(&__origouts_3000_shr[((_fuseiter10745 * 896UL) + ((_fuseiter10746 * 64UL) + _fuseiter10747))]);
          vec_f32x16 __cached_7;
          __cached_7 = (vec_f32x16)(__cached_6);
          vec_f32x16 __cached_8;
          __cached_8 = vec_f32x16::load(&__ins_2[((fused_0k_o__n_2839 * 64UL) + _fuseiter10747)]);
          __cached_7 = (__cached_7 * __cached_8);
          vec_f32x16 __cached_9;
          __cached_9 = vec_f32x16::load(&__ins_3[((fused_0k_o__n_2839 * 64UL) + _fuseiter10747)]);
          __cached_7 = (__cached_7 + __cached_9);
          __cached_7 = sc_max(__cached_7, vec_f32x16(0.f));
          vec_s8x16 __cached_10;
          __cached_10 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_7));
          vec_s8x16 __cached_11;
          __cached_11 = __cached_10;
          vec_s8x16::store(__cached_11, &__outs_0[((((_fuseiter10747 + (fused_0k_o__n_2839 * 64UL)) / 128UL) * 25088UL) + ((_fuseiter10745 * 1792UL) + ((_fuseiter10746 * 128UL) + ((_fuseiter10747 + (fused_0k_o__n_2839 * 64UL)) % 128UL))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_3000_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4f_conv_2_cast_mul_add_cast_add_cast__170(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_282 = *(void**)(__module_data + 272);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0k__n_2840 = 0UL; fused_0k__n_2840 < 4UL; fused_0k__n_2840 += 1UL) {
    int32_t* __origouts_3010_shr = (int32_t*)sc_aligned_malloc(__stream, 200704UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(c * 25088UL)];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[((fused_0k__n_2840 * 65536UL) + (c * 32768UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_41 = &__origouts_3010_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_282, A_list, B_list, &__origouts_3010_shr[0UL], 1, 1, 1, 2, 7, 7, __stream);
    for (uint64_t _fuseiter10780 = 0UL; _fuseiter10780 < 14UL; _fuseiter10780 += 1UL) {
      for (uint64_t _fuseiter10781 = 0UL; _fuseiter10781 < 14UL; _fuseiter10781 += 1UL) {
        for (uint64_t _fuseiter10782 = 0UL; _fuseiter10782 < 256UL; _fuseiter10782 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_3010_shr[((_fuseiter10780 * 3584UL) + ((_fuseiter10781 * 256UL) + _fuseiter10782))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_2840 * 256UL) + _fuseiter10782)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_2840 * 256UL) + _fuseiter10782)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_u8x16 __cached_7;
          __cached_7 = vec_u8x16::load(&__ins_4[((fused_0k__n_2840 * 50176UL) + ((_fuseiter10780 * 3584UL) + ((_fuseiter10781 * 256UL) + _fuseiter10782)))]);
          __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16::store(__cached_8, &__outs_0[((fused_0k__n_2840 * 50176UL) + ((_fuseiter10780 * 3584UL) + ((_fuseiter10781 * 256UL) + _fuseiter10782)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_3010_shr);
  }
  return true;
}

static bool reorder__531(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_10814 = 0UL; _fuseiter_10814 < 128UL; _fuseiter_10814 += 1UL) {
    for (uint64_t _fuseiter_10815 = 0UL; _fuseiter_10815 < 4UL; _fuseiter_10815 += 1UL) {
      for (uint64_t _fuseiter_10818 = 0UL; _fuseiter_10818 < 64UL; _fuseiter_10818 += 1UL) {
        for (uint64_t _fuseiter_10819 = 0UL; _fuseiter_10819 < 16UL; _fuseiter_10819 += 1UL) {
          for (uint64_t _fuseiter_10820 = 0UL; _fuseiter_10820 < 4UL; _fuseiter_10820 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_10819 + (_fuseiter_10814 * 16UL)) * 1024UL) + ((_fuseiter_10820 + (_fuseiter_10818 * 4UL)) + (_fuseiter_10815 * 256UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_10814 * 16384UL) + ((_fuseiter_10815 * 4096UL) + ((_fuseiter_10818 * 64UL) + ((_fuseiter_10819 * 4UL) + _fuseiter_10820))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__654(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2841____itr_2_2842____itr_3_2843 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2841____itr_2_2842____itr_3_2843 < 128UL; fused_0fused_0fused_0__itr_0____itr_1_2841____itr_2_2842____itr_3_2843 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0__itr_0____itr_1_2841____itr_2_2842____itr_3_2843 / 128UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2841____itr_2_2842____itr_3_2843 % 128UL) * 16UL))]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[(((fused_0fused_0fused_0__itr_0____itr_1_2841____itr_2_2842____itr_3_2843 / 128UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2841____itr_2_2842____itr_3_2843 % 128UL) * 16UL))]);
  }
  return true;
}

static bool mul__653(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2844____itr_2_2845____itr_3_2846 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2844____itr_2_2845____itr_3_2846 < 128UL; fused_0fused_0fused_0__itr_0____itr_1_2844____itr_2_2845____itr_3_2846 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0__itr_0____itr_1_2844____itr_2_2845____itr_3_2846 / 128UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2844____itr_2_2845____itr_3_2846 % 128UL) * 16UL))]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[(((fused_0fused_0fused_0__itr_0____itr_1_2844____itr_2_2845____itr_3_2846 / 128UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2844____itr_2_2845____itr_3_2846 % 128UL) * 16UL))]);
  }
  return true;
}

static bool mul__240(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2847____itr_2_2848 = 0UL; fused_0fused_0__itr_0____itr_1_2847____itr_2_2848 < 786432UL; fused_0fused_0__itr_0____itr_1_2847____itr_2_2848 += 1UL) {
    for (uint64_t _fuseiter_10837 = 0UL; _fuseiter_10837 < 3UL; _fuseiter_10837 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2847____itr_2_2848 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2847____itr_2_2848 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2847____itr_2_2848 % 3UL) * 3UL))) + _fuseiter_10837)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2847____itr_2_2848 / 1536UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2847____itr_2_2848 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2847____itr_2_2848 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2847____itr_2_2848 % 3UL) * 3UL))) + _fuseiter_10837)] = __cached_2;
    }
  }
  return true;
}

static bool cast__241(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2849____itr_2_2850 = 0UL; fused_0fused_0__itr_0____itr_1_2849____itr_2_2850 < 786432UL; fused_0fused_0__itr_0____itr_1_2849____itr_2_2850 += 1UL) {
    for (uint64_t _fuseiter10842 = 0UL; _fuseiter10842 < 3UL; _fuseiter10842 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2849____itr_2_2850 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2849____itr_2_2850 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2849____itr_2_2850 % 3UL) * 3UL))) + _fuseiter10842)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2849____itr_2_2850 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2849____itr_2_2850 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2849____itr_2_2850 % 3UL) * 3UL))) + _fuseiter10842)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__537(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_10843___fuseiter_10844_2851___fuseiter_10845_2852 = 0UL; fused_0fused_0_fuseiter_10843___fuseiter_10844_2851___fuseiter_10845_2852 < 24UL; fused_0fused_0_fuseiter_10843___fuseiter_10844_2851___fuseiter_10845_2852 += 1UL) {
    for (uint64_t _fuseiter_10846 = 0UL; _fuseiter_10846 < 3UL; _fuseiter_10846 += 1UL) {
      for (uint64_t _fuseiter_10847 = 0UL; _fuseiter_10847 < 64UL; _fuseiter_10847 += 1UL) {
        for (uint64_t _fuseiter_10848 = 0UL; _fuseiter_10848 < 128UL; _fuseiter_10848 += 1UL) {
          for (uint64_t _fuseiter_10849 = 0UL; _fuseiter_10849 < 4UL; _fuseiter_10849 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_10848 + ((fused_0fused_0_fuseiter_10843___fuseiter_10844_2851___fuseiter_10845_2852 / 6UL) * 128UL)) * 4608UL) + ((((_fuseiter_10849 + (_fuseiter_10847 * 4UL)) + (((fused_0fused_0_fuseiter_10843___fuseiter_10844_2851___fuseiter_10845_2852 / 3UL) % 2UL) * 256UL)) * 9UL) + (((fused_0fused_0_fuseiter_10843___fuseiter_10844_2851___fuseiter_10845_2852 % 3UL) * 3UL) + _fuseiter_10846)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_10843___fuseiter_10844_2851___fuseiter_10845_2852 / 6UL) * 589824UL) + ((((fused_0fused_0_fuseiter_10843___fuseiter_10844_2851___fuseiter_10845_2852 / 3UL) % 2UL) * 294912UL) + (((fused_0fused_0_fuseiter_10843___fuseiter_10844_2851___fuseiter_10845_2852 % 3UL) * 98304UL) + ((_fuseiter_10846 * 32768UL) + ((_fuseiter_10847 * 512UL) + ((_fuseiter_10848 * 4UL) + _fuseiter_10849))))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool res5a_conv_0_cast_mul_add_relu_cast_reorder__681(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_284 = *(void**)(__module_data + 280);
  for (uint64_t fused_0n__k_2853 = 0UL; fused_0n__k_2853 < 4UL; fused_0n__k_2853 += 1UL) {
    memset(&__outs_0[(((fused_0n__k_2853 / 2UL) * 131072UL) + ((fused_0n__k_2853 % 2UL) * 65536UL))], 0, 4096UL);
    for (uint64_t p1 = 0UL; p1 < 14UL; p1 += 1UL) {
      memset(&__outs_0[(((fused_0n__k_2853 / 2UL) * 131072UL) + (((fused_0n__k_2853 % 2UL) * 65536UL) + ((p1 + 1UL) * 4096UL)))], 0, 256UL);
      memset(&__outs_0[((((fused_0n__k_2853 / 2UL) * 131072UL) + (((fused_0n__k_2853 % 2UL) * 65536UL) + ((p1 + 1UL) * 4096UL))) + 3840UL)], 0, 256UL);
    }
    memset(&__outs_0[((((fused_0n__k_2853 / 2UL) * 131072UL) + ((fused_0n__k_2853 % 2UL) * 65536UL)) + 61440UL)], 0, 4096UL);
  }
  for (uint64_t fused_0n__k_2854 = 0UL; fused_0n__k_2854 < 16UL; fused_0n__k_2854 += 1UL) {
    alignas(64) int8_t __rescheduled_2[128UL];
    int32_t* __origouts_3020_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
    void** A_list = (void**)&__rescheduled_2[0UL];
    void** B_list = (void**)&__rescheduled_2[64UL];
    for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0n__k_2854 / 8UL) * 200704UL) + (c * 50176UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(((fused_0n__k_2854 % 8UL) * 65536UL) + (c * 16384UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_42 = &__origouts_3020_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_284, A_list, B_list, &__origouts_3020_shr[0UL], 1, 1, 1, 4, 8, 7, __stream);
    for (uint64_t _fuseiter10852 = 0UL; _fuseiter10852 < 14UL; _fuseiter10852 += 1UL) {
      for (uint64_t _fuseiter10853 = 0UL; _fuseiter10853 < 14UL; _fuseiter10853 += 1UL) {
        for (uint64_t _fuseiter10854 = 0UL; _fuseiter10854 < 64UL; _fuseiter10854 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_3020_shr[((_fuseiter10852 * 896UL) + ((_fuseiter10853 * 64UL) + _fuseiter10854))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0n__k_2854 % 8UL) * 64UL) + _fuseiter10854)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_2854 % 8UL) * 64UL) + _fuseiter10854)]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[(((fused_0n__k_2854 / 8UL) * 131072UL) + ((((_fuseiter10854 + ((fused_0n__k_2854 % 8UL) * 64UL)) / 256UL) * 65536UL) + (((_fuseiter10852 + 1UL) * 4096UL) + (((_fuseiter10853 + 1UL) * 256UL) + ((_fuseiter10854 + ((fused_0n__k_2854 % 8UL) * 64UL)) % 256UL)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_3020_shr);
  }
  return true;
}

static bool mul__658(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2855____itr_2_2856____itr_3_2857 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2855____itr_2_2856____itr_3_2857 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2855____itr_2_2856____itr_3_2857 += 1UL) {
    for (uint64_t _fuseiter_10889 = 0UL; _fuseiter_10889 < 128UL; _fuseiter_10889 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2855____itr_2_2856____itr_3_2857 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2855____itr_2_2856____itr_3_2857 % 4UL) * 128UL)) + _fuseiter_10889)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2855____itr_2_2856____itr_3_2857 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2855____itr_2_2856____itr_3_2857 % 4UL) * 128UL)) + _fuseiter_10889)]);
    }
  }
  return true;
}

static bool mul__657(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2858____itr_2_2859____itr_3_2860 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2858____itr_2_2859____itr_3_2860 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2858____itr_2_2859____itr_3_2860 += 1UL) {
    for (uint64_t _fuseiter_10895 = 0UL; _fuseiter_10895 < 128UL; _fuseiter_10895 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2858____itr_2_2859____itr_3_2860 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2858____itr_2_2859____itr_3_2860 % 4UL) * 128UL)) + _fuseiter_10895)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2858____itr_2_2859____itr_3_2860 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2858____itr_2_2859____itr_3_2860 % 4UL) * 128UL)) + _fuseiter_10895)]);
    }
  }
  return true;
}

static bool res5a_conv_1_cast_mul_add_relu_cast_reorder__680(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_288 = (void**)&__uninitialized_data[23657560UL];
  alignas(64) int8_t __rescheduled_0[128UL];
  int32_t* conv_os_blk_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[64UL];
  int32_t __cached_0;
  __cached_0 = 49;
  conv_os_blk_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 0;
  conv_os_acc_size[0] = __cached_1;
  for (uint64_t fused_0k_o__n_2861 = 0UL; fused_0k_o__n_2861 < 8UL; fused_0k_o__n_2861 += 1UL) {
    int8_t* __rescheduled_1 = (int8_t*)sc_aligned_malloc(__stream, 384UL);
    int32_t* __origouts_3030_shr = (int32_t*)sc_aligned_malloc(__stream, 25088UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[192UL];
    int32_t __cached_2;
    __cached_2 = conv_os_acc_size[0UL];
    int32_t __cached_3;
    __cached_3 = conv_os_blk_size[0UL];
    memset(&__origouts_3030_shr[(uint64_t)(((__cached_2 / 7) * 896) + ((__cached_2 % 7) * 128))], 0, ((uint64_t)(__cached_3 * 128) * 4UL));
    for (uint64_t c_o = 0UL; c_o < 2UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_4;
          __cached_4 = &__ins_0[(((fused_0k_o__n_2861 % 2UL) * 131072UL) + ((c_o * 65536UL) + ((r * 4096UL) + (s * 256UL))))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_4;
          void* __cached_5;
          __cached_5 = &__ins_1[(((fused_0k_o__n_2861 / 2UL) * 589824UL) + ((c_o * 294912UL) + ((r * 98304UL) + (s * 32768UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_5;
        }
      }
    }
    void* _arg_cache_43 = &__origouts_3030_shr[(uint64_t)(((__cached_2 / 7) * 896) + ((__cached_2 % 7) * 128))];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_288[0UL], A_list, B_list, &__origouts_3030_shr[(uint64_t)(((__cached_2 / 7) * 896) + ((__cached_2 % 7) * 128))], 1, 256, 32768, 18, 7, 7, __stream);
    for (uint64_t _fuseiter10899 = 0UL; _fuseiter10899 < 7UL; _fuseiter10899 += 1UL) {
      for (uint64_t _fuseiter10900 = 0UL; _fuseiter10900 < 7UL; _fuseiter10900 += 1UL) {
        for (uint64_t _fuseiter10901 = 0UL; _fuseiter10901 < 128UL; _fuseiter10901 += 16UL) {
          vec_s32x16 __cached_6;
          __cached_6 = vec_s32x16::load(&__origouts_3030_shr[((_fuseiter10899 * 896UL) + ((_fuseiter10900 * 128UL) + _fuseiter10901))]);
          vec_f32x16 __cached_7;
          __cached_7 = (vec_f32x16)(__cached_6);
          vec_f32x16 __cached_8;
          __cached_8 = vec_f32x16::load(&__ins_2[(((fused_0k_o__n_2861 / 2UL) * 128UL) + _fuseiter10901)]);
          __cached_7 = (__cached_7 * __cached_8);
          vec_f32x16 __cached_9;
          __cached_9 = vec_f32x16::load(&__ins_3[(((fused_0k_o__n_2861 / 2UL) * 128UL) + _fuseiter10901)]);
          __cached_7 = (__cached_7 + __cached_9);
          __cached_7 = sc_max(__cached_7, vec_f32x16(0.f));
          vec_s8x16 __cached_10;
          __cached_10 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_7));
          vec_s8x16 __cached_11;
          __cached_11 = __cached_10;
          vec_s8x16::store(__cached_11, &__outs_0[(((fused_0k_o__n_2861 % 2UL) * 25088UL) + ((((_fuseiter10901 + ((fused_0k_o__n_2861 / 2UL) * 128UL)) / 256UL) * 12544UL) + ((_fuseiter10899 * 1792UL) + ((_fuseiter10900 * 256UL) + ((_fuseiter10901 + ((fused_0k_o__n_2861 / 2UL) * 128UL)) % 256UL)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_3030_shr);
    sc_aligned_free(__stream, __rescheduled_1);
  }
  return true;
}

static bool res5a_conv_b_cast_mul_add_cast_reorder__682(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_290 = *(void**)(__module_data + 288);
  uint8_t* input_tmp = (uint8_t*)sc_aligned_malloc(__stream, 100352UL);
  for (uint64_t fused_0n__c_o_2862 = 0UL; fused_0n__c_o_2862 < 8UL; fused_0n__c_o_2862 += 1UL) {
    for (uint64_t p = 0UL; p < 7UL; p += 1UL) {
      for (uint64_t q = 0UL; q < 7UL; q += 1UL) {
        for (uint64_t c_i = 0UL; c_i < 256UL; c_i += 64UL) {
          vec_u8x64 __cached_0;
          __cached_0 = vec_u8x64::load(&__ins_0[(((fused_0n__c_o_2862 / 4UL) * 200704UL) + (((fused_0n__c_o_2862 % 4UL) * 50176UL) + ((p * 7168UL) + ((q * 512UL) + c_i))))]);
          vec_u8x64 __cached_1;
          __cached_1 = __cached_0;
          vec_u8x64::store(__cached_1, &input_tmp[(((fused_0n__c_o_2862 / 4UL) * 50176UL) + (((fused_0n__c_o_2862 % 4UL) * 12544UL) + ((p * 1792UL) + ((q * 256UL) + c_i))))]);
        }
      }
    }
  }
  for (uint64_t fused_0k__n_2863 = 0UL; fused_0k__n_2863 < 256UL; fused_0k__n_2863 += 1UL) {
    alignas(64) int8_t __rescheduled_2[128UL];
    int32_t* __origouts_3040_shr = (int32_t*)sc_aligned_malloc(__stream, 3136UL);
    void** A_list = (void**)&__rescheduled_2[0UL];
    void** B_list = (void**)&__rescheduled_2[64UL];
    for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
      void* __cached_2;
      __cached_2 = &input_tmp[(((fused_0k__n_2863 % 2UL) * 50176UL) + (c * 12544UL))];
      A_list[c] = __cached_2;
      void* __cached_3;
      __cached_3 = &__ins_1[(((fused_0k__n_2863 / 2UL) * 16384UL) + (c * 4096UL))];
      B_list[c] = __cached_3;
    }
    void* _arg_cache_44 = &__origouts_3040_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_290, A_list, B_list, &__origouts_3040_shr[0UL], 1, 1, 1, 4, 8, 7, __stream);
    for (uint64_t _fuseiter10934 = 0UL; _fuseiter10934 < 7UL; _fuseiter10934 += 1UL) {
      for (uint64_t _fuseiter10935 = 0UL; _fuseiter10935 < 7UL; _fuseiter10935 += 1UL) {
        vec_s32x16 __cached_4;
        __cached_4 = vec_s32x16::load(&__origouts_3040_shr[((_fuseiter10934 * 112UL) + (_fuseiter10935 * 16UL))]);
        vec_f32x16 __cached_5;
        __cached_5 = (vec_f32x16)(__cached_4);
        vec_f32x16 __cached_6;
        __cached_6 = vec_f32x16::load(&__ins_2[((fused_0k__n_2863 / 2UL) * 16UL)]);
        __cached_5 = (__cached_5 * __cached_6);
        vec_f32x16 __cached_7;
        __cached_7 = vec_f32x16::load(&__ins_3[((fused_0k__n_2863 / 2UL) * 16UL)]);
        __cached_5 = (__cached_5 + __cached_7);
        vec_s8x16 __cached_8;
        __cached_8 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_5));
        vec_s8x16 __cached_9;
        __cached_9 = __cached_8;
        vec_s8x16::store(__cached_9, &__outs_0[(((fused_0k__n_2863 % 2UL) * 100352UL) + (((((fused_0k__n_2863 / 2UL) * 16UL) / 64UL) * 3136UL) + ((_fuseiter10934 * 448UL) + ((_fuseiter10935 * 64UL) + (((fused_0k__n_2863 / 2UL) * 16UL) % 64UL)))))]);
      }
    }
    sc_aligned_free(__stream, __origouts_3040_shr);
  }
  sc_aligned_free(__stream, input_tmp);
  return true;
}

static bool mul__660(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2864____itr_2_2865____itr_3_2866 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2864____itr_2_2865____itr_3_2866 < 32UL; fused_0fused_0fused_0__itr_0____itr_1_2864____itr_2_2865____itr_3_2866 += 1UL) {
    for (uint64_t _fuseiter_10965 = 0UL; _fuseiter_10965 < 64UL; _fuseiter_10965 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2864____itr_2_2865____itr_3_2866 / 32UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2864____itr_2_2865____itr_3_2866 % 32UL) * 64UL)) + _fuseiter_10965)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2864____itr_2_2865____itr_3_2866 / 32UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2864____itr_2_2865____itr_3_2866 % 32UL) * 64UL)) + _fuseiter_10965)]);
    }
  }
  return true;
}

static bool mul__659(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2867____itr_2_2868____itr_3_2869 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2867____itr_2_2868____itr_3_2869 < 32UL; fused_0fused_0fused_0__itr_0____itr_1_2867____itr_2_2868____itr_3_2869 += 1UL) {
    for (uint64_t _fuseiter_10971 = 0UL; _fuseiter_10971 < 64UL; _fuseiter_10971 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2867____itr_2_2868____itr_3_2869 / 32UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2867____itr_2_2868____itr_3_2869 % 32UL) * 64UL)) + _fuseiter_10971)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2867____itr_2_2868____itr_3_2869 / 32UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2867____itr_2_2868____itr_3_2869 % 32UL) * 64UL)) + _fuseiter_10971)]);
    }
  }
  return true;
}

static bool reorder__540(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_10973 = 0UL; _fuseiter_10973 < 32UL; _fuseiter_10973 += 1UL) {
    for (uint64_t _fuseiter_10974 = 0UL; _fuseiter_10974 < 2UL; _fuseiter_10974 += 1UL) {
      for (uint64_t _fuseiter_10977 = 0UL; _fuseiter_10977 < 64UL; _fuseiter_10977 += 1UL) {
        for (uint64_t _fuseiter_10978 = 0UL; _fuseiter_10978 < 64UL; _fuseiter_10978 += 1UL) {
          for (uint64_t _fuseiter_10979 = 0UL; _fuseiter_10979 < 4UL; _fuseiter_10979 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_10978 + (_fuseiter_10973 * 64UL)) * 512UL) + ((_fuseiter_10979 + (_fuseiter_10977 * 4UL)) + (_fuseiter_10974 * 256UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_10973 * 32768UL) + ((_fuseiter_10974 * 16384UL) + ((_fuseiter_10977 * 256UL) + ((_fuseiter_10978 * 4UL) + _fuseiter_10979))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__662(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2870____itr_2_2871____itr_3_2872 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2870____itr_2_2871____itr_3_2872 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2870____itr_2_2871____itr_3_2872 += 1UL) {
    for (uint64_t _fuseiter_10984 = 0UL; _fuseiter_10984 < 128UL; _fuseiter_10984 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2870____itr_2_2871____itr_3_2872 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2870____itr_2_2871____itr_3_2872 % 4UL) * 128UL)) + _fuseiter_10984)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2870____itr_2_2871____itr_3_2872 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2870____itr_2_2871____itr_3_2872 % 4UL) * 128UL)) + _fuseiter_10984)]);
    }
  }
  return true;
}

static bool mul__661(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2873____itr_2_2874____itr_3_2875 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2873____itr_2_2874____itr_3_2875 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2873____itr_2_2874____itr_3_2875 += 1UL) {
    for (uint64_t _fuseiter_10990 = 0UL; _fuseiter_10990 < 128UL; _fuseiter_10990 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2873____itr_2_2874____itr_3_2875 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2873____itr_2_2874____itr_3_2875 % 4UL) * 128UL)) + _fuseiter_10990)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2873____itr_2_2874____itr_3_2875 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2873____itr_2_2874____itr_3_2875 % 4UL) * 128UL)) + _fuseiter_10990)]);
    }
  }
  return true;
}

static bool reorder__543(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_10992___fuseiter_10993_2876 = 0UL; fused_0_fuseiter_10992___fuseiter_10993_2876 < 128UL; fused_0_fuseiter_10992___fuseiter_10993_2876 += 1UL) {
    for (uint64_t _fuseiter_10996 = 0UL; _fuseiter_10996 < 16UL; _fuseiter_10996 += 1UL) {
      for (uint64_t _fuseiter_10997 = 0UL; _fuseiter_10997 < 128UL; _fuseiter_10997 += 1UL) {
        for (uint64_t _fuseiter_10998 = 0UL; _fuseiter_10998 < 4UL; _fuseiter_10998 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_10997 + ((fused_0_fuseiter_10992___fuseiter_10993_2876 / 32UL) * 128UL)) * 2048UL) + ((_fuseiter_10998 + (_fuseiter_10996 * 4UL)) + ((fused_0_fuseiter_10992___fuseiter_10993_2876 % 32UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_10992___fuseiter_10993_2876 / 32UL) * 262144UL) + (((fused_0_fuseiter_10992___fuseiter_10993_2876 % 32UL) * 8192UL) + ((_fuseiter_10996 * 512UL) + ((_fuseiter_10997 * 4UL) + _fuseiter_10998))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__249(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2877____itr_2_2878 = 0UL; fused_0fused_0__itr_0____itr_1_2877____itr_2_2878 < 786432UL; fused_0fused_0__itr_0____itr_1_2877____itr_2_2878 += 1UL) {
    for (uint64_t _fuseiter_11003 = 0UL; _fuseiter_11003 < 3UL; _fuseiter_11003 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2877____itr_2_2878 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2877____itr_2_2878 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2877____itr_2_2878 % 3UL) * 3UL))) + _fuseiter_11003)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2877____itr_2_2878 / 1536UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2877____itr_2_2878 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2877____itr_2_2878 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2877____itr_2_2878 % 3UL) * 3UL))) + _fuseiter_11003)] = __cached_2;
    }
  }
  return true;
}

static bool cast__250(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2879____itr_2_2880 = 0UL; fused_0fused_0__itr_0____itr_1_2879____itr_2_2880 < 786432UL; fused_0fused_0__itr_0____itr_1_2879____itr_2_2880 += 1UL) {
    for (uint64_t _fuseiter11008 = 0UL; _fuseiter11008 < 3UL; _fuseiter11008 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2879____itr_2_2880 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2879____itr_2_2880 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2879____itr_2_2880 % 3UL) * 3UL))) + _fuseiter11008)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2879____itr_2_2880 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2879____itr_2_2880 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2879____itr_2_2880 % 3UL) * 3UL))) + _fuseiter11008)] = __cached_1;
    }
  }
  return true;
}

static bool mul__258(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2881____itr_2_2882 = 0UL; fused_0fused_0__itr_0____itr_1_2881____itr_2_2882 < 786432UL; fused_0fused_0__itr_0____itr_1_2881____itr_2_2882 += 1UL) {
    for (uint64_t _fuseiter_11013 = 0UL; _fuseiter_11013 < 3UL; _fuseiter_11013 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2881____itr_2_2882 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2881____itr_2_2882 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2881____itr_2_2882 % 3UL) * 3UL))) + _fuseiter_11013)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2881____itr_2_2882 / 1536UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2881____itr_2_2882 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2881____itr_2_2882 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2881____itr_2_2882 % 3UL) * 3UL))) + _fuseiter_11013)] = __cached_2;
    }
  }
  return true;
}

static bool cast__259(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2883____itr_2_2884 = 0UL; fused_0fused_0__itr_0____itr_1_2883____itr_2_2884 < 786432UL; fused_0fused_0__itr_0____itr_1_2883____itr_2_2884 += 1UL) {
    for (uint64_t _fuseiter11018 = 0UL; _fuseiter11018 < 3UL; _fuseiter11018 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2883____itr_2_2884 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2883____itr_2_2884 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2883____itr_2_2884 % 3UL) * 3UL))) + _fuseiter11018)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2883____itr_2_2884 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2883____itr_2_2884 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2883____itr_2_2884 % 3UL) * 3UL))) + _fuseiter11018)] = __cached_1;
    }
  }
  return true;
}

static bool mul__664(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2885____itr_2_2886____itr_3_2887 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2885____itr_2_2886____itr_3_2887 < 32UL; fused_0fused_0fused_0__itr_0____itr_1_2885____itr_2_2886____itr_3_2887 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0__itr_0____itr_1_2885____itr_2_2886____itr_3_2887 / 32UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2885____itr_2_2886____itr_3_2887 % 32UL) * 16UL))]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[(((fused_0fused_0fused_0__itr_0____itr_1_2885____itr_2_2886____itr_3_2887 / 32UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2885____itr_2_2886____itr_3_2887 % 32UL) * 16UL))]);
  }
  return true;
}

static bool mul__663(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2888____itr_2_2889____itr_3_2890 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2888____itr_2_2889____itr_3_2890 < 32UL; fused_0fused_0fused_0__itr_0____itr_1_2888____itr_2_2889____itr_3_2890 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0__itr_0____itr_1_2888____itr_2_2889____itr_3_2890 / 32UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2888____itr_2_2889____itr_3_2890 % 32UL) * 16UL))]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[(((fused_0fused_0fused_0__itr_0____itr_1_2888____itr_2_2889____itr_3_2890 / 32UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2888____itr_2_2889____itr_3_2890 % 32UL) * 16UL))]);
  }
  return true;
}

static bool reorder__546(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_11031 = 0UL; _fuseiter_11031 < 32UL; _fuseiter_11031 += 1UL) {
    for (uint64_t _fuseiter_11033 = 0UL; _fuseiter_11033 < 3UL; _fuseiter_11033 += 1UL) {
      for (uint64_t _fuseiter_11034 = 0UL; _fuseiter_11034 < 3UL; _fuseiter_11034 += 1UL) {
        for (uint64_t _fuseiter_11035 = 0UL; _fuseiter_11035 < 128UL; _fuseiter_11035 += 1UL) {
          for (uint64_t _fuseiter_11036 = 0UL; _fuseiter_11036 < 16UL; _fuseiter_11036 += 1UL) {
            for (uint64_t _fuseiter_11037 = 0UL; _fuseiter_11037 < 4UL; _fuseiter_11037 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_11036 + (_fuseiter_11031 * 16UL)) * 4608UL) + (((_fuseiter_11037 + (_fuseiter_11035 * 4UL)) * 9UL) + ((_fuseiter_11033 * 3UL) + _fuseiter_11034)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[((_fuseiter_11031 * 73728UL) + ((_fuseiter_11033 * 24576UL) + ((_fuseiter_11034 * 8192UL) + ((_fuseiter_11035 * 64UL) + ((_fuseiter_11036 * 4UL) + _fuseiter_11037)))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool res5a_conv_2_cast_mul_add_cast_add_cast__679(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_292 = *(void**)(__module_data + 296);
  for (uint64_t fused_0k__n_2891 = 0UL; fused_0k__n_2891 < 64UL; fused_0k__n_2891 += 1UL) {
    alignas(64) int8_t __rescheduled_1[128UL];
    int32_t* __origouts_3050_shr = (int32_t*)sc_aligned_malloc(__stream, 12544UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[64UL];
    for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0k__n_2891 % 2UL) * 25088UL) + (c * 12544UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(((fused_0k__n_2891 / 2UL) * 32768UL) + (c * 16384UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_45 = &__origouts_3050_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_292, A_list, B_list, &__origouts_3050_shr[0UL], 1, 1, 1, 2, 7, 7, __stream);
    for (uint64_t _fuseiter11040 = 0UL; _fuseiter11040 < 7UL; _fuseiter11040 += 1UL) {
      for (uint64_t _fuseiter11041 = 0UL; _fuseiter11041 < 7UL; _fuseiter11041 += 1UL) {
        for (uint64_t _fuseiter11042 = 0UL; _fuseiter11042 < 64UL; _fuseiter11042 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_3050_shr[((_fuseiter11040 * 448UL) + ((_fuseiter11041 * 64UL) + _fuseiter11042))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0k__n_2891 / 2UL) * 64UL) + _fuseiter11042)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0k__n_2891 / 2UL) * 64UL) + _fuseiter11042)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0k__n_2891 % 2UL) * 100352UL) + ((fused_0k__n_2891 / 2UL) * 3136UL)) + ((_fuseiter11040 * 448UL) + ((_fuseiter11041 * 64UL) + _fuseiter11042)))]);
          __cached_6 = (__cached_6 + __cached_7);
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16::store(__cached_8, &__outs_0[((((fused_0k__n_2891 % 2UL) * 100352UL) + ((fused_0k__n_2891 / 2UL) * 3136UL)) + ((_fuseiter11040 * 448UL) + ((_fuseiter11041 * 64UL) + _fuseiter11042)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_3050_shr);
  }
  return true;
}

static bool res5b_conv_0_cast_mul_add_relu_cast_reorder__678(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_294 = *(void**)(__module_data + 304);
  for (uint64_t fused_0n__k_2892 = 0UL; fused_0n__k_2892 < 2UL; fused_0n__k_2892 += 1UL) {
    memset(&__outs_0[(fused_0n__k_2892 * 41472UL)], 0, 4608UL);
    for (uint64_t p1 = 0UL; p1 < 7UL; p1 += 1UL) {
      memset(&__outs_0[((fused_0n__k_2892 * 41472UL) + ((p1 + 1UL) * 4608UL))], 0, 512UL);
      memset(&__outs_0[(((fused_0n__k_2892 * 41472UL) + ((p1 + 1UL) * 4608UL)) + 4096UL)], 0, 512UL);
    }
    memset(&__outs_0[((fused_0n__k_2892 * 41472UL) + 36864UL)], 0, 4608UL);
  }
  for (uint64_t fused_0n__k_2893 = 0UL; fused_0n__k_2893 < 8UL; fused_0n__k_2893 += 1UL) {
    int8_t* __rescheduled_2 = (int8_t*)sc_aligned_malloc(__stream, 512UL);
    int32_t* __origouts_3060_shr = (int32_t*)sc_aligned_malloc(__stream, 25088UL);
    void** A_list = (void**)&__rescheduled_2[0UL];
    void** B_list = (void**)&__rescheduled_2[256UL];
    for (uint64_t c = 0UL; c < 32UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0n__k_2893 / 4UL) * 100352UL) + (c * 3136UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(((fused_0n__k_2893 % 4UL) * 262144UL) + (c * 8192UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_46 = &__origouts_3060_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_294, A_list, B_list, &__origouts_3060_shr[0UL], 1, 1, 1, 32, 8, 7, __stream);
    for (uint64_t _fuseiter11076 = 0UL; _fuseiter11076 < 7UL; _fuseiter11076 += 1UL) {
      for (uint64_t _fuseiter11077 = 0UL; _fuseiter11077 < 7UL; _fuseiter11077 += 1UL) {
        for (uint64_t _fuseiter11078 = 0UL; _fuseiter11078 < 128UL; _fuseiter11078 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_3060_shr[((_fuseiter11076 * 896UL) + ((_fuseiter11077 * 128UL) + _fuseiter11078))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0n__k_2893 % 4UL) * 128UL) + _fuseiter11078)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_2893 % 4UL) * 128UL) + _fuseiter11078)]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[(((fused_0n__k_2893 / 4UL) * 41472UL) + ((((_fuseiter11078 + ((fused_0n__k_2893 % 4UL) * 128UL)) / 512UL) * 41472UL) + (((_fuseiter11076 + 1UL) * 4608UL) + (((_fuseiter11077 + 1UL) * 512UL) + ((_fuseiter11078 + ((fused_0n__k_2893 % 4UL) * 128UL)) % 512UL)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_3060_shr);
    sc_aligned_free(__stream, __rescheduled_2);
  }
  return true;
}

static bool res5b_conv_1_cast_mul_add_relu_cast_reorder__677(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_298 = (void**)&__uninitialized_data[23657576UL];
  alignas(64) int8_t __rescheduled_0[128UL];
  int32_t* conv_os_blk_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[64UL];
  int32_t __cached_0;
  __cached_0 = 49;
  conv_os_blk_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 0;
  conv_os_acc_size[0] = __cached_1;
  for (uint64_t fused_0k_o__n_2894 = 0UL; fused_0k_o__n_2894 < 64UL; fused_0k_o__n_2894 += 1UL) {
    int8_t* __rescheduled_1 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
    int32_t* __origouts_3070_shr = (int32_t*)sc_aligned_malloc(__stream, 3136UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[128UL];
    int32_t __cached_2;
    __cached_2 = conv_os_acc_size[0UL];
    int32_t __cached_3;
    __cached_3 = conv_os_blk_size[0UL];
    memset(&__origouts_3070_shr[(uint64_t)(((__cached_2 / 7) * 112) + ((__cached_2 % 7) * 16))], 0, ((uint64_t)(__cached_3 * 16) * 4UL));
    for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
      for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
        void* __cached_4;
        __cached_4 = &__ins_0[(((fused_0k_o__n_2894 % 2UL) * 41472UL) + ((r * 4608UL) + (s * 512UL)))];
        A_list[((r * 3UL) + s)] = __cached_4;
        void* __cached_5;
        __cached_5 = &__ins_1[(((fused_0k_o__n_2894 / 2UL) * 73728UL) + ((r * 24576UL) + (s * 8192UL)))];
        B_list[((r * 3UL) + s)] = __cached_5;
      }
    }
    void* _arg_cache_47 = &__origouts_3070_shr[(uint64_t)(((__cached_2 / 7) * 112) + ((__cached_2 % 7) * 16))];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_298[0UL], A_list, B_list, &__origouts_3070_shr[(uint64_t)(((__cached_2 / 7) * 112) + ((__cached_2 % 7) * 16))], 1, 512, 8192, 9, 7, 7, __stream);
    for (uint64_t _fuseiter11111 = 0UL; _fuseiter11111 < 7UL; _fuseiter11111 += 1UL) {
      for (uint64_t _fuseiter11112 = 0UL; _fuseiter11112 < 7UL; _fuseiter11112 += 1UL) {
        vec_s32x16 __cached_6;
        __cached_6 = vec_s32x16::load(&__origouts_3070_shr[((_fuseiter11111 * 112UL) + (_fuseiter11112 * 16UL))]);
        vec_f32x16 __cached_7;
        __cached_7 = (vec_f32x16)(__cached_6);
        vec_f32x16 __cached_8;
        __cached_8 = vec_f32x16::load(&__ins_2[((fused_0k_o__n_2894 / 2UL) * 16UL)]);
        __cached_7 = (__cached_7 * __cached_8);
        vec_f32x16 __cached_9;
        __cached_9 = vec_f32x16::load(&__ins_3[((fused_0k_o__n_2894 / 2UL) * 16UL)]);
        __cached_7 = (__cached_7 + __cached_9);
        __cached_7 = sc_max(__cached_7, vec_f32x16(0.f));
        vec_s8x16 __cached_10;
        __cached_10 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_7));
        vec_s8x16 __cached_11;
        __cached_11 = __cached_10;
        vec_s8x16::store(__cached_11, &__outs_0[(((fused_0k_o__n_2894 % 2UL) * 25088UL) + (((((fused_0k_o__n_2894 / 2UL) * 16UL) / 256UL) * 12544UL) + ((_fuseiter11111 * 1792UL) + ((_fuseiter11112 * 256UL) + (((fused_0k_o__n_2894 / 2UL) * 16UL) % 256UL)))))]);
      }
    }
    sc_aligned_free(__stream, __origouts_3070_shr);
    sc_aligned_free(__stream, __rescheduled_1);
  }
  return true;
}

static bool mul__666(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2895____itr_2_2896____itr_3_2897 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2895____itr_2_2896____itr_3_2897 < 32UL; fused_0fused_0fused_0__itr_0____itr_1_2895____itr_2_2896____itr_3_2897 += 1UL) {
    for (uint64_t _fuseiter_11148 = 0UL; _fuseiter_11148 < 64UL; _fuseiter_11148 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2895____itr_2_2896____itr_3_2897 / 32UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2895____itr_2_2896____itr_3_2897 % 32UL) * 64UL)) + _fuseiter_11148)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2895____itr_2_2896____itr_3_2897 / 32UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2895____itr_2_2896____itr_3_2897 % 32UL) * 64UL)) + _fuseiter_11148)]);
    }
  }
  return true;
}

static bool mul__665(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2898____itr_2_2899____itr_3_2900 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2898____itr_2_2899____itr_3_2900 < 32UL; fused_0fused_0fused_0__itr_0____itr_1_2898____itr_2_2899____itr_3_2900 += 1UL) {
    for (uint64_t _fuseiter_11154 = 0UL; _fuseiter_11154 < 64UL; _fuseiter_11154 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2898____itr_2_2899____itr_3_2900 / 32UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2898____itr_2_2899____itr_3_2900 % 32UL) * 64UL)) + _fuseiter_11154)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2898____itr_2_2899____itr_3_2900 / 32UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2898____itr_2_2899____itr_3_2900 % 32UL) * 64UL)) + _fuseiter_11154)]);
    }
  }
  return true;
}

static bool reorder__549(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_11156 = 0UL; _fuseiter_11156 < 32UL; _fuseiter_11156 += 1UL) {
    for (uint64_t _fuseiter_11157 = 0UL; _fuseiter_11157 < 2UL; _fuseiter_11157 += 1UL) {
      for (uint64_t _fuseiter_11160 = 0UL; _fuseiter_11160 < 64UL; _fuseiter_11160 += 1UL) {
        for (uint64_t _fuseiter_11161 = 0UL; _fuseiter_11161 < 64UL; _fuseiter_11161 += 1UL) {
          for (uint64_t _fuseiter_11162 = 0UL; _fuseiter_11162 < 4UL; _fuseiter_11162 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_11161 + (_fuseiter_11156 * 64UL)) * 512UL) + ((_fuseiter_11162 + (_fuseiter_11160 * 4UL)) + (_fuseiter_11157 * 256UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_11156 * 32768UL) + ((_fuseiter_11157 * 16384UL) + ((_fuseiter_11160 * 256UL) + ((_fuseiter_11161 * 4UL) + _fuseiter_11162))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__668(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2901____itr_2_2902____itr_3_2903 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2901____itr_2_2902____itr_3_2903 < 16UL; fused_0fused_0fused_0__itr_0____itr_1_2901____itr_2_2902____itr_3_2903 += 1UL) {
    for (uint64_t _fuseiter_11167 = 0UL; _fuseiter_11167 < 32UL; _fuseiter_11167 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2901____itr_2_2902____itr_3_2903 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2901____itr_2_2902____itr_3_2903 % 16UL) * 32UL)) + _fuseiter_11167)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2901____itr_2_2902____itr_3_2903 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2901____itr_2_2902____itr_3_2903 % 16UL) * 32UL)) + _fuseiter_11167)]);
    }
  }
  return true;
}

static bool mul__667(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2904____itr_2_2905____itr_3_2906 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2904____itr_2_2905____itr_3_2906 < 16UL; fused_0fused_0fused_0__itr_0____itr_1_2904____itr_2_2905____itr_3_2906 += 1UL) {
    for (uint64_t _fuseiter_11173 = 0UL; _fuseiter_11173 < 32UL; _fuseiter_11173 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2904____itr_2_2905____itr_3_2906 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2904____itr_2_2905____itr_3_2906 % 16UL) * 32UL)) + _fuseiter_11173)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2904____itr_2_2905____itr_3_2906 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2904____itr_2_2905____itr_3_2906 % 16UL) * 32UL)) + _fuseiter_11173)]);
    }
  }
  return true;
}

static bool reorder__552(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_11175 = 0UL; _fuseiter_11175 < 16UL; _fuseiter_11175 += 1UL) {
    for (uint64_t _fuseiter_11176 = 0UL; _fuseiter_11176 < 4UL; _fuseiter_11176 += 1UL) {
      for (uint64_t _fuseiter_11179 = 0UL; _fuseiter_11179 < 128UL; _fuseiter_11179 += 1UL) {
        for (uint64_t _fuseiter_11180 = 0UL; _fuseiter_11180 < 32UL; _fuseiter_11180 += 1UL) {
          for (uint64_t _fuseiter_11181 = 0UL; _fuseiter_11181 < 4UL; _fuseiter_11181 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_11180 + (_fuseiter_11175 * 32UL)) * 2048UL) + ((_fuseiter_11181 + (_fuseiter_11179 * 4UL)) + (_fuseiter_11176 * 512UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_11175 * 65536UL) + ((_fuseiter_11176 * 16384UL) + ((_fuseiter_11179 * 128UL) + ((_fuseiter_11180 * 4UL) + _fuseiter_11181))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__670(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2907____itr_2_2908____itr_3_2909 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2907____itr_2_2908____itr_3_2909 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2907____itr_2_2908____itr_3_2909 += 1UL) {
    for (uint64_t _fuseiter_11186 = 0UL; _fuseiter_11186 < 128UL; _fuseiter_11186 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2907____itr_2_2908____itr_3_2909 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2907____itr_2_2908____itr_3_2909 % 4UL) * 128UL)) + _fuseiter_11186)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2907____itr_2_2908____itr_3_2909 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2907____itr_2_2908____itr_3_2909 % 4UL) * 128UL)) + _fuseiter_11186)]);
    }
  }
  return true;
}

static bool mul__669(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2910____itr_2_2911____itr_3_2912 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2910____itr_2_2911____itr_3_2912 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2910____itr_2_2911____itr_3_2912 += 1UL) {
    for (uint64_t _fuseiter_11192 = 0UL; _fuseiter_11192 < 128UL; _fuseiter_11192 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2910____itr_2_2911____itr_3_2912 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2910____itr_2_2911____itr_3_2912 % 4UL) * 128UL)) + _fuseiter_11192)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2910____itr_2_2911____itr_3_2912 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2910____itr_2_2911____itr_3_2912 % 4UL) * 128UL)) + _fuseiter_11192)]);
    }
  }
  return true;
}

static bool reorder__555(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_11194___fuseiter_11195_2913___fuseiter_11196_2914 = 0UL; fused_0fused_0_fuseiter_11194___fuseiter_11195_2913___fuseiter_11196_2914 < 24UL; fused_0fused_0_fuseiter_11194___fuseiter_11195_2913___fuseiter_11196_2914 += 1UL) {
    for (uint64_t _fuseiter_11197 = 0UL; _fuseiter_11197 < 3UL; _fuseiter_11197 += 1UL) {
      for (uint64_t _fuseiter_11198 = 0UL; _fuseiter_11198 < 64UL; _fuseiter_11198 += 1UL) {
        for (uint64_t _fuseiter_11199 = 0UL; _fuseiter_11199 < 128UL; _fuseiter_11199 += 1UL) {
          for (uint64_t _fuseiter_11200 = 0UL; _fuseiter_11200 < 4UL; _fuseiter_11200 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_11199 + ((fused_0fused_0_fuseiter_11194___fuseiter_11195_2913___fuseiter_11196_2914 / 6UL) * 128UL)) * 4608UL) + ((((_fuseiter_11200 + (_fuseiter_11198 * 4UL)) + (((fused_0fused_0_fuseiter_11194___fuseiter_11195_2913___fuseiter_11196_2914 / 3UL) % 2UL) * 256UL)) * 9UL) + (((fused_0fused_0_fuseiter_11194___fuseiter_11195_2913___fuseiter_11196_2914 % 3UL) * 3UL) + _fuseiter_11197)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_11194___fuseiter_11195_2913___fuseiter_11196_2914 / 6UL) * 589824UL) + ((((fused_0fused_0_fuseiter_11194___fuseiter_11195_2913___fuseiter_11196_2914 / 3UL) % 2UL) * 294912UL) + (((fused_0fused_0_fuseiter_11194___fuseiter_11195_2913___fuseiter_11196_2914 % 3UL) * 98304UL) + ((_fuseiter_11197 * 32768UL) + ((_fuseiter_11198 * 512UL) + ((_fuseiter_11199 * 4UL) + _fuseiter_11200))))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool res5b_conv_2_cast_mul_add_cast_add_cast_reorder__676(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_300 = *(void**)(__module_data + 312);
  for (uint64_t fused_0n__k_2915 = 0UL; fused_0n__k_2915 < 64UL; fused_0n__k_2915 += 1UL) {
    alignas(64) int8_t __rescheduled_1[128UL];
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_3080_shr = (int32_t*)sc_aligned_malloc(__stream, 1792UL);
      void** A_list = (void**)&__rescheduled_1[0UL];
      void** B_list = (void**)&__rescheduled_1[64UL];
      for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0n__k_2915 / 32UL) * 25088UL) + ((c * 12544UL) + (p_o * 1792UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0n__k_2915 % 32UL) * 32768UL) + (c * 16384UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_48 = &__origouts_3080_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_300, A_list, B_list, &__origouts_3080_shr[0UL], 1, 1, 1, 2, 7, 7, __stream);
      for (uint64_t _fuseiter11204 = 0UL; _fuseiter11204 < 7UL; _fuseiter11204 += 1UL) {
        for (uint64_t _fuseiter11205 = 0UL; _fuseiter11205 < 64UL; _fuseiter11205 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_3080_shr[((_fuseiter11204 * 64UL) + _fuseiter11205)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0n__k_2915 % 32UL) * 64UL) + _fuseiter11205)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_2915 % 32UL) * 64UL) + _fuseiter11205)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_u8x16 __cached_7;
          __cached_7 = vec_u8x16::load(&__ins_4[((((fused_0n__k_2915 / 32UL) * 100352UL) + (((fused_0n__k_2915 % 32UL) * 3136UL) + (p_o * 448UL))) + ((_fuseiter11204 * 64UL) + _fuseiter11205))]);
          __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16 __cached_9;
          __cached_9 = __cached_8;
          vec_u8x16::store(__cached_9, &__outs_0[(((fused_0n__k_2915 / 32UL) * 100352UL) + ((((_fuseiter11205 + ((fused_0n__k_2915 % 32UL) * 64UL)) / 512UL) * 25088UL) + ((p_o * 3584UL) + ((_fuseiter11204 * 512UL) + ((_fuseiter11205 + ((fused_0n__k_2915 % 32UL) * 64UL)) % 512UL)))))]);
        }
      }
      sc_aligned_free(__stream, __origouts_3080_shr);
    }
  }
  return true;
}

static bool res5c_conv_0_cast_mul_add_relu_cast_reorder__675(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_302 = *(void**)(__module_data + 320);
  for (uint64_t fused_0n__k_2916 = 0UL; fused_0n__k_2916 < 4UL; fused_0n__k_2916 += 1UL) {
    memset(&__outs_0[(((fused_0n__k_2916 / 2UL) * 41472UL) + ((fused_0n__k_2916 % 2UL) * 20736UL))], 0, 2304UL);
    for (uint64_t p1 = 0UL; p1 < 7UL; p1 += 1UL) {
      memset(&__outs_0[(((fused_0n__k_2916 / 2UL) * 41472UL) + (((fused_0n__k_2916 % 2UL) * 20736UL) + ((p1 + 1UL) * 2304UL)))], 0, 256UL);
      memset(&__outs_0[((((fused_0n__k_2916 / 2UL) * 41472UL) + (((fused_0n__k_2916 % 2UL) * 20736UL) + ((p1 + 1UL) * 2304UL))) + 2048UL)], 0, 256UL);
    }
    memset(&__outs_0[((((fused_0n__k_2916 / 2UL) * 41472UL) + ((fused_0n__k_2916 % 2UL) * 20736UL)) + 18432UL)], 0, 2304UL);
  }
  for (uint64_t fused_0n__k_2917 = 0UL; fused_0n__k_2917 < 32UL; fused_0n__k_2917 += 1UL) {
    alignas(64) int8_t __rescheduled_2[128UL];
    int32_t* __origouts_3090_shr = (int32_t*)sc_aligned_malloc(__stream, 6272UL);
    void** A_list = (void**)&__rescheduled_2[0UL];
    void** B_list = (void**)&__rescheduled_2[64UL];
    for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0n__k_2917 / 16UL) * 100352UL) + (c * 25088UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(((fused_0n__k_2917 % 16UL) * 65536UL) + (c * 16384UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_49 = &__origouts_3090_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_302, A_list, B_list, &__origouts_3090_shr[0UL], 1, 1, 1, 4, 8, 7, __stream);
    for (uint64_t _fuseiter11244 = 0UL; _fuseiter11244 < 7UL; _fuseiter11244 += 1UL) {
      for (uint64_t _fuseiter11245 = 0UL; _fuseiter11245 < 7UL; _fuseiter11245 += 1UL) {
        for (uint64_t _fuseiter11246 = 0UL; _fuseiter11246 < 32UL; _fuseiter11246 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_3090_shr[((_fuseiter11244 * 224UL) + ((_fuseiter11245 * 32UL) + _fuseiter11246))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0n__k_2917 % 16UL) * 32UL) + _fuseiter11246)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_2917 % 16UL) * 32UL) + _fuseiter11246)]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[(((fused_0n__k_2917 / 16UL) * 41472UL) + ((((_fuseiter11246 + ((fused_0n__k_2917 % 16UL) * 32UL)) / 256UL) * 20736UL) + (((_fuseiter11244 + 1UL) * 2304UL) + (((_fuseiter11245 + 1UL) * 256UL) + ((_fuseiter11246 + ((fused_0n__k_2917 % 16UL) * 32UL)) % 256UL)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_3090_shr);
  }
  return true;
}

static bool res5c_conv_1_cast_mul_add_relu_cast_reorder__674(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_304 = (void**)&__uninitialized_data[23657584UL];
  alignas(64) int8_t __rescheduled_0[128UL];
  int32_t* conv_os_blk_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[64UL];
  int32_t __cached_0;
  __cached_0 = 49;
  conv_os_blk_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 0;
  conv_os_acc_size[0] = __cached_1;
  for (uint64_t fused_0k_o__n_2918 = 0UL; fused_0k_o__n_2918 < 8UL; fused_0k_o__n_2918 += 1UL) {
    int8_t* __rescheduled_1 = (int8_t*)sc_aligned_malloc(__stream, 384UL);
    int32_t* __origouts_3100_shr = (int32_t*)sc_aligned_malloc(__stream, 25088UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[192UL];
    int32_t __cached_2;
    __cached_2 = conv_os_acc_size[0UL];
    int32_t __cached_3;
    __cached_3 = conv_os_blk_size[0UL];
    memset(&__origouts_3100_shr[(uint64_t)(((__cached_2 / 7) * 896) + ((__cached_2 % 7) * 128))], 0, ((uint64_t)(__cached_3 * 128) * 4UL));
    for (uint64_t c_o = 0UL; c_o < 2UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_4;
          __cached_4 = &__ins_0[(((fused_0k_o__n_2918 % 2UL) * 41472UL) + ((c_o * 20736UL) + ((r * 2304UL) + (s * 256UL))))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_4;
          void* __cached_5;
          __cached_5 = &__ins_1[(((fused_0k_o__n_2918 / 2UL) * 589824UL) + ((c_o * 294912UL) + ((r * 98304UL) + (s * 32768UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_5;
        }
      }
    }
    void* _arg_cache_50 = &__origouts_3100_shr[(uint64_t)(((__cached_2 / 7) * 896) + ((__cached_2 % 7) * 128))];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_304[0UL], A_list, B_list, &__origouts_3100_shr[(uint64_t)(((__cached_2 / 7) * 896) + ((__cached_2 % 7) * 128))], 1, 256, 32768, 18, 7, 7, __stream);
    for (uint64_t _fuseiter11279 = 0UL; _fuseiter11279 < 7UL; _fuseiter11279 += 1UL) {
      for (uint64_t _fuseiter11280 = 0UL; _fuseiter11280 < 7UL; _fuseiter11280 += 1UL) {
        for (uint64_t _fuseiter11281 = 0UL; _fuseiter11281 < 128UL; _fuseiter11281 += 16UL) {
          vec_s32x16 __cached_6;
          __cached_6 = vec_s32x16::load(&__origouts_3100_shr[((_fuseiter11279 * 896UL) + ((_fuseiter11280 * 128UL) + _fuseiter11281))]);
          vec_f32x16 __cached_7;
          __cached_7 = (vec_f32x16)(__cached_6);
          vec_f32x16 __cached_8;
          __cached_8 = vec_f32x16::load(&__ins_2[(((fused_0k_o__n_2918 / 2UL) * 128UL) + _fuseiter11281)]);
          __cached_7 = (__cached_7 * __cached_8);
          vec_f32x16 __cached_9;
          __cached_9 = vec_f32x16::load(&__ins_3[(((fused_0k_o__n_2918 / 2UL) * 128UL) + _fuseiter11281)]);
          __cached_7 = (__cached_7 + __cached_9);
          __cached_7 = sc_max(__cached_7, vec_f32x16(0.f));
          vec_s8x16 __cached_10;
          __cached_10 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_7));
          vec_s8x16 __cached_11;
          __cached_11 = __cached_10;
          vec_s8x16::store(__cached_11, &__outs_0[(((fused_0k_o__n_2918 % 2UL) * 25088UL) + ((((_fuseiter11281 + ((fused_0k_o__n_2918 / 2UL) * 128UL)) / 512UL) * 25088UL) + ((_fuseiter11279 * 3584UL) + ((_fuseiter11280 * 512UL) + ((_fuseiter11281 + ((fused_0k_o__n_2918 / 2UL) * 128UL)) % 512UL)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_3100_shr);
    sc_aligned_free(__stream, __rescheduled_1);
  }
  return true;
}

static bool mul__672(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2919____itr_2_2920____itr_3_2921 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2919____itr_2_2920____itr_3_2921 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2919____itr_2_2920____itr_3_2921 += 1UL) {
    for (uint64_t _fuseiter_11316 = 0UL; _fuseiter_11316 < 512UL; _fuseiter_11316 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2919____itr_2_2920____itr_3_2921 / 4UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2919____itr_2_2920____itr_3_2921 % 4UL) * 512UL)) + _fuseiter_11316)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2919____itr_2_2920____itr_3_2921 / 4UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2919____itr_2_2920____itr_3_2921 % 4UL) * 512UL)) + _fuseiter_11316)]);
    }
  }
  return true;
}

static bool mul__671(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_2922____itr_2_2923____itr_3_2924 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_2922____itr_2_2923____itr_3_2924 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_2922____itr_2_2923____itr_3_2924 += 1UL) {
    for (uint64_t _fuseiter_11322 = 0UL; _fuseiter_11322 < 512UL; _fuseiter_11322 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_2922____itr_2_2923____itr_3_2924 / 4UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2922____itr_2_2923____itr_3_2924 % 4UL) * 512UL)) + _fuseiter_11322)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_2922____itr_2_2923____itr_3_2924 / 4UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_2922____itr_2_2923____itr_3_2924 % 4UL) * 512UL)) + _fuseiter_11322)]);
    }
  }
  return true;
}

static bool reorder__558(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_11324___fuseiter_11325_2925___fuseiter_11326_2926___fuseiter_11327_2927___fuseiter_11328_2928 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_11324___fuseiter_11325_2925___fuseiter_11326_2926___fuseiter_11327_2927___fuseiter_11328_2928 < 512UL; fused_0fused_0fused_0fused_0_fuseiter_11324___fuseiter_11325_2925___fuseiter_11326_2926___fuseiter_11327_2927___fuseiter_11328_2928 += 1UL) {
    for (uint64_t _fuseiter_11329 = 0UL; _fuseiter_11329 < 512UL; _fuseiter_11329 += 1UL) {
      for (uint64_t _fuseiter_11330 = 0UL; _fuseiter_11330 < 4UL; _fuseiter_11330 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_11329 + ((fused_0fused_0fused_0fused_0_fuseiter_11324___fuseiter_11325_2925___fuseiter_11326_2926___fuseiter_11327_2927___fuseiter_11328_2928 / 128UL) * 512UL)) * 512UL) + (_fuseiter_11330 + ((fused_0fused_0fused_0fused_0_fuseiter_11324___fuseiter_11325_2925___fuseiter_11326_2926___fuseiter_11327_2927___fuseiter_11328_2928 % 128UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_11324___fuseiter_11325_2925___fuseiter_11326_2926___fuseiter_11327_2927___fuseiter_11328_2928 / 128UL) * 262144UL) + (((fused_0fused_0fused_0fused_0_fuseiter_11324___fuseiter_11325_2925___fuseiter_11326_2926___fuseiter_11327_2927___fuseiter_11328_2928 % 128UL) * 2048UL) + ((_fuseiter_11329 * 4UL) + _fuseiter_11330)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool res5c_conv_2_cast_mul_add_cast_add_cast_cast__673(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_306 = *(void**)(__module_data + 328);
  for (uint64_t fused_0k__n_2929 = 0UL; fused_0k__n_2929 < 8UL; fused_0k__n_2929 += 1UL) {
    alignas(64) int8_t __rescheduled_1[128UL];
    int32_t* __origouts_3110_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[64UL];
    void* __cached_0;
    __cached_0 = &__ins_0[((fused_0k__n_2929 % 2UL) * 25088UL)];
    A_list[0UL] = __cached_0;
    void* __cached_1;
    __cached_1 = &__ins_1[((fused_0k__n_2929 / 2UL) * 262144UL)];
    B_list[0UL] = __cached_1;
    void* _arg_cache_51 = &__origouts_3110_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_306, A_list, B_list, &__origouts_3110_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
    for (uint64_t _fuseiter11333 = 0UL; _fuseiter11333 < 7UL; _fuseiter11333 += 1UL) {
      for (uint64_t _fuseiter11334 = 0UL; _fuseiter11334 < 7UL; _fuseiter11334 += 1UL) {
        for (uint64_t _fuseiter11335 = 0UL; _fuseiter11335 < 512UL; _fuseiter11335 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_3110_shr[((_fuseiter11333 * 3584UL) + ((_fuseiter11334 * 512UL) + _fuseiter11335))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0k__n_2929 / 2UL) * 512UL) + _fuseiter11335)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0k__n_2929 / 2UL) * 512UL) + _fuseiter11335)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_u8x16 __cached_7;
          __cached_7 = vec_u8x16::load(&__ins_4[((((fused_0k__n_2929 % 2UL) * 100352UL) + ((fused_0k__n_2929 / 2UL) * 25088UL)) + ((_fuseiter11333 * 3584UL) + ((_fuseiter11334 * 512UL) + _fuseiter11335)))]);
          __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_s8x16 __cached_9;
          __cached_9 = (vec_s8x16)(__cached_8);
          vec_s8x16::store(__cached_9, &__outs_0[((((fused_0k__n_2929 % 2UL) * 100352UL) + ((fused_0k__n_2929 / 2UL) * 25088UL)) + ((_fuseiter11333 * 3584UL) + ((_fuseiter11334 * 512UL) + _fuseiter11335)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_3110_shr);
  }
  return true;
}

static bool reorder__105(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0n__h_2930 = 0UL; fused_0n__h_2930 < 14UL; fused_0n__h_2930 += 1UL) {
    for (uint64_t w = 0UL; w < 7UL; w += 1UL) {
      for (uint64_t c = 0UL; c < 2048UL; c += 64UL) {
        vec_s8x64 zmm0;
        vec_s8x64 __cached_0;
        __cached_0 = vec_s8x64::load(&__ins_0[(((fused_0n__h_2930 / 7UL) * 100352UL) + (((c / 512UL) * 25088UL) + (((fused_0n__h_2930 % 7UL) * 3584UL) + ((w * 512UL) + (c % 512UL)))))]);
        zmm0 = __cached_0;
        vec_s8x64 __cached_1;
        __cached_1 = zmm0;
        vec_s8x64::store(__cached_1, &__outs_0[(((fused_0n__h_2930 / 7UL) * 100352UL) + (((fused_0n__h_2930 % 7UL) * 14336UL) + ((w * 2048UL) + c)))]);
      }
    }
  }
  return true;
}

static void __init_const_globals(int8_t* __restrict__ backbone_output, int8_t* __restrict__ backbone_input, float* __restrict__ res2a_weight_b, float* __restrict__ res2a_bias_b, float* __restrict__ res2a_weight_0, float* __restrict__ res2a_bias_0, float* __restrict__ res2a_weight_1, float* __restrict__ res2a_bias_1, float* __restrict__ res2a_weight_2, float* __restrict__ res2a_bias_2, float* __restrict__ res2b_weight_0, float* __restrict__ res2b_bias_0, float* __restrict__ res2b_weight_1, float* __restrict__ res2b_bias_1, float* __restrict__ res2b_weight_2, float* __restrict__ res2b_bias_2, float* __restrict__ res2c_weight_0, float* __restrict__ res2c_bias_0, float* __restrict__ res2c_weight_1, float* __restrict__ res2c_bias_1, float* __restrict__ res2c_weight_2, float* __restrict__ res2c_bias_2, float* __restrict__ res3a_weight_b, float* __restrict__ res3a_bias_b, float* __restrict__ res3a_weight_0, float* __restrict__ res3a_bias_0, float* __restrict__ res3a_weight_1, float* __restrict__ res3a_bias_1, float* __restrict__ res3a_weight_2, float* __restrict__ res3a_bias_2, float* __restrict__ res3b_weight_0, float* __restrict__ res3b_bias_0, float* __restrict__ res3b_weight_1, float* __restrict__ res3b_bias_1, float* __restrict__ res3b_weight_2, float* __restrict__ res3b_bias_2, float* __restrict__ res3c_weight_0, float* __restrict__ res3c_bias_0, float* __restrict__ res3c_weight_1, float* __restrict__ res3c_bias_1, float* __restrict__ res3c_weight_2, float* __restrict__ res3c_bias_2, float* __restrict__ res3d_weight_0, float* __restrict__ res3d_bias_0, float* __restrict__ res3d_weight_1, float* __restrict__ res3d_bias_1, float* __restrict__ res3d_weight_2, float* __restrict__ res3d_bias_2, float* __restrict__ res4a_weight_b, float* __restrict__ res4a_bias_b, float* __restrict__ res4a_weight_0, float* __restrict__ res4a_bias_0, float* __restrict__ res4a_weight_1, float* __restrict__ res4a_bias_1, float* __restrict__ res4a_weight_2, float* __restrict__ res4a_bias_2, float* __restrict__ res4b_weight_0, float* __restrict__ res4b_bias_0, float* __restrict__ res4b_weight_1, float* __restrict__ res4b_bias_1, float* __restrict__ res4b_weight_2, float* __restrict__ res4b_bias_2, float* __restrict__ res4c_weight_0, float* __restrict__ res4c_bias_0, float* __restrict__ res4c_weight_1, float* __restrict__ res4c_bias_1, float* __restrict__ res4c_weight_2, float* __restrict__ res4c_bias_2, float* __restrict__ res4d_weight_0, float* __restrict__ res4d_bias_0, float* __restrict__ res4d_weight_1, float* __restrict__ res4d_bias_1, float* __restrict__ res4d_weight_2, float* __restrict__ res4d_bias_2, float* __restrict__ res4e_weight_0, float* __restrict__ res4e_bias_0, float* __restrict__ res4e_weight_1, float* __restrict__ res4e_bias_1, float* __restrict__ res4e_weight_2, float* __restrict__ res4e_bias_2, float* __restrict__ res4f_weight_0, float* __restrict__ res4f_bias_0, float* __restrict__ res4f_weight_1, float* __restrict__ res4f_bias_1, float* __restrict__ res4f_weight_2, float* __restrict__ res4f_bias_2, float* __restrict__ res5a_weight_b, float* __restrict__ res5a_bias_b, float* __restrict__ res5a_weight_0, float* __restrict__ res5a_bias_0, float* __restrict__ res5a_weight_1, float* __restrict__ res5a_bias_1, float* __restrict__ res5a_weight_2, float* __restrict__ res5a_bias_2, float* __restrict__ res5b_weight_0, float* __restrict__ res5b_bias_0, float* __restrict__ res5b_weight_1, float* __restrict__ res5b_bias_1, float* __restrict__ res5b_weight_2, float* __restrict__ res5b_bias_2, float* __restrict__ res5c_weight_0, float* __restrict__ res5c_bias_0, float* __restrict__ res5c_weight_1, float* __restrict__ res5c_bias_1, float* __restrict__ res5c_weight_2, float* __restrict__ res5c_bias_2) noexcept{
  float* folded_const_46 = (float*)&__module_data[83456UL];
  float* folded_const_41 = (float*)&__module_data[77184UL];
  float* folded_const_31 = (float*)&__module_data[64640UL];
  float* folded_const_26 = (float*)&__module_data[58368UL];
  float* folded_const_21 = (float*)&__module_data[52096UL];
  float* folded_const_16 = (float*)&__module_data[45824UL];
  float* folded_const_103 = (float*)&__module_data[107456UL];
  float* folded_const_80 = (float*)&__module_data[106816UL];
  float* folded_const_78 = (float*)&__module_data[105728UL];
  float* folded_const_75 = (float*)&__module_data[105152UL];
  float* folded_const_73 = (float*)&__module_data[104064UL];
  float* folded_const_72 = (float*)&__module_data[103808UL];
  float* folded_const_70 = (float*)&__module_data[103488UL];
  float* folded_const_68 = (float*)&__module_data[102400UL];
  float* folded_const_67 = (float*)&__module_data[100352UL];
  float* folded_const_66 = (float*)&__module_data[99840UL];
  float* folded_const_64 = (float*)&__module_data[99264UL];
  float* folded_const_62 = (float*)&__module_data[97152UL];
  float* folded_const_61 = (float*)&__module_data[96640UL];
  float* folded_const_57 = (float*)&__module_data[93952UL];
  float* folded_const_54 = (float*)&__module_data[92864UL];
  float* folded_const_52 = (float*)&__module_data[90752UL];
  float* folded_const_51 = (float*)&__module_data[90240UL];
  float* folded_const_49 = (float*)&__module_data[89664UL];
  float* folded_const_47 = (float*)&__module_data[87552UL];
  float* folded_const_43 = (float*)&__module_data[81344UL];
  float* folded_const_40 = (float*)&__module_data[76160UL];
  float* folded_const_38 = (float*)&__module_data[75072UL];
  float* folded_const_35 = (float*)&__module_data[69888UL];
  float* folded_const_33 = (float*)&__module_data[68800UL];
  float* folded_const_30 = (float*)&__module_data[63616UL];
  float* folded_const_28 = (float*)&__module_data[62528UL];
  float* folded_const_23 = (float*)&__module_data[56256UL];
  float* folded_const_20 = (float*)&__module_data[51072UL];
  float* folded_const_18 = (float*)&__module_data[49984UL];
  float* folded_const_15 = (float*)&__module_data[37632UL];
  float* folded_const_14 = (float*)&__module_data[35584UL];
  float* folded_const_12 = (float*)&__module_data[33472UL];
  float* folded_const_10 = (float*)&__module_data[25216UL];
  float* folded_const_9 = (float*)&__module_data[23168UL];
  float* folded_const_7 = (float*)&__module_data[21056UL];
  float* folded_const_5 = (float*)&__module_data[12800UL];
  float* folded_const_4 = (float*)&__module_data[10752UL];
  float* folded_const_2 = (float*)&__module_data[8640UL];
  float* folded_const_0 = (float*)&__module_data[384UL];
  float* folded_const_154 = (float*)&__module_data[213184UL];
  float* folded_const_155 = (float*)&__module_data[213440UL];
  float* folded_const_152 = (float*)&__module_data[211904UL];
  float* folded_const_149 = (float*)&__module_data[210368UL];
  float* folded_const_146 = (float*)&__module_data[208832UL];
  float* folded_const_151 = (float*)&__module_data[211648UL];
  float* folded_const_148 = (float*)&__module_data[210112UL];
  float* folded_const_144 = (float*)&__module_data[206272UL];
  float* folded_const_153 = (float*)&__module_data[212928UL];
  float* folded_const_150 = (float*)&__module_data[211392UL];
  float* folded_const_147 = (float*)&__module_data[209856UL];
  float* folded_const_142 = (float*)&__module_data[203712UL];
  float* folded_const_139 = (float*)&__module_data[200640UL];
  float* folded_const_136 = (float*)&__module_data[197568UL];
  float* folded_const_133 = (float*)&__module_data[194496UL];
  float* folded_const_141 = (float*)&__module_data[203200UL];
  float* folded_const_138 = (float*)&__module_data[200128UL];
  float* folded_const_135 = (float*)&__module_data[197056UL];
  float* folded_const_145 = (float*)&__module_data[206784UL];
  float* folded_const_131 = (float*)&__module_data[189376UL];
  float* folded_const_143 = (float*)&__module_data[205760UL];
  float* folded_const_140 = (float*)&__module_data[202688UL];
  float* folded_const_137 = (float*)&__module_data[199616UL];
  float* folded_const_134 = (float*)&__module_data[196544UL];
  float* folded_const_129 = (float*)&__module_data[184256UL];
  float* folded_const_126 = (float*)&__module_data[178112UL];
  float* folded_const_123 = (float*)&__module_data[171968UL];
  float* folded_const_120 = (float*)&__module_data[165824UL];
  float* folded_const_117 = (float*)&__module_data[159680UL];
  float* folded_const_114 = (float*)&__module_data[153536UL];
  float* folded_const_128 = (float*)&__module_data[183232UL];
  float* folded_const_125 = (float*)&__module_data[177088UL];
  float* folded_const_122 = (float*)&__module_data[170944UL];
  float* folded_const_119 = (float*)&__module_data[164800UL];
  float* folded_const_116 = (float*)&__module_data[158656UL];
  float* folded_const_132 = (float*)&__module_data[190400UL];
  float* folded_const_112 = (float*)&__module_data[143296UL];
  float* folded_const_130 = (float*)&__module_data[188352UL];
  float* folded_const_127 = (float*)&__module_data[182208UL];
  float* folded_const_124 = (float*)&__module_data[176064UL];
  float* folded_const_121 = (float*)&__module_data[169920UL];
  float* folded_const_118 = (float*)&__module_data[163776UL];
  float* folded_const_115 = (float*)&__module_data[157632UL];
  int8_t* folded_const_156 = (int8_t*)&__uninitialized_data[0UL];
  float* folded_const_157 = (float*)&__uninitialized_data[589824UL];
  float* folded_const_86 = (float*)&__module_data[107092UL];
  float* folded_const_158 = (float*)&__uninitialized_data[593920UL];
  float* folded_const_159 = (float*)&__uninitialized_data[598016UL];
  float* folded_const_87 = (float*)&__module_data[107096UL];
  float* folded_const_160 = (float*)&__uninitialized_data[602112UL];
  float* folded_const_161 = (float*)&__uninitialized_data[606208UL];
  float* folded_const_88 = (float*)&__module_data[107100UL];
  float* folded_const_162 = (float*)&__uninitialized_data[610304UL];
  float* folded_const_163 = (float*)&__uninitialized_data[614400UL];
  float* folded_const_89 = (float*)&__module_data[107104UL];
  float* folded_const_164 = (float*)&__uninitialized_data[618496UL];
  float* folded_const_165 = (float*)&__uninitialized_data[622592UL];
  float* folded_const_90 = (float*)&__module_data[107108UL];
  float* folded_const_166 = (float*)&__uninitialized_data[626688UL];
  float* folded_const_36 = (float*)&__module_data[70912UL];
  float* folded_const_167 = (float*)&__uninitialized_data[630784UL];
  float* folded_const_91 = (float*)&__module_data[107112UL];
  float* folded_const_168 = (float*)&__uninitialized_data[634880UL];
  float* folded_const_169 = (float*)&__uninitialized_data[638976UL];
  float* folded_const_92 = (float*)&__module_data[107116UL];
  float* folded_const_170 = (float*)&__uninitialized_data[643072UL];
  float* folded_const_171 = (float*)&__uninitialized_data[647168UL];
  float* folded_const_93 = (float*)&__module_data[107120UL];
  float* folded_const_172 = (float*)&__uninitialized_data[649216UL];
  float* folded_const_173 = (float*)&__uninitialized_data[651264UL];
  float* folded_const_94 = (float*)&__module_data[107124UL];
  float* folded_const_174 = (float*)&__uninitialized_data[653312UL];
  float* folded_const_175 = (float*)&__uninitialized_data[655360UL];
  float* folded_const_95 = (float*)&__module_data[107128UL];
  float* folded_const_176 = (float*)&__uninitialized_data[657408UL];
  float* folded_const_177 = (float*)&__uninitialized_data[659456UL];
  float* folded_const_96 = (float*)&__module_data[107132UL];
  float* folded_const_178 = (float*)&__uninitialized_data[661504UL];
  float* folded_const_179 = (float*)&__uninitialized_data[663552UL];
  float* folded_const_97 = (float*)&__module_data[107136UL];
  float* folded_const_180 = (float*)&__uninitialized_data[665600UL];
  float* folded_const_181 = (float*)&__uninitialized_data[667648UL];
  float* folded_const_17 = (float*)&__module_data[49920UL];
  float* folded_const_182 = (float*)&__uninitialized_data[668672UL];
  float* folded_const_183 = (float*)&__uninitialized_data[669696UL];
  float* folded_const_19 = (float*)&__module_data[51008UL];
  float* folded_const_184 = (float*)&__uninitialized_data[670720UL];
  float* folded_const_185 = (float*)&__uninitialized_data[671744UL];
  float* folded_const_22 = (float*)&__module_data[56192UL];
  float* folded_const_186 = (float*)&__uninitialized_data[672768UL];
  float* folded_const_187 = (float*)&__uninitialized_data[673792UL];
  float* folded_const_24 = (float*)&__module_data[57280UL];
  float* folded_const_188 = (float*)&__uninitialized_data[674816UL];
  float* folded_const_25 = (float*)&__module_data[57344UL];
  float* folded_const_189 = (float*)&__uninitialized_data[675840UL];
  float* folded_const_27 = (float*)&__module_data[62464UL];
  float* folded_const_190 = (float*)&__uninitialized_data[676864UL];
  float* folded_const_191 = (float*)&__uninitialized_data[677888UL];
  float* folded_const_29 = (float*)&__module_data[63552UL];
  float* folded_const_192 = (float*)&__uninitialized_data[678912UL];
  float* folded_const_193 = (float*)&__uninitialized_data[679936UL];
  float* folded_const_32 = (float*)&__module_data[68736UL];
  float* folded_const_194 = (float*)&__uninitialized_data[680960UL];
  float* folded_const_195 = (float*)&__uninitialized_data[681984UL];
  float* folded_const_34 = (float*)&__module_data[69824UL];
  float* folded_const_196 = (float*)&__uninitialized_data[683008UL];
  float* folded_const_197 = (float*)&__uninitialized_data[684032UL];
  float* folded_const_37 = (float*)&__module_data[75008UL];
  float* folded_const_198 = (float*)&__uninitialized_data[685056UL];
  float* folded_const_199 = (float*)&__uninitialized_data[686080UL];
  float* folded_const_39 = (float*)&__module_data[76096UL];
  float* folded_const_200 = (float*)&__uninitialized_data[687104UL];
  float* folded_const_201 = (float*)&__uninitialized_data[688128UL];
  float* folded_const_42 = (float*)&__module_data[81280UL];
  float* folded_const_202 = (float*)&__uninitialized_data[689152UL];
  float* folded_const_203 = (float*)&__uninitialized_data[690176UL];
  float* folded_const_44 = (float*)&__module_data[82368UL];
  float* folded_const_204 = (float*)&__uninitialized_data[691200UL];
  float* folded_const_45 = (float*)&__module_data[82432UL];
  float* folded_const_205 = (float*)&__uninitialized_data[692224UL];
  float* folded_const_98 = (float*)&__module_data[107140UL];
  float* folded_const_206 = (float*)&__uninitialized_data[693248UL];
  float* folded_const_207 = (float*)&__uninitialized_data[694272UL];
  float* folded_const_99 = (float*)&__module_data[107144UL];
  float* folded_const_208 = (float*)&__uninitialized_data[695296UL];
  float* folded_const_209 = (float*)&__uninitialized_data[696320UL];
  float* folded_const_100 = (float*)&__module_data[107148UL];
  float* folded_const_210 = (float*)&__uninitialized_data[697344UL];
  float* folded_const_211 = (float*)&__uninitialized_data[698368UL];
  float* folded_const_101 = (float*)&__module_data[107152UL];
  float* folded_const_212 = (float*)&__uninitialized_data[699392UL];
  int8_t* folded_const_213 = (int8_t*)&__uninitialized_data[700416UL];
  int8_t* folded_const_214 = (int8_t*)&__uninitialized_data[962560UL];
  int8_t* folded_const_215 = (int8_t*)&__uninitialized_data[1224704UL];
  int8_t* folded_const_216 = (int8_t*)&__uninitialized_data[1486848UL];
  int8_t* folded_const_217 = (int8_t*)&__uninitialized_data[1748992UL];
  int8_t* folded_const_218 = (int8_t*)&__uninitialized_data[2011136UL];
  int8_t* folded_const_219 = (int8_t*)&__uninitialized_data[2273280UL];
  int8_t* folded_const_220 = (int8_t*)&__uninitialized_data[2535424UL];
  int8_t* folded_const_221 = (int8_t*)&__uninitialized_data[2797568UL];
  int8_t* folded_const_222 = (int8_t*)&__uninitialized_data[3059712UL];
  int8_t* folded_const_223 = (int8_t*)&__uninitialized_data[3321856UL];
  int8_t* folded_const_224 = (int8_t*)&__uninitialized_data[3584000UL];
  float* folded_const_225 = (float*)&__uninitialized_data[3588096UL];
  float* folded_const_48 = (float*)&__module_data[89600UL];
  float* folded_const_226 = (float*)&__uninitialized_data[3588608UL];
  float* folded_const_227 = (float*)&__uninitialized_data[3589120UL];
  float* folded_const_50 = (float*)&__module_data[90176UL];
  float* folded_const_228 = (float*)&__uninitialized_data[3589632UL];
  float* folded_const_229 = (float*)&__uninitialized_data[3590144UL];
  float* folded_const_53 = (float*)&__module_data[92800UL];
  float* folded_const_230 = (float*)&__uninitialized_data[3590656UL];
  float* folded_const_231 = (float*)&__uninitialized_data[3591168UL];
  float* folded_const_55 = (float*)&__module_data[93376UL];
  float* folded_const_232 = (float*)&__uninitialized_data[3591680UL];
  float* folded_const_56 = (float*)&__module_data[93440UL];
  float* folded_const_233 = (float*)&__uninitialized_data[3592192UL];
  float* folded_const_58 = (float*)&__module_data[96000UL];
  float* folded_const_234 = (float*)&__uninitialized_data[3592704UL];
  float* folded_const_59 = (float*)&__module_data[96064UL];
  float* folded_const_235 = (float*)&__uninitialized_data[3593216UL];
  float* folded_const_60 = (float*)&__module_data[96576UL];
  float* folded_const_236 = (float*)&__uninitialized_data[3593728UL];
  float* folded_const_237 = (float*)&__uninitialized_data[3594240UL];
  float* folded_const_63 = (float*)&__module_data[99200UL];
  float* folded_const_238 = (float*)&__uninitialized_data[3594752UL];
  float* folded_const_239 = (float*)&__uninitialized_data[3595264UL];
  float* folded_const_65 = (float*)&__module_data[99776UL];
  float* folded_const_240 = (float*)&__uninitialized_data[3595776UL];
  float* folded_const_241 = (float*)&__uninitialized_data[3596288UL];
  float* folded_const_69 = (float*)&__module_data[103424UL];
  float* folded_const_242 = (float*)&__uninitialized_data[3596544UL];
  float* folded_const_243 = (float*)&__uninitialized_data[3596800UL];
  float* folded_const_71 = (float*)&__module_data[103744UL];
  float* folded_const_244 = (float*)&__uninitialized_data[3597056UL];
  float* folded_const_245 = (float*)&__uninitialized_data[3597312UL];
  float* folded_const_74 = (float*)&__module_data[105088UL];
  float* folded_const_246 = (float*)&__uninitialized_data[3597568UL];
  float* folded_const_247 = (float*)&__uninitialized_data[3597824UL];
  float* folded_const_76 = (float*)&__module_data[105408UL];
  float* folded_const_248 = (float*)&__uninitialized_data[3598080UL];
  float* folded_const_77 = (float*)&__module_data[105472UL];
  float* folded_const_249 = (float*)&__uninitialized_data[3598336UL];
  float* folded_const_79 = (float*)&__module_data[106752UL];
  float* folded_const_250 = (float*)&__uninitialized_data[3598592UL];
  float* folded_const_251 = (float*)&__uninitialized_data[3598848UL];
  float* folded_const_81 = (float*)&__module_data[107072UL];
  float* folded_const_252 = (float*)&__uninitialized_data[3599104UL];
  float* folded_const_102 = (float*)&__module_data[107200UL];
  int8_t* folded_const_253 = (int8_t*)&__uninitialized_data[3599360UL];
  int8_t* folded_const_254 = (int8_t*)&__uninitialized_data[4189184UL];
  int8_t* folded_const_255 = (int8_t*)&__uninitialized_data[4779008UL];
  int8_t* folded_const_256 = (int8_t*)&__uninitialized_data[5368832UL];
  int8_t* folded_const_257 = (int8_t*)&__uninitialized_data[5958656UL];
  int8_t* folded_const_258 = (int8_t*)&__uninitialized_data[6548480UL];
  int8_t* folded_const_259 = (int8_t*)&__uninitialized_data[7072768UL];
  int8_t* folded_const_260 = (int8_t*)&__uninitialized_data[7220224UL];
  int8_t* folded_const_261 = (int8_t*)&__uninitialized_data[7367680UL];
  int8_t* folded_const_262 = (int8_t*)&__uninitialized_data[7515136UL];
  int8_t* folded_const_263 = (int8_t*)&__uninitialized_data[7662592UL];
  int8_t* folded_const_264 = (int8_t*)&__uninitialized_data[7793664UL];
  int8_t* folded_const_265 = (int8_t*)&__uninitialized_data[7924736UL];
  int8_t* folded_const_266 = (int8_t*)&__uninitialized_data[7990272UL];
  int8_t* folded_const_267 = (int8_t*)&__uninitialized_data[8055808UL];
  int8_t* folded_const_268 = (int8_t*)&__uninitialized_data[8121344UL];
  int8_t* folded_const_269 = (int8_t*)&__uninitialized_data[8186880UL];
  int8_t* folded_const_270 = (int8_t*)&__uninitialized_data[8252416UL];
  int8_t* folded_const_271 = (int8_t*)&__uninitialized_data[8317952UL];
  int8_t* folded_const_272 = (int8_t*)&__uninitialized_data[8383488UL];
  int8_t* folded_const_273 = (int8_t*)&__uninitialized_data[8420352UL];
  int8_t* folded_const_274 = (int8_t*)&__uninitialized_data[8457216UL];
  int8_t* folded_const_275 = (int8_t*)&__uninitialized_data[8494080UL];
  int8_t* folded_const_276 = (int8_t*)&__uninitialized_data[8526848UL];
  int8_t* folded_const_277 = (int8_t*)&__uninitialized_data[8543232UL];
  int8_t* folded_const_278 = (int8_t*)&__uninitialized_data[8559616UL];
  int8_t* folded_const_279 = (int8_t*)&__uninitialized_data[8576000UL];
  int8_t* folded_const_280 = (int8_t*)&__uninitialized_data[8592384UL];
  int8_t* folded_const_281 = (int8_t*)&__uninitialized_data[8608768UL];
  float* folded_const_282 = (float*)&__uninitialized_data[8625152UL];
  float* folded_const_13 = (float*)&__module_data[35520UL];
  float* folded_const_283 = (float*)&__uninitialized_data[8627200UL];
  int8_t* folded_const_284 = (int8_t*)&__uninitialized_data[8629248UL];
  float* folded_const_110 = (float*)&__module_data[133056UL];
  float* folded_const_107 = (float*)&__module_data[120768UL];
  float* folded_const_104 = (float*)&__module_data[108480UL];
  float* folded_const_109 = (float*)&__module_data[131008UL];
  float* folded_const_106 = (float*)&__module_data[118720UL];
  float* folded_const_113 = (float*)&__module_data[145344UL];
  int8_t* folded_const_285 = (int8_t*)&__uninitialized_data[9153536UL];
  float* folded_const_286 = (float*)&__uninitialized_data[11250688UL];
  float* folded_const_85 = (float*)&__module_data[107088UL];
  float* folded_const_287 = (float*)&__uninitialized_data[11258880UL];
  float* folded_const_111 = (float*)&__module_data[141248UL];
  int8_t* folded_const_288 = (int8_t*)&__uninitialized_data[11267072UL];
  float* folded_const_289 = (float*)&__uninitialized_data[13626368UL];
  float* folded_const_11 = (float*)&__module_data[33408UL];
  float* folded_const_290 = (float*)&__uninitialized_data[13628416UL];
  float* folded_const_291 = (float*)&__uninitialized_data[13630464UL];
  float* folded_const_84 = (float*)&__module_data[107084UL];
  float* folded_const_292 = (float*)&__uninitialized_data[13638656UL];
  int8_t* folded_const_293 = (int8_t*)&__uninitialized_data[13646848UL];
  float* folded_const_294 = (float*)&__uninitialized_data[14695424UL];
  float* folded_const_8 = (float*)&__module_data[23104UL];
  float* folded_const_295 = (float*)&__uninitialized_data[14697472UL];
  int8_t* folded_const_296 = (int8_t*)&__uninitialized_data[14699520UL];
  float* folded_const_108 = (float*)&__module_data[128960UL];
  float* folded_const_105 = (float*)&__module_data[116672UL];
  float* folded_const_297 = (float*)&__uninitialized_data[15748096UL];
  float* folded_const_6 = (float*)&__module_data[20992UL];
  float* folded_const_298 = (float*)&__uninitialized_data[15750144UL];
  int8_t* folded_const_299 = (int8_t*)&__uninitialized_data[15752192UL];
  float* folded_const_300 = (float*)&__uninitialized_data[18111488UL];
  float* folded_const_83 = (float*)&__module_data[107080UL];
  float* folded_const_301 = (float*)&__uninitialized_data[18119680UL];
  int8_t* folded_const_302 = (int8_t*)&__uninitialized_data[18127872UL];
  float* folded_const_303 = (float*)&__uninitialized_data[19176448UL];
  float* folded_const_3 = (float*)&__module_data[10688UL];
  float* folded_const_304 = (float*)&__uninitialized_data[19178496UL];
  int8_t* folded_const_305 = (int8_t*)&__uninitialized_data[19180544UL];
  float* folded_const_306 = (float*)&__uninitialized_data[20229120UL];
  float* folded_const_1 = (float*)&__module_data[8576UL];
  float* folded_const_307 = (float*)&__uninitialized_data[20231168UL];
  int8_t* folded_const_308 = (int8_t*)&__uninitialized_data[20233216UL];
  float* folded_const_309 = (float*)&__uninitialized_data[22592512UL];
  float* folded_const_82 = (float*)&__module_data[107076UL];
  float* folded_const_310 = (float*)&__uninitialized_data[22600704UL];
  int8_t* folded_const_311 = (int8_t*)&__uninitialized_data[22608896UL];
  bool& is_init = *(bool*)(__module_data + 0);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 18546688UL);
  // [f32 [1, 32, 1, 1, 32] @ ABCD32b]
  float* buffer_261 = (float*)&__rescheduled_0[0UL];
  reorder__481(buffer_261, folded_const_46);
  // [f32 [1, 8, 1, 1, 128] @ ABCD128b]
  float* buffer_262 = (float*)&__rescheduled_0[4096UL];
  reorder__488(buffer_262, folded_const_41);
  // [f32 [1, 8, 1, 1, 128] @ ABCD128b]
  float* buffer_263 = (float*)&__rescheduled_0[8192UL];
  reorder__504(buffer_263, folded_const_31);
  // [f32 [1, 8, 1, 1, 128] @ ABCD128b]
  float* buffer_264 = (float*)&__rescheduled_0[12288UL];
  reorder__513(buffer_264, folded_const_26);
  // [f32 [1, 8, 1, 1, 128] @ ABCD128b]
  float* buffer_265 = (float*)&__rescheduled_0[16384UL];
  reorder__520(buffer_265, folded_const_21);
  // [f32 [1, 4, 1, 1, 256] @ ABCD256b]
  float* buffer_266 = (float*)&__rescheduled_0[20480UL];
  reorder__529(buffer_266, folded_const_16);
  // [f32 [1, 16, 1, 1, 16] @ ABCD16b]
  float* buffer_267 = (float*)&__rescheduled_0[24576UL];
  reorder__420(buffer_267, folded_const_103);
  // [f32 [1, 2, 1, 1, 32] @ ABCD32b]
  float* buffer_268 = (float*)&__rescheduled_0[25600UL];
  reorder__424(buffer_268, folded_const_80);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_269 = (float*)&__rescheduled_0[25856UL];
  reorder__427(buffer_269, folded_const_78);
  // [f32 [1, 4, 1, 1, 16] @ ABCD16b]
  float* buffer_270 = (float*)&__rescheduled_0[26880UL];
  reorder__431(buffer_270, folded_const_75);
  // [f32 [1, 8, 1, 1, 32] @ ABCD32b]
  float* buffer_271 = (float*)&__rescheduled_0[27136UL];
  reorder__434(buffer_271, folded_const_73);
  // [f32 [1, 2, 1, 1, 32] @ ABCD32b]
  float* buffer_272 = (float*)&__rescheduled_0[28160UL];
  reorder__437(buffer_272, folded_const_72);
  // [f32 [1, 2, 1, 1, 32] @ ABCD32b]
  float* buffer_273 = (float*)&__rescheduled_0[28416UL];
  reorder__440(buffer_273, folded_const_70);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_274 = (float*)&__rescheduled_0[28672UL];
  reorder__443(buffer_274, folded_const_68);
  // [f32 [1, 16, 1, 1, 32] @ ABCD32b]
  float* buffer_275 = (float*)&__rescheduled_0[29696UL];
  reorder__446(buffer_275, folded_const_67);
  // [f32 [1, 2, 1, 1, 64] @ ABCD64b]
  float* buffer_276 = (float*)&__rescheduled_0[31744UL];
  reorder__449(buffer_276, folded_const_66);
  // [f32 [1, 4, 1, 1, 32] @ ABCD32b]
  float* buffer_277 = (float*)&__rescheduled_0[32256UL];
  reorder__452(buffer_277, folded_const_64);
  // [f32 [1, 16, 1, 1, 32] @ ABCD32b]
  float* buffer_278 = (float*)&__rescheduled_0[32768UL];
  reorder__455(buffer_278, folded_const_62);
  // [f32 [1, 4, 1, 1, 32] @ ABCD32b]
  float* buffer_279 = (float*)&__rescheduled_0[34816UL];
  reorder__458(buffer_279, folded_const_61);
  // [f32 [1, 4, 1, 1, 128] @ ABCD128b]
  float* buffer_280 = (float*)&__rescheduled_0[35328UL];
  reorder__462(buffer_280, folded_const_57);
  // [f32 [1, 4, 1, 1, 32] @ ABCD32b]
  float* buffer_281 = (float*)&__rescheduled_0[37376UL];
  reorder__466(buffer_281, folded_const_54);
  // [f32 [1, 8, 1, 1, 64] @ ABCD64b]
  float* buffer_282 = (float*)&__rescheduled_0[37888UL];
  reorder__469(buffer_282, folded_const_52);
  // [f32 [1, 2, 1, 1, 64] @ ABCD64b]
  float* buffer_283 = (float*)&__rescheduled_0[39936UL];
  reorder__472(buffer_283, folded_const_51);
  // [f32 [1, 4, 1, 1, 32] @ ABCD32b]
  float* buffer_284 = (float*)&__rescheduled_0[40448UL];
  reorder__475(buffer_284, folded_const_49);
  // [f32 [1, 4, 1, 1, 128] @ ABCD128b]
  float* buffer_285 = (float*)&__rescheduled_0[40960UL];
  reorder__478(buffer_285, folded_const_47);
  // [f32 [1, 2, 1, 1, 128] @ ABCD128b]
  float* buffer_286 = (float*)&__rescheduled_0[43008UL];
  reorder__485(buffer_286, folded_const_43);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_287 = (float*)&__rescheduled_0[44032UL];
  reorder__491(buffer_287, folded_const_40);
  // [f32 [1, 16, 1, 1, 16] @ ABCD16b]
  float* buffer_288 = (float*)&__rescheduled_0[45056UL];
  reorder__494(buffer_288, folded_const_38);
  // [f32 [1, 8, 1, 1, 32] @ ABCD32b]
  float* buffer_289 = (float*)&__rescheduled_0[46080UL];
  reorder__498(buffer_289, folded_const_35);
  // [f32 [1, 8, 1, 1, 32] @ ABCD32b]
  float* buffer_290 = (float*)&__rescheduled_0[47104UL];
  reorder__501(buffer_290, folded_const_33);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_291 = (float*)&__rescheduled_0[48128UL];
  reorder__507(buffer_291, folded_const_30);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_292 = (float*)&__rescheduled_0[49152UL];
  reorder__510(buffer_292, folded_const_28);
  // [f32 [1, 8, 1, 1, 32] @ ABCD32b]
  float* buffer_293 = (float*)&__rescheduled_0[50176UL];
  reorder__517(buffer_293, folded_const_23);
  // [f32 [1, 8, 1, 1, 32] @ ABCD32b]
  float* buffer_294 = (float*)&__rescheduled_0[51200UL];
  reorder__523(buffer_294, folded_const_20);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_295 = (float*)&__rescheduled_0[52224UL];
  reorder__526(buffer_295, folded_const_18);
  // [f32 [1, 128, 1, 1, 16] @ ABCD16b]
  float* buffer_296 = (float*)&__rescheduled_0[53248UL];
  reorder__532(buffer_296, folded_const_15);
  // [f32 [1, 8, 1, 1, 64] @ ABCD64b]
  float* buffer_297 = (float*)&__rescheduled_0[61440UL];
  reorder__535(buffer_297, folded_const_14);
  // [f32 [1, 4, 1, 1, 128] @ ABCD128b]
  float* buffer_298 = (float*)&__rescheduled_0[63488UL];
  reorder__538(buffer_298, folded_const_12);
  // [f32 [1, 32, 1, 1, 64] @ ABCD64b]
  float* buffer_299 = (float*)&__rescheduled_0[65536UL];
  reorder__541(buffer_299, folded_const_10);
  // [f32 [1, 4, 1, 1, 128] @ ABCD128b]
  float* buffer_300 = (float*)&__rescheduled_0[73728UL];
  reorder__544(buffer_300, folded_const_9);
  // [f32 [1, 32, 1, 1, 16] @ ABCD16b]
  float* buffer_301 = (float*)&__rescheduled_0[75776UL];
  reorder__547(buffer_301, folded_const_7);
  // [f32 [1, 32, 1, 1, 64] @ ABCD64b]
  float* buffer_302 = (float*)&__rescheduled_0[77824UL];
  reorder__550(buffer_302, folded_const_5);
  // [f32 [1, 16, 1, 1, 32] @ ABCD32b]
  float* buffer_303 = (float*)&__rescheduled_0[86016UL];
  reorder__553(buffer_303, folded_const_4);
  // [f32 [1, 4, 1, 1, 128] @ ABCD128b]
  float* buffer_304 = (float*)&__rescheduled_0[88064UL];
  reorder__556(buffer_304, folded_const_2);
  // [f32 [1, 4, 1, 1, 512] @ ABCD512b]
  float* buffer_305 = (float*)&__rescheduled_0[90112UL];
  reorder__559(buffer_305, folded_const_0);
  // [f32 [1, 2, 1, 1, 32] @ ABCD32b]
  float* buffer_306 = (float*)&__rescheduled_0[98304UL];
  reorder__425(buffer_306, &res2a_bias_1[0]);
  // [f32 [1, 4, 1, 1, 16] @ ABCD16b]
  float* buffer_307 = (float*)&__rescheduled_0[98560UL];
  reorder__432(buffer_307, &res2b_bias_1[0]);
  // [f32 [1, 2, 1, 1, 32] @ ABCD32b]
  float* buffer_308 = (float*)&__rescheduled_0[98816UL];
  reorder__438(buffer_308, &res2c_bias_0[0]);
  // [f32 [1, 2, 1, 1, 32] @ ABCD32b]
  float* buffer_309 = (float*)&__rescheduled_0[99072UL];
  reorder__441(buffer_309, &res2c_bias_1[0]);
  // [f32 [1, 2, 1, 1, 64] @ ABCD64b]
  float* buffer_310 = (float*)&__rescheduled_0[99328UL];
  reorder__450(buffer_310, &res3a_bias_0[0]);
  // [f32 [1, 4, 1, 1, 32] @ ABCD32b]
  float* buffer_311 = (float*)&__rescheduled_0[99840UL];
  reorder__453(buffer_311, &res3a_bias_1[0]);
  // [f32 [1, 4, 1, 1, 32] @ ABCD32b]
  float* buffer_312 = (float*)&__rescheduled_0[100352UL];
  reorder__459(buffer_312, &res3b_bias_0[0]);
  // [f32 [1, 4, 1, 1, 32] @ ABCD32b]
  float* buffer_313 = (float*)&__rescheduled_0[100864UL];
  reorder__467(buffer_313, &res3c_bias_1[0]);
  // [f32 [1, 2, 1, 1, 64] @ ABCD64b]
  float* buffer_314 = (float*)&__rescheduled_0[101376UL];
  reorder__473(buffer_314, &res3d_bias_0[0]);
  // [f32 [1, 4, 1, 1, 32] @ ABCD32b]
  float* buffer_315 = (float*)&__rescheduled_0[101888UL];
  reorder__476(buffer_315, &res3d_bias_1[0]);
  // [f32 [1, 16, 1, 1, 16] @ ABCD16b]
  float* buffer_316 = (float*)&__rescheduled_0[102400UL];
  reorder__421(buffer_316, &res2a_bias_b[0]);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_317 = (float*)&__rescheduled_0[103424UL];
  reorder__428(buffer_317, &res2a_bias_2[0]);
  // [f32 [1, 8, 1, 1, 32] @ ABCD32b]
  float* buffer_318 = (float*)&__rescheduled_0[104448UL];
  reorder__435(buffer_318, &res2b_bias_2[0]);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_319 = (float*)&__rescheduled_0[105472UL];
  reorder__444(buffer_319, &res2c_bias_2[0]);
  // [f32 [1, 2, 1, 1, 128] @ ABCD128b]
  float* buffer_320 = (float*)&__rescheduled_0[106496UL];
  reorder__486(buffer_320, &res4a_bias_1[0]);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_321 = (float*)&__rescheduled_0[107520UL];
  reorder__492(buffer_321, &res4b_bias_0[0]);
  // [f32 [1, 16, 1, 1, 16] @ ABCD16b]
  float* buffer_322 = (float*)&__rescheduled_0[108544UL];
  reorder__495(buffer_322, &res4b_bias_1[0]);
  // [f32 [1, 8, 1, 1, 32] @ ABCD32b]
  float* buffer_323 = (float*)&__rescheduled_0[109568UL];
  reorder__499(buffer_323, &res4c_bias_0[0]);
  // [f32 [1, 8, 1, 1, 32] @ ABCD32b]
  float* buffer_324 = (float*)&__rescheduled_0[110592UL];
  reorder__502(buffer_324, &res4c_bias_1[0]);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_325 = (float*)&__rescheduled_0[111616UL];
  reorder__508(buffer_325, &res4d_bias_0[0]);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_326 = (float*)&__rescheduled_0[112640UL];
  reorder__511(buffer_326, &res4d_bias_1[0]);
  // [f32 [1, 8, 1, 1, 32] @ ABCD32b]
  float* buffer_327 = (float*)&__rescheduled_0[113664UL];
  reorder__518(buffer_327, &res4e_bias_1[0]);
  // [f32 [1, 8, 1, 1, 32] @ ABCD32b]
  float* buffer_328 = (float*)&__rescheduled_0[114688UL];
  reorder__524(buffer_328, &res4f_bias_0[0]);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_329 = (float*)&__rescheduled_0[115712UL];
  reorder__527(buffer_329, &res4f_bias_1[0]);
  // [f32 [1, 16, 1, 1, 32] @ ABCD32b]
  float* buffer_330 = (float*)&__rescheduled_0[116736UL];
  reorder__447(buffer_330, &res3a_bias_b[0]);
  // [f32 [1, 16, 1, 1, 32] @ ABCD32b]
  float* buffer_331 = (float*)&__rescheduled_0[118784UL];
  reorder__456(buffer_331, &res3a_bias_2[0]);
  // [f32 [1, 4, 1, 1, 128] @ ABCD128b]
  float* buffer_332 = (float*)&__rescheduled_0[120832UL];
  reorder__463(buffer_332, &res3b_bias_2[0]);
  // [f32 [1, 8, 1, 1, 64] @ ABCD64b]
  float* buffer_333 = (float*)&__rescheduled_0[122880UL];
  reorder__470(buffer_333, &res3c_bias_2[0]);
  // [f32 [1, 4, 1, 1, 128] @ ABCD128b]
  float* buffer_334 = (float*)&__rescheduled_0[124928UL];
  reorder__479(buffer_334, &res3d_bias_2[0]);
  // [f32 [1, 8, 1, 1, 64] @ ABCD64b]
  float* buffer_335 = (float*)&__rescheduled_0[126976UL];
  reorder__536(buffer_335, &res5a_bias_0[0]);
  // [f32 [1, 4, 1, 1, 128] @ ABCD128b]
  float* buffer_336 = (float*)&__rescheduled_0[129024UL];
  reorder__539(buffer_336, &res5a_bias_1[0]);
  // [f32 [1, 4, 1, 1, 128] @ ABCD128b]
  float* buffer_337 = (float*)&__rescheduled_0[131072UL];
  reorder__545(buffer_337, &res5b_bias_0[0]);
  // [f32 [1, 32, 1, 1, 16] @ ABCD16b]
  float* buffer_338 = (float*)&__rescheduled_0[133120UL];
  reorder__548(buffer_338, &res5b_bias_1[0]);
  // [f32 [1, 16, 1, 1, 32] @ ABCD32b]
  float* buffer_339 = (float*)&__rescheduled_0[135168UL];
  reorder__554(buffer_339, &res5c_bias_0[0]);
  // [f32 [1, 4, 1, 1, 128] @ ABCD128b]
  float* buffer_340 = (float*)&__rescheduled_0[137216UL];
  reorder__557(buffer_340, &res5c_bias_1[0]);
  // [f32 [1, 32, 1, 1, 32] @ ABCD32b]
  float* buffer_341 = (float*)&__rescheduled_0[139264UL];
  reorder__482(buffer_341, &res4a_bias_b[0]);
  // [f32 [1, 8, 1, 1, 128] @ ABCD128b]
  float* buffer_342 = (float*)&__rescheduled_0[143360UL];
  reorder__489(buffer_342, &res4a_bias_2[0]);
  // [f32 [1, 8, 1, 1, 128] @ ABCD128b]
  float* buffer_343 = (float*)&__rescheduled_0[147456UL];
  reorder__505(buffer_343, &res4c_bias_2[0]);
  // [f32 [1, 8, 1, 1, 128] @ ABCD128b]
  float* buffer_344 = (float*)&__rescheduled_0[151552UL];
  reorder__514(buffer_344, &res4d_bias_2[0]);
  // [f32 [1, 8, 1, 1, 128] @ ABCD128b]
  float* buffer_345 = (float*)&__rescheduled_0[155648UL];
  reorder__521(buffer_345, &res4e_bias_2[0]);
  // [f32 [1, 4, 1, 1, 256] @ ABCD256b]
  float* buffer_346 = (float*)&__rescheduled_0[159744UL];
  reorder__530(buffer_346, &res4f_bias_2[0]);
  // [f32 [1, 128, 1, 1, 16] @ ABCD16b]
  float* buffer_347 = (float*)&__rescheduled_0[163840UL];
  reorder__533(buffer_347, &res5a_bias_b[0]);
  // [f32 [1, 32, 1, 1, 64] @ ABCD64b]
  float* buffer_348 = (float*)&__rescheduled_0[172032UL];
  reorder__542(buffer_348, &res5a_bias_2[0]);
  // [f32 [1, 32, 1, 1, 64] @ ABCD64b]
  float* buffer_349 = (float*)&__rescheduled_0[180224UL];
  reorder__551(buffer_349, &res5b_bias_2[0]);
  // [f32 [1, 4, 1, 1, 512] @ ABCD512b]
  float* buffer_350 = (float*)&__rescheduled_0[188416UL];
  reorder__560(buffer_350, &res5c_bias_2[0]);
  // [f32 [64, 64, 1, 1] @ ABCD]
  float* buffer_351 = (float*)&__rescheduled_0[196608UL];
  mul__111(buffer_351, res2a_weight_0, folded_const_154);
  // [s8 [64, 64, 1, 1] @ ABCD]
  int8_t* buffer_352 = (int8_t*)&__rescheduled_0[2555904UL];
  cast__112(buffer_352, buffer_351);
  // [f32 [256, 64, 1, 1] @ ABCD]
  float* buffer_353 = (float*)&__rescheduled_0[196608UL];
  mul__108(buffer_353, res2a_weight_b, folded_const_155);
  // [s8 [256, 64, 1, 1] @ ABCD]
  int8_t* buffer_354 = (int8_t*)&__rescheduled_0[2560000UL];
  cast__109(buffer_354, buffer_353);
  // [f32 [256, 64, 1, 1] @ ABCD]
  float* buffer_355 = (float*)&__rescheduled_0[196608UL];
  mul__117(buffer_355, res2a_weight_2, folded_const_152);
  // [s8 [256, 64, 1, 1] @ ABCD]
  int8_t* buffer_356 = (int8_t*)&__rescheduled_0[2576384UL];
  cast__118(buffer_356, buffer_355);
  // [f32 [256, 64, 1, 1] @ ABCD]
  float* buffer_357 = (float*)&__rescheduled_0[196608UL];
  mul__126(buffer_357, res2b_weight_2, folded_const_149);
  // [s8 [256, 64, 1, 1] @ ABCD]
  int8_t* buffer_358 = (int8_t*)&__rescheduled_0[2592768UL];
  cast__127(buffer_358, buffer_357);
  // [f32 [256, 64, 1, 1] @ ABCD]
  float* buffer_359 = (float*)&__rescheduled_0[196608UL];
  mul__135(buffer_359, res2c_weight_2, folded_const_146);
  // [s8 [256, 64, 1, 1] @ ABCD]
  int8_t* buffer_360 = (int8_t*)&__rescheduled_0[2609152UL];
  cast__136(buffer_360, buffer_359);
  // [f32 [64, 256, 1, 1] @ ABCD]
  float* buffer_361 = (float*)&__rescheduled_0[196608UL];
  mul__120(buffer_361, res2b_weight_0, folded_const_151);
  // [s8 [64, 256, 1, 1] @ ABCD]
  int8_t* buffer_362 = (int8_t*)&__rescheduled_0[2625536UL];
  cast__121(buffer_362, buffer_361);
  // [f32 [64, 256, 1, 1] @ ABCD]
  float* buffer_363 = (float*)&__rescheduled_0[196608UL];
  mul__129(buffer_363, res2c_weight_0, folded_const_148);
  // [s8 [64, 256, 1, 1] @ ABCD]
  int8_t* buffer_364 = (int8_t*)&__rescheduled_0[2641920UL];
  cast__130(buffer_364, buffer_363);
  // [f32 [128, 256, 1, 1] @ ABCD]
  float* buffer_365 = (float*)&__rescheduled_0[196608UL];
  mul__141(buffer_365, res3a_weight_0, folded_const_144);
  // [s8 [128, 256, 1, 1] @ ABCD]
  int8_t* buffer_366 = (int8_t*)&__rescheduled_0[2658304UL];
  cast__142(buffer_366, buffer_365);
  // [f32 [64, 64, 3, 3] @ ABCD]
  float* buffer_367 = (float*)&__rescheduled_0[196608UL];
  mul__114(buffer_367, res2a_weight_1, folded_const_153);
  // [s8 [64, 64, 3, 3] @ ABCD]
  int8_t* buffer_368 = (int8_t*)&__rescheduled_0[2691072UL];
  cast__115(buffer_368, buffer_367);
  // [f32 [64, 64, 3, 3] @ ABCD]
  float* buffer_369 = (float*)&__rescheduled_0[196608UL];
  mul__123(buffer_369, res2b_weight_1, folded_const_150);
  // [s8 [64, 64, 3, 3] @ ABCD]
  int8_t* buffer_370 = (int8_t*)&__rescheduled_0[2727936UL];
  cast__124(buffer_370, buffer_369);
  // [f32 [64, 64, 3, 3] @ ABCD]
  float* buffer_371 = (float*)&__rescheduled_0[196608UL];
  mul__132(buffer_371, res2c_weight_1, folded_const_147);
  // [s8 [64, 64, 3, 3] @ ABCD]
  int8_t* buffer_372 = (int8_t*)&__rescheduled_0[2764800UL];
  cast__133(buffer_372, buffer_371);
  // [f32 [512, 128, 1, 1] @ ABCD]
  float* buffer_373 = (float*)&__rescheduled_0[196608UL];
  mul__147(buffer_373, res3a_weight_2, folded_const_142);
  // [s8 [512, 128, 1, 1] @ ABCD]
  int8_t* buffer_374 = (int8_t*)&__rescheduled_0[2801664UL];
  cast__148(buffer_374, buffer_373);
  // [f32 [512, 128, 1, 1] @ ABCD]
  float* buffer_375 = (float*)&__rescheduled_0[196608UL];
  mul__156(buffer_375, res3b_weight_2, folded_const_139);
  // [s8 [512, 128, 1, 1] @ ABCD]
  int8_t* buffer_376 = (int8_t*)&__rescheduled_0[2867200UL];
  cast__157(buffer_376, buffer_375);
  // [f32 [512, 128, 1, 1] @ ABCD]
  float* buffer_377 = (float*)&__rescheduled_0[196608UL];
  mul__165(buffer_377, res3c_weight_2, folded_const_136);
  // [s8 [512, 128, 1, 1] @ ABCD]
  int8_t* buffer_378 = (int8_t*)&__rescheduled_0[2932736UL];
  cast__166(buffer_378, buffer_377);
  // [f32 [512, 128, 1, 1] @ ABCD]
  float* buffer_379 = (float*)&__rescheduled_0[196608UL];
  mul__174(buffer_379, res3d_weight_2, folded_const_133);
  // [s8 [512, 128, 1, 1] @ ABCD]
  int8_t* buffer_380 = (int8_t*)&__rescheduled_0[2998272UL];
  cast__175(buffer_380, buffer_379);
  // [f32 [128, 512, 1, 1] @ ABCD]
  float* buffer_381 = (float*)&__rescheduled_0[196608UL];
  mul__150(buffer_381, res3b_weight_0, folded_const_141);
  // [s8 [128, 512, 1, 1] @ ABCD]
  int8_t* buffer_382 = (int8_t*)&__rescheduled_0[3063808UL];
  cast__151(buffer_382, buffer_381);
  // [f32 [128, 512, 1, 1] @ ABCD]
  float* buffer_383 = (float*)&__rescheduled_0[196608UL];
  mul__159(buffer_383, res3c_weight_0, folded_const_138);
  // [s8 [128, 512, 1, 1] @ ABCD]
  int8_t* buffer_384 = (int8_t*)&__rescheduled_0[3129344UL];
  cast__160(buffer_384, buffer_383);
  // [f32 [128, 512, 1, 1] @ ABCD]
  float* buffer_385 = (float*)&__rescheduled_0[196608UL];
  mul__168(buffer_385, res3d_weight_0, folded_const_135);
  // [s8 [128, 512, 1, 1] @ ABCD]
  int8_t* buffer_386 = (int8_t*)&__rescheduled_0[3194880UL];
  cast__169(buffer_386, buffer_385);
  // [f32 [512, 256, 1, 1] @ ABCD]
  float* buffer_387 = (float*)&__rescheduled_0[196608UL];
  mul__138(buffer_387, res3a_weight_b, folded_const_145);
  // [s8 [512, 256, 1, 1] @ ABCD]
  int8_t* buffer_388 = (int8_t*)&__rescheduled_0[3260416UL];
  cast__139(buffer_388, buffer_387);
  // [f32 [256, 512, 1, 1] @ ABCD]
  float* buffer_389 = (float*)&__rescheduled_0[196608UL];
  mul__180(buffer_389, res4a_weight_0, folded_const_131);
  // [s8 [256, 512, 1, 1] @ ABCD]
  int8_t* buffer_390 = (int8_t*)&__rescheduled_0[3391488UL];
  cast__181(buffer_390, buffer_389);
  // [f32 [128, 128, 3, 3] @ ABCD]
  float* buffer_391 = (float*)&__rescheduled_0[196608UL];
  mul__144(buffer_391, res3a_weight_1, folded_const_143);
  // [s8 [128, 128, 3, 3] @ ABCD]
  int8_t* buffer_392 = (int8_t*)&__rescheduled_0[3522560UL];
  cast__145(buffer_392, buffer_391);
  // [f32 [128, 128, 3, 3] @ ABCD]
  float* buffer_393 = (float*)&__rescheduled_0[196608UL];
  mul__153(buffer_393, res3b_weight_1, folded_const_140);
  // [s8 [128, 128, 3, 3] @ ABCD]
  int8_t* buffer_394 = (int8_t*)&__rescheduled_0[3670016UL];
  cast__154(buffer_394, buffer_393);
  // [f32 [128, 128, 3, 3] @ ABCD]
  float* buffer_395 = (float*)&__rescheduled_0[196608UL];
  mul__162(buffer_395, res3c_weight_1, folded_const_137);
  // [s8 [128, 128, 3, 3] @ ABCD]
  int8_t* buffer_396 = (int8_t*)&__rescheduled_0[3817472UL];
  cast__163(buffer_396, buffer_395);
  // [f32 [128, 128, 3, 3] @ ABCD]
  float* buffer_397 = (float*)&__rescheduled_0[196608UL];
  mul__171(buffer_397, res3d_weight_1, folded_const_134);
  // [s8 [128, 128, 3, 3] @ ABCD]
  int8_t* buffer_398 = (int8_t*)&__rescheduled_0[3964928UL];
  cast__172(buffer_398, buffer_397);
  // [f32 [1024, 256, 1, 1] @ ABCD]
  float* buffer_399 = (float*)&__rescheduled_0[196608UL];
  mul__186(buffer_399, res4a_weight_2, folded_const_129);
  // [s8 [1024, 256, 1, 1] @ ABCD]
  int8_t* buffer_400 = (int8_t*)&__rescheduled_0[4112384UL];
  cast__187(buffer_400, buffer_399);
  // [f32 [1024, 256, 1, 1] @ ABCD]
  float* buffer_401 = (float*)&__rescheduled_0[196608UL];
  mul__195(buffer_401, res4b_weight_2, folded_const_126);
  // [s8 [1024, 256, 1, 1] @ ABCD]
  int8_t* buffer_402 = (int8_t*)&__rescheduled_0[4374528UL];
  cast__196(buffer_402, buffer_401);
  // [f32 [1024, 256, 1, 1] @ ABCD]
  float* buffer_403 = (float*)&__rescheduled_0[196608UL];
  mul__204(buffer_403, res4c_weight_2, folded_const_123);
  // [s8 [1024, 256, 1, 1] @ ABCD]
  int8_t* buffer_404 = (int8_t*)&__rescheduled_0[4636672UL];
  cast__205(buffer_404, buffer_403);
  // [f32 [1024, 256, 1, 1] @ ABCD]
  float* buffer_405 = (float*)&__rescheduled_0[196608UL];
  mul__213(buffer_405, res4d_weight_2, folded_const_120);
  // [s8 [1024, 256, 1, 1] @ ABCD]
  int8_t* buffer_406 = (int8_t*)&__rescheduled_0[4898816UL];
  cast__214(buffer_406, buffer_405);
  // [f32 [1024, 256, 1, 1] @ ABCD]
  float* buffer_407 = (float*)&__rescheduled_0[196608UL];
  mul__222(buffer_407, res4e_weight_2, folded_const_117);
  // [s8 [1024, 256, 1, 1] @ ABCD]
  int8_t* buffer_408 = (int8_t*)&__rescheduled_0[5160960UL];
  cast__223(buffer_408, buffer_407);
  // [f32 [1024, 256, 1, 1] @ ABCD]
  float* buffer_409 = (float*)&__rescheduled_0[196608UL];
  mul__231(buffer_409, res4f_weight_2, folded_const_114);
  // [s8 [1024, 256, 1, 1] @ ABCD]
  int8_t* buffer_410 = (int8_t*)&__rescheduled_0[5423104UL];
  cast__232(buffer_410, buffer_409);
  // [f32 [256, 1024, 1, 1] @ ABCD]
  float* buffer_411 = (float*)&__rescheduled_0[196608UL];
  mul__189(buffer_411, res4b_weight_0, folded_const_128);
  // [s8 [256, 1024, 1, 1] @ ABCD]
  int8_t* buffer_412 = (int8_t*)&__rescheduled_0[5685248UL];
  cast__190(buffer_412, buffer_411);
  // [f32 [256, 1024, 1, 1] @ ABCD]
  float* buffer_413 = (float*)&__rescheduled_0[196608UL];
  mul__198(buffer_413, res4c_weight_0, folded_const_125);
  // [s8 [256, 1024, 1, 1] @ ABCD]
  int8_t* buffer_414 = (int8_t*)&__rescheduled_0[5947392UL];
  cast__199(buffer_414, buffer_413);
  // [f32 [256, 1024, 1, 1] @ ABCD]
  float* buffer_415 = (float*)&__rescheduled_0[196608UL];
  mul__207(buffer_415, res4d_weight_0, folded_const_122);
  // [s8 [256, 1024, 1, 1] @ ABCD]
  int8_t* buffer_416 = (int8_t*)&__rescheduled_0[6209536UL];
  cast__208(buffer_416, buffer_415);
  // [f32 [256, 1024, 1, 1] @ ABCD]
  float* buffer_417 = (float*)&__rescheduled_0[196608UL];
  mul__216(buffer_417, res4e_weight_0, folded_const_119);
  // [s8 [256, 1024, 1, 1] @ ABCD]
  int8_t* buffer_418 = (int8_t*)&__rescheduled_0[6471680UL];
  cast__217(buffer_418, buffer_417);
  // [f32 [256, 1024, 1, 1] @ ABCD]
  float* buffer_419 = (float*)&__rescheduled_0[196608UL];
  mul__225(buffer_419, res4f_weight_0, folded_const_116);
  // [s8 [256, 1024, 1, 1] @ ABCD]
  int8_t* buffer_420 = (int8_t*)&__rescheduled_0[6733824UL];
  cast__226(buffer_420, buffer_419);
  // [f32 [1024, 512, 1, 1] @ ABCD]
  float* buffer_421 = (float*)&__rescheduled_0[196608UL];
  mul__177(buffer_421, res4a_weight_b, folded_const_132);
  // [s8 [1024, 512, 1, 1] @ ABCD]
  int8_t* buffer_422 = (int8_t*)&__rescheduled_0[6995968UL];
  cast__178(buffer_422, buffer_421);
  // [f32 [512, 1024, 1, 1] @ ABCD]
  float* buffer_423 = (float*)&__rescheduled_0[196608UL];
  mul__237(buffer_423, res5a_weight_0, folded_const_112);
  // [s8 [512, 1024, 1, 1] @ ABCD]
  int8_t* buffer_424 = (int8_t*)&__rescheduled_0[7520256UL];
  cast__238(buffer_424, buffer_423);
  // [f32 [256, 256, 3, 3] @ ABCD]
  float* buffer_425 = (float*)&__rescheduled_0[196608UL];
  mul__183(buffer_425, res4a_weight_1, folded_const_130);
  // [s8 [256, 256, 3, 3] @ ABCD]
  int8_t* buffer_426 = (int8_t*)&__rescheduled_0[8044544UL];
  cast__184(buffer_426, buffer_425);
  // [f32 [256, 256, 3, 3] @ ABCD]
  float* buffer_427 = (float*)&__rescheduled_0[196608UL];
  mul__192(buffer_427, res4b_weight_1, folded_const_127);
  // [s8 [256, 256, 3, 3] @ ABCD]
  int8_t* buffer_428 = (int8_t*)&__rescheduled_0[8634368UL];
  cast__193(buffer_428, buffer_427);
  // [f32 [256, 256, 3, 3] @ ABCD]
  float* buffer_429 = (float*)&__rescheduled_0[196608UL];
  mul__201(buffer_429, res4c_weight_1, folded_const_124);
  // [s8 [256, 256, 3, 3] @ ABCD]
  int8_t* buffer_430 = (int8_t*)&__rescheduled_0[9224192UL];
  cast__202(buffer_430, buffer_429);
  // [f32 [256, 256, 3, 3] @ ABCD]
  float* buffer_431 = (float*)&__rescheduled_0[196608UL];
  mul__210(buffer_431, res4d_weight_1, folded_const_121);
  // [s8 [256, 256, 3, 3] @ ABCD]
  int8_t* buffer_432 = (int8_t*)&__rescheduled_0[9814016UL];
  cast__211(buffer_432, buffer_431);
  // [f32 [256, 256, 3, 3] @ ABCD]
  float* buffer_433 = (float*)&__rescheduled_0[196608UL];
  mul__219(buffer_433, res4e_weight_1, folded_const_118);
  // [s8 [256, 256, 3, 3] @ ABCD]
  int8_t* buffer_434 = (int8_t*)&__rescheduled_0[10403840UL];
  cast__220(buffer_434, buffer_433);
  // [f32 [256, 256, 3, 3] @ ABCD]
  float* buffer_435 = (float*)&__rescheduled_0[196608UL];
  mul__228(buffer_435, res4f_weight_1, folded_const_115);
  // [s8 [256, 256, 3, 3] @ ABCD]
  int8_t* buffer_436 = (int8_t*)&__rescheduled_0[10993664UL];
  cast__229(buffer_436, buffer_435);
  reorder__525(folded_const_156, buffer_436);
  mul__652(folded_const_157, buffer_346, folded_const_86);
  mul__651(folded_const_158, buffer_266, folded_const_86);
  mul__646(folded_const_159, buffer_345, folded_const_87);
  mul__645(folded_const_160, buffer_265, folded_const_87);
  mul__640(folded_const_161, buffer_344, folded_const_88);
  mul__639(folded_const_162, buffer_264, folded_const_88);
  mul__634(folded_const_163, buffer_343, folded_const_89);
  mul__633(folded_const_164, buffer_263, folded_const_89);
  mul__628(folded_const_165, &res4b_bias_2[0], folded_const_90);
  mul__627(folded_const_166, &folded_const_36[0UL], folded_const_90);
  mul__622(folded_const_167, buffer_342, folded_const_91);
  mul__621(folded_const_168, buffer_262, folded_const_91);
  mul__616(folded_const_169, buffer_341, folded_const_92);
  mul__615(folded_const_170, buffer_261, folded_const_92);
  mul__614(folded_const_171, buffer_334, folded_const_93);
  mul__613(folded_const_172, buffer_285, folded_const_93);
  mul__608(folded_const_173, buffer_333, folded_const_94);
  mul__607(folded_const_174, buffer_282, folded_const_94);
  mul__602(folded_const_175, buffer_332, folded_const_95);
  mul__601(folded_const_176, buffer_280, folded_const_95);
  mul__596(folded_const_177, buffer_331, folded_const_96);
  mul__595(folded_const_178, buffer_278, folded_const_96);
  mul__590(folded_const_179, buffer_330, folded_const_97);
  mul__589(folded_const_180, buffer_275, folded_const_97);
  mul__650(folded_const_181, buffer_329, folded_const_17);
  mul__649(folded_const_182, buffer_295, folded_const_17);
  mul__648(folded_const_183, buffer_328, folded_const_19);
  mul__647(folded_const_184, buffer_294, folded_const_19);
  mul__644(folded_const_185, buffer_327, folded_const_22);
  mul__643(folded_const_186, buffer_293, folded_const_22);
  mul__642(folded_const_187, &res4e_bias_0[0], folded_const_24);
  mul__641(folded_const_188, &folded_const_25[0UL], folded_const_24);
  mul__638(folded_const_189, buffer_326, folded_const_27);
  mul__637(folded_const_190, buffer_292, folded_const_27);
  mul__636(folded_const_191, buffer_325, folded_const_29);
  mul__635(folded_const_192, buffer_291, folded_const_29);
  mul__632(folded_const_193, buffer_324, folded_const_32);
  mul__631(folded_const_194, buffer_290, folded_const_32);
  mul__630(folded_const_195, buffer_323, folded_const_34);
  mul__629(folded_const_196, buffer_289, folded_const_34);
  mul__626(folded_const_197, buffer_322, folded_const_37);
  mul__625(folded_const_198, buffer_288, folded_const_37);
  mul__624(folded_const_199, buffer_321, folded_const_39);
  mul__623(folded_const_200, buffer_287, folded_const_39);
  mul__620(folded_const_201, buffer_320, folded_const_42);
  mul__619(folded_const_202, buffer_286, folded_const_42);
  mul__618(folded_const_203, &res4a_bias_0[0], folded_const_44);
  mul__617(folded_const_204, &folded_const_45[0UL], folded_const_44);
  mul__588(folded_const_205, buffer_319, folded_const_98);
  mul__587(folded_const_206, buffer_274, folded_const_98);
  mul__582(folded_const_207, buffer_318, folded_const_99);
  mul__581(folded_const_208, buffer_271, folded_const_99);
  mul__576(folded_const_209, buffer_317, folded_const_100);
  mul__575(folded_const_210, buffer_269, folded_const_100);
  mul__570(folded_const_211, buffer_316, folded_const_101);
  mul__569(folded_const_212, buffer_267, folded_const_101);
  reorder__522(folded_const_213, buffer_420);
  reorder__515(folded_const_214, buffer_418);
  reorder__506(folded_const_215, buffer_416);
  reorder__497(folded_const_216, buffer_414);
  reorder__490(folded_const_217, buffer_412);
  reorder__528(folded_const_218, buffer_410);
  reorder__519(folded_const_219, buffer_408);
  reorder__512(folded_const_220, buffer_406);
  reorder__503(folded_const_221, buffer_404);
  reorder__496(folded_const_222, buffer_402);
  reorder__487(folded_const_223, buffer_400);
  reorder__422(folded_const_224, buffer_352);
  mul__612(folded_const_225, buffer_315, folded_const_48);
  mul__611(folded_const_226, buffer_284, folded_const_48);
  mul__610(folded_const_227, buffer_314, folded_const_50);
  mul__609(folded_const_228, buffer_283, folded_const_50);
  mul__606(folded_const_229, buffer_313, folded_const_53);
  mul__605(folded_const_230, buffer_281, folded_const_53);
  mul__604(folded_const_231, &res3c_bias_0[0], folded_const_55);
  mul__603(folded_const_232, &folded_const_56[0UL], folded_const_55);
  mul__600(folded_const_233, &res3b_bias_1[0], folded_const_58);
  mul__599(folded_const_234, &folded_const_59[0UL], folded_const_58);
  mul__598(folded_const_235, buffer_312, folded_const_60);
  mul__597(folded_const_236, buffer_279, folded_const_60);
  mul__594(folded_const_237, buffer_311, folded_const_63);
  mul__593(folded_const_238, buffer_277, folded_const_63);
  mul__592(folded_const_239, buffer_310, folded_const_65);
  mul__591(folded_const_240, buffer_276, folded_const_65);
  mul__586(folded_const_241, buffer_309, folded_const_69);
  mul__585(folded_const_242, buffer_273, folded_const_69);
  mul__584(folded_const_243, buffer_308, folded_const_71);
  mul__583(folded_const_244, buffer_272, folded_const_71);
  mul__580(folded_const_245, buffer_307, folded_const_74);
  mul__579(folded_const_246, buffer_270, folded_const_74);
  mul__578(folded_const_247, &res2b_bias_0[0], folded_const_76);
  mul__577(folded_const_248, &folded_const_77[0UL], folded_const_76);
  mul__574(folded_const_249, buffer_306, folded_const_79);
  mul__573(folded_const_250, buffer_268, folded_const_79);
  mul__572(folded_const_251, &res2a_bias_0[0], folded_const_81);
  mul__571(folded_const_252, &folded_const_102[0UL], folded_const_81);
  reorder__516(folded_const_253, buffer_434);
  reorder__509(folded_const_254, buffer_432);
  reorder__500(folded_const_255, buffer_430);
  reorder__493(folded_const_256, buffer_428);
  reorder__484(folded_const_257, buffer_426);
  reorder__480(folded_const_258, buffer_422);
  reorder__474(folded_const_259, buffer_398);
  reorder__465(folded_const_260, buffer_396);
  reorder__460(folded_const_261, buffer_394);
  reorder__451(folded_const_262, buffer_392);
  reorder__483(folded_const_263, buffer_390);
  reorder__445(folded_const_264, buffer_388);
  reorder__471(folded_const_265, buffer_386);
  reorder__464(folded_const_266, buffer_384);
  reorder__457(folded_const_267, buffer_382);
  reorder__477(folded_const_268, buffer_380);
  reorder__468(folded_const_269, buffer_378);
  reorder__461(folded_const_270, buffer_376);
  reorder__454(folded_const_271, buffer_374);
  reorder__439(folded_const_272, buffer_372);
  reorder__430(folded_const_273, buffer_370);
  reorder__423(folded_const_274, buffer_368);
  reorder__448(folded_const_275, buffer_366);
  reorder__436(folded_const_276, buffer_364);
  reorder__429(folded_const_277, buffer_362);
  reorder__442(folded_const_278, buffer_360);
  reorder__433(folded_const_279, buffer_358);
  reorder__426(folded_const_280, buffer_356);
  reorder__419(folded_const_281, buffer_354);
  mul__656(folded_const_282, buffer_335, folded_const_13);
  mul__655(folded_const_283, buffer_297, folded_const_13);
  reorder__534(folded_const_284, buffer_424);
  // [f32 [2048, 512, 1, 1] @ ABCD]
  float* buffer_566 = (float*)&__rescheduled_0[196608UL];
  mul__243(buffer_566, res5a_weight_2, folded_const_110);
  // [s8 [2048, 512, 1, 1] @ ABCD]
  int8_t* buffer_567 = (int8_t*)&__rescheduled_0[9633792UL];
  cast__244(buffer_567, buffer_566);
  // [f32 [2048, 512, 1, 1] @ ABCD]
  float* buffer_568 = (float*)&__rescheduled_0[196608UL];
  mul__252(buffer_568, res5b_weight_2, folded_const_107);
  // [s8 [2048, 512, 1, 1] @ ABCD]
  int8_t* buffer_569 = (int8_t*)&__rescheduled_0[11993088UL];
  cast__253(buffer_569, buffer_568);
  // [f32 [2048, 512, 1, 1] @ ABCD]
  float* buffer_570 = (float*)&__rescheduled_0[196608UL];
  mul__261(buffer_570, res5c_weight_2, folded_const_104);
  // [s8 [2048, 512, 1, 1] @ ABCD]
  int8_t* buffer_571 = (int8_t*)&__rescheduled_0[13041664UL];
  cast__262(buffer_571, buffer_570);
  // [f32 [512, 2048, 1, 1] @ ABCD]
  float* buffer_572 = (float*)&__rescheduled_0[196608UL];
  mul__246(buffer_572, res5b_weight_0, folded_const_109);
  // [s8 [512, 2048, 1, 1] @ ABCD]
  int8_t* buffer_573 = (int8_t*)&__rescheduled_0[14090240UL];
  cast__247(buffer_573, buffer_572);
  // [f32 [512, 2048, 1, 1] @ ABCD]
  float* buffer_574 = (float*)&__rescheduled_0[196608UL];
  mul__255(buffer_574, res5c_weight_0, folded_const_106);
  // [s8 [512, 2048, 1, 1] @ ABCD]
  int8_t* buffer_575 = (int8_t*)&__rescheduled_0[15138816UL];
  cast__256(buffer_575, buffer_574);
  // [f32 [2048, 1024, 1, 1] @ ABCD]
  float* buffer_576 = (float*)&__rescheduled_0[196608UL];
  mul__234(buffer_576, res5a_weight_b, folded_const_113);
  // [s8 [2048, 1024, 1, 1] @ ABCD]
  int8_t* buffer_577 = (int8_t*)&__rescheduled_0[16187392UL];
  cast__235(buffer_577, buffer_576);
  reorder__531(folded_const_285, buffer_577);
  mul__654(folded_const_286, buffer_347, folded_const_85);
  mul__653(folded_const_287, buffer_296, folded_const_85);
  // [f32 [512, 512, 3, 3] @ ABCD]
  float* buffer_582 = (float*)&__rescheduled_0[196608UL];
  mul__240(buffer_582, res5a_weight_1, folded_const_111);
  // [s8 [512, 512, 3, 3] @ ABCD]
  int8_t* buffer_583 = (int8_t*)&__rescheduled_0[16187392UL];
  cast__241(buffer_583, buffer_582);
  reorder__537(folded_const_288, buffer_583);
  mul__658(folded_const_289, buffer_336, folded_const_11);
  mul__657(folded_const_290, buffer_298, folded_const_11);
  mul__660(folded_const_291, buffer_348, folded_const_84);
  mul__659(folded_const_292, buffer_299, folded_const_84);
  reorder__540(folded_const_293, buffer_567);
  mul__662(folded_const_294, buffer_337, folded_const_8);
  mul__661(folded_const_295, buffer_300, folded_const_8);
  reorder__543(folded_const_296, buffer_573);
  // [f32 [512, 512, 3, 3] @ ABCD]
  float* buffer_596 = (float*)&__rescheduled_0[196608UL];
  mul__249(buffer_596, res5b_weight_1, folded_const_108);
  // [s8 [512, 512, 3, 3] @ ABCD]
  int8_t* buffer_597 = (int8_t*)&__rescheduled_0[16187392UL];
  cast__250(buffer_597, buffer_596);
  // [f32 [512, 512, 3, 3] @ ABCD]
  float* buffer_598 = (float*)&__rescheduled_0[196608UL];
  mul__258(buffer_598, res5c_weight_1, folded_const_105);
  // [s8 [512, 512, 3, 3] @ ABCD]
  int8_t* buffer_599 = (int8_t*)&__rescheduled_0[9633792UL];
  cast__259(buffer_599, buffer_598);
  mul__664(folded_const_297, buffer_338, folded_const_6);
  mul__663(folded_const_298, buffer_301, folded_const_6);
  reorder__546(folded_const_299, buffer_597);
  mul__666(folded_const_300, buffer_349, folded_const_83);
  mul__665(folded_const_301, buffer_302, folded_const_83);
  reorder__549(folded_const_302, buffer_569);
  mul__668(folded_const_303, buffer_339, folded_const_3);
  mul__667(folded_const_304, buffer_303, folded_const_3);
  reorder__552(folded_const_305, buffer_575);
  mul__670(folded_const_306, buffer_340, folded_const_1);
  mul__669(folded_const_307, buffer_304, folded_const_1);
  reorder__555(folded_const_308, buffer_599);
  mul__672(folded_const_309, buffer_350, folded_const_82);
  mul__671(folded_const_310, buffer_305, folded_const_82);
  reorder__558(folded_const_311, buffer_571);
  is_init = true;
  sc_aligned_free(__stream, __rescheduled_0);
}

extern "C" void sc_init_rn50_backbone_bs2() {
  bool& is_init = *(bool*)(__module_data + 0);
  void*& __sc_kernel_cache = *(void**)(__module_data + 8);
  uint8_t* __brgemm_attrs = (uint8_t*)&__module_data[214464UL];
  void*& __sc_kernel_cache_206 = *(void**)(__module_data + 16);
  uint8_t* __brgemm_attrs_205 = (uint8_t*)&__module_data[214592UL];
  void*& __sc_kernel_cache_208 = *(void**)(__module_data + 24);
  uint8_t* __brgemm_attrs_207 = (uint8_t*)&__module_data[214720UL];
  void*& __sc_kernel_cache_210 = *(void**)(__module_data + 32);
  uint8_t* __brgemm_attrs_209 = (uint8_t*)&__module_data[214848UL];
  void*& __sc_kernel_cache_212 = *(void**)(__module_data + 40);
  uint8_t* __brgemm_attrs_211 = (uint8_t*)&__module_data[214976UL];
  void*& __sc_kernel_cache_214 = *(void**)(__module_data + 48);
  uint8_t* __brgemm_attrs_213 = (uint8_t*)&__module_data[215104UL];
  void*& __sc_kernel_cache_216 = *(void**)(__module_data + 56);
  uint8_t* __brgemm_attrs_215 = (uint8_t*)&__module_data[215232UL];
  void*& __sc_kernel_cache_218 = *(void**)(__module_data + 64);
  uint8_t* __brgemm_attrs_217 = (uint8_t*)&__module_data[215360UL];
  void*& __sc_kernel_cache_220 = *(void**)(__module_data + 72);
  uint8_t* __brgemm_attrs_219 = (uint8_t*)&__module_data[215488UL];
  void*& __sc_kernel_cache_222 = *(void**)(__module_data + 80);
  uint8_t* __brgemm_attrs_221 = (uint8_t*)&__module_data[215616UL];
  void*& __sc_kernel_cache_224 = *(void**)(__module_data + 88);
  uint8_t* __brgemm_attrs_223 = (uint8_t*)&__module_data[215744UL];
  void*& __sc_kernel_cache_226 = *(void**)(__module_data + 96);
  uint8_t* __brgemm_attrs_225 = (uint8_t*)&__module_data[215872UL];
  void*& __sc_kernel_cache_228 = *(void**)(__module_data + 104);
  uint8_t* __brgemm_attrs_227 = (uint8_t*)&__module_data[216000UL];
  void*& __sc_kernel_cache_230 = *(void**)(__module_data + 112);
  uint8_t* __brgemm_attrs_229 = (uint8_t*)&__module_data[216128UL];
  void*& __sc_kernel_cache_232 = *(void**)(__module_data + 120);
  uint8_t* __brgemm_attrs_231 = (uint8_t*)&__module_data[216256UL];
  void*& __sc_kernel_cache_235 = *(void**)(__module_data + 128);
  uint8_t* __brgemm_attrs_234 = (uint8_t*)&__module_data[217408UL];
  void*& __sc_kernel_cache_237 = *(void**)(__module_data + 136);
  uint8_t* __brgemm_attrs_236 = (uint8_t*)&__module_data[217536UL];
  void*& __sc_kernel_cache_239 = *(void**)(__module_data + 144);
  uint8_t* __brgemm_attrs_238 = (uint8_t*)&__module_data[217664UL];
  void*& __sc_kernel_cache_241 = *(void**)(__module_data + 152);
  uint8_t* __brgemm_attrs_240 = (uint8_t*)&__module_data[217792UL];
  void*& __sc_kernel_cache_243 = *(void**)(__module_data + 160);
  uint8_t* __brgemm_attrs_242 = (uint8_t*)&__module_data[217920UL];
  void*& __sc_kernel_cache_244 = *(void**)(__module_data + 168);
  void*& __sc_kernel_cache_246 = *(void**)(__module_data + 176);
  uint8_t* __brgemm_attrs_245 = (uint8_t*)&__module_data[218048UL];
  void*& __sc_kernel_cache_248 = *(void**)(__module_data + 184);
  uint8_t* __brgemm_attrs_247 = (uint8_t*)&__module_data[218176UL];
  void*& __sc_kernel_cache_254 = *(void**)(__module_data + 192);
  uint8_t* __brgemm_attrs_253 = (uint8_t*)&__module_data[218880UL];
  void*& __sc_kernel_cache_256 = *(void**)(__module_data + 200);
  uint8_t* __brgemm_attrs_255 = (uint8_t*)&__module_data[219008UL];
  void*& __sc_kernel_cache_258 = *(void**)(__module_data + 208);
  uint8_t* __brgemm_attrs_257 = (uint8_t*)&__module_data[219136UL];
  void*& __sc_kernel_cache_264 = *(void**)(__module_data + 216);
  uint8_t* __brgemm_attrs_263 = (uint8_t*)&__module_data[219648UL];
  void*& __sc_kernel_cache_266 = *(void**)(__module_data + 224);
  uint8_t* __brgemm_attrs_265 = (uint8_t*)&__module_data[219776UL];
  void*& __sc_kernel_cache_270 = *(void**)(__module_data + 232);
  uint8_t* __brgemm_attrs_269 = (uint8_t*)&__module_data[220032UL];
  void*& __sc_kernel_cache_272 = *(void**)(__module_data + 240);
  uint8_t* __brgemm_attrs_271 = (uint8_t*)&__module_data[220160UL];
  void*& __sc_kernel_cache_276 = *(void**)(__module_data + 248);
  uint8_t* __brgemm_attrs_275 = (uint8_t*)&__module_data[220416UL];
  void*& __sc_kernel_cache_278 = *(void**)(__module_data + 256);
  uint8_t* __brgemm_attrs_277 = (uint8_t*)&__module_data[220544UL];
  void*& __sc_kernel_cache_280 = *(void**)(__module_data + 264);
  uint8_t* __brgemm_attrs_279 = (uint8_t*)&__module_data[220672UL];
  void*& __sc_kernel_cache_282 = *(void**)(__module_data + 272);
  uint8_t* __brgemm_attrs_281 = (uint8_t*)&__module_data[220800UL];
  void*& __sc_kernel_cache_284 = *(void**)(__module_data + 280);
  uint8_t* __brgemm_attrs_283 = (uint8_t*)&__module_data[220928UL];
  void*& __sc_kernel_cache_290 = *(void**)(__module_data + 288);
  uint8_t* __brgemm_attrs_289 = (uint8_t*)&__module_data[221312UL];
  void*& __sc_kernel_cache_292 = *(void**)(__module_data + 296);
  uint8_t* __brgemm_attrs_291 = (uint8_t*)&__module_data[221440UL];
  void*& __sc_kernel_cache_294 = *(void**)(__module_data + 304);
  uint8_t* __brgemm_attrs_293 = (uint8_t*)&__module_data[221568UL];
  void*& __sc_kernel_cache_300 = *(void**)(__module_data + 312);
  uint8_t* __brgemm_attrs_299 = (uint8_t*)&__module_data[221888UL];
  void*& __sc_kernel_cache_302 = *(void**)(__module_data + 320);
  uint8_t* __brgemm_attrs_301 = (uint8_t*)&__module_data[222016UL];
  void*& __sc_kernel_cache_306 = *(void**)(__module_data + 328);
  uint8_t* __brgemm_attrs_305 = (uint8_t*)&__module_data[222272UL];
  void** __brgemm_bd_mask_arr = (void**)&__uninitialized_data[23657472UL];
  uint8_t* __brgemm_full_bd_mask = (uint8_t*)&__module_data[216512UL];
  void** __brgemm_bd_mask_arr_251 = (void**)&__uninitialized_data[23657504UL];
  uint8_t* __brgemm_full_bd_mask_250 = (uint8_t*)&__module_data[218432UL];
  void** __brgemm_bd_mask_arr_261 = (void**)&__uninitialized_data[23657520UL];
  uint8_t* __brgemm_full_bd_mask_260 = (uint8_t*)&__module_data[219392UL];
  void** __brgemm_bd_mask_arr_287 = (void**)&__uninitialized_data[23657552UL];
  uint8_t* __brgemm_full_bd_mask_286 = (uint8_t*)&__module_data[221184UL];
  void** __brgemm_bd_mask_arr_297 = (void**)&__uninitialized_data[23657568UL];
  uint8_t* __brgemm_full_bd_mask_296 = (uint8_t*)&__module_data[221816UL];
  void** __sc_kernel_cache_arr = (void**)&__uninitialized_data[23657488UL];
  uint8_t* __brgemm_attrs_233 = (uint8_t*)&__module_data[216384UL];
  void** __sc_kernel_cache_arr_252 = (void**)&__uninitialized_data[23657512UL];
  uint8_t* __brgemm_attrs_249 = (uint8_t*)&__module_data[218304UL];
  void** __sc_kernel_cache_arr_262 = (void**)&__uninitialized_data[23657528UL];
  uint8_t* __brgemm_attrs_259 = (uint8_t*)&__module_data[219264UL];
  void** __sc_kernel_cache_arr_268 = (void**)&__uninitialized_data[23657536UL];
  uint8_t* __brgemm_attrs_267 = (uint8_t*)&__module_data[219904UL];
  void** __sc_kernel_cache_arr_274 = (void**)&__uninitialized_data[23657544UL];
  uint8_t* __brgemm_attrs_273 = (uint8_t*)&__module_data[220288UL];
  void** __sc_kernel_cache_arr_288 = (void**)&__uninitialized_data[23657560UL];
  uint8_t* __brgemm_attrs_285 = (uint8_t*)&__module_data[221056UL];
  void** __sc_kernel_cache_arr_298 = (void**)&__uninitialized_data[23657576UL];
  uint8_t* __brgemm_attrs_295 = (uint8_t*)&__module_data[221696UL];
  void** __sc_kernel_cache_arr_304 = (void**)&__uninitialized_data[23657584UL];
  uint8_t* __brgemm_attrs_303 = (uint8_t*)&__module_data[222144UL];
  is_init = false;
  __sc_kernel_cache = dnnl_brgemm_list_func(112, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs, ((void*)0), ((void*)0));
  __sc_kernel_cache_206 = dnnl_brgemm_list_func(14, 32, 64, 64, 32, 32, 0.f, 7, 7, __brgemm_attrs_205, ((void*)0), ((void*)0));
  __sc_kernel_cache_208 = dnnl_brgemm_list_func(112, 16, 64, 64, 16, 16, 0.f, 7, 7, __brgemm_attrs_207, ((void*)0), ((void*)0));
  __sc_kernel_cache_210 = dnnl_brgemm_list_func(56, 64, 32, 32, 64, 64, 0.f, 7, 7, __brgemm_attrs_209, ((void*)0), ((void*)0));
  __sc_kernel_cache_212 = dnnl_brgemm_list_func(392, 64, 32, 32, 64, 64, 0.f, 8, 7, __brgemm_attrs_211, ((void*)0), ((void*)0));
  __sc_kernel_cache_214 = dnnl_brgemm_list_func(28, 16, 64, 64, 16, 16, 0.f, 7, 7, __brgemm_attrs_213, ((void*)0), ((void*)0));
  __sc_kernel_cache_216 = dnnl_brgemm_list_func(56, 32, 64, 64, 32, 32, 0.f, 7, 7, __brgemm_attrs_215, ((void*)0), ((void*)0));
  __sc_kernel_cache_218 = dnnl_brgemm_list_func(56, 32, 64, 64, 32, 32, 0.f, 8, 7, __brgemm_attrs_217, ((void*)0), ((void*)0));
  __sc_kernel_cache_220 = dnnl_brgemm_list_func(28, 32, 64, 64, 32, 32, 0.f, 7, 7, __brgemm_attrs_219, ((void*)0), ((void*)0));
  __sc_kernel_cache_222 = dnnl_brgemm_list_func(56, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_221, ((void*)0), ((void*)0));
  __sc_kernel_cache_224 = dnnl_brgemm_list_func(112, 64, 128, 128, 64, 64, 0.f, 8, 7, __brgemm_attrs_223, ((void*)0), ((void*)0));
  __sc_kernel_cache_226 = dnnl_brgemm_list_func(28, 32, 128, 256, 32, 32, 0.f, 7, 7, __brgemm_attrs_225, ((void*)0), ((void*)0));
  __sc_kernel_cache_228 = dnnl_brgemm_list_func(196, 32, 128, 128, 32, 32, 0.f, 8, 7, __brgemm_attrs_227, ((void*)0), ((void*)0));
  __sc_kernel_cache_230 = dnnl_brgemm_list_func(392, 32, 32, 32, 32, 32, 0.f, 7, 7, __brgemm_attrs_229, ((void*)0), ((void*)0));
  __sc_kernel_cache_232 = dnnl_brgemm_list_func(112, 32, 128, 128, 32, 32, 0.f, 8, 7, __brgemm_attrs_231, ((void*)0), ((void*)0));
  __sc_kernel_cache_235 = dnnl_brgemm_list_func(28, 128, 64, 64, 128, 128, 0.f, 7, 7, __brgemm_attrs_234, ((void*)0), ((void*)0));
  __sc_kernel_cache_237 = dnnl_brgemm_list_func(784, 128, 64, 64, 128, 128, 0.f, 8, 7, __brgemm_attrs_236, ((void*)0), ((void*)0));
  __sc_kernel_cache_239 = dnnl_brgemm_list_func(28, 32, 64, 64, 32, 32, 0.f, 7, 7, __brgemm_attrs_238, ((void*)0), ((void*)0));
  __sc_kernel_cache_241 = dnnl_brgemm_list_func(56, 64, 128, 128, 64, 64, 0.f, 7, 7, __brgemm_attrs_240, ((void*)0), ((void*)0));
  __sc_kernel_cache_243 = dnnl_brgemm_list_func(56, 64, 128, 128, 64, 64, 0.f, 8, 7, __brgemm_attrs_242, ((void*)0), ((void*)0));
  __sc_kernel_cache_244 = dnnl_brgemm_list_func(28, 32, 128, 128, 32, 32, 0.f, 7, 7, __brgemm_attrs_225, ((void*)0), ((void*)0));
  __sc_kernel_cache_246 = dnnl_brgemm_list_func(196, 128, 32, 32, 128, 128, 0.f, 7, 7, __brgemm_attrs_245, ((void*)0), ((void*)0));
  __sc_kernel_cache_248 = dnnl_brgemm_list_func(56, 256, 128, 128, 256, 256, 0.f, 8, 7, __brgemm_attrs_247, ((void*)0), ((void*)0));
  __sc_kernel_cache_254 = dnnl_brgemm_list_func(98, 32, 128, 128, 32, 32, 0.f, 8, 7, __brgemm_attrs_253, ((void*)0), ((void*)0));
  __sc_kernel_cache_256 = dnnl_brgemm_list_func(14, 128, 256, 256, 128, 128, 0.f, 7, 7, __brgemm_attrs_255, ((void*)0), ((void*)0));
  __sc_kernel_cache_258 = dnnl_brgemm_list_func(28, 64, 1024, 1024, 64, 64, 0.f, 8, 7, __brgemm_attrs_257, ((void*)0), ((void*)0));
  __sc_kernel_cache_264 = dnnl_brgemm_list_func(98, 1024, 64, 64, 1024, 1024, 0.f, 7, 7, __brgemm_attrs_263, ((void*)0), ((void*)0));
  __sc_kernel_cache_266 = dnnl_brgemm_list_func(196, 32, 128, 128, 32, 32, 0.f, 8, 7, __brgemm_attrs_265, ((void*)0), ((void*)0));
  __sc_kernel_cache_270 = dnnl_brgemm_list_func(196, 128, 256, 256, 128, 128, 0.f, 7, 7, __brgemm_attrs_269, ((void*)0), ((void*)0));
  __sc_kernel_cache_272 = dnnl_brgemm_list_func(196, 64, 128, 128, 64, 64, 0.f, 8, 7, __brgemm_attrs_271, ((void*)0), ((void*)0));
  __sc_kernel_cache_276 = dnnl_brgemm_list_func(196, 128, 128, 128, 128, 128, 0.f, 7, 7, __brgemm_attrs_275, ((void*)0), ((void*)0));
  __sc_kernel_cache_278 = dnnl_brgemm_list_func(98, 256, 128, 128, 256, 256, 0.f, 8, 7, __brgemm_attrs_277, ((void*)0), ((void*)0));
  __sc_kernel_cache_280 = dnnl_brgemm_list_func(28, 32, 512, 512, 32, 32, 0.f, 8, 7, __brgemm_attrs_279, ((void*)0), ((void*)0));
  __sc_kernel_cache_282 = dnnl_brgemm_list_func(196, 256, 128, 128, 256, 256, 0.f, 7, 7, __brgemm_attrs_281, ((void*)0), ((void*)0));
  __sc_kernel_cache_284 = dnnl_brgemm_list_func(196, 64, 256, 256, 64, 64, 0.f, 8, 7, __brgemm_attrs_283, ((void*)0), ((void*)0));
  __sc_kernel_cache_290 = dnnl_brgemm_list_func(49, 16, 256, 256, 16, 16, 0.f, 8, 7, __brgemm_attrs_289, ((void*)0), ((void*)0));
  __sc_kernel_cache_292 = dnnl_brgemm_list_func(49, 64, 256, 256, 64, 64, 0.f, 7, 7, __brgemm_attrs_291, ((void*)0), ((void*)0));
  __sc_kernel_cache_294 = dnnl_brgemm_list_func(49, 128, 64, 64, 128, 128, 0.f, 8, 7, __brgemm_attrs_293, ((void*)0), ((void*)0));
  __sc_kernel_cache_300 = dnnl_brgemm_list_func(7, 64, 256, 256, 64, 64, 0.f, 7, 7, __brgemm_attrs_299, ((void*)0), ((void*)0));
  __sc_kernel_cache_302 = dnnl_brgemm_list_func(49, 32, 512, 512, 32, 32, 0.f, 8, 7, __brgemm_attrs_301, ((void*)0), ((void*)0));
  __sc_kernel_cache_306 = dnnl_brgemm_list_func(49, 512, 512, 512, 512, 512, 0.f, 7, 7, __brgemm_attrs_305, ((void*)0), ((void*)0));
  __brgemm_bd_mask_arr[0] = &__brgemm_full_bd_mask[(0 * 419)];
  __brgemm_bd_mask_arr[1] = &__brgemm_full_bd_mask[(1 * 419)];
  __brgemm_bd_mask_arr_251[0] = &__brgemm_full_bd_mask_250[(0 * 404)];
  __brgemm_bd_mask_arr_261[0] = &__brgemm_full_bd_mask_260[(0 * 222)];
  __brgemm_bd_mask_arr_287[0] = &__brgemm_full_bd_mask_286[(0 * 103)];
  __brgemm_bd_mask_arr_297[0] = &__brgemm_full_bd_mask_296[(0 * 61)];
  __sc_kernel_cache_arr[0] = dnnl_brgemm_list_func(419, 128, 64, 64, 128, 128, 0.f, 7, 7, __brgemm_attrs_233, __brgemm_bd_mask_arr[0], ((void*)0));
  __sc_kernel_cache_arr[1] = dnnl_brgemm_list_func(419, 128, 64, 64, 128, 128, 0.f, 7, 7, __brgemm_attrs_233, __brgemm_bd_mask_arr[1], ((void*)0));
  __sc_kernel_cache_arr_252[0] = dnnl_brgemm_list_func(404, 128, 256, 512, 128, 128, 0.f, 7, 7, __brgemm_attrs_249, __brgemm_bd_mask_arr_251[0], ((void*)0));
  __sc_kernel_cache_arr_262[0] = dnnl_brgemm_list_func(222, 16, 256, 256, 16, 16, 0.f, 7, 7, __brgemm_attrs_259, __brgemm_bd_mask_arr_261[0], ((void*)0));
  __sc_kernel_cache_arr_268[0] = dnnl_brgemm_list_func(222, 32, 128, 128, 32, 32, 0.f, 7, 7, __brgemm_attrs_267, __brgemm_bd_mask_arr_261[0], ((void*)0));
  __sc_kernel_cache_arr_274[0] = dnnl_brgemm_list_func(222, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_273, __brgemm_bd_mask_arr_261[0], ((void*)0));
  __sc_kernel_cache_arr_288[0] = dnnl_brgemm_list_func(103, 128, 256, 512, 128, 128, 0.f, 7, 7, __brgemm_attrs_285, __brgemm_bd_mask_arr_287[0], ((void*)0));
  __sc_kernel_cache_arr_298[0] = dnnl_brgemm_list_func(61, 16, 512, 512, 16, 16, 0.f, 7, 7, __brgemm_attrs_295, __brgemm_bd_mask_arr_297[0], ((void*)0));
  __sc_kernel_cache_arr_304[0] = dnnl_brgemm_list_func(61, 128, 256, 256, 128, 128, 0.f, 7, 7, __brgemm_attrs_303, __brgemm_bd_mask_arr_297[0], ((void*)0));
}

