import os
import numpy as np



def f32tobf20(src_array,rnd=1):
    uint32_np_tensor = np.frombuffer(src_array.tobytes(),dtype="uint32")
    shape = uint32_np_tensor.shape
    # print(shape)
    uint32_np_tensor_new = np.ones(shape, dtype="uint32")
    EU_N_BIT_1 = 1 << 11 - 1
    RND_RDNE = 1

    for i,item in enumerate(uint32_np_tensor):
        src_sign = (item>>31) & 0x1
        src_exp = (item>>23) & 0xFF
        src_mant = item & 0x7FFFFF
        dst_sign = src_sign
        if src_exp == 0:
            dst_sign = src_sign
            dst_exp = 0
            dst_mant = 0
        elif src_exp == 0xFF and src_mant !=0:
            dst_sign = 0
            dst_exp = 0xFF
            dst_mant = 0x7FF
        elif src_exp == 0xFF and src_mant ==0:
            dst_sign = src_sign
            dst_exp = 0xFF
            dst_mant = 0x0
        else:
            dst_exp = src_exp
            dst_mant = src_mant >> 12

            bit_before_point = (src_mant >> 12) & 0x1
            bit_after_point = (src_mant >> 11) & 0x1
            s = 0x0
            if (src_mant & EU_N_BIT_1) != 0:
                s= 0x1
            if rnd == RND_RDNE:
                # << 0.5
                if bit_after_point == 0:
                    dst_mant = dst_mant
                # > 0.5
                elif bit_after_point == 1 and s == 1:
                    dst_mant = dst_mant + 0x1
                    if dst_mant & 0x7FF == 0x0:
                        dst_exp = dst_exp + 1
                # = 0.5
                elif bit_after_point == 1 and s == 0 and bit_before_point == 1:
                    dst_mant = dst_mant + 0x1
                    if dst_mant & 0x7FF == 0x0:
                        dst_exp = dst_exp + 1
                # = 0.5
                elif bit_after_point == 1 and s == 0 and bit_before_point == 0:
                    dst_mant = dst_mant
                else:
                    assert(0)
        
        dst = int(dst_sign<<31) + int(dst_exp<<23) + int(dst_mant<<12)
        # print(src_sign,src_exp,src_mant)
        # print(dst_sign,dst_exp,dst_mant)
        uint32_np_tensor_new[i] = dst
    return uint32_np_tensor_new

if __name__ == '__main__':
    a = [0.1]
    src_array = np.array(a,dtype='float32')
    print(src_array.shape)
    uint32_np_tensor_new = f32tobf20(src_array)
    print(bin(uint32_np_tensor_new[0]))
    uint32_np_tensor_new_float = np.frombuffer(uint32_np_tensor_new.tobytes(),dtype="float32")
    print(float(uint32_np_tensor_new_float[0]))