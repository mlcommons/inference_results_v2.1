#pragma once
#include <stdint.h>


/**
 * Do reorder from ACDB format to ABCD format. Plain dims is 2x2048x7x7
**/
extern "C" bool reorder_2x2048x7x7_ACDB_ABCD(uint16_t* out, uint16_t* in);


/**
 * Do reorder from ABCD format to ACDB format. Plain dims is 2x64x56x56
**/
extern "C" bool reorder_2x64x56x56_ABCD_ACDB(uint16_t* out, uint16_t* in);


/**
 * Do reorder from ACDB format to ABCD format. Plain dims is 4x2048x7x7
**/
extern "C" bool reorder_4x2048x7x7_ACDB_ABCD(uint16_t* out, uint16_t* in);


/**
 * Do reorder from ABCD format to ACDB format. Plain dims is 4x64x56x56
**/
extern "C" bool reorder_4x64x56x56_ABCD_ACDB(uint16_t* out, uint16_t* in);


/**
 * Do reorder from ACDB format to ABCD format. Plain dims is 9x2048x7x7
**/
extern "C" bool reorder_9x2048x7x7_ACDB_ABCD(uint16_t* out, uint16_t* in);


/**
 * Do reorder from ABCD format to ACDB format. Plain dims is 9x64x56x56
**/
extern "C" bool reorder_9x64x56x56_ABCD_ACDB(uint16_t* out, uint16_t* in);
