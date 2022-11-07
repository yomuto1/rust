#include <stdio.h>
#include <time.h>
#include <string.h>
#include <arm_neon.h>

#define ITERATION (1000u)
#define INP_WID (512u)
#define INP_HEI (512u)
#define OUT_WID (INP_WID / 2u)
#define OUT_HEI (INP_HEI / 2u)
#define VAL_SHIFT (5)

void func_natc(const unsigned int inp_wid_u32, const unsigned int inp_hei_u32, const unsigned int out_wid_u32, const unsigned int out_hei_u32, const signed int val_shift_s32, const signed char* p_ker_s8, const unsigned char* p_inp_u8, unsigned char* p_out_u8);
void func_neon(const unsigned int inp_wid_u32, const unsigned int inp_hei_u32, const unsigned int out_wid_u32, const unsigned int out_hei_u32, const signed int val_shift_s32, const signed char* p_ker_s8, const unsigned char* p_inp_u8, unsigned char* p_out_u8);

void main (void) {
    FILE* fp;
    static unsigned char sa_inp_0_u8[INP_WID * INP_HEI];
    static unsigned char sa_out_0_u8[OUT_WID * OUT_HEI]; // max pool result
    static unsigned char sa_out_1_u8[OUT_WID * OUT_HEI]; // max pool result, neon
    const signed char a_ker_s8[3 * 3] = { 3, 7, -5, -10, 19, 7, 5, -9, 7 }; // non-separable filter is considered
    unsigned int i_u32;
    unsigned int j_u32;
    unsigned int z_u32;
    clock_t time_start;

    fp = fopen("/home/hyuk/code/barbara_gray.raw", "rb");

    fread(sa_inp_0_u8, sizeof(unsigned char), INP_WID * INP_HEI, fp);

    fclose(fp);

    memset(&sa_out_0_u8[0], 0, OUT_WID * OUT_HEI * sizeof(unsigned char));
    memset(&sa_out_1_u8[0], 0, OUT_WID * OUT_HEI * sizeof(unsigned char));

    // natural C code version
    time_start = clock();

    for (z_u32 = 0u; z_u32 < ITERATION; z_u32++)    
    {
        func_natc(INP_WID, INP_HEI, OUT_WID, OUT_HEI, VAL_SHIFT, a_ker_s8, sa_inp_0_u8, sa_out_0_u8);
    }

    printf("natural C code version time: %f sec\n", (float)(clock() - time_start) / CLOCKS_PER_SEC);

    fp = fopen("out_c_nat.bin", "wb");
    fwrite(&sa_out_0_u8[0], sizeof(unsigned char), OUT_WID * OUT_HEI, fp);
    fclose(fp);
 
    // arm neon version
    time_start = clock();

    for (z_u32 = 0u; z_u32 < ITERATION; z_u32++)    
    {
        func_neon(INP_WID, INP_HEI, OUT_WID, OUT_HEI, VAL_SHIFT, a_ker_s8, sa_inp_0_u8, sa_out_1_u8);
    }

    printf("neon C code version time: %f sec\n", (float)(clock() - time_start) / CLOCKS_PER_SEC);

    fp = fopen("out_c_neon.bin", "wb");
    fwrite(&sa_out_1_u8[0], sizeof(unsigned char), OUT_WID * OUT_HEI, fp);
    fclose(fp); 

    for (j_u32 = 0u; j_u32 < OUT_HEI; j_u32++)
    //for (j_u32 = 0u; j_u32 < 4u; j_u32++)
    {
        for (i_u32 = 0u; i_u32 < OUT_WID; i_u32++)
        {
            if (sa_out_0_u8[i_u32 + (j_u32 * OUT_WID)] != sa_out_1_u8[i_u32 + (j_u32 * OUT_WID)])
            {
                printf("mismatch (%u, %u): %d, %d\n", i_u32, j_u32, sa_out_0_u8[i_u32 + (j_u32 * OUT_WID)], sa_out_1_u8[i_u32 + (j_u32 * OUT_WID)]);
            }
        }
    }
}

void func_natc(const unsigned int inp_wid_u32, const unsigned int inp_hei_u32, const unsigned int out_wid_u32, const unsigned int out_hei_u32, const signed int val_shift_s32, const signed char* p_ker_s8, const unsigned char* p_inp_u8, unsigned char* p_out_u8)
{
    unsigned int i_u32;
    unsigned int j_u32;
    unsigned int k_w_u32;
    unsigned int k_h_u32;

    signed char ker_s8;
    unsigned char inp_u8;
    signed short sum_s16;
    unsigned char max_u8;
    unsigned char tmp_u8;

    static unsigned char sa_tmp_0_u8[INP_WID * INP_HEI] = { 0, }; // filtered image

    // filtering
    for (j_u32 = 0u; j_u32 < (inp_hei_u32 - 2u); j_u32++)
    {
        for (i_u32 = 0u; i_u32 < (inp_wid_u32 - 2u); i_u32++)
        {
            sum_s16 = 0;

            for (k_h_u32 = 0u; k_h_u32 < 3u; k_h_u32++)
            {
                for (k_w_u32 = 0u; k_w_u32 < 3u; k_w_u32++)
                {
                    ker_s8 = p_ker_s8[k_w_u32 + (k_h_u32 * 3u)];
                    inp_u8 = p_inp_u8[k_w_u32 + i_u32 + ((k_h_u32 + j_u32) * inp_wid_u32)];

                    sum_s16 += ker_s8 * inp_u8;
                }
            }

            sum_s16 >>= val_shift_s32;
            if (sum_s16 > 255)
            {
                sum_s16 = 255;
            }
            if (sum_s16 < 0)
            {
                sum_s16 = 0;
            }

            sa_tmp_0_u8[i_u32 + (j_u32 * inp_wid_u32)] = (unsigned char)sum_s16;
        }
    }

    // 2x2 max pooling
    for (j_u32 = 0u; j_u32 < out_hei_u32; j_u32++)
    {
        for (i_u32 = 0u; i_u32 < out_wid_u32; i_u32++)
        {
            max_u8 = sa_tmp_0_u8[(i_u32 * 2u) + 0u + (((j_u32 * 2u) + 0u) * inp_wid_u32)];
            tmp_u8 = sa_tmp_0_u8[(i_u32 * 2u) + 1u + (((j_u32 * 2u) + 0u) * inp_wid_u32)];
            if (max_u8 < tmp_u8)
            {
                max_u8 = tmp_u8;
            }
            tmp_u8 = sa_tmp_0_u8[(i_u32 * 2u) + 0u + (((j_u32 * 2u) + 1u) * inp_wid_u32)];
            if (max_u8 < tmp_u8)
            {
                max_u8 = tmp_u8;
            }
            tmp_u8 = sa_tmp_0_u8[(i_u32 * 2u) + 1u + (((j_u32 * 2u) + 1u) * inp_wid_u32)];
            if (max_u8 < tmp_u8)
            {
                max_u8 = tmp_u8;
            }
            p_out_u8[i_u32 + (j_u32 * out_wid_u32)] = max_u8;
        }
    }
}

void func_neon(const unsigned int inp_wid_u32, const unsigned int inp_hei_u32, const unsigned int out_wid_u32, const unsigned int out_hei_u32, const signed int val_shift_s32, const signed char* p_ker_s8, const unsigned char* p_inp_u8, unsigned char* p_out_u8)
{
    unsigned int i_u32;
    unsigned int j_u32;

    const int16x8_t zero_s16x8 = vdupq_n_s16(0);
    const int16x8_t u8max_s16x8 = vdupq_n_s16(255);

#if 0
    for (j_u32 = 0u; j_u32 < (out_hei_u32 - 1u); j_u32++)
    {
        const unsigned char* restrict p_inp_line0_u8 = &sa_inp_0_u8[((2u * j_u32) + 0u) * inp_wid_u32];
        const unsigned char* restrict p_inp_line1_u8 = &sa_inp_0_u8[((2u * j_u32) + 1u) * inp_wid_u32];
        const unsigned char* restrict p_inp_line2_u8 = &sa_inp_0_u8[((2u * j_u32) + 2u) * inp_wid_u32];
        const unsigned char* restrict p_inp_line3_u8 = &sa_inp_0_u8[((2u * j_u32) + 3u) * inp_wid_u32];
        unsigned char* restrict p_out_u8 = &sa_out_1_u8[j_u32 * out_wid_u32];

        for (i_u32 = 0u; i_u32 < out_wid_u32; i_u32 += 8u)
        {
            uint8x16_t inp_line00_0_u8x16 = vld1q_u8(&p_inp_line0_u8[0]);
            uint8x16_t inp_line00_1_u8x16 = vld1q_u8(&p_inp_line0_u8[1]);
            uint8x16_t inp_line00_2_u8x16 = vld1q_u8(&p_inp_line0_u8[2]);
            uint8x16_t inp_line10_0_u8x16 = vld1q_u8(&p_inp_line1_u8[0]);
            uint8x16_t inp_line10_1_u8x16 = vld1q_u8(&p_inp_line1_u8[1]);
            uint8x16_t inp_line10_2_u8x16 = vld1q_u8(&p_inp_line1_u8[2]);
            uint8x16_t inp_line20_0_u8x16 = vld1q_u8(&p_inp_line2_u8[0]);
            uint8x16_t inp_line20_1_u8x16 = vld1q_u8(&p_inp_line2_u8[1]);
            uint8x16_t inp_line20_2_u8x16 = vld1q_u8(&p_inp_line2_u8[2]);
            uint8x16_t inp_line30_0_u8x16 = vld1q_u8(&p_inp_line3_u8[0]);
            uint8x16_t inp_line30_1_u8x16 = vld1q_u8(&p_inp_line3_u8[1]);
            uint8x16_t inp_line30_2_u8x16 = vld1q_u8(&p_inp_line3_u8[2]);

            p_inp_line0_u8 += 16u;
            p_inp_line1_u8 += 16u;
            p_inp_line2_u8 += 16u;
            p_inp_line3_u8 += 16u;

            uint8x8_t inp_line00_0_u8x8 = vget_low_u8(inp_line00_0_u8x16);
            uint8x8_t inp_line00_1_u8x8 = vget_low_u8(inp_line00_1_u8x16);
            uint8x8_t inp_line00_2_u8x8 = vget_low_u8(inp_line00_2_u8x16);
            uint8x8_t inp_line10_0_u8x8 = vget_low_u8(inp_line10_0_u8x16);
            uint8x8_t inp_line10_1_u8x8 = vget_low_u8(inp_line10_1_u8x16);
            uint8x8_t inp_line10_2_u8x8 = vget_low_u8(inp_line10_2_u8x16);
            uint8x8_t inp_line20_0_u8x8 = vget_low_u8(inp_line20_0_u8x16);
            uint8x8_t inp_line20_1_u8x8 = vget_low_u8(inp_line20_1_u8x16);
            uint8x8_t inp_line20_2_u8x8 = vget_low_u8(inp_line20_2_u8x16);
            uint8x8_t inp_line30_0_u8x8 = vget_low_u8(inp_line30_0_u8x16);
            uint8x8_t inp_line30_1_u8x8 = vget_low_u8(inp_line30_1_u8x16);
            uint8x8_t inp_line30_2_u8x8 = vget_low_u8(inp_line30_2_u8x16);
            uint8x8_t inp_line01_0_u8x8 = vget_high_u8(inp_line00_0_u8x16);
            uint8x8_t inp_line01_1_u8x8 = vget_high_u8(inp_line00_1_u8x16);
            uint8x8_t inp_line01_2_u8x8 = vget_high_u8(inp_line00_2_u8x16);
            uint8x8_t inp_line11_0_u8x8 = vget_high_u8(inp_line10_0_u8x16);
            uint8x8_t inp_line11_1_u8x8 = vget_high_u8(inp_line10_1_u8x16);
            uint8x8_t inp_line11_2_u8x8 = vget_high_u8(inp_line10_2_u8x16);
            uint8x8_t inp_line21_0_u8x8 = vget_high_u8(inp_line20_0_u8x16);
            uint8x8_t inp_line21_1_u8x8 = vget_high_u8(inp_line20_1_u8x16);
            uint8x8_t inp_line21_2_u8x8 = vget_high_u8(inp_line20_2_u8x16);
            uint8x8_t inp_line31_0_u8x8 = vget_high_u8(inp_line30_0_u8x16);
            uint8x8_t inp_line31_1_u8x8 = vget_high_u8(inp_line30_1_u8x16);
            uint8x8_t inp_line31_2_u8x8 = vget_high_u8(inp_line30_2_u8x16);

            inp_line0_0_u16x8 = vmovl_u8(inp_line00_0_u8x8);
            inp_line0_1_u16x8 = vmovl_u8(inp_line00_1_u8x8);
            inp_line0_2_u16x8 = vmovl_u8(inp_line00_2_u8x8);
            inp_line1_0_u16x8 = vmovl_u8(inp_line10_0_u8x8);
            inp_line1_1_u16x8 = vmovl_u8(inp_line10_1_u8x8);
            inp_line1_2_u16x8 = vmovl_u8(inp_line10_2_u8x8);
            inp_line2_0_u16x8 = vmovl_u8(inp_line20_0_u8x8);
            inp_line2_1_u16x8 = vmovl_u8(inp_line20_1_u8x8);
            inp_line2_2_u16x8 = vmovl_u8(inp_line20_2_u8x8);
            inp_line3_0_u16x8 = vmovl_u8(inp_line30_0_u8x8);
            inp_line3_1_u16x8 = vmovl_u8(inp_line30_1_u8x8);
            inp_line3_2_u16x8 = vmovl_u8(inp_line30_2_u8x8);
            inp_line0_0_s16x8 = vreinterpretq_s16_u16(inp_line0_0_u16x8);
            inp_line0_1_s16x8 = vreinterpretq_s16_u16(inp_line0_1_u16x8);
            inp_line0_2_s16x8 = vreinterpretq_s16_u16(inp_line0_2_u16x8);
            inp_line1_0_s16x8 = vreinterpretq_s16_u16(inp_line1_0_u16x8);
            inp_line1_1_s16x8 = vreinterpretq_s16_u16(inp_line1_1_u16x8);
            inp_line1_2_s16x8 = vreinterpretq_s16_u16(inp_line1_2_u16x8);
            inp_line2_0_s16x8 = vreinterpretq_s16_u16(inp_line2_0_u16x8);
            inp_line2_1_s16x8 = vreinterpretq_s16_u16(inp_line2_1_u16x8);
            inp_line2_2_s16x8 = vreinterpretq_s16_u16(inp_line2_2_u16x8);
            inp_line3_0_s16x8 = vreinterpretq_s16_u16(inp_line3_0_u16x8);
            inp_line3_1_s16x8 = vreinterpretq_s16_u16(inp_line3_1_u16x8);
            inp_line3_2_s16x8 = vreinterpretq_s16_u16(inp_line3_2_u16x8);
            inp_line0_0_h_s16x4 = vget_high_s16(inp_line0_0_s16x8);
            inp_line0_1_h_s16x4 = vget_high_s16(inp_line0_1_s16x8);
            inp_line0_2_h_s16x4 = vget_high_s16(inp_line0_2_s16x8);
            inp_line1_0_h_s16x4 = vget_high_s16(inp_line1_0_s16x8);
            inp_line1_1_h_s16x4 = vget_high_s16(inp_line1_1_s16x8);
            inp_line1_2_h_s16x4 = vget_high_s16(inp_line1_2_s16x8);
            inp_line2_0_h_s16x4 = vget_high_s16(inp_line2_0_s16x8);
            inp_line2_1_h_s16x4 = vget_high_s16(inp_line2_1_s16x8);
            inp_line2_2_h_s16x4 = vget_high_s16(inp_line2_2_s16x8);
            inp_line3_0_h_s16x4 = vget_high_s16(inp_line3_0_s16x8);
            inp_line3_1_h_s16x4 = vget_high_s16(inp_line3_1_s16x8);
            inp_line3_2_h_s16x4 = vget_high_s16(inp_line3_2_s16x8);
            inp_line0_0_l_s16x4 = vget_low_s16(inp_line0_0_s16x8);
            inp_line0_1_l_s16x4 = vget_low_s16(inp_line0_1_s16x8);
            inp_line0_2_l_s16x4 = vget_low_s16(inp_line0_2_s16x8);
            inp_line1_0_l_s16x4 = vget_low_s16(inp_line1_0_s16x8);
            inp_line1_1_l_s16x4 = vget_low_s16(inp_line1_1_s16x8);
            inp_line1_2_l_s16x4 = vget_low_s16(inp_line1_2_s16x8);
            inp_line2_0_l_s16x4 = vget_low_s16(inp_line2_0_s16x8);
            inp_line2_1_l_s16x4 = vget_low_s16(inp_line2_1_s16x8);
            inp_line2_2_l_s16x4 = vget_low_s16(inp_line2_2_s16x8);
            inp_line3_0_l_s16x4 = vget_low_s16(inp_line3_0_s16x8);
            inp_line3_1_l_s16x4 = vget_low_s16(inp_line3_1_s16x8);
            inp_line3_2_l_s16x4 = vget_low_s16(inp_line3_2_s16x8);
            inp_line0_0_h_s32x4 = vmovl_s16(inp_line0_0_h_s16x4);
            inp_line0_1_h_s32x4 = vmovl_s16(inp_line0_1_h_s16x4);
            inp_line0_2_h_s32x4 = vmovl_s16(inp_line0_2_h_s16x4);
            inp_line1_0_h_s32x4 = vmovl_s16(inp_line1_0_h_s16x4);
            inp_line1_1_h_s32x4 = vmovl_s16(inp_line1_1_h_s16x4);
            inp_line1_2_h_s32x4 = vmovl_s16(inp_line1_2_h_s16x4);
            inp_line2_0_h_s32x4 = vmovl_s16(inp_line2_0_h_s16x4);
            inp_line2_1_h_s32x4 = vmovl_s16(inp_line2_1_h_s16x4);
            inp_line2_2_h_s32x4 = vmovl_s16(inp_line2_2_h_s16x4);
            inp_line3_0_h_s32x4 = vmovl_s16(inp_line3_0_h_s16x4);
            inp_line3_1_h_s32x4 = vmovl_s16(inp_line3_1_h_s16x4);
            inp_line3_2_h_s32x4 = vmovl_s16(inp_line3_2_h_s16x4);
            inp_line0_0_l_s32x4 = vmovl_s16(inp_line0_0_l_s16x4);
            inp_line0_1_l_s32x4 = vmovl_s16(inp_line0_1_l_s16x4);
            inp_line0_2_l_s32x4 = vmovl_s16(inp_line0_2_l_s16x4);
            inp_line1_0_l_s32x4 = vmovl_s16(inp_line1_0_l_s16x4);
            inp_line1_1_l_s32x4 = vmovl_s16(inp_line1_1_l_s16x4);
            inp_line1_2_l_s32x4 = vmovl_s16(inp_line1_2_l_s16x4);
            inp_line2_0_l_s32x4 = vmovl_s16(inp_line2_0_l_s16x4);
            inp_line2_1_l_s32x4 = vmovl_s16(inp_line2_1_l_s16x4);
            inp_line2_2_l_s32x4 = vmovl_s16(inp_line2_2_l_s16x4);
            inp_line3_0_l_s32x4 = vmovl_s16(inp_line3_0_l_s16x4);
            inp_line3_1_l_s32x4 = vmovl_s16(inp_line3_1_l_s16x4);
            inp_line3_2_l_s32x4 = vmovl_s16(inp_line3_2_l_s16x4);
            sum_line0_h_s32x4 = vdupq_n_s32(0);
            sum_line1_h_s32x4 = vdupq_n_s32(0);
            sum_line0_l_s32x4 = vdupq_n_s32(0);
            sum_line1_l_s32x4 = vdupq_n_s32(0);

            sum_line0_h_s32x4 = vmlaq_n_s32(sum_line0_h_s32x4, inp_line0_0_h_s32x4, (signed int)a_ker_s16[0]); 
            sum_line0_h_s32x4 = vmlaq_n_s32(sum_line0_h_s32x4, inp_line0_1_h_s32x4, (signed int)a_ker_s16[1]); 
            sum_line0_h_s32x4 = vmlaq_n_s32(sum_line0_h_s32x4, inp_line0_2_h_s32x4, (signed int)a_ker_s16[2]); 
            sum_line0_h_s32x4 = vmlaq_n_s32(sum_line0_h_s32x4, inp_line1_0_h_s32x4, (signed int)a_ker_s16[3]); 
            sum_line0_h_s32x4 = vmlaq_n_s32(sum_line0_h_s32x4, inp_line1_1_h_s32x4, (signed int)a_ker_s16[4]); 
            sum_line0_h_s32x4 = vmlaq_n_s32(sum_line0_h_s32x4, inp_line1_2_h_s32x4, (signed int)a_ker_s16[5]); 
            sum_line0_h_s32x4 = vmlaq_n_s32(sum_line0_h_s32x4, inp_line2_0_h_s32x4, (signed int)a_ker_s16[6]); 
            sum_line0_h_s32x4 = vmlaq_n_s32(sum_line0_h_s32x4, inp_line2_1_h_s32x4, (signed int)a_ker_s16[7]); 
            sum_line0_h_s32x4 = vmlaq_n_s32(sum_line0_h_s32x4, inp_line2_2_h_s32x4, (signed int)a_ker_s16[8]); 
            sum_line1_h_s32x4 = vmlaq_n_s32(sum_line1_h_s32x4, inp_line1_0_h_s32x4, (signed int)a_ker_s16[0]); 
            sum_line1_h_s32x4 = vmlaq_n_s32(sum_line1_h_s32x4, inp_line1_1_h_s32x4, (signed int)a_ker_s16[1]); 
            sum_line1_h_s32x4 = vmlaq_n_s32(sum_line1_h_s32x4, inp_line1_2_h_s32x4, (signed int)a_ker_s16[2]); 
            sum_line1_h_s32x4 = vmlaq_n_s32(sum_line1_h_s32x4, inp_line2_0_h_s32x4, (signed int)a_ker_s16[3]); 
            sum_line1_h_s32x4 = vmlaq_n_s32(sum_line1_h_s32x4, inp_line2_1_h_s32x4, (signed int)a_ker_s16[4]); 
            sum_line1_h_s32x4 = vmlaq_n_s32(sum_line1_h_s32x4, inp_line2_2_h_s32x4, (signed int)a_ker_s16[5]); 
            sum_line1_h_s32x4 = vmlaq_n_s32(sum_line1_h_s32x4, inp_line3_0_h_s32x4, (signed int)a_ker_s16[6]); 
            sum_line1_h_s32x4 = vmlaq_n_s32(sum_line1_h_s32x4, inp_line3_1_h_s32x4, (signed int)a_ker_s16[7]); 
            sum_line1_h_s32x4 = vmlaq_n_s32(sum_line1_h_s32x4, inp_line3_2_h_s32x4, (signed int)a_ker_s16[8]); 
            sum_line0_l_s32x4 = vmlaq_n_s32(sum_line0_l_s32x4, inp_line0_0_l_s32x4, (signed int)a_ker_s16[0]); 
            sum_line0_l_s32x4 = vmlaq_n_s32(sum_line0_l_s32x4, inp_line0_1_l_s32x4, (signed int)a_ker_s16[1]); 
            sum_line0_l_s32x4 = vmlaq_n_s32(sum_line0_l_s32x4, inp_line0_2_l_s32x4, (signed int)a_ker_s16[2]); 
            sum_line0_l_s32x4 = vmlaq_n_s32(sum_line0_l_s32x4, inp_line1_0_l_s32x4, (signed int)a_ker_s16[3]); 
            sum_line0_l_s32x4 = vmlaq_n_s32(sum_line0_l_s32x4, inp_line1_1_l_s32x4, (signed int)a_ker_s16[4]); 
            sum_line0_l_s32x4 = vmlaq_n_s32(sum_line0_l_s32x4, inp_line1_2_l_s32x4, (signed int)a_ker_s16[5]); 
            sum_line0_l_s32x4 = vmlaq_n_s32(sum_line0_l_s32x4, inp_line2_0_l_s32x4, (signed int)a_ker_s16[6]); 
            sum_line0_l_s32x4 = vmlaq_n_s32(sum_line0_l_s32x4, inp_line2_1_l_s32x4, (signed int)a_ker_s16[7]); 
            sum_line0_l_s32x4 = vmlaq_n_s32(sum_line0_l_s32x4, inp_line2_2_l_s32x4, (signed int)a_ker_s16[8]); 
            sum_line1_l_s32x4 = vmlaq_n_s32(sum_line1_l_s32x4, inp_line1_0_l_s32x4, (signed int)a_ker_s16[0]); 
            sum_line1_l_s32x4 = vmlaq_n_s32(sum_line1_l_s32x4, inp_line1_1_l_s32x4, (signed int)a_ker_s16[1]); 
            sum_line1_l_s32x4 = vmlaq_n_s32(sum_line1_l_s32x4, inp_line1_2_l_s32x4, (signed int)a_ker_s16[2]); 
            sum_line1_l_s32x4 = vmlaq_n_s32(sum_line1_l_s32x4, inp_line2_0_l_s32x4, (signed int)a_ker_s16[3]); 
            sum_line1_l_s32x4 = vmlaq_n_s32(sum_line1_l_s32x4, inp_line2_1_l_s32x4, (signed int)a_ker_s16[4]); 
            sum_line1_l_s32x4 = vmlaq_n_s32(sum_line1_l_s32x4, inp_line2_2_l_s32x4, (signed int)a_ker_s16[5]); 
            sum_line1_l_s32x4 = vmlaq_n_s32(sum_line1_l_s32x4, inp_line3_0_l_s32x4, (signed int)a_ker_s16[6]); 
            sum_line1_l_s32x4 = vmlaq_n_s32(sum_line1_l_s32x4, inp_line3_1_l_s32x4, (signed int)a_ker_s16[7]); 
            sum_line1_l_s32x4 = vmlaq_n_s32(sum_line1_l_s32x4, inp_line3_2_l_s32x4, (signed int)a_ker_s16[8]); 

            sum_line0_h_h_s32x2 = vget_high_s32(sum_line0_h_s32x4); 
            sum_line1_h_h_s32x2 = vget_high_s32(sum_line1_h_s32x4); 
            sum_line0_l_h_s32x2 = vget_high_s32(sum_line0_l_s32x4); 
            sum_line1_l_h_s32x2 = vget_high_s32(sum_line1_l_s32x4); 
            sum_line0_h_l_s32x2 = vget_low_s32(sum_line0_h_s32x4); 
            sum_line1_h_l_s32x2 = vget_low_s32(sum_line1_h_s32x4); 
            sum_line0_l_l_s32x2 = vget_low_s32(sum_line0_l_s32x4); 
            sum_line1_l_l_s32x2 = vget_low_s32(sum_line1_l_s32x4);

            max_line0_h_s32x2 = vpmax_s32(sum_line0_h_l_s32x2, sum_line0_h_h_s32x2);
            max_line0_l_s32x2 = vpmax_s32(sum_line0_l_l_s32x2, sum_line0_l_h_s32x2);
            max_line1_h_s32x2 = vpmax_s32(sum_line1_h_l_s32x2, sum_line1_h_h_s32x2);
            max_line1_l_s32x2 = vpmax_s32(sum_line1_l_l_s32x2, sum_line1_l_h_s32x2);

            max_h_s32x2 = vmax_s32(max_line0_h_s32x2, max_line1_h_s32x2);
            max_l_s32x2 = vmax_s32(max_line0_l_s32x2, max_line1_l_s32x2);

            max_s32x4 = vcombine_s32(max_l_s32x2, max_h_s32x2);

            shift_s32x4 = vshrq_n_s32(max_s32x4, val_shift_s32);
            shift_s16x4 = vqmovn_s32(shift_s32x4);

            shift_s16x4 = vmax_s16(shift_s16x4, zero_s16x4);
            shift_s16x4 = vmin_s16(shift_s16x4, u8max_s16x4);

            uint16x4_t shift_0_u16x4 = vreinterpret_u16_s16(shift_s16x4);

            inp_line0_0_u16x8 = vmovl_u8(inp_line01_0_u8x8);
            inp_line0_1_u16x8 = vmovl_u8(inp_line01_1_u8x8);
            inp_line0_2_u16x8 = vmovl_u8(inp_line01_2_u8x8);
            inp_line1_0_u16x8 = vmovl_u8(inp_line11_0_u8x8);
            inp_line1_1_u16x8 = vmovl_u8(inp_line11_1_u8x8);
            inp_line1_2_u16x8 = vmovl_u8(inp_line11_2_u8x8);
            inp_line2_0_u16x8 = vmovl_u8(inp_line21_0_u8x8);
            inp_line2_1_u16x8 = vmovl_u8(inp_line21_1_u8x8);
            inp_line2_2_u16x8 = vmovl_u8(inp_line21_2_u8x8);
            inp_line3_0_u16x8 = vmovl_u8(inp_line31_0_u8x8);
            inp_line3_1_u16x8 = vmovl_u8(inp_line31_1_u8x8);
            inp_line3_2_u16x8 = vmovl_u8(inp_line31_2_u8x8);
            inp_line0_0_s16x8 = vreinterpretq_s16_u16(inp_line0_0_u16x8);
            inp_line0_1_s16x8 = vreinterpretq_s16_u16(inp_line0_1_u16x8);
            inp_line0_2_s16x8 = vreinterpretq_s16_u16(inp_line0_2_u16x8);
            inp_line1_0_s16x8 = vreinterpretq_s16_u16(inp_line1_0_u16x8);
            inp_line1_1_s16x8 = vreinterpretq_s16_u16(inp_line1_1_u16x8);
            inp_line1_2_s16x8 = vreinterpretq_s16_u16(inp_line1_2_u16x8);
            inp_line2_0_s16x8 = vreinterpretq_s16_u16(inp_line2_0_u16x8);
            inp_line2_1_s16x8 = vreinterpretq_s16_u16(inp_line2_1_u16x8);
            inp_line2_2_s16x8 = vreinterpretq_s16_u16(inp_line2_2_u16x8);
            inp_line3_0_s16x8 = vreinterpretq_s16_u16(inp_line3_0_u16x8);
            inp_line3_1_s16x8 = vreinterpretq_s16_u16(inp_line3_1_u16x8);
            inp_line3_2_s16x8 = vreinterpretq_s16_u16(inp_line3_2_u16x8);
            inp_line0_0_h_s16x4 = vget_high_s16(inp_line0_0_s16x8);
            inp_line0_1_h_s16x4 = vget_high_s16(inp_line0_1_s16x8);
            inp_line0_2_h_s16x4 = vget_high_s16(inp_line0_2_s16x8);
            inp_line1_0_h_s16x4 = vget_high_s16(inp_line1_0_s16x8);
            inp_line1_1_h_s16x4 = vget_high_s16(inp_line1_1_s16x8);
            inp_line1_2_h_s16x4 = vget_high_s16(inp_line1_2_s16x8);
            inp_line2_0_h_s16x4 = vget_high_s16(inp_line2_0_s16x8);
            inp_line2_1_h_s16x4 = vget_high_s16(inp_line2_1_s16x8);
            inp_line2_2_h_s16x4 = vget_high_s16(inp_line2_2_s16x8);
            inp_line3_0_h_s16x4 = vget_high_s16(inp_line3_0_s16x8);
            inp_line3_1_h_s16x4 = vget_high_s16(inp_line3_1_s16x8);
            inp_line3_2_h_s16x4 = vget_high_s16(inp_line3_2_s16x8);
            inp_line0_0_l_s16x4 = vget_low_s16(inp_line0_0_s16x8);
            inp_line0_1_l_s16x4 = vget_low_s16(inp_line0_1_s16x8);
            inp_line0_2_l_s16x4 = vget_low_s16(inp_line0_2_s16x8);
            inp_line1_0_l_s16x4 = vget_low_s16(inp_line1_0_s16x8);
            inp_line1_1_l_s16x4 = vget_low_s16(inp_line1_1_s16x8);
            inp_line1_2_l_s16x4 = vget_low_s16(inp_line1_2_s16x8);
            inp_line2_0_l_s16x4 = vget_low_s16(inp_line2_0_s16x8);
            inp_line2_1_l_s16x4 = vget_low_s16(inp_line2_1_s16x8);
            inp_line2_2_l_s16x4 = vget_low_s16(inp_line2_2_s16x8);
            inp_line3_0_l_s16x4 = vget_low_s16(inp_line3_0_s16x8);
            inp_line3_1_l_s16x4 = vget_low_s16(inp_line3_1_s16x8);
            inp_line3_2_l_s16x4 = vget_low_s16(inp_line3_2_s16x8);
            inp_line0_0_h_s32x4 = vmovl_s16(inp_line0_0_h_s16x4);
            inp_line0_1_h_s32x4 = vmovl_s16(inp_line0_1_h_s16x4);
            inp_line0_2_h_s32x4 = vmovl_s16(inp_line0_2_h_s16x4);
            inp_line1_0_h_s32x4 = vmovl_s16(inp_line1_0_h_s16x4);
            inp_line1_1_h_s32x4 = vmovl_s16(inp_line1_1_h_s16x4);
            inp_line1_2_h_s32x4 = vmovl_s16(inp_line1_2_h_s16x4);
            inp_line2_0_h_s32x4 = vmovl_s16(inp_line2_0_h_s16x4);
            inp_line2_1_h_s32x4 = vmovl_s16(inp_line2_1_h_s16x4);
            inp_line2_2_h_s32x4 = vmovl_s16(inp_line2_2_h_s16x4);
            inp_line3_0_h_s32x4 = vmovl_s16(inp_line3_0_h_s16x4);
            inp_line3_1_h_s32x4 = vmovl_s16(inp_line3_1_h_s16x4);
            inp_line3_2_h_s32x4 = vmovl_s16(inp_line3_2_h_s16x4);
            inp_line0_0_l_s32x4 = vmovl_s16(inp_line0_0_l_s16x4);
            inp_line0_1_l_s32x4 = vmovl_s16(inp_line0_1_l_s16x4);
            inp_line0_2_l_s32x4 = vmovl_s16(inp_line0_2_l_s16x4);
            inp_line1_0_l_s32x4 = vmovl_s16(inp_line1_0_l_s16x4);
            inp_line1_1_l_s32x4 = vmovl_s16(inp_line1_1_l_s16x4);
            inp_line1_2_l_s32x4 = vmovl_s16(inp_line1_2_l_s16x4);
            inp_line2_0_l_s32x4 = vmovl_s16(inp_line2_0_l_s16x4);
            inp_line2_1_l_s32x4 = vmovl_s16(inp_line2_1_l_s16x4);
            inp_line2_2_l_s32x4 = vmovl_s16(inp_line2_2_l_s16x4);
            inp_line3_0_l_s32x4 = vmovl_s16(inp_line3_0_l_s16x4);
            inp_line3_1_l_s32x4 = vmovl_s16(inp_line3_1_l_s16x4);
            inp_line3_2_l_s32x4 = vmovl_s16(inp_line3_2_l_s16x4);
            sum_line0_h_s32x4 = vdupq_n_s32(0);
            sum_line1_h_s32x4 = vdupq_n_s32(0);
            sum_line0_l_s32x4 = vdupq_n_s32(0);
            sum_line1_l_s32x4 = vdupq_n_s32(0);

            sum_line0_h_s32x4 = vmlaq_n_s32(sum_line0_h_s32x4, inp_line0_0_h_s32x4, (signed int)a_ker_s16[0]); 
            sum_line0_h_s32x4 = vmlaq_n_s32(sum_line0_h_s32x4, inp_line0_1_h_s32x4, (signed int)a_ker_s16[1]); 
            sum_line0_h_s32x4 = vmlaq_n_s32(sum_line0_h_s32x4, inp_line0_2_h_s32x4, (signed int)a_ker_s16[2]); 
            sum_line0_h_s32x4 = vmlaq_n_s32(sum_line0_h_s32x4, inp_line1_0_h_s32x4, (signed int)a_ker_s16[3]); 
            sum_line0_h_s32x4 = vmlaq_n_s32(sum_line0_h_s32x4, inp_line1_1_h_s32x4, (signed int)a_ker_s16[4]); 
            sum_line0_h_s32x4 = vmlaq_n_s32(sum_line0_h_s32x4, inp_line1_2_h_s32x4, (signed int)a_ker_s16[5]); 
            sum_line0_h_s32x4 = vmlaq_n_s32(sum_line0_h_s32x4, inp_line2_0_h_s32x4, (signed int)a_ker_s16[6]); 
            sum_line0_h_s32x4 = vmlaq_n_s32(sum_line0_h_s32x4, inp_line2_1_h_s32x4, (signed int)a_ker_s16[7]); 
            sum_line0_h_s32x4 = vmlaq_n_s32(sum_line0_h_s32x4, inp_line2_2_h_s32x4, (signed int)a_ker_s16[8]); 
            sum_line1_h_s32x4 = vmlaq_n_s32(sum_line1_h_s32x4, inp_line1_0_h_s32x4, (signed int)a_ker_s16[0]); 
            sum_line1_h_s32x4 = vmlaq_n_s32(sum_line1_h_s32x4, inp_line1_1_h_s32x4, (signed int)a_ker_s16[1]); 
            sum_line1_h_s32x4 = vmlaq_n_s32(sum_line1_h_s32x4, inp_line1_2_h_s32x4, (signed int)a_ker_s16[2]); 
            sum_line1_h_s32x4 = vmlaq_n_s32(sum_line1_h_s32x4, inp_line2_0_h_s32x4, (signed int)a_ker_s16[3]); 
            sum_line1_h_s32x4 = vmlaq_n_s32(sum_line1_h_s32x4, inp_line2_1_h_s32x4, (signed int)a_ker_s16[4]); 
            sum_line1_h_s32x4 = vmlaq_n_s32(sum_line1_h_s32x4, inp_line2_2_h_s32x4, (signed int)a_ker_s16[5]); 
            sum_line1_h_s32x4 = vmlaq_n_s32(sum_line1_h_s32x4, inp_line3_0_h_s32x4, (signed int)a_ker_s16[6]); 
            sum_line1_h_s32x4 = vmlaq_n_s32(sum_line1_h_s32x4, inp_line3_1_h_s32x4, (signed int)a_ker_s16[7]); 
            sum_line1_h_s32x4 = vmlaq_n_s32(sum_line1_h_s32x4, inp_line3_2_h_s32x4, (signed int)a_ker_s16[8]); 
            sum_line0_l_s32x4 = vmlaq_n_s32(sum_line0_l_s32x4, inp_line0_0_l_s32x4, (signed int)a_ker_s16[0]); 
            sum_line0_l_s32x4 = vmlaq_n_s32(sum_line0_l_s32x4, inp_line0_1_l_s32x4, (signed int)a_ker_s16[1]); 
            sum_line0_l_s32x4 = vmlaq_n_s32(sum_line0_l_s32x4, inp_line0_2_l_s32x4, (signed int)a_ker_s16[2]); 
            sum_line0_l_s32x4 = vmlaq_n_s32(sum_line0_l_s32x4, inp_line1_0_l_s32x4, (signed int)a_ker_s16[3]); 
            sum_line0_l_s32x4 = vmlaq_n_s32(sum_line0_l_s32x4, inp_line1_1_l_s32x4, (signed int)a_ker_s16[4]); 
            sum_line0_l_s32x4 = vmlaq_n_s32(sum_line0_l_s32x4, inp_line1_2_l_s32x4, (signed int)a_ker_s16[5]); 
            sum_line0_l_s32x4 = vmlaq_n_s32(sum_line0_l_s32x4, inp_line2_0_l_s32x4, (signed int)a_ker_s16[6]); 
            sum_line0_l_s32x4 = vmlaq_n_s32(sum_line0_l_s32x4, inp_line2_1_l_s32x4, (signed int)a_ker_s16[7]); 
            sum_line0_l_s32x4 = vmlaq_n_s32(sum_line0_l_s32x4, inp_line2_2_l_s32x4, (signed int)a_ker_s16[8]); 
            sum_line1_l_s32x4 = vmlaq_n_s32(sum_line1_l_s32x4, inp_line1_0_l_s32x4, (signed int)a_ker_s16[0]); 
            sum_line1_l_s32x4 = vmlaq_n_s32(sum_line1_l_s32x4, inp_line1_1_l_s32x4, (signed int)a_ker_s16[1]); 
            sum_line1_l_s32x4 = vmlaq_n_s32(sum_line1_l_s32x4, inp_line1_2_l_s32x4, (signed int)a_ker_s16[2]); 
            sum_line1_l_s32x4 = vmlaq_n_s32(sum_line1_l_s32x4, inp_line2_0_l_s32x4, (signed int)a_ker_s16[3]); 
            sum_line1_l_s32x4 = vmlaq_n_s32(sum_line1_l_s32x4, inp_line2_1_l_s32x4, (signed int)a_ker_s16[4]); 
            sum_line1_l_s32x4 = vmlaq_n_s32(sum_line1_l_s32x4, inp_line2_2_l_s32x4, (signed int)a_ker_s16[5]); 
            sum_line1_l_s32x4 = vmlaq_n_s32(sum_line1_l_s32x4, inp_line3_0_l_s32x4, (signed int)a_ker_s16[6]); 
            sum_line1_l_s32x4 = vmlaq_n_s32(sum_line1_l_s32x4, inp_line3_1_l_s32x4, (signed int)a_ker_s16[7]); 
            sum_line1_l_s32x4 = vmlaq_n_s32(sum_line1_l_s32x4, inp_line3_2_l_s32x4, (signed int)a_ker_s16[8]); 

            sum_line0_h_h_s32x2 = vget_high_s32(sum_line0_h_s32x4); 
            sum_line1_h_h_s32x2 = vget_high_s32(sum_line1_h_s32x4); 
            sum_line0_l_h_s32x2 = vget_high_s32(sum_line0_l_s32x4); 
            sum_line1_l_h_s32x2 = vget_high_s32(sum_line1_l_s32x4); 
            sum_line0_h_l_s32x2 = vget_low_s32(sum_line0_h_s32x4); 
            sum_line1_h_l_s32x2 = vget_low_s32(sum_line1_h_s32x4); 
            sum_line0_l_l_s32x2 = vget_low_s32(sum_line0_l_s32x4); 
            sum_line1_l_l_s32x2 = vget_low_s32(sum_line1_l_s32x4);

            max_line0_h_s32x2 = vpmax_s32(sum_line0_h_l_s32x2, sum_line0_h_h_s32x2);
            max_line0_l_s32x2 = vpmax_s32(sum_line0_l_l_s32x2, sum_line0_l_h_s32x2);
            max_line1_h_s32x2 = vpmax_s32(sum_line1_h_l_s32x2, sum_line1_h_h_s32x2);
            max_line1_l_s32x2 = vpmax_s32(sum_line1_l_l_s32x2, sum_line1_l_h_s32x2);

            max_h_s32x2 = vmax_s32(max_line0_h_s32x2, max_line1_h_s32x2);
            max_l_s32x2 = vmax_s32(max_line0_l_s32x2, max_line1_l_s32x2);

            max_s32x4 = vcombine_s32(max_l_s32x2, max_h_s32x2);

            shift_s32x4 = vshrq_n_s32(max_s32x4, val_shift_s32);
            shift_s16x4 = vqmovn_s32(shift_s32x4);

            shift_s16x4 = vmax_s16(shift_s16x4, zero_s16x4);
            shift_s16x4 = vmin_s16(shift_s16x4, u8max_s16x4);

            uint16x4_t shift_1_u16x4 = vreinterpret_u16_s16(shift_s16x4);

            uint16x8_t shift_u16x8 = vcombine_u16(shift_0_u16x4, shift_1_u16x4);
            uint8x8_t shift_u8x8 = vmovn_u16(shift_u16x8);

            vst1_u8(p_out_u8, shift_u8x8);
            p_out_u8 += 8u;
        }

        p_out_u8 -= 1u;
        *p_out_u8 = 0u;
    }
#elif 0
    for (j_u32 = 0u; j_u32 < (out_hei_u32 - 1u); j_u32++)
    {
        const unsigned char* restrict p_inp_line0_u8 = &sa_inp_0_u8[((2u * j_u32) + 0u) * inp_wid_u32];
        const unsigned char* restrict p_inp_line1_u8 = &sa_inp_0_u8[((2u * j_u32) + 1u) * inp_wid_u32];
        const unsigned char* restrict p_inp_line2_u8 = &sa_inp_0_u8[((2u * j_u32) + 2u) * inp_wid_u32];
        const unsigned char* restrict p_inp_line3_u8 = &sa_inp_0_u8[((2u * j_u32) + 3u) * inp_wid_u32];
        unsigned char* restrict p_out_u8 = &sa_out_1_u8[j_u32 * out_wid_u32];

        for (i_u32 = 0u; i_u32 < out_wid_u32; i_u32 += 8u)
        {
            uint8x8_t inp_line00_0_u8x8 = vld1_u8(&p_inp_line0_u8[0]);
            uint8x8_t inp_line00_1_u8x8 = vld1_u8(&p_inp_line0_u8[1]);
            uint8x8_t inp_line00_2_u8x8 = vld1_u8(&p_inp_line0_u8[2]);
            uint8x8_t inp_line10_0_u8x8 = vld1_u8(&p_inp_line1_u8[0]);
            uint8x8_t inp_line10_1_u8x8 = vld1_u8(&p_inp_line1_u8[1]);
            uint8x8_t inp_line10_2_u8x8 = vld1_u8(&p_inp_line1_u8[2]);
            uint8x8_t inp_line20_0_u8x8 = vld1_u8(&p_inp_line2_u8[0]);
            uint8x8_t inp_line20_1_u8x8 = vld1_u8(&p_inp_line2_u8[1]);
            uint8x8_t inp_line20_2_u8x8 = vld1_u8(&p_inp_line2_u8[2]);
            uint8x8_t inp_line30_0_u8x8 = vld1_u8(&p_inp_line3_u8[0]);
            uint8x8_t inp_line30_1_u8x8 = vld1_u8(&p_inp_line3_u8[1]);
            uint8x8_t inp_line30_2_u8x8 = vld1_u8(&p_inp_line3_u8[2]);

            p_inp_line0_u8 += 8u;
            p_inp_line1_u8 += 8u;
            p_inp_line2_u8 += 8u;
            p_inp_line3_u8 += 8u;

            uint16x8_t inp_line00_0_u16x8 = vmovl_u8(inp_line00_0_u8x8);
            uint16x8_t inp_line00_1_u16x8 = vmovl_u8(inp_line00_1_u8x8);
            uint16x8_t inp_line00_2_u16x8 = vmovl_u8(inp_line00_2_u8x8);
            uint16x8_t inp_line10_0_u16x8 = vmovl_u8(inp_line10_0_u8x8);
            uint16x8_t inp_line10_1_u16x8 = vmovl_u8(inp_line10_1_u8x8);
            uint16x8_t inp_line10_2_u16x8 = vmovl_u8(inp_line10_2_u8x8);
            uint16x8_t inp_line20_0_u16x8 = vmovl_u8(inp_line20_0_u8x8);
            uint16x8_t inp_line20_1_u16x8 = vmovl_u8(inp_line20_1_u8x8);
            uint16x8_t inp_line20_2_u16x8 = vmovl_u8(inp_line20_2_u8x8);
            uint16x8_t inp_line30_0_u16x8 = vmovl_u8(inp_line30_0_u8x8);
            uint16x8_t inp_line30_1_u16x8 = vmovl_u8(inp_line30_1_u8x8);
            uint16x8_t inp_line30_2_u16x8 = vmovl_u8(inp_line30_2_u8x8);
            int16x8_t inp_line00_0_s16x8 = vreinterpretq_s16_u16(inp_line00_0_u16x8);
            int16x8_t inp_line00_1_s16x8 = vreinterpretq_s16_u16(inp_line00_1_u16x8);
            int16x8_t inp_line00_2_s16x8 = vreinterpretq_s16_u16(inp_line00_2_u16x8);
            int16x8_t inp_line10_0_s16x8 = vreinterpretq_s16_u16(inp_line10_0_u16x8);
            int16x8_t inp_line10_1_s16x8 = vreinterpretq_s16_u16(inp_line10_1_u16x8);
            int16x8_t inp_line10_2_s16x8 = vreinterpretq_s16_u16(inp_line10_2_u16x8);
            int16x8_t inp_line20_0_s16x8 = vreinterpretq_s16_u16(inp_line20_0_u16x8);
            int16x8_t inp_line20_1_s16x8 = vreinterpretq_s16_u16(inp_line20_1_u16x8);
            int16x8_t inp_line20_2_s16x8 = vreinterpretq_s16_u16(inp_line20_2_u16x8);
            int16x8_t inp_line30_0_s16x8 = vreinterpretq_s16_u16(inp_line30_0_u16x8);
            int16x8_t inp_line30_1_s16x8 = vreinterpretq_s16_u16(inp_line30_1_u16x8);
            int16x8_t inp_line30_2_s16x8 = vreinterpretq_s16_u16(inp_line30_2_u16x8);
            int16x4_t inp_line00_0_h_s16x4 = vget_high_s16(inp_line00_0_s16x8);
            int16x4_t inp_line00_1_h_s16x4 = vget_high_s16(inp_line00_1_s16x8);
            int16x4_t inp_line00_2_h_s16x4 = vget_high_s16(inp_line00_2_s16x8);
            int16x4_t inp_line10_0_h_s16x4 = vget_high_s16(inp_line10_0_s16x8);
            int16x4_t inp_line10_1_h_s16x4 = vget_high_s16(inp_line10_1_s16x8);
            int16x4_t inp_line10_2_h_s16x4 = vget_high_s16(inp_line10_2_s16x8);
            int16x4_t inp_line20_0_h_s16x4 = vget_high_s16(inp_line20_0_s16x8);
            int16x4_t inp_line20_1_h_s16x4 = vget_high_s16(inp_line20_1_s16x8);
            int16x4_t inp_line20_2_h_s16x4 = vget_high_s16(inp_line20_2_s16x8);
            int16x4_t inp_line30_0_h_s16x4 = vget_high_s16(inp_line30_0_s16x8);
            int16x4_t inp_line30_1_h_s16x4 = vget_high_s16(inp_line30_1_s16x8);
            int16x4_t inp_line30_2_h_s16x4 = vget_high_s16(inp_line30_2_s16x8);
            int16x4_t inp_line00_0_l_s16x4 = vget_low_s16(inp_line00_0_s16x8);
            int16x4_t inp_line00_1_l_s16x4 = vget_low_s16(inp_line00_1_s16x8);
            int16x4_t inp_line00_2_l_s16x4 = vget_low_s16(inp_line00_2_s16x8);
            int16x4_t inp_line10_0_l_s16x4 = vget_low_s16(inp_line10_0_s16x8);
            int16x4_t inp_line10_1_l_s16x4 = vget_low_s16(inp_line10_1_s16x8);
            int16x4_t inp_line10_2_l_s16x4 = vget_low_s16(inp_line10_2_s16x8);
            int16x4_t inp_line20_0_l_s16x4 = vget_low_s16(inp_line20_0_s16x8);
            int16x4_t inp_line20_1_l_s16x4 = vget_low_s16(inp_line20_1_s16x8);
            int16x4_t inp_line20_2_l_s16x4 = vget_low_s16(inp_line20_2_s16x8);
            int16x4_t inp_line30_0_l_s16x4 = vget_low_s16(inp_line30_0_s16x8);
            int16x4_t inp_line30_1_l_s16x4 = vget_low_s16(inp_line30_1_s16x8);
            int16x4_t inp_line30_2_l_s16x4 = vget_low_s16(inp_line30_2_s16x8);
            int32x4_t inp_line00_0_h_s32x4 = vmovl_s16(inp_line00_0_h_s16x4);
            int32x4_t inp_line00_1_h_s32x4 = vmovl_s16(inp_line00_1_h_s16x4);
            int32x4_t inp_line00_2_h_s32x4 = vmovl_s16(inp_line00_2_h_s16x4);
            int32x4_t inp_line10_0_h_s32x4 = vmovl_s16(inp_line10_0_h_s16x4);
            int32x4_t inp_line10_1_h_s32x4 = vmovl_s16(inp_line10_1_h_s16x4);
            int32x4_t inp_line10_2_h_s32x4 = vmovl_s16(inp_line10_2_h_s16x4);
            int32x4_t inp_line20_0_h_s32x4 = vmovl_s16(inp_line20_0_h_s16x4);
            int32x4_t inp_line20_1_h_s32x4 = vmovl_s16(inp_line20_1_h_s16x4);
            int32x4_t inp_line20_2_h_s32x4 = vmovl_s16(inp_line20_2_h_s16x4);
            int32x4_t inp_line30_0_h_s32x4 = vmovl_s16(inp_line30_0_h_s16x4);
            int32x4_t inp_line30_1_h_s32x4 = vmovl_s16(inp_line30_1_h_s16x4);
            int32x4_t inp_line30_2_h_s32x4 = vmovl_s16(inp_line30_2_h_s16x4);
            int32x4_t inp_line00_0_l_s32x4 = vmovl_s16(inp_line00_0_l_s16x4);
            int32x4_t inp_line00_1_l_s32x4 = vmovl_s16(inp_line00_1_l_s16x4);
            int32x4_t inp_line00_2_l_s32x4 = vmovl_s16(inp_line00_2_l_s16x4);
            int32x4_t inp_line10_0_l_s32x4 = vmovl_s16(inp_line10_0_l_s16x4);
            int32x4_t inp_line10_1_l_s32x4 = vmovl_s16(inp_line10_1_l_s16x4);
            int32x4_t inp_line10_2_l_s32x4 = vmovl_s16(inp_line10_2_l_s16x4);
            int32x4_t inp_line20_0_l_s32x4 = vmovl_s16(inp_line20_0_l_s16x4);
            int32x4_t inp_line20_1_l_s32x4 = vmovl_s16(inp_line20_1_l_s16x4);
            int32x4_t inp_line20_2_l_s32x4 = vmovl_s16(inp_line20_2_l_s16x4);
            int32x4_t inp_line30_0_l_s32x4 = vmovl_s16(inp_line30_0_l_s16x4);
            int32x4_t inp_line30_1_l_s32x4 = vmovl_s16(inp_line30_1_l_s16x4);
            int32x4_t inp_line30_2_l_s32x4 = vmovl_s16(inp_line30_2_l_s16x4);
            int32x4_t sum_line00_h_s32x4 = vdupq_n_s32(0);
            int32x4_t sum_line10_h_s32x4 = vdupq_n_s32(0);
            int32x4_t sum_line00_l_s32x4 = vdupq_n_s32(0);
            int32x4_t sum_line10_l_s32x4 = vdupq_n_s32(0);

            sum_line00_h_s32x4 = vmlaq_n_s32(sum_line00_h_s32x4, inp_line00_0_h_s32x4, (signed int)a_ker_s16[0]); 
            sum_line00_h_s32x4 = vmlaq_n_s32(sum_line00_h_s32x4, inp_line00_1_h_s32x4, (signed int)a_ker_s16[1]); 
            sum_line00_h_s32x4 = vmlaq_n_s32(sum_line00_h_s32x4, inp_line00_2_h_s32x4, (signed int)a_ker_s16[2]); 
            sum_line00_h_s32x4 = vmlaq_n_s32(sum_line00_h_s32x4, inp_line10_0_h_s32x4, (signed int)a_ker_s16[3]); 
            sum_line00_h_s32x4 = vmlaq_n_s32(sum_line00_h_s32x4, inp_line10_1_h_s32x4, (signed int)a_ker_s16[4]); 
            sum_line00_h_s32x4 = vmlaq_n_s32(sum_line00_h_s32x4, inp_line10_2_h_s32x4, (signed int)a_ker_s16[5]); 
            sum_line00_h_s32x4 = vmlaq_n_s32(sum_line00_h_s32x4, inp_line20_0_h_s32x4, (signed int)a_ker_s16[6]); 
            sum_line00_h_s32x4 = vmlaq_n_s32(sum_line00_h_s32x4, inp_line20_1_h_s32x4, (signed int)a_ker_s16[7]); 
            sum_line00_h_s32x4 = vmlaq_n_s32(sum_line00_h_s32x4, inp_line20_2_h_s32x4, (signed int)a_ker_s16[8]); 
            sum_line10_h_s32x4 = vmlaq_n_s32(sum_line10_h_s32x4, inp_line10_0_h_s32x4, (signed int)a_ker_s16[0]); 
            sum_line10_h_s32x4 = vmlaq_n_s32(sum_line10_h_s32x4, inp_line10_1_h_s32x4, (signed int)a_ker_s16[1]); 
            sum_line10_h_s32x4 = vmlaq_n_s32(sum_line10_h_s32x4, inp_line10_2_h_s32x4, (signed int)a_ker_s16[2]); 
            sum_line10_h_s32x4 = vmlaq_n_s32(sum_line10_h_s32x4, inp_line20_0_h_s32x4, (signed int)a_ker_s16[3]); 
            sum_line10_h_s32x4 = vmlaq_n_s32(sum_line10_h_s32x4, inp_line20_1_h_s32x4, (signed int)a_ker_s16[4]); 
            sum_line10_h_s32x4 = vmlaq_n_s32(sum_line10_h_s32x4, inp_line20_2_h_s32x4, (signed int)a_ker_s16[5]); 
            sum_line10_h_s32x4 = vmlaq_n_s32(sum_line10_h_s32x4, inp_line30_0_h_s32x4, (signed int)a_ker_s16[6]); 
            sum_line10_h_s32x4 = vmlaq_n_s32(sum_line10_h_s32x4, inp_line30_1_h_s32x4, (signed int)a_ker_s16[7]); 
            sum_line10_h_s32x4 = vmlaq_n_s32(sum_line10_h_s32x4, inp_line30_2_h_s32x4, (signed int)a_ker_s16[8]); 
            sum_line00_l_s32x4 = vmlaq_n_s32(sum_line00_l_s32x4, inp_line00_0_l_s32x4, (signed int)a_ker_s16[0]); 
            sum_line00_l_s32x4 = vmlaq_n_s32(sum_line00_l_s32x4, inp_line00_1_l_s32x4, (signed int)a_ker_s16[1]); 
            sum_line00_l_s32x4 = vmlaq_n_s32(sum_line00_l_s32x4, inp_line00_2_l_s32x4, (signed int)a_ker_s16[2]); 
            sum_line00_l_s32x4 = vmlaq_n_s32(sum_line00_l_s32x4, inp_line10_0_l_s32x4, (signed int)a_ker_s16[3]); 
            sum_line00_l_s32x4 = vmlaq_n_s32(sum_line00_l_s32x4, inp_line10_1_l_s32x4, (signed int)a_ker_s16[4]); 
            sum_line00_l_s32x4 = vmlaq_n_s32(sum_line00_l_s32x4, inp_line10_2_l_s32x4, (signed int)a_ker_s16[5]); 
            sum_line00_l_s32x4 = vmlaq_n_s32(sum_line00_l_s32x4, inp_line20_0_l_s32x4, (signed int)a_ker_s16[6]); 
            sum_line00_l_s32x4 = vmlaq_n_s32(sum_line00_l_s32x4, inp_line20_1_l_s32x4, (signed int)a_ker_s16[7]); 
            sum_line00_l_s32x4 = vmlaq_n_s32(sum_line00_l_s32x4, inp_line20_2_l_s32x4, (signed int)a_ker_s16[8]); 
            sum_line10_l_s32x4 = vmlaq_n_s32(sum_line10_l_s32x4, inp_line10_0_l_s32x4, (signed int)a_ker_s16[0]); 
            sum_line10_l_s32x4 = vmlaq_n_s32(sum_line10_l_s32x4, inp_line10_1_l_s32x4, (signed int)a_ker_s16[1]); 
            sum_line10_l_s32x4 = vmlaq_n_s32(sum_line10_l_s32x4, inp_line10_2_l_s32x4, (signed int)a_ker_s16[2]); 
            sum_line10_l_s32x4 = vmlaq_n_s32(sum_line10_l_s32x4, inp_line20_0_l_s32x4, (signed int)a_ker_s16[3]); 
            sum_line10_l_s32x4 = vmlaq_n_s32(sum_line10_l_s32x4, inp_line20_1_l_s32x4, (signed int)a_ker_s16[4]); 
            sum_line10_l_s32x4 = vmlaq_n_s32(sum_line10_l_s32x4, inp_line20_2_l_s32x4, (signed int)a_ker_s16[5]); 
            sum_line10_l_s32x4 = vmlaq_n_s32(sum_line10_l_s32x4, inp_line30_0_l_s32x4, (signed int)a_ker_s16[6]); 
            sum_line10_l_s32x4 = vmlaq_n_s32(sum_line10_l_s32x4, inp_line30_1_l_s32x4, (signed int)a_ker_s16[7]); 
            sum_line10_l_s32x4 = vmlaq_n_s32(sum_line10_l_s32x4, inp_line30_2_l_s32x4, (signed int)a_ker_s16[8]); 

            int32x2_t sum_line00_h_h_s32x2 = vget_high_s32(sum_line00_h_s32x4); 
            int32x2_t sum_line10_h_h_s32x2 = vget_high_s32(sum_line10_h_s32x4); 
            int32x2_t sum_line00_l_h_s32x2 = vget_high_s32(sum_line00_l_s32x4); 
            int32x2_t sum_line10_l_h_s32x2 = vget_high_s32(sum_line10_l_s32x4); 
            int32x2_t sum_line00_h_l_s32x2 = vget_low_s32(sum_line00_h_s32x4); 
            int32x2_t sum_line10_h_l_s32x2 = vget_low_s32(sum_line10_h_s32x4); 
            int32x2_t sum_line00_l_l_s32x2 = vget_low_s32(sum_line00_l_s32x4); 
            int32x2_t sum_line10_l_l_s32x2 = vget_low_s32(sum_line10_l_s32x4);

            int32x2_t max_line00_h_s32x2 = vpmax_s32(sum_line00_h_l_s32x2, sum_line00_h_h_s32x2);
            int32x2_t max_line00_l_s32x2 = vpmax_s32(sum_line00_l_l_s32x2, sum_line00_l_h_s32x2);
            int32x2_t max_line10_h_s32x2 = vpmax_s32(sum_line10_h_l_s32x2, sum_line10_h_h_s32x2);
            int32x2_t max_line10_l_s32x2 = vpmax_s32(sum_line10_l_l_s32x2, sum_line10_l_h_s32x2);

            int32x2_t max_0_h_s32x2 = vmax_s32(max_line00_h_s32x2, max_line10_h_s32x2);
            int32x2_t max_0_l_s32x2 = vmax_s32(max_line00_l_s32x2, max_line10_l_s32x2);

            int32x4_t max_0_s32x4 = vcombine_s32(max_0_l_s32x2, max_0_h_s32x2);

            int32x4_t shift_0_s32x4 = vshrq_n_s32(max_0_s32x4, val_shift_s32);
            int16x4_t shift_0_s16x4 = vqmovn_s32(shift_0_s32x4);

            int16x4_t shift_zero_0_s16x4 = vmax_s16(shift_0_s16x4, zero_s16x4);
            int16x4_t shift_255_0_s16x4 = vmin_s16(shift_zero_0_s16x4, u8max_s16x4);

            uint16x4_t shift_0_u16x4 = vreinterpret_u16_s16(shift_255_0_s16x4);

            uint8x8_t inp_line01_0_u8x8 = vld1_u8(&p_inp_line0_u8[0]);
            uint8x8_t inp_line01_1_u8x8 = vld1_u8(&p_inp_line0_u8[1]);
            uint8x8_t inp_line01_2_u8x8 = vld1_u8(&p_inp_line0_u8[2]);
            uint8x8_t inp_line11_0_u8x8 = vld1_u8(&p_inp_line1_u8[0]);
            uint8x8_t inp_line11_1_u8x8 = vld1_u8(&p_inp_line1_u8[1]);
            uint8x8_t inp_line11_2_u8x8 = vld1_u8(&p_inp_line1_u8[2]);
            uint8x8_t inp_line21_0_u8x8 = vld1_u8(&p_inp_line2_u8[0]);
            uint8x8_t inp_line21_1_u8x8 = vld1_u8(&p_inp_line2_u8[1]);
            uint8x8_t inp_line21_2_u8x8 = vld1_u8(&p_inp_line2_u8[2]);
            uint8x8_t inp_line31_0_u8x8 = vld1_u8(&p_inp_line3_u8[0]);
            uint8x8_t inp_line31_1_u8x8 = vld1_u8(&p_inp_line3_u8[1]);
            uint8x8_t inp_line31_2_u8x8 = vld1_u8(&p_inp_line3_u8[2]);
            
            p_inp_line0_u8 += 8u;
            p_inp_line1_u8 += 8u;
            p_inp_line2_u8 += 8u;
            p_inp_line3_u8 += 8u;

            uint16x8_t inp_line01_0_u16x8 = vmovl_u8(inp_line01_0_u8x8);
            uint16x8_t inp_line01_1_u16x8 = vmovl_u8(inp_line01_1_u8x8);
            uint16x8_t inp_line01_2_u16x8 = vmovl_u8(inp_line01_2_u8x8);
            uint16x8_t inp_line11_0_u16x8 = vmovl_u8(inp_line11_0_u8x8);
            uint16x8_t inp_line11_1_u16x8 = vmovl_u8(inp_line11_1_u8x8);
            uint16x8_t inp_line11_2_u16x8 = vmovl_u8(inp_line11_2_u8x8);
            uint16x8_t inp_line21_0_u16x8 = vmovl_u8(inp_line21_0_u8x8);
            uint16x8_t inp_line21_1_u16x8 = vmovl_u8(inp_line21_1_u8x8);
            uint16x8_t inp_line21_2_u16x8 = vmovl_u8(inp_line21_2_u8x8);
            uint16x8_t inp_line31_0_u16x8 = vmovl_u8(inp_line31_0_u8x8);
            uint16x8_t inp_line31_1_u16x8 = vmovl_u8(inp_line31_1_u8x8);
            uint16x8_t inp_line31_2_u16x8 = vmovl_u8(inp_line31_2_u8x8);
            int16x8_t inp_line01_0_s16x8 = vreinterpretq_s16_u16(inp_line01_0_u16x8);
            int16x8_t inp_line01_1_s16x8 = vreinterpretq_s16_u16(inp_line01_1_u16x8);
            int16x8_t inp_line01_2_s16x8 = vreinterpretq_s16_u16(inp_line01_2_u16x8);
            int16x8_t inp_line11_0_s16x8 = vreinterpretq_s16_u16(inp_line11_0_u16x8);
            int16x8_t inp_line11_1_s16x8 = vreinterpretq_s16_u16(inp_line11_1_u16x8);
            int16x8_t inp_line11_2_s16x8 = vreinterpretq_s16_u16(inp_line11_2_u16x8);
            int16x8_t inp_line21_0_s16x8 = vreinterpretq_s16_u16(inp_line21_0_u16x8);
            int16x8_t inp_line21_1_s16x8 = vreinterpretq_s16_u16(inp_line21_1_u16x8);
            int16x8_t inp_line21_2_s16x8 = vreinterpretq_s16_u16(inp_line21_2_u16x8);
            int16x8_t inp_line31_0_s16x8 = vreinterpretq_s16_u16(inp_line31_0_u16x8);
            int16x8_t inp_line31_1_s16x8 = vreinterpretq_s16_u16(inp_line31_1_u16x8);
            int16x8_t inp_line31_2_s16x8 = vreinterpretq_s16_u16(inp_line31_2_u16x8);
            int16x4_t inp_line01_0_h_s16x4 = vget_high_s16(inp_line01_0_s16x8);
            int16x4_t inp_line01_1_h_s16x4 = vget_high_s16(inp_line01_1_s16x8);
            int16x4_t inp_line01_2_h_s16x4 = vget_high_s16(inp_line01_2_s16x8);
            int16x4_t inp_line11_0_h_s16x4 = vget_high_s16(inp_line11_0_s16x8);
            int16x4_t inp_line11_1_h_s16x4 = vget_high_s16(inp_line11_1_s16x8);
            int16x4_t inp_line11_2_h_s16x4 = vget_high_s16(inp_line11_2_s16x8);
            int16x4_t inp_line21_0_h_s16x4 = vget_high_s16(inp_line21_0_s16x8);
            int16x4_t inp_line21_1_h_s16x4 = vget_high_s16(inp_line21_1_s16x8);
            int16x4_t inp_line21_2_h_s16x4 = vget_high_s16(inp_line21_2_s16x8);
            int16x4_t inp_line31_0_h_s16x4 = vget_high_s16(inp_line31_0_s16x8);
            int16x4_t inp_line31_1_h_s16x4 = vget_high_s16(inp_line31_1_s16x8);
            int16x4_t inp_line31_2_h_s16x4 = vget_high_s16(inp_line31_2_s16x8);
            int16x4_t inp_line01_0_l_s16x4 = vget_low_s16(inp_line01_0_s16x8);
            int16x4_t inp_line01_1_l_s16x4 = vget_low_s16(inp_line01_1_s16x8);
            int16x4_t inp_line01_2_l_s16x4 = vget_low_s16(inp_line01_2_s16x8);
            int16x4_t inp_line11_0_l_s16x4 = vget_low_s16(inp_line11_0_s16x8);
            int16x4_t inp_line11_1_l_s16x4 = vget_low_s16(inp_line11_1_s16x8);
            int16x4_t inp_line11_2_l_s16x4 = vget_low_s16(inp_line11_2_s16x8);
            int16x4_t inp_line21_0_l_s16x4 = vget_low_s16(inp_line21_0_s16x8);
            int16x4_t inp_line21_1_l_s16x4 = vget_low_s16(inp_line21_1_s16x8);
            int16x4_t inp_line21_2_l_s16x4 = vget_low_s16(inp_line21_2_s16x8);
            int16x4_t inp_line31_0_l_s16x4 = vget_low_s16(inp_line31_0_s16x8);
            int16x4_t inp_line31_1_l_s16x4 = vget_low_s16(inp_line31_1_s16x8);
            int16x4_t inp_line31_2_l_s16x4 = vget_low_s16(inp_line31_2_s16x8);
            int32x4_t inp_line01_0_h_s32x4 = vmovl_s16(inp_line01_0_h_s16x4);
            int32x4_t inp_line01_1_h_s32x4 = vmovl_s16(inp_line01_1_h_s16x4);
            int32x4_t inp_line01_2_h_s32x4 = vmovl_s16(inp_line01_2_h_s16x4);
            int32x4_t inp_line11_0_h_s32x4 = vmovl_s16(inp_line11_0_h_s16x4);
            int32x4_t inp_line11_1_h_s32x4 = vmovl_s16(inp_line11_1_h_s16x4);
            int32x4_t inp_line11_2_h_s32x4 = vmovl_s16(inp_line11_2_h_s16x4);
            int32x4_t inp_line21_0_h_s32x4 = vmovl_s16(inp_line21_0_h_s16x4);
            int32x4_t inp_line21_1_h_s32x4 = vmovl_s16(inp_line21_1_h_s16x4);
            int32x4_t inp_line21_2_h_s32x4 = vmovl_s16(inp_line21_2_h_s16x4);
            int32x4_t inp_line31_0_h_s32x4 = vmovl_s16(inp_line31_0_h_s16x4);
            int32x4_t inp_line31_1_h_s32x4 = vmovl_s16(inp_line31_1_h_s16x4);
            int32x4_t inp_line31_2_h_s32x4 = vmovl_s16(inp_line31_2_h_s16x4);
            int32x4_t inp_line01_0_l_s32x4 = vmovl_s16(inp_line01_0_l_s16x4);
            int32x4_t inp_line01_1_l_s32x4 = vmovl_s16(inp_line01_1_l_s16x4);
            int32x4_t inp_line01_2_l_s32x4 = vmovl_s16(inp_line01_2_l_s16x4);
            int32x4_t inp_line11_0_l_s32x4 = vmovl_s16(inp_line11_0_l_s16x4);
            int32x4_t inp_line11_1_l_s32x4 = vmovl_s16(inp_line11_1_l_s16x4);
            int32x4_t inp_line11_2_l_s32x4 = vmovl_s16(inp_line11_2_l_s16x4);
            int32x4_t inp_line21_0_l_s32x4 = vmovl_s16(inp_line21_0_l_s16x4);
            int32x4_t inp_line21_1_l_s32x4 = vmovl_s16(inp_line21_1_l_s16x4);
            int32x4_t inp_line21_2_l_s32x4 = vmovl_s16(inp_line21_2_l_s16x4);
            int32x4_t inp_line31_0_l_s32x4 = vmovl_s16(inp_line31_0_l_s16x4);
            int32x4_t inp_line31_1_l_s32x4 = vmovl_s16(inp_line31_1_l_s16x4);
            int32x4_t inp_line31_2_l_s32x4 = vmovl_s16(inp_line31_2_l_s16x4);
            int32x4_t sum_line01_h_s32x4 = vdupq_n_s32(0);
            int32x4_t sum_line11_h_s32x4 = vdupq_n_s32(0);
            int32x4_t sum_line01_l_s32x4 = vdupq_n_s32(0);
            int32x4_t sum_line11_l_s32x4 = vdupq_n_s32(0);

            sum_line01_h_s32x4 = vmlaq_n_s32(sum_line01_h_s32x4, inp_line01_0_h_s32x4, (signed int)a_ker_s16[0]); 
            sum_line01_h_s32x4 = vmlaq_n_s32(sum_line01_h_s32x4, inp_line01_1_h_s32x4, (signed int)a_ker_s16[1]); 
            sum_line01_h_s32x4 = vmlaq_n_s32(sum_line01_h_s32x4, inp_line01_2_h_s32x4, (signed int)a_ker_s16[2]); 
            sum_line01_h_s32x4 = vmlaq_n_s32(sum_line01_h_s32x4, inp_line11_0_h_s32x4, (signed int)a_ker_s16[3]); 
            sum_line01_h_s32x4 = vmlaq_n_s32(sum_line01_h_s32x4, inp_line11_1_h_s32x4, (signed int)a_ker_s16[4]); 
            sum_line01_h_s32x4 = vmlaq_n_s32(sum_line01_h_s32x4, inp_line11_2_h_s32x4, (signed int)a_ker_s16[5]); 
            sum_line01_h_s32x4 = vmlaq_n_s32(sum_line01_h_s32x4, inp_line21_0_h_s32x4, (signed int)a_ker_s16[6]); 
            sum_line01_h_s32x4 = vmlaq_n_s32(sum_line01_h_s32x4, inp_line21_1_h_s32x4, (signed int)a_ker_s16[7]); 
            sum_line01_h_s32x4 = vmlaq_n_s32(sum_line01_h_s32x4, inp_line21_2_h_s32x4, (signed int)a_ker_s16[8]); 
            sum_line11_h_s32x4 = vmlaq_n_s32(sum_line11_h_s32x4, inp_line11_0_h_s32x4, (signed int)a_ker_s16[0]); 
            sum_line11_h_s32x4 = vmlaq_n_s32(sum_line11_h_s32x4, inp_line11_1_h_s32x4, (signed int)a_ker_s16[1]); 
            sum_line11_h_s32x4 = vmlaq_n_s32(sum_line11_h_s32x4, inp_line11_2_h_s32x4, (signed int)a_ker_s16[2]); 
            sum_line11_h_s32x4 = vmlaq_n_s32(sum_line11_h_s32x4, inp_line21_0_h_s32x4, (signed int)a_ker_s16[3]); 
            sum_line11_h_s32x4 = vmlaq_n_s32(sum_line11_h_s32x4, inp_line21_1_h_s32x4, (signed int)a_ker_s16[4]); 
            sum_line11_h_s32x4 = vmlaq_n_s32(sum_line11_h_s32x4, inp_line21_2_h_s32x4, (signed int)a_ker_s16[5]); 
            sum_line11_h_s32x4 = vmlaq_n_s32(sum_line11_h_s32x4, inp_line31_0_h_s32x4, (signed int)a_ker_s16[6]); 
            sum_line11_h_s32x4 = vmlaq_n_s32(sum_line11_h_s32x4, inp_line31_1_h_s32x4, (signed int)a_ker_s16[7]); 
            sum_line11_h_s32x4 = vmlaq_n_s32(sum_line11_h_s32x4, inp_line31_2_h_s32x4, (signed int)a_ker_s16[8]); 
            sum_line01_l_s32x4 = vmlaq_n_s32(sum_line01_l_s32x4, inp_line01_0_l_s32x4, (signed int)a_ker_s16[0]); 
            sum_line01_l_s32x4 = vmlaq_n_s32(sum_line01_l_s32x4, inp_line01_1_l_s32x4, (signed int)a_ker_s16[1]); 
            sum_line01_l_s32x4 = vmlaq_n_s32(sum_line01_l_s32x4, inp_line01_2_l_s32x4, (signed int)a_ker_s16[2]); 
            sum_line01_l_s32x4 = vmlaq_n_s32(sum_line01_l_s32x4, inp_line11_0_l_s32x4, (signed int)a_ker_s16[3]); 
            sum_line01_l_s32x4 = vmlaq_n_s32(sum_line01_l_s32x4, inp_line11_1_l_s32x4, (signed int)a_ker_s16[4]); 
            sum_line01_l_s32x4 = vmlaq_n_s32(sum_line01_l_s32x4, inp_line11_2_l_s32x4, (signed int)a_ker_s16[5]); 
            sum_line01_l_s32x4 = vmlaq_n_s32(sum_line01_l_s32x4, inp_line21_0_l_s32x4, (signed int)a_ker_s16[6]); 
            sum_line01_l_s32x4 = vmlaq_n_s32(sum_line01_l_s32x4, inp_line21_1_l_s32x4, (signed int)a_ker_s16[7]); 
            sum_line01_l_s32x4 = vmlaq_n_s32(sum_line01_l_s32x4, inp_line21_2_l_s32x4, (signed int)a_ker_s16[8]); 
            sum_line11_l_s32x4 = vmlaq_n_s32(sum_line11_l_s32x4, inp_line11_0_l_s32x4, (signed int)a_ker_s16[0]); 
            sum_line11_l_s32x4 = vmlaq_n_s32(sum_line11_l_s32x4, inp_line11_1_l_s32x4, (signed int)a_ker_s16[1]); 
            sum_line11_l_s32x4 = vmlaq_n_s32(sum_line11_l_s32x4, inp_line11_2_l_s32x4, (signed int)a_ker_s16[2]); 
            sum_line11_l_s32x4 = vmlaq_n_s32(sum_line11_l_s32x4, inp_line21_0_l_s32x4, (signed int)a_ker_s16[3]); 
            sum_line11_l_s32x4 = vmlaq_n_s32(sum_line11_l_s32x4, inp_line21_1_l_s32x4, (signed int)a_ker_s16[4]); 
            sum_line11_l_s32x4 = vmlaq_n_s32(sum_line11_l_s32x4, inp_line21_2_l_s32x4, (signed int)a_ker_s16[5]); 
            sum_line11_l_s32x4 = vmlaq_n_s32(sum_line11_l_s32x4, inp_line31_0_l_s32x4, (signed int)a_ker_s16[6]); 
            sum_line11_l_s32x4 = vmlaq_n_s32(sum_line11_l_s32x4, inp_line31_1_l_s32x4, (signed int)a_ker_s16[7]); 
            sum_line11_l_s32x4 = vmlaq_n_s32(sum_line11_l_s32x4, inp_line31_2_l_s32x4, (signed int)a_ker_s16[8]); 

            int32x2_t sum_line01_h_h_s32x2 = vget_high_s32(sum_line01_h_s32x4); 
            int32x2_t sum_line11_h_h_s32x2 = vget_high_s32(sum_line11_h_s32x4); 
            int32x2_t sum_line01_l_h_s32x2 = vget_high_s32(sum_line01_l_s32x4); 
            int32x2_t sum_line11_l_h_s32x2 = vget_high_s32(sum_line11_l_s32x4); 
            int32x2_t sum_line01_h_l_s32x2 = vget_low_s32(sum_line01_h_s32x4); 
            int32x2_t sum_line11_h_l_s32x2 = vget_low_s32(sum_line11_h_s32x4); 
            int32x2_t sum_line01_l_l_s32x2 = vget_low_s32(sum_line01_l_s32x4); 
            int32x2_t sum_line11_l_l_s32x2 = vget_low_s32(sum_line11_l_s32x4);

            int32x2_t max_line01_h_s32x2 = vpmax_s32(sum_line01_h_l_s32x2, sum_line01_h_h_s32x2);
            int32x2_t max_line01_l_s32x2 = vpmax_s32(sum_line01_l_l_s32x2, sum_line01_l_h_s32x2);
            int32x2_t max_line11_h_s32x2 = vpmax_s32(sum_line11_h_l_s32x2, sum_line11_h_h_s32x2);
            int32x2_t max_line11_l_s32x2 = vpmax_s32(sum_line11_l_l_s32x2, sum_line11_l_h_s32x2);

            int32x2_t max_1_h_s32x2 = vmax_s32(max_line01_h_s32x2, max_line11_h_s32x2);
            int32x2_t max_1_l_s32x2 = vmax_s32(max_line01_l_s32x2, max_line11_l_s32x2);

            int32x4_t max_1_s32x4 = vcombine_s32(max_1_l_s32x2, max_1_h_s32x2);

            int32x4_t shift_1_s32x4 = vshrq_n_s32(max_1_s32x4, val_shift_s32);
            int16x4_t shift_1_s16x4 = vqmovn_s32(shift_1_s32x4);

            int16x4_t shift_zero_1_s16x4 = vmax_s16(shift_1_s16x4, zero_s16x4);
            int16x4_t shift_255_1_s16x4 = vmin_s16(shift_zero_1_s16x4, u8max_s16x4);

            uint16x4_t shift_1_u16x4 = vreinterpret_u16_s16(shift_255_1_s16x4);

            uint16x8_t shift_u16x8 = vcombine_u16(shift_0_u16x4, shift_1_u16x4);
            uint8x8_t shift_u8x8 = vmovn_u16(shift_u16x8);

            vst1_u8(p_out_u8, shift_u8x8);
            p_out_u8 += 8u;
        }

        p_out_u8 -= 1u;
        *p_out_u8 = 0u;
    }
#else
    for (j_u32 = 0u; j_u32 < (out_hei_u32 - 1u); j_u32++)
    {
        const unsigned char* restrict p_inp_line0_u8 = &p_inp_u8[((2u * j_u32) + 0u) * inp_wid_u32];
        const unsigned char* restrict p_inp_line1_u8 = &p_inp_u8[((2u * j_u32) + 1u) * inp_wid_u32];
        const unsigned char* restrict p_inp_line2_u8 = &p_inp_u8[((2u * j_u32) + 2u) * inp_wid_u32];
        const unsigned char* restrict p_inp_line3_u8 = &p_inp_u8[((2u * j_u32) + 3u) * inp_wid_u32];
        unsigned char* restrict p_out_line_u8 = &p_out_u8[j_u32 * out_wid_u32];

        for (i_u32 = 0u; i_u32 < out_wid_u32; i_u32 += 8u)
        {
            uint8x16_t inp_line0_pel0_u8x16 = vld1q_u8(&p_inp_line0_u8[0]);
            uint8x16_t inp_line0_pel1_u8x16 = vld1q_u8(&p_inp_line0_u8[1]);
            uint8x16_t inp_line0_pel2_u8x16 = vld1q_u8(&p_inp_line0_u8[2]);
            uint8x16_t inp_line1_pel0_u8x16 = vld1q_u8(&p_inp_line1_u8[0]);
            uint8x16_t inp_line1_pel1_u8x16 = vld1q_u8(&p_inp_line1_u8[1]);
            uint8x16_t inp_line1_pel2_u8x16 = vld1q_u8(&p_inp_line1_u8[2]);
            uint8x16_t inp_line2_pel0_u8x16 = vld1q_u8(&p_inp_line2_u8[0]);
            uint8x16_t inp_line2_pel1_u8x16 = vld1q_u8(&p_inp_line2_u8[1]);
            uint8x16_t inp_line2_pel2_u8x16 = vld1q_u8(&p_inp_line2_u8[2]);
            uint8x16_t inp_line3_pel0_u8x16 = vld1q_u8(&p_inp_line3_u8[0]);
            uint8x16_t inp_line3_pel1_u8x16 = vld1q_u8(&p_inp_line3_u8[1]);
            uint8x16_t inp_line3_pel2_u8x16 = vld1q_u8(&p_inp_line3_u8[2]);

            p_inp_line0_u8 += 16u;
            p_inp_line1_u8 += 16u;
            p_inp_line2_u8 += 16u;
            p_inp_line3_u8 += 16u;

            uint8x8_t inp_line0_pel0_l_u8x8 = vget_low_u8(inp_line0_pel0_u8x16); 
            uint8x8_t inp_line0_pel1_l_u8x8 = vget_low_u8(inp_line0_pel1_u8x16); 
            uint8x8_t inp_line0_pel2_l_u8x8 = vget_low_u8(inp_line0_pel2_u8x16); 
            uint8x8_t inp_line1_pel0_l_u8x8 = vget_low_u8(inp_line1_pel0_u8x16); 
            uint8x8_t inp_line1_pel1_l_u8x8 = vget_low_u8(inp_line1_pel1_u8x16); 
            uint8x8_t inp_line1_pel2_l_u8x8 = vget_low_u8(inp_line1_pel2_u8x16); 
            uint8x8_t inp_line2_pel0_l_u8x8 = vget_low_u8(inp_line2_pel0_u8x16); 
            uint8x8_t inp_line2_pel1_l_u8x8 = vget_low_u8(inp_line2_pel1_u8x16); 
            uint8x8_t inp_line2_pel2_l_u8x8 = vget_low_u8(inp_line2_pel2_u8x16); 
            uint8x8_t inp_line3_pel0_l_u8x8 = vget_low_u8(inp_line3_pel0_u8x16); 
            uint8x8_t inp_line3_pel1_l_u8x8 = vget_low_u8(inp_line3_pel1_u8x16); 
            uint8x8_t inp_line3_pel2_l_u8x8 = vget_low_u8(inp_line3_pel2_u8x16); 
            uint8x8_t inp_line0_pel0_h_u8x8 = vget_high_u8(inp_line0_pel0_u8x16); 
            uint8x8_t inp_line0_pel1_h_u8x8 = vget_high_u8(inp_line0_pel1_u8x16); 
            uint8x8_t inp_line0_pel2_h_u8x8 = vget_high_u8(inp_line0_pel2_u8x16); 
            uint8x8_t inp_line1_pel0_h_u8x8 = vget_high_u8(inp_line1_pel0_u8x16); 
            uint8x8_t inp_line1_pel1_h_u8x8 = vget_high_u8(inp_line1_pel1_u8x16); 
            uint8x8_t inp_line1_pel2_h_u8x8 = vget_high_u8(inp_line1_pel2_u8x16); 
            uint8x8_t inp_line2_pel0_h_u8x8 = vget_high_u8(inp_line2_pel0_u8x16); 
            uint8x8_t inp_line2_pel1_h_u8x8 = vget_high_u8(inp_line2_pel1_u8x16); 
            uint8x8_t inp_line2_pel2_h_u8x8 = vget_high_u8(inp_line2_pel2_u8x16); 
            uint8x8_t inp_line3_pel0_h_u8x8 = vget_high_u8(inp_line3_pel0_u8x16); 
            uint8x8_t inp_line3_pel1_h_u8x8 = vget_high_u8(inp_line3_pel1_u8x16); 
            uint8x8_t inp_line3_pel2_h_u8x8 = vget_high_u8(inp_line3_pel2_u8x16); 

            uint16x8_t inp_line0_pel0_l_u16x8 = vmovl_u8(inp_line0_pel0_l_u8x8);
            uint16x8_t inp_line0_pel1_l_u16x8 = vmovl_u8(inp_line0_pel1_l_u8x8);
            uint16x8_t inp_line0_pel2_l_u16x8 = vmovl_u8(inp_line0_pel2_l_u8x8);
            uint16x8_t inp_line1_pel0_l_u16x8 = vmovl_u8(inp_line1_pel0_l_u8x8);
            uint16x8_t inp_line1_pel1_l_u16x8 = vmovl_u8(inp_line1_pel1_l_u8x8);
            uint16x8_t inp_line1_pel2_l_u16x8 = vmovl_u8(inp_line1_pel2_l_u8x8);
            uint16x8_t inp_line2_pel0_l_u16x8 = vmovl_u8(inp_line2_pel0_l_u8x8);
            uint16x8_t inp_line2_pel1_l_u16x8 = vmovl_u8(inp_line2_pel1_l_u8x8);
            uint16x8_t inp_line2_pel2_l_u16x8 = vmovl_u8(inp_line2_pel2_l_u8x8);
            uint16x8_t inp_line3_pel0_l_u16x8 = vmovl_u8(inp_line3_pel0_l_u8x8);
            uint16x8_t inp_line3_pel1_l_u16x8 = vmovl_u8(inp_line3_pel1_l_u8x8);
            uint16x8_t inp_line3_pel2_l_u16x8 = vmovl_u8(inp_line3_pel2_l_u8x8);
            uint16x8_t inp_line0_pel0_h_u16x8 = vmovl_u8(inp_line0_pel0_h_u8x8);
            uint16x8_t inp_line0_pel1_h_u16x8 = vmovl_u8(inp_line0_pel1_h_u8x8);
            uint16x8_t inp_line0_pel2_h_u16x8 = vmovl_u8(inp_line0_pel2_h_u8x8);
            uint16x8_t inp_line1_pel0_h_u16x8 = vmovl_u8(inp_line1_pel0_h_u8x8);
            uint16x8_t inp_line1_pel1_h_u16x8 = vmovl_u8(inp_line1_pel1_h_u8x8);
            uint16x8_t inp_line1_pel2_h_u16x8 = vmovl_u8(inp_line1_pel2_h_u8x8);
            uint16x8_t inp_line2_pel0_h_u16x8 = vmovl_u8(inp_line2_pel0_h_u8x8);
            uint16x8_t inp_line2_pel1_h_u16x8 = vmovl_u8(inp_line2_pel1_h_u8x8);
            uint16x8_t inp_line2_pel2_h_u16x8 = vmovl_u8(inp_line2_pel2_h_u8x8);
            uint16x8_t inp_line3_pel0_h_u16x8 = vmovl_u8(inp_line3_pel0_h_u8x8);
            uint16x8_t inp_line3_pel1_h_u16x8 = vmovl_u8(inp_line3_pel1_h_u8x8);
            uint16x8_t inp_line3_pel2_h_u16x8 = vmovl_u8(inp_line3_pel2_h_u8x8);

            int16x8_t inp_line0_pel0_l_s16x8 = vreinterpretq_s16_u16(inp_line0_pel0_l_u16x8);
            int16x8_t inp_line0_pel1_l_s16x8 = vreinterpretq_s16_u16(inp_line0_pel1_l_u16x8);
            int16x8_t inp_line0_pel2_l_s16x8 = vreinterpretq_s16_u16(inp_line0_pel2_l_u16x8);
            int16x8_t inp_line1_pel0_l_s16x8 = vreinterpretq_s16_u16(inp_line1_pel0_l_u16x8);
            int16x8_t inp_line1_pel1_l_s16x8 = vreinterpretq_s16_u16(inp_line1_pel1_l_u16x8);
            int16x8_t inp_line1_pel2_l_s16x8 = vreinterpretq_s16_u16(inp_line1_pel2_l_u16x8);
            int16x8_t inp_line2_pel0_l_s16x8 = vreinterpretq_s16_u16(inp_line2_pel0_l_u16x8);
            int16x8_t inp_line2_pel1_l_s16x8 = vreinterpretq_s16_u16(inp_line2_pel1_l_u16x8);
            int16x8_t inp_line2_pel2_l_s16x8 = vreinterpretq_s16_u16(inp_line2_pel2_l_u16x8);
            int16x8_t inp_line3_pel0_l_s16x8 = vreinterpretq_s16_u16(inp_line3_pel0_l_u16x8);
            int16x8_t inp_line3_pel1_l_s16x8 = vreinterpretq_s16_u16(inp_line3_pel1_l_u16x8);
            int16x8_t inp_line3_pel2_l_s16x8 = vreinterpretq_s16_u16(inp_line3_pel2_l_u16x8);
            int16x8_t inp_line0_pel0_h_s16x8 = vreinterpretq_s16_u16(inp_line0_pel0_h_u16x8);
            int16x8_t inp_line0_pel1_h_s16x8 = vreinterpretq_s16_u16(inp_line0_pel1_h_u16x8);
            int16x8_t inp_line0_pel2_h_s16x8 = vreinterpretq_s16_u16(inp_line0_pel2_h_u16x8);
            int16x8_t inp_line1_pel0_h_s16x8 = vreinterpretq_s16_u16(inp_line1_pel0_h_u16x8);
            int16x8_t inp_line1_pel1_h_s16x8 = vreinterpretq_s16_u16(inp_line1_pel1_h_u16x8);
            int16x8_t inp_line1_pel2_h_s16x8 = vreinterpretq_s16_u16(inp_line1_pel2_h_u16x8);
            int16x8_t inp_line2_pel0_h_s16x8 = vreinterpretq_s16_u16(inp_line2_pel0_h_u16x8);
            int16x8_t inp_line2_pel1_h_s16x8 = vreinterpretq_s16_u16(inp_line2_pel1_h_u16x8);
            int16x8_t inp_line2_pel2_h_s16x8 = vreinterpretq_s16_u16(inp_line2_pel2_h_u16x8);
            int16x8_t inp_line3_pel0_h_s16x8 = vreinterpretq_s16_u16(inp_line3_pel0_h_u16x8);
            int16x8_t inp_line3_pel1_h_s16x8 = vreinterpretq_s16_u16(inp_line3_pel1_h_u16x8);
            int16x8_t inp_line3_pel2_h_s16x8 = vreinterpretq_s16_u16(inp_line3_pel2_h_u16x8);

            int16x8_t sum_line0_l_s16x8 = vdupq_n_s16(0);
            int16x8_t sum_line1_l_s16x8 = vdupq_n_s16(0);
            int16x8_t sum_line0_h_s16x8 = vdupq_n_s16(0);
            int16x8_t sum_line1_h_s16x8 = vdupq_n_s16(0);

            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line0_pel0_l_s16x8, (signed short)p_ker_s8[0]); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line0_pel1_l_s16x8, (signed short)p_ker_s8[1]); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line0_pel2_l_s16x8, (signed short)p_ker_s8[2]); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line1_pel0_l_s16x8, (signed short)p_ker_s8[3]); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line1_pel1_l_s16x8, (signed short)p_ker_s8[4]); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line1_pel2_l_s16x8, (signed short)p_ker_s8[5]); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line2_pel0_l_s16x8, (signed short)p_ker_s8[6]); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line2_pel1_l_s16x8, (signed short)p_ker_s8[7]); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line2_pel2_l_s16x8, (signed short)p_ker_s8[8]); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line1_pel0_l_s16x8, (signed short)p_ker_s8[0]); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line1_pel1_l_s16x8, (signed short)p_ker_s8[1]); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line1_pel2_l_s16x8, (signed short)p_ker_s8[2]); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line2_pel0_l_s16x8, (signed short)p_ker_s8[3]); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line2_pel1_l_s16x8, (signed short)p_ker_s8[4]); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line2_pel2_l_s16x8, (signed short)p_ker_s8[5]); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line3_pel0_l_s16x8, (signed short)p_ker_s8[6]); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line3_pel1_l_s16x8, (signed short)p_ker_s8[7]); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line3_pel2_l_s16x8, (signed short)p_ker_s8[8]); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line0_pel0_h_s16x8, (signed short)p_ker_s8[0]); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line0_pel1_h_s16x8, (signed short)p_ker_s8[1]); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line0_pel2_h_s16x8, (signed short)p_ker_s8[2]); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line1_pel0_h_s16x8, (signed short)p_ker_s8[3]); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line1_pel1_h_s16x8, (signed short)p_ker_s8[4]); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line1_pel2_h_s16x8, (signed short)p_ker_s8[5]); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line2_pel0_h_s16x8, (signed short)p_ker_s8[6]); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line2_pel1_h_s16x8, (signed short)p_ker_s8[7]); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line2_pel2_h_s16x8, (signed short)p_ker_s8[8]); 
            sum_line1_h_s16x8 = vmlaq_n_s16(sum_line1_h_s16x8, inp_line1_pel0_h_s16x8, (signed short)p_ker_s8[0]); 
            sum_line1_h_s16x8 = vmlaq_n_s16(sum_line1_h_s16x8, inp_line1_pel1_h_s16x8, (signed short)p_ker_s8[1]); 
            sum_line1_h_s16x8 = vmlaq_n_s16(sum_line1_h_s16x8, inp_line1_pel2_h_s16x8, (signed short)p_ker_s8[2]); 
            sum_line1_h_s16x8 = vmlaq_n_s16(sum_line1_h_s16x8, inp_line2_pel0_h_s16x8, (signed short)p_ker_s8[3]); 
            sum_line1_h_s16x8 = vmlaq_n_s16(sum_line1_h_s16x8, inp_line2_pel1_h_s16x8, (signed short)p_ker_s8[4]); 
            sum_line1_h_s16x8 = vmlaq_n_s16(sum_line1_h_s16x8, inp_line2_pel2_h_s16x8, (signed short)p_ker_s8[5]); 
            sum_line1_h_s16x8 = vmlaq_n_s16(sum_line1_h_s16x8, inp_line3_pel0_h_s16x8, (signed short)p_ker_s8[6]); 
            sum_line1_h_s16x8 = vmlaq_n_s16(sum_line1_h_s16x8, inp_line3_pel1_h_s16x8, (signed short)p_ker_s8[7]); 
            sum_line1_h_s16x8 = vmlaq_n_s16(sum_line1_h_s16x8, inp_line3_pel2_h_s16x8, (signed short)p_ker_s8[8]); 

            int16x8_t max_line0_s16x8 = vpmaxq_s16(sum_line0_l_s16x8, sum_line0_h_s16x8);
            int16x8_t max_line1_s16x8 = vpmaxq_s16(sum_line1_l_s16x8, sum_line1_h_s16x8);

            int16x8_t max_s16x8 = vmaxq_s16(max_line0_s16x8, max_line1_s16x8);

            int16x8_t shift_s16x8 = vshrq_n_s16(max_s16x8, val_shift_s32);

            int16x8_t shift_zero_s16x8 = vmaxq_s16(shift_s16x8, zero_s16x8);
            int16x8_t shift_255_s16x8 = vminq_s16(shift_zero_s16x8, u8max_s16x8);

            uint16x8_t shift_u16x8 = vreinterpretq_u16_s16(shift_255_s16x8);

            uint8x8_t shift_u8x8 = vmovn_u16(shift_u16x8);

            vst1_u8(p_out_line_u8, shift_u8x8);
            p_out_line_u8 += 8u;
        }

        p_out_line_u8 -= 1u;
        *p_out_line_u8 = 0u;
    }
#endif
}
