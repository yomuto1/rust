#include <stdio.h>
#include <time.h>
#include <string.h>
#include <arm_neon.h>

#define ITERATION (10000u)
#define INP_WID (512u)
#define INP_HEI (512u)
#define OUT_WID (INP_WID / 2u)
#define OUT_HEI (INP_HEI / 2u)

void func_natc(const unsigned int inp_wid_u32, const unsigned int inp_hei_u32, const unsigned int out_wid_u32, const unsigned int out_hei_u32, const unsigned char* p_inp_u8, unsigned char* p_out_u8);
void func_neon(const unsigned int inp_wid_u32, const unsigned int inp_hei_u32, const unsigned int out_wid_u32, const unsigned int out_hei_u32, const unsigned char* p_inp_u8, unsigned char* p_out_u8);

void main (void) {
    FILE* fp;
    static unsigned char sa_inp_0_u8[INP_WID * INP_HEI];
    static unsigned char sa_out_0_u8[OUT_WID * OUT_HEI]; // max pool result
    static unsigned char sa_out_1_u8[OUT_WID * OUT_HEI]; // max pool result, neon
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
        func_natc(INP_WID, INP_HEI, OUT_WID, OUT_HEI, sa_inp_0_u8, sa_out_0_u8);
    }

    printf("natural C code version time: %f sec\n", (float)(clock() - time_start) / CLOCKS_PER_SEC);

    fp = fopen("out_c_natc_resize_barbara_256x256.y", "wb");
    fwrite(&sa_out_0_u8[0], sizeof(unsigned char), OUT_WID * OUT_HEI, fp);
    fclose(fp);
 
    // arm neon version
    time_start = clock();

    for (z_u32 = 0u; z_u32 < ITERATION; z_u32++)    
    {
        func_neon(INP_WID, INP_HEI, OUT_WID, OUT_HEI, sa_inp_0_u8, sa_out_1_u8);
    }

    printf("neon C code version time: %f sec\n", (float)(clock() - time_start) / CLOCKS_PER_SEC);

    fp = fopen("out_c_neon_resize_barbara_256x256.y", "wb");
    fwrite(&sa_out_1_u8[0], sizeof(unsigned char), OUT_WID * OUT_HEI, fp);
    fclose(fp); 

    for (j_u32 = 0u; j_u32 < OUT_HEI; j_u32++)
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

void func_natc(const unsigned int inp_wid_u32, const unsigned int inp_hei_u32, const unsigned int out_wid_u32, const unsigned int out_hei_u32, const unsigned char* p_inp_u8, unsigned char* p_out_u8)
{
    unsigned int i_u32;
    unsigned int j_u32;

    unsigned char inp_0_u8;
    unsigned char inp_1_u8;
    unsigned char inp_2_u8;
    unsigned char inp_3_u8;

    for (j_u32 = 0u; j_u32 < out_hei_u32; j_u32++)
    {
        for (i_u32 = 0u; i_u32 < out_wid_u32; i_u32++)
        {
            inp_0_u8 = p_inp_u8[(i_u32 * 2u) + 0u + (((j_u32 * 2u) + 0u) * inp_wid_u32)];
            inp_1_u8 = p_inp_u8[(i_u32 * 2u) + 1u + (((j_u32 * 2u) + 0u) * inp_wid_u32)];
            inp_2_u8 = p_inp_u8[(i_u32 * 2u) + 0u + (((j_u32 * 2u) + 1u) * inp_wid_u32)];
            inp_3_u8 = p_inp_u8[(i_u32 * 2u) + 1u + (((j_u32 * 2u) + 1u) * inp_wid_u32)];

            p_out_u8[i_u32 + (j_u32 * out_wid_u32)] = (inp_0_u8 + inp_1_u8 + inp_2_u8 + inp_3_u8) / 4u;

            //if (i_u32 < 8u && j_u32 == 0)
            //{
            //    printf("natc: (%u, %u): %d, %d, %d, %d,  %d\n", i_u32, j_u32, inp_0_u8, inp_1_u8, inp_2_u8, inp_3_u8, p_out_u8[i_u32 + (j_u32 * out_wid_u32)]);
            //}
        }
    }
}

void func_neon(const unsigned int inp_wid_u32, const unsigned int inp_hei_u32, const unsigned int out_wid_u32, const unsigned int out_hei_u32, const unsigned char* p_inp_u8, unsigned char* p_out_u8)
{
    unsigned int i_u32;
    unsigned int j_u32;

    for (j_u32 = 0u; j_u32 < out_hei_u32; j_u32++)
    {
        const unsigned char* restrict p_inp_line0_u8 = &p_inp_u8[((2u * j_u32) + 0u) * inp_wid_u32];
        const unsigned char* restrict p_inp_line1_u8 = &p_inp_u8[((2u * j_u32) + 1u) * inp_wid_u32];
        unsigned char* restrict p_out_line_u8 = &p_out_u8[j_u32 * out_wid_u32];

        for (i_u32 = 0u; i_u32 < out_wid_u32; i_u32 += 8u)
        {
            uint8x16_t inp_line0_u8x16 = vld1q_u8(&p_inp_line0_u8[0]);
            uint8x16_t inp_line1_u8x16 = vld1q_u8(&p_inp_line1_u8[0]);

            p_inp_line0_u8 += 16u;
            p_inp_line1_u8 += 16u;

            uint16x8_t add_01_u16x8 = vpaddlq_u8(inp_line0_u8x16);
            uint16x8_t add_23_u16x8 = vpaddlq_u8(inp_line1_u8x16);

            uint16x8_t add_u16x8 = vaddq_u16(add_01_u16x8, add_23_u16x8);
            
            uint8x8_t shift_u8x8 = vshrn_n_u16(add_u16x8, 2);

            //if (i_u32 == 0 && j_u32 == 0)
            //{
            //    printf("neon: (0, 0): %d, %d, %d, %d,  %d\n", vgetq_lane_u8(inp_line0_u8x16, 0), vgetq_lane_u8(inp_line0_u8x16, 1), vgetq_lane_u8(inp_line1_u8x16, 0), vgetq_lane_u8(inp_line1_u8x16, 1), vget_lane_u8(shift_u8x8, 0));
            //    printf("%d, %d, %d\n", vgetq_lane_u16(add_01_u16x8, 0), vgetq_lane_u16(add_23_u16x8, 0), vgetq_lane_u16(add_u16x8, 0));
            //    printf("neon: (7, 0): %d, %d, %d, %d,  %d\n", vgetq_lane_u8(inp_line0_u8x16, 14), vgetq_lane_u8(inp_line0_u8x16, 15), vgetq_lane_u8(inp_line1_u8x16, 14), vgetq_lane_u8(inp_line1_u8x16, 15), vget_lane_u8(shift_u8x8, 7));
            //    printf("%d, %d, %d\n", vgetq_lane_u16(add_01_u16x8, 7), vgetq_lane_u16(add_23_u16x8, 7), vgetq_lane_u16(add_u16x8, 7));
            //}

            vst1_u8(p_out_line_u8, shift_u8x8);
            p_out_line_u8 += 8u;
        }
    }
}
