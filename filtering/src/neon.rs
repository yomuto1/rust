use core::arch::aarch64::*;

pub unsafe fn func_neon<const VAL_SHIFT: i32>(inp_wid: usize, out_wid: usize, out_hei: usize, p_ker_s8: &[i8;3 * 3], p_inp_u8: &Vec<u8>, p_out_u8: &mut Vec<u8>) {
    let zero_s16x8 = vdupq_n_s16(0);
    let u8max_s16x8 = vdupq_n_s16(255);

    for j in 0..(out_hei - 1) {
        for i in 0..(out_wid / 8) {
            let p_inp_line0_pel0_u8: *const u8 = &p_inp_u8[0 + (i * 16) + ((2 * j) + 0) * inp_wid];
            let p_inp_line0_pel1_u8: *const u8 = &p_inp_u8[1 + (i * 16) + ((2 * j) + 0) * inp_wid];
            let p_inp_line0_pel2_u8: *const u8 = &p_inp_u8[2 + (i * 16) + ((2 * j) + 0) * inp_wid];
            let p_inp_line1_pel0_u8: *const u8 = &p_inp_u8[0 + (i * 16) + ((2 * j) + 1) * inp_wid];
            let p_inp_line1_pel1_u8: *const u8 = &p_inp_u8[1 + (i * 16) + ((2 * j) + 1) * inp_wid];
            let p_inp_line1_pel2_u8: *const u8 = &p_inp_u8[2 + (i * 16) + ((2 * j) + 1) * inp_wid];
            let p_inp_line2_pel0_u8: *const u8 = &p_inp_u8[0 + (i * 16) + ((2 * j) + 2) * inp_wid];
            let p_inp_line2_pel1_u8: *const u8 = &p_inp_u8[1 + (i * 16) + ((2 * j) + 2) * inp_wid];
            let p_inp_line2_pel2_u8: *const u8 = &p_inp_u8[2 + (i * 16) + ((2 * j) + 2) * inp_wid];
            let p_inp_line3_pel0_u8: *const u8 = &p_inp_u8[0 + (i * 16) + ((2 * j) + 3) * inp_wid];
            let p_inp_line3_pel1_u8: *const u8 = &p_inp_u8[1 + (i * 16) + ((2 * j) + 3) * inp_wid];
            let p_inp_line3_pel2_u8: *const u8 = &p_inp_u8[2 + (i * 16) + ((2 * j) + 3) * inp_wid];
            //let p_inp_line0_pel0_u8: *const u8 = p_inp_u8.get_unchecked(0 + (i * 16) + ((2 * j) + 0) * inp_wid);
            //let p_inp_line0_pel1_u8: *const u8 = p_inp_u8.get_unchecked(1 + (i * 16) + ((2 * j) + 0) * inp_wid);
            //let p_inp_line0_pel2_u8: *const u8 = p_inp_u8.get_unchecked(2 + (i * 16) + ((2 * j) + 0) * inp_wid);
            //let p_inp_line1_pel0_u8: *const u8 = p_inp_u8.get_unchecked(0 + (i * 16) + ((2 * j) + 1) * inp_wid);
            //let p_inp_line1_pel1_u8: *const u8 = p_inp_u8.get_unchecked(1 + (i * 16) + ((2 * j) + 1) * inp_wid);
            //let p_inp_line1_pel2_u8: *const u8 = p_inp_u8.get_unchecked(2 + (i * 16) + ((2 * j) + 1) * inp_wid);
            //let p_inp_line2_pel0_u8: *const u8 = p_inp_u8.get_unchecked(0 + (i * 16) + ((2 * j) + 2) * inp_wid);
            //let p_inp_line2_pel1_u8: *const u8 = p_inp_u8.get_unchecked(1 + (i * 16) + ((2 * j) + 2) * inp_wid);
            //let p_inp_line2_pel2_u8: *const u8 = p_inp_u8.get_unchecked(2 + (i * 16) + ((2 * j) + 2) * inp_wid);
            //let p_inp_line3_pel0_u8: *const u8 = p_inp_u8.get_unchecked(0 + (i * 16) + ((2 * j) + 3) * inp_wid);
            //let p_inp_line3_pel1_u8: *const u8 = p_inp_u8.get_unchecked(1 + (i * 16) + ((2 * j) + 3) * inp_wid);
            //let p_inp_line3_pel2_u8: *const u8 = p_inp_u8.get_unchecked(2 + (i * 16) + ((2 * j) + 3) * inp_wid);

            let inp_line0_pel0_u8x16 = vld1q_u8(p_inp_line0_pel0_u8);
            let inp_line0_pel1_u8x16 = vld1q_u8(p_inp_line0_pel1_u8);
            let inp_line0_pel2_u8x16 = vld1q_u8(p_inp_line0_pel2_u8);
            let inp_line1_pel0_u8x16 = vld1q_u8(p_inp_line1_pel0_u8);
            let inp_line1_pel1_u8x16 = vld1q_u8(p_inp_line1_pel1_u8);
            let inp_line1_pel2_u8x16 = vld1q_u8(p_inp_line1_pel2_u8);
            let inp_line2_pel0_u8x16 = vld1q_u8(p_inp_line2_pel0_u8);
            let inp_line2_pel1_u8x16 = vld1q_u8(p_inp_line2_pel1_u8);
            let inp_line2_pel2_u8x16 = vld1q_u8(p_inp_line2_pel2_u8);
            let inp_line3_pel0_u8x16 = vld1q_u8(p_inp_line3_pel0_u8);
            let inp_line3_pel1_u8x16 = vld1q_u8(p_inp_line3_pel1_u8);
            let inp_line3_pel2_u8x16 = vld1q_u8(p_inp_line3_pel2_u8);

            let inp_line0_pel0_l_u8x8 = vget_low_u8(inp_line0_pel0_u8x16); 
            let inp_line0_pel1_l_u8x8 = vget_low_u8(inp_line0_pel1_u8x16); 
            let inp_line0_pel2_l_u8x8 = vget_low_u8(inp_line0_pel2_u8x16); 
            let inp_line1_pel0_l_u8x8 = vget_low_u8(inp_line1_pel0_u8x16); 
            let inp_line1_pel1_l_u8x8 = vget_low_u8(inp_line1_pel1_u8x16); 
            let inp_line1_pel2_l_u8x8 = vget_low_u8(inp_line1_pel2_u8x16); 
            let inp_line2_pel0_l_u8x8 = vget_low_u8(inp_line2_pel0_u8x16); 
            let inp_line2_pel1_l_u8x8 = vget_low_u8(inp_line2_pel1_u8x16); 
            let inp_line2_pel2_l_u8x8 = vget_low_u8(inp_line2_pel2_u8x16); 
            let inp_line3_pel0_l_u8x8 = vget_low_u8(inp_line3_pel0_u8x16); 
            let inp_line3_pel1_l_u8x8 = vget_low_u8(inp_line3_pel1_u8x16); 
            let inp_line3_pel2_l_u8x8 = vget_low_u8(inp_line3_pel2_u8x16); 
            let inp_line0_pel0_h_u8x8 = vget_high_u8(inp_line0_pel0_u8x16); 
            let inp_line0_pel1_h_u8x8 = vget_high_u8(inp_line0_pel1_u8x16); 
            let inp_line0_pel2_h_u8x8 = vget_high_u8(inp_line0_pel2_u8x16); 
            let inp_line1_pel0_h_u8x8 = vget_high_u8(inp_line1_pel0_u8x16); 
            let inp_line1_pel1_h_u8x8 = vget_high_u8(inp_line1_pel1_u8x16); 
            let inp_line1_pel2_h_u8x8 = vget_high_u8(inp_line1_pel2_u8x16); 
            let inp_line2_pel0_h_u8x8 = vget_high_u8(inp_line2_pel0_u8x16); 
            let inp_line2_pel1_h_u8x8 = vget_high_u8(inp_line2_pel1_u8x16); 
            let inp_line2_pel2_h_u8x8 = vget_high_u8(inp_line2_pel2_u8x16); 
            let inp_line3_pel0_h_u8x8 = vget_high_u8(inp_line3_pel0_u8x16); 
            let inp_line3_pel1_h_u8x8 = vget_high_u8(inp_line3_pel1_u8x16); 
            let inp_line3_pel2_h_u8x8 = vget_high_u8(inp_line3_pel2_u8x16); 

            let inp_line0_pel0_l_u16x8 = vmovl_u8(inp_line0_pel0_l_u8x8);
            let inp_line0_pel1_l_u16x8 = vmovl_u8(inp_line0_pel1_l_u8x8);
            let inp_line0_pel2_l_u16x8 = vmovl_u8(inp_line0_pel2_l_u8x8);
            let inp_line1_pel0_l_u16x8 = vmovl_u8(inp_line1_pel0_l_u8x8);
            let inp_line1_pel1_l_u16x8 = vmovl_u8(inp_line1_pel1_l_u8x8);
            let inp_line1_pel2_l_u16x8 = vmovl_u8(inp_line1_pel2_l_u8x8);
            let inp_line2_pel0_l_u16x8 = vmovl_u8(inp_line2_pel0_l_u8x8);
            let inp_line2_pel1_l_u16x8 = vmovl_u8(inp_line2_pel1_l_u8x8);
            let inp_line2_pel2_l_u16x8 = vmovl_u8(inp_line2_pel2_l_u8x8);
            let inp_line3_pel0_l_u16x8 = vmovl_u8(inp_line3_pel0_l_u8x8);
            let inp_line3_pel1_l_u16x8 = vmovl_u8(inp_line3_pel1_l_u8x8);
            let inp_line3_pel2_l_u16x8 = vmovl_u8(inp_line3_pel2_l_u8x8);
            let inp_line0_pel0_h_u16x8 = vmovl_u8(inp_line0_pel0_h_u8x8);
            let inp_line0_pel1_h_u16x8 = vmovl_u8(inp_line0_pel1_h_u8x8);
            let inp_line0_pel2_h_u16x8 = vmovl_u8(inp_line0_pel2_h_u8x8);
            let inp_line1_pel0_h_u16x8 = vmovl_u8(inp_line1_pel0_h_u8x8);
            let inp_line1_pel1_h_u16x8 = vmovl_u8(inp_line1_pel1_h_u8x8);
            let inp_line1_pel2_h_u16x8 = vmovl_u8(inp_line1_pel2_h_u8x8);
            let inp_line2_pel0_h_u16x8 = vmovl_u8(inp_line2_pel0_h_u8x8);
            let inp_line2_pel1_h_u16x8 = vmovl_u8(inp_line2_pel1_h_u8x8);
            let inp_line2_pel2_h_u16x8 = vmovl_u8(inp_line2_pel2_h_u8x8);
            let inp_line3_pel0_h_u16x8 = vmovl_u8(inp_line3_pel0_h_u8x8);
            let inp_line3_pel1_h_u16x8 = vmovl_u8(inp_line3_pel1_h_u8x8);
            let inp_line3_pel2_h_u16x8 = vmovl_u8(inp_line3_pel2_h_u8x8);

            let inp_line0_pel0_l_s16x8 = vreinterpretq_s16_u16(inp_line0_pel0_l_u16x8);
            let inp_line0_pel1_l_s16x8 = vreinterpretq_s16_u16(inp_line0_pel1_l_u16x8);
            let inp_line0_pel2_l_s16x8 = vreinterpretq_s16_u16(inp_line0_pel2_l_u16x8);
            let inp_line1_pel0_l_s16x8 = vreinterpretq_s16_u16(inp_line1_pel0_l_u16x8);
            let inp_line1_pel1_l_s16x8 = vreinterpretq_s16_u16(inp_line1_pel1_l_u16x8);
            let inp_line1_pel2_l_s16x8 = vreinterpretq_s16_u16(inp_line1_pel2_l_u16x8);
            let inp_line2_pel0_l_s16x8 = vreinterpretq_s16_u16(inp_line2_pel0_l_u16x8);
            let inp_line2_pel1_l_s16x8 = vreinterpretq_s16_u16(inp_line2_pel1_l_u16x8);
            let inp_line2_pel2_l_s16x8 = vreinterpretq_s16_u16(inp_line2_pel2_l_u16x8);
            let inp_line3_pel0_l_s16x8 = vreinterpretq_s16_u16(inp_line3_pel0_l_u16x8);
            let inp_line3_pel1_l_s16x8 = vreinterpretq_s16_u16(inp_line3_pel1_l_u16x8);
            let inp_line3_pel2_l_s16x8 = vreinterpretq_s16_u16(inp_line3_pel2_l_u16x8);
            let inp_line0_pel0_h_s16x8 = vreinterpretq_s16_u16(inp_line0_pel0_h_u16x8);
            let inp_line0_pel1_h_s16x8 = vreinterpretq_s16_u16(inp_line0_pel1_h_u16x8);
            let inp_line0_pel2_h_s16x8 = vreinterpretq_s16_u16(inp_line0_pel2_h_u16x8);
            let inp_line1_pel0_h_s16x8 = vreinterpretq_s16_u16(inp_line1_pel0_h_u16x8);
            let inp_line1_pel1_h_s16x8 = vreinterpretq_s16_u16(inp_line1_pel1_h_u16x8);
            let inp_line1_pel2_h_s16x8 = vreinterpretq_s16_u16(inp_line1_pel2_h_u16x8);
            let inp_line2_pel0_h_s16x8 = vreinterpretq_s16_u16(inp_line2_pel0_h_u16x8);
            let inp_line2_pel1_h_s16x8 = vreinterpretq_s16_u16(inp_line2_pel1_h_u16x8);
            let inp_line2_pel2_h_s16x8 = vreinterpretq_s16_u16(inp_line2_pel2_h_u16x8);
            let inp_line3_pel0_h_s16x8 = vreinterpretq_s16_u16(inp_line3_pel0_h_u16x8);
            let inp_line3_pel1_h_s16x8 = vreinterpretq_s16_u16(inp_line3_pel1_h_u16x8);
            let inp_line3_pel2_h_s16x8 = vreinterpretq_s16_u16(inp_line3_pel2_h_u16x8);

            let mut sum_line0_l_s16x8 = vmulq_n_s16(inp_line0_pel0_l_s16x8, p_ker_s8[0] as i16);
            let mut sum_line1_l_s16x8 = vmulq_n_s16(inp_line1_pel0_l_s16x8, p_ker_s8[0] as i16);
            let mut sum_line0_h_s16x8 = vmulq_n_s16(inp_line0_pel0_h_s16x8, p_ker_s8[0] as i16);
            let mut sum_line1_h_s16x8 = vmulq_n_s16(inp_line1_pel0_h_s16x8, p_ker_s8[0] as i16);

            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line0_pel1_l_s16x8, p_ker_s8[1] as i16); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line0_pel2_l_s16x8, p_ker_s8[2] as i16); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line1_pel0_l_s16x8, p_ker_s8[3] as i16); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line1_pel1_l_s16x8, p_ker_s8[4] as i16); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line1_pel2_l_s16x8, p_ker_s8[5] as i16); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line2_pel0_l_s16x8, p_ker_s8[6] as i16); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line2_pel1_l_s16x8, p_ker_s8[7] as i16); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line2_pel2_l_s16x8, p_ker_s8[8] as i16); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line1_pel1_l_s16x8, p_ker_s8[1] as i16); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line1_pel2_l_s16x8, p_ker_s8[2] as i16); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line2_pel0_l_s16x8, p_ker_s8[3] as i16); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line2_pel1_l_s16x8, p_ker_s8[4] as i16); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line2_pel2_l_s16x8, p_ker_s8[5] as i16); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line3_pel0_l_s16x8, p_ker_s8[6] as i16); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line3_pel1_l_s16x8, p_ker_s8[7] as i16); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line3_pel2_l_s16x8, p_ker_s8[8] as i16); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line0_pel1_h_s16x8, p_ker_s8[1] as i16); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line0_pel2_h_s16x8, p_ker_s8[2] as i16); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line1_pel0_h_s16x8, p_ker_s8[3] as i16); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line1_pel1_h_s16x8, p_ker_s8[4] as i16); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line1_pel2_h_s16x8, p_ker_s8[5] as i16); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line2_pel0_h_s16x8, p_ker_s8[6] as i16); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line2_pel1_h_s16x8, p_ker_s8[7] as i16); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line2_pel2_h_s16x8, p_ker_s8[8] as i16); 
            sum_line1_h_s16x8 = vmlaq_n_s16(sum_line1_h_s16x8, inp_line1_pel1_h_s16x8, p_ker_s8[1] as i16); 
            sum_line1_h_s16x8 = vmlaq_n_s16(sum_line1_h_s16x8, inp_line1_pel2_h_s16x8, p_ker_s8[2] as i16); 
            sum_line1_h_s16x8 = vmlaq_n_s16(sum_line1_h_s16x8, inp_line2_pel0_h_s16x8, p_ker_s8[3] as i16); 
            sum_line1_h_s16x8 = vmlaq_n_s16(sum_line1_h_s16x8, inp_line2_pel1_h_s16x8, p_ker_s8[4] as i16); 
            sum_line1_h_s16x8 = vmlaq_n_s16(sum_line1_h_s16x8, inp_line2_pel2_h_s16x8, p_ker_s8[5] as i16); 
            sum_line1_h_s16x8 = vmlaq_n_s16(sum_line1_h_s16x8, inp_line3_pel0_h_s16x8, p_ker_s8[6] as i16); 
            sum_line1_h_s16x8 = vmlaq_n_s16(sum_line1_h_s16x8, inp_line3_pel1_h_s16x8, p_ker_s8[7] as i16); 
            sum_line1_h_s16x8 = vmlaq_n_s16(sum_line1_h_s16x8, inp_line3_pel2_h_s16x8, p_ker_s8[8] as i16); 

            let max_line0_s16x8 = vpmaxq_s16(sum_line0_l_s16x8, sum_line0_h_s16x8);
            let max_line1_s16x8 = vpmaxq_s16(sum_line1_l_s16x8, sum_line1_h_s16x8);

            let max_s16x8 = vmaxq_s16(max_line0_s16x8, max_line1_s16x8);

            let shift_s16x8 = vshrq_n_s16(max_s16x8, VAL_SHIFT);

            let shift_zero_s16x8 = vmaxq_s16(shift_s16x8, zero_s16x8);
            let shift_255_s16x8 = vminq_s16(shift_zero_s16x8, u8max_s16x8);

            let shift_u16x8 = vreinterpretq_u16_s16(shift_255_s16x8);

            let shift_u8x8 = vmovn_u16(shift_u16x8);

            let p_out_line_u8: *mut u8 = &mut p_out_u8[(i * 8) + (j * out_wid)];

            vst1_u8(p_out_line_u8, shift_u8x8);
        }

        p_out_u8[out_wid - 1 + (j * out_wid)] = 0;
    }
}
