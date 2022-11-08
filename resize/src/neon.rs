use core::arch::aarch64::*;

pub unsafe fn func_neon(inp_wid: usize, out_wid: usize, out_hei: usize, p_inp_u8: &Vec<u8>, p_out_u8: &mut Vec<u8>) {
    for j in 0..out_hei {
        for i in 0..(out_wid / 8) {
            let p_inp_line0_u8: *const u8 = &p_inp_u8[(i * 16) + ((2 * j) + 0) * inp_wid];
            let p_inp_line1_u8: *const u8 = &p_inp_u8[(i * 16) + ((2 * j) + 1) * inp_wid];

            let inp_line0_u8x16 = vld1q_u8(p_inp_line0_u8);
            let inp_line1_u8x16 = vld1q_u8(p_inp_line1_u8);

            let add_01_u16x8 = vpaddlq_u8(inp_line0_u8x16);
            let add_23_u16x8 = vpaddlq_u8(inp_line1_u8x16);

            let add_u16x8 = vaddq_u16(add_01_u16x8, add_23_u16x8);
            
            let shift_u8x8 = vshrn_n_u16(add_u16x8, 2);

            let p_out_line_u8: *mut u8 = &mut p_out_u8[(i * 8) + (j * out_wid)];

            vst1_u8(p_out_line_u8, shift_u8x8);
        }
    }
}
