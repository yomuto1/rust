pub fn func_natr<const VAL_SHIFT: i32>(inp_wid: usize, out_wid: usize, out_hei: usize, p_ker_s8: &[i8;3 * 3], p_inp_u8: &Vec<u8>, p_out_u8: &mut Vec<u8>) {
    for j in 0..(out_hei - 1) {
        for i in 0..(out_wid - 1) {
            let mut sum_s16: i16 = 0;
            
            for k_h in 0..3 {
                for k_w in 0..3 {
                    let ker_s8: i8 = p_ker_s8[k_w + k_h * 3];
                    let inp_u8: u8 = p_inp_u8[k_w + 2 * i + 0 + (k_h + 2 * j + 0) * inp_wid];

                    sum_s16 += inp_u8 as i16 * ker_s8 as i16;
                }
            }
            
            sum_s16 >>= VAL_SHIFT;
            if sum_s16 > 255 {
                sum_s16 = 255;
            }
            if sum_s16 < 0 {
                sum_s16 = 0;
            }

            let sum_0_u8: u8 = sum_s16 as u8;

            let mut sum_s16: i16 = 0;
            
            for k_h in 0..3 {
                for k_w in 0..3 {
                    let ker_s8: i8 = p_ker_s8[k_w + k_h * 3];
                    let inp_u8: u8 = p_inp_u8[k_w + 2 * i + 1 + (k_h + 2 * j + 0) * inp_wid];

                    sum_s16 += inp_u8 as i16 * ker_s8 as i16;
                }
            }
            
            sum_s16 >>= VAL_SHIFT;
            if sum_s16 > 255 {
                sum_s16 = 255;
            }
            if sum_s16 < 0 {
                sum_s16 = 0;
            }

            let sum_1_u8: u8 = sum_s16 as u8;

            let mut sum_s16: i16 = 0;
            
            for k_h in 0..3 {
                for k_w in 0..3 {
                    let ker_s8: i8 = p_ker_s8[k_w + k_h * 3];
                    let inp_u8: u8 = p_inp_u8[k_w + 2 * i + 0 + (k_h + 2 * j + 1) * inp_wid];

                    sum_s16 += inp_u8 as i16 * ker_s8 as i16;
                }
            }
            
            sum_s16 >>= VAL_SHIFT;
            if sum_s16 > 255 {
                sum_s16 = 255;
            }
            if sum_s16 < 0 {
                sum_s16 = 0;
            }

            let sum_2_u8: u8 = sum_s16 as u8;

            let mut sum_s16: i16 = 0;
            
            for k_h in 0..3 {
                for k_w in 0..3 {
                    let ker_s8: i8 = p_ker_s8[k_w + k_h * 3];
                    let inp_u8: u8 = p_inp_u8[k_w + 2 * i + 1 + (k_h + 2 * j + 1) * inp_wid];

                    sum_s16 += inp_u8 as i16 * ker_s8 as i16;
                }
            }
            
            sum_s16 >>= VAL_SHIFT;
            if sum_s16 > 255 {
                sum_s16 = 255;
            }
            if sum_s16 < 0 {
                sum_s16 = 0;
            }

            let sum_3_u8: u8 = sum_s16 as u8;

            let mut max_u8: u8 = sum_0_u8;
            
            if max_u8 < sum_1_u8 {
                max_u8 = sum_1_u8;
            }
            if max_u8 < sum_2_u8 {
                max_u8 = sum_2_u8;
            }
            if max_u8 < sum_3_u8 {
                max_u8 = sum_3_u8;
            }
            
            p_out_u8[i + j * out_wid] = max_u8;
        }
    }
}
//pub unsafe fn func_natr<const VAL_SHIFT: i32>(inp_wid: usize, out_wid: usize, out_hei: usize, p_ker_s8: &[i8;3 * 3], p_inp_u8: &Vec<u8>, p_out_u8: &mut Vec<u8>) {
//    for j in 0..(out_hei - 1) {
//        for i in 0..(out_wid - 1) {
//            let mut sum_s16: i16 = 0;
//            
//            for k_h in 0..3 {
//                for k_w in 0..3 {
//                    let ker_s8: i8 = *p_ker_s8.get_unchecked(k_w + k_h * 3);
//                    let inp_u8: u8 = *p_inp_u8.get_unchecked(k_w + 2 * i + 0 + (k_h + 2 * j + 0) * inp_wid);
//
//                    sum_s16 += inp_u8 as i16 * ker_s8 as i16;
//                }
//            }
//            
//            sum_s16 >>= VAL_SHIFT;
//            if sum_s16 > 255 {
//                sum_s16 = 255;
//            }
//            if sum_s16 < 0 {
//                sum_s16 = 0;
//            }
//
//            let sum_0_u8: u8 = sum_s16 as u8;
//
//            let mut sum_s16: i16 = 0;
//            
//            for k_h in 0..3 {
//                for k_w in 0..3 {
//                    let ker_s8: i8 = *p_ker_s8.get_unchecked(k_w + k_h * 3);
//                    let inp_u8: u8 = *p_inp_u8.get_unchecked(k_w + 2 * i + 1 + (k_h + 2 * j + 0) * inp_wid);
//
//                    sum_s16 += inp_u8 as i16 * ker_s8 as i16;
//                }
//            }
//            
//            sum_s16 >>= VAL_SHIFT;
//            if sum_s16 > 255 {
//                sum_s16 = 255;
//            }
//            if sum_s16 < 0 {
//                sum_s16 = 0;
//            }
//
//            let sum_1_u8: u8 = sum_s16 as u8;
//
//            let mut sum_s16: i16 = 0;
//            
//            for k_h in 0..3 {
//                for k_w in 0..3 {
//                    let ker_s8: i8 = *p_ker_s8.get_unchecked(k_w + k_h * 3);
//                    let inp_u8: u8 = *p_inp_u8.get_unchecked(k_w + 2 * i + 0 + (k_h + 2 * j + 1) * inp_wid);
//
//                    sum_s16 += inp_u8 as i16 * ker_s8 as i16;
//                }
//            }
//            
//            sum_s16 >>= VAL_SHIFT;
//            if sum_s16 > 255 {
//                sum_s16 = 255;
//            }
//            if sum_s16 < 0 {
//                sum_s16 = 0;
//            }
//
//            let sum_2_u8: u8 = sum_s16 as u8;
//
//            let mut sum_s16: i16 = 0;
//            
//            for k_h in 0..3 {
//                for k_w in 0..3 {
//                    let ker_s8: i8 = *p_ker_s8.get_unchecked(k_w + k_h * 3);
//                    let inp_u8: u8 = *p_inp_u8.get_unchecked(k_w + 2 * i + 1 + (k_h + 2 * j + 1) * inp_wid);
//
//                    sum_s16 += inp_u8 as i16 * ker_s8 as i16;
//                }
//            }
//            
//            sum_s16 >>= VAL_SHIFT;
//            if sum_s16 > 255 {
//                sum_s16 = 255;
//            }
//            if sum_s16 < 0 {
//                sum_s16 = 0;
//            }
//
//            let sum_3_u8: u8 = sum_s16 as u8;
//
//            let mut max_u8: u8 = sum_0_u8;
//            
//            if max_u8 < sum_1_u8 {
//                max_u8 = sum_1_u8;
//            }
//            if max_u8 < sum_2_u8 {
//                max_u8 = sum_2_u8;
//            }
//            if max_u8 < sum_3_u8 {
//                max_u8 = sum_3_u8;
//            }
//            
//            p_out_u8[i + j * out_wid] = max_u8;
//        }
//    }
//}
