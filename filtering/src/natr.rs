pub fn func_natr<const VAL_SHIFT: i32>(inp_wid: usize, inp_hei: usize, out_wid: usize, out_hei: usize, p_ker_s8: &[i8;3 * 3], p_inp_u8: &Vec<u8>, p_out_u8: &mut Vec<u8>) {
    let mut a_out_filtered_y_u8: Vec<u8> = vec![0; inp_wid * inp_hei];

    // filtering
    for j in 0..(inp_hei - 2) {
        for i in 0..(inp_wid - 2) {
            let mut sum_s16: i16 = 0;
            
            for k_h in 0..3 {
                for k_w in 0..3 {
                    let ker_s8: i8 = p_ker_s8[k_w + k_h * 3];
                    let inp_u8: u8 = p_inp_u8[k_w + i + (k_h + j) * inp_wid];

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
            
            a_out_filtered_y_u8[i + j * inp_wid] = sum_s16 as u8;
        }
    }
    
    // max pooling
    for j in 0..out_hei {
        for i in 0..out_wid {
            let pel_00_u8: u8 = a_out_filtered_y_u8[(i * 2 + 0) + (j * 2 + 0) * inp_wid];
            let pel_10_u8: u8 = a_out_filtered_y_u8[(i * 2 + 1) + (j * 2 + 0) * inp_wid];
            let pel_01_u8: u8 = a_out_filtered_y_u8[(i * 2 + 0) + (j * 2 + 1) * inp_wid];
            let pel_11_u8: u8 = a_out_filtered_y_u8[(i * 2 + 1) + (j * 2 + 1) * inp_wid];
            
            let mut max_u8: u8 = pel_00_u8;
            
            if max_u8 < pel_10_u8 {
                max_u8 = pel_10_u8;
            }
            if max_u8 < pel_01_u8 {
                max_u8 = pel_01_u8;
            }
            if max_u8 < pel_11_u8 {
                max_u8 = pel_11_u8;
            }
            
            p_out_u8[i + j * out_wid] = max_u8;
        }
    }
}
