pub fn func_natr(inp_wid: usize, out_wid: usize, out_hei: usize, p_inp_u8: &Vec<u8>, p_out_u8: &mut Vec<u8>) {
    for j in 0..out_hei {
        for i in 0..out_wid {
            let inp_0_u8: u8 = p_inp_u8[2 * i + 0 + (2 * j + 0) * inp_wid];
            let inp_1_u8: u8 = p_inp_u8[2 * i + 1 + (2 * j + 0) * inp_wid];
            let inp_2_u8: u8 = p_inp_u8[2 * i + 0 + (2 * j + 1) * inp_wid];
            let inp_3_u8: u8 = p_inp_u8[2 * i + 1 + (2 * j + 1) * inp_wid];
            
            p_out_u8[i + j * out_wid] = ((inp_0_u8 as u16 + inp_1_u8 as u16 + inp_2_u8 as u16 + inp_3_u8 as u16) / 4) as u8;
        }
    }
}
//pub unsafe fn func_natr(inp_wid: usize, out_wid: usize, out_hei: usize, p_inp_u8: &Vec<u8>, p_out_u8: &mut Vec<u8>) {
//    for j in 0..out_hei {
//        for i in 0..out_wid {
//            let inp_0_u8: u8 = *p_inp_u8.get_unchecked(2 * i + 0 + (2 * j + 0) * inp_wid);
//            let inp_1_u8: u8 = *p_inp_u8.get_unchecked(2 * i + 1 + (2 * j + 0) * inp_wid);
//            let inp_2_u8: u8 = *p_inp_u8.get_unchecked(2 * i + 0 + (2 * j + 1) * inp_wid);
//            let inp_3_u8: u8 = *p_inp_u8.get_unchecked(2 * i + 1 + (2 * j + 1) * inp_wid);
//            
//            p_out_u8[i + j * out_wid] = ((inp_0_u8 as u16 + inp_1_u8 as u16 + inp_2_u8 as u16 + inp_3_u8 as u16) / 4) as u8;
//        }
//    }
//}
