mod natr;
mod neon;

pub use crate::natr::func_natr;
pub use crate::neon::func_neon;

pub const INP_WID: usize = 512;
pub const INP_HEI: usize = 512;
pub const OUT_WID: usize = 256;
pub const OUT_HEI: usize = 256;
pub const VAL_SHIFT: i32 = 5;

pub fn run_filter_natr(p_ker_s8: &[i8;3 * 3], p_inp_u8: &Vec<u8>, p_out_u8: &mut Vec<u8>) {
    natr::func_natr::<VAL_SHIFT>(INP_WID, INP_HEI, OUT_WID, OUT_HEI, &p_ker_s8, &p_inp_u8, p_out_u8);
}

pub fn run_filter_neon(p_ker_s8: &[i8;3 * 3], p_inp_u8: &Vec<u8>, p_out_u8: &mut Vec<u8>) {
    unsafe {
        neon::func_neon::<VAL_SHIFT>(INP_WID, OUT_WID, OUT_HEI, &p_ker_s8, &p_inp_u8, p_out_u8);
    }
}
