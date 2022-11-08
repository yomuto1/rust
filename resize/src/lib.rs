mod natr;
mod neon;

pub use crate::natr::func_natr;
pub use crate::neon::func_neon;

pub const INP_WID: usize = 512;
pub const INP_HEI: usize = 512;
pub const OUT_WID: usize = 256;
pub const OUT_HEI: usize = 256;

pub fn run_resize_natr(p_inp_u8: &Vec<u8>, p_out_u8: &mut Vec<u8>) {
    natr::func_natr(INP_WID, OUT_WID, OUT_HEI, &p_inp_u8, p_out_u8);
}

pub fn run_resize_neon(p_inp_u8: &Vec<u8>, p_out_u8: &mut Vec<u8>) {
    unsafe {
        neon::func_neon(INP_WID, OUT_WID, OUT_HEI, &p_inp_u8, p_out_u8);
    }
}
