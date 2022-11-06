//use std::fs;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::time::{Instant};
use core::arch::aarch64::*;

const ITERATION: usize = 1000;
const INP_WID: usize = 512;
const INP_HEI: usize = 512;
const OUT_WID: usize = 256;
const OUT_HEI: usize = 256;
const VAL_SHIFT: i32 = 5;

fn main() {
    let inp_filename = r"/home/hyuk/code/barbara_gray.raw";
    let out_filename = r"barbara_512x512.y";
    let out_pool_natr_filename = r"barbara_256x256_pool_natr.y";
    let out_pool_neon_filename = r"barbara_256x256_pool_neon.y";
    
    let mut inp_file = File::open(&inp_filename).expect("no inp file found");
    //let metadata = fs::metadata(&inp_filename).expect("unable to read metadata");
    let mut buffer = vec![0; INP_WID * INP_HEI];
    inp_file.read(&mut buffer).expect("buffer overflow");
    
    //let mut a_out_y_u8: [u8; INP_WID * INP_HEI] = [0; INP_WID * INP_HEI];
    let mut a_out_y_u8: Vec<u8> = vec![0; INP_WID * INP_HEI];
    
    for i in 0..(INP_WID * INP_HEI - 1) {
        let val_u8: u8 = buffer[i];
        a_out_y_u8[i] = val_u8;
    }

    let mut out_file = File::create(out_filename).expect("failed to open out file");
    
    out_file.write_all(&a_out_y_u8).expect("failed to write file");

    let a_coefficient_s8: [i8; 3 * 3] = [3, 7, -5, -10, 19, 7, 5, -9, 7];
    let mut a_out_pool_y_natr_u8: Vec<u8> = vec![0; OUT_WID * OUT_HEI];
    
    let start = Instant::now();

    let mut count = 0;
    
    loop {
        count += 1;

        func_natr(&a_coefficient_s8, &a_out_y_u8, &mut a_out_pool_y_natr_u8);

        if count == ITERATION {
            break;
        }
    }
    
    let duration = start.elapsed();

    println!("natural rust code version time: {:?}", duration);

    let mut out_file = File::create(out_pool_natr_filename).expect("failed to open out pool file");
    
    out_file.write_all(&a_out_pool_y_natr_u8).expect("failed to write file");

    let mut a_out_pool_y_neon_u8: Vec<u8> = vec![0; OUT_WID * OUT_HEI];

    let start = Instant::now();

    let mut count = 0;
    
    loop {
        count += 1;

        unsafe {
            func_neon(&a_coefficient_s8, &a_out_y_u8, &mut a_out_pool_y_neon_u8);
        }

        if count == ITERATION {
            break;
        }
    }
    
    let duration = start.elapsed();

    println!("neon rust code version time: {:?}", duration);

    let mut out_file = File::create(out_pool_neon_filename).expect("failed to open out pool file");
    
    out_file.write_all(&a_out_pool_y_neon_u8).expect("failed to write file");

    for j in 0..OUT_HEI {
        for i in 0..OUT_WID {
            let natr_value = a_out_pool_y_natr_u8[i + (j * OUT_WID)];
            let neon_value = a_out_pool_y_neon_u8[i + (j * OUT_WID)];

            if natr_value != neon_value {
                println!("mismatch ({i}, {j}): {natr_value}, {neon_value}");
            }
        }
    }
}

fn func_natr(p_ker_s8: &[i8;3 * 3], p_inp_u8: &Vec<u8>, p_out_u8: &mut Vec<u8>) {
    let mut a_out_filtered_y_u8: Vec<u8> = vec![0; INP_WID * INP_HEI];

    // filtering
    for j in 0..(INP_HEI - 2) {
        for i in 0..(INP_WID - 2) {
            let mut sum_s16: i16 = 0;
            
            for k_h in 0..3 {
                for k_w in 0..3 {
                    let ker_s8: i8 = p_ker_s8[k_w + k_h * 3];
                    let inp_u8: u8 = p_inp_u8[k_w + i + (k_h + j) * INP_WID];

                    sum_s16 += inp_u8 as i16 * ker_s8 as i16;

                    //if i == 0 && j == 0 {
                    //    println!("{k_w}, {k_h}: {ker_s16}, {inp_u8}, {sum_s32}");
                    //}
                }
            }

            //if i == 0 && j == 0 {
            //    println!("{i}, {j}: {sum_s32}");
            //}
            
            sum_s16 >>= VAL_SHIFT;
            if sum_s16 > 255 {
                sum_s16 = 255;
            }
            if sum_s16 < 0 {
                sum_s16 = 0;
            }
            
            //if i == 0 && j == 0 {
            //    println!("{sum_s32}");
            //}
            
            a_out_filtered_y_u8[i + j * INP_WID] = sum_s16 as u8;
        }
    }
    
    // max pooling
    for j in 0..OUT_HEI {
        for i in 0..OUT_WID {
            let pel_00_u8: u8 = a_out_filtered_y_u8[(i * 2 + 0) + (j * 2 + 0) * INP_WID];
            let pel_10_u8: u8 = a_out_filtered_y_u8[(i * 2 + 1) + (j * 2 + 0) * INP_WID];
            let pel_01_u8: u8 = a_out_filtered_y_u8[(i * 2 + 0) + (j * 2 + 1) * INP_WID];
            let pel_11_u8: u8 = a_out_filtered_y_u8[(i * 2 + 1) + (j * 2 + 1) * INP_WID];
            
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
            
            p_out_u8[i + j * OUT_WID] = max_u8;
        }
    }
}

unsafe fn func_neon(p_ker_s8: &[i8;3 * 3], p_inp_u8: &Vec<u8>, p_out_u8: &mut Vec<u8>) {
    let zero_s16x8 = vdupq_n_s16(0);
    let u8max_s16x8 = vdupq_n_s16(255);

    for j in 0..(OUT_HEI - 1) {
        for i in 0..(OUT_WID / 8) {
            let p_inp_line0_pel0_u8: *const u8 = &p_inp_u8[0 + (i * 16) + ((2 * j) + 0) * INP_WID];
            let p_inp_line0_pel1_u8: *const u8 = &p_inp_u8[1 + (i * 16) + ((2 * j) + 0) * INP_WID];
            let p_inp_line0_pel2_u8: *const u8 = &p_inp_u8[2 + (i * 16) + ((2 * j) + 0) * INP_WID];
            let p_inp_line1_pel0_u8: *const u8 = &p_inp_u8[0 + (i * 16) + ((2 * j) + 1) * INP_WID];
            let p_inp_line1_pel1_u8: *const u8 = &p_inp_u8[1 + (i * 16) + ((2 * j) + 1) * INP_WID];
            let p_inp_line1_pel2_u8: *const u8 = &p_inp_u8[2 + (i * 16) + ((2 * j) + 1) * INP_WID];
            let p_inp_line2_pel0_u8: *const u8 = &p_inp_u8[0 + (i * 16) + ((2 * j) + 2) * INP_WID];
            let p_inp_line2_pel1_u8: *const u8 = &p_inp_u8[1 + (i * 16) + ((2 * j) + 2) * INP_WID];
            let p_inp_line2_pel2_u8: *const u8 = &p_inp_u8[2 + (i * 16) + ((2 * j) + 2) * INP_WID];
            let p_inp_line3_pel0_u8: *const u8 = &p_inp_u8[0 + (i * 16) + ((2 * j) + 3) * INP_WID];
            let p_inp_line3_pel1_u8: *const u8 = &p_inp_u8[1 + (i * 16) + ((2 * j) + 3) * INP_WID];
            let p_inp_line3_pel2_u8: *const u8 = &p_inp_u8[2 + (i * 16) + ((2 * j) + 3) * INP_WID];

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

            let mut sum_line0_l_s16x8 = vdupq_n_s16(0);
            let mut sum_line1_l_s16x8 = vdupq_n_s16(0);
            let mut sum_line0_h_s16x8 = vdupq_n_s16(0);
            let mut sum_line1_h_s16x8 = vdupq_n_s16(0);

            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line0_pel0_l_s16x8, p_ker_s8[0] as i16); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line0_pel1_l_s16x8, p_ker_s8[1] as i16); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line0_pel2_l_s16x8, p_ker_s8[2] as i16); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line1_pel0_l_s16x8, p_ker_s8[3] as i16); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line1_pel1_l_s16x8, p_ker_s8[4] as i16); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line1_pel2_l_s16x8, p_ker_s8[5] as i16); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line2_pel0_l_s16x8, p_ker_s8[6] as i16); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line2_pel1_l_s16x8, p_ker_s8[7] as i16); 
            sum_line0_l_s16x8 = vmlaq_n_s16(sum_line0_l_s16x8, inp_line2_pel2_l_s16x8, p_ker_s8[8] as i16); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line1_pel0_l_s16x8, p_ker_s8[0] as i16); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line1_pel1_l_s16x8, p_ker_s8[1] as i16); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line1_pel2_l_s16x8, p_ker_s8[2] as i16); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line2_pel0_l_s16x8, p_ker_s8[3] as i16); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line2_pel1_l_s16x8, p_ker_s8[4] as i16); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line2_pel2_l_s16x8, p_ker_s8[5] as i16); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line3_pel0_l_s16x8, p_ker_s8[6] as i16); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line3_pel1_l_s16x8, p_ker_s8[7] as i16); 
            sum_line1_l_s16x8 = vmlaq_n_s16(sum_line1_l_s16x8, inp_line3_pel2_l_s16x8, p_ker_s8[8] as i16); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line0_pel0_h_s16x8, p_ker_s8[0] as i16); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line0_pel1_h_s16x8, p_ker_s8[1] as i16); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line0_pel2_h_s16x8, p_ker_s8[2] as i16); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line1_pel0_h_s16x8, p_ker_s8[3] as i16); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line1_pel1_h_s16x8, p_ker_s8[4] as i16); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line1_pel2_h_s16x8, p_ker_s8[5] as i16); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line2_pel0_h_s16x8, p_ker_s8[6] as i16); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line2_pel1_h_s16x8, p_ker_s8[7] as i16); 
            sum_line0_h_s16x8 = vmlaq_n_s16(sum_line0_h_s16x8, inp_line2_pel2_h_s16x8, p_ker_s8[8] as i16); 
            sum_line1_h_s16x8 = vmlaq_n_s16(sum_line1_h_s16x8, inp_line1_pel0_h_s16x8, p_ker_s8[0] as i16); 
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

            let p_out_line_u8: *mut u8 = &mut p_out_u8[(i * 8) + (j * OUT_WID)];

            vst1_u8(p_out_line_u8, shift_u8x8);
        }

        p_out_u8[OUT_WID - 1 + (j * OUT_WID)] = 0;
    }
}
