//use std::fs;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::time::{Instant};

use filtering::INP_WID;
use filtering::INP_HEI;
use filtering::OUT_WID;
use filtering::OUT_HEI;
//use crate::natr::func_natr;
//use crate::neon::func_neon;

//pub mod defines;
//pub mod natr;
//pub mod neon;

const ITERATION: usize = 1000;

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

        filtering::run_filter_natr(&a_coefficient_s8, &a_out_y_u8, &mut a_out_pool_y_natr_u8);

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

        filtering::run_filter_neon(&a_coefficient_s8, &a_out_y_u8, &mut a_out_pool_y_neon_u8);

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
