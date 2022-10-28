//use std::fs;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::time::{Instant};

const ITERATION: usize = 1000;
const INP_WID: usize = 1280;
const INP_HEI: usize = 720;
const OUT_WID: usize = 640;
const OUT_HEI: usize = 360;

fn main() {
	let inp_filename = r"E:\ICSP\TestSequences\1280x720\bigships_1280x720.yuv";
	let out_filename = r"bigships_1280x720.y";
	let out_filtered_filename = r"bigships_1280x720_filtered.y";
	let out_pool_filename = r"bigships_640x360_pool.y";
	
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

	let mut a_out_filtered_y_u8: Vec<u8> = vec![0; INP_WID * INP_HEI];
	let a_coefficient_s16: [i16; 3 * 3] = [1, 2, 1, 2, 4, 2, 1, 2, 1];
	let mut a_out_pool_y_u8: Vec<u8> = vec![0; OUT_WID * OUT_HEI];
	
	let start = Instant::now();

	let mut count = 0;
	
    loop {
        count += 1;

		// filtering
		for j in 0..(INP_HEI - 1) {
			for i in 0..(INP_WID - 1) {
				let mut sum_s32: i32 = 0;
				
				for k_h in 0..2 {
					for k_w in 0..2 {
						sum_s32 += (a_out_y_u8[k_w + k_h * INP_WID + i + j * INP_WID] as i16 * a_coefficient_s16[k_w + k_h * 3]) as i32;
					}
				}
				
				sum_s32 >>= 3;
				if sum_s32 > 255 {
					sum_s32 = 255;
				}
				if sum_s32 < 0 {
					sum_s32 = 0;
				}
				
				a_out_filtered_y_u8[i + j * INP_WID] = sum_s32 as u8;
			}
		}
		
		// max pooling
		for j in 0..(OUT_HEI - 1) {
			for i in 0..(OUT_WID - 1) {
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
				
				a_out_pool_y_u8[i + j * OUT_WID] = max_u8;
			}
		}

        if count == ITERATION {
            break;
        }
    }
	
	let duration = start.elapsed();

    println!("Time elapsed in expensive_function() is: {:?}", duration);

	let mut out_file = File::create(out_filtered_filename).expect("failed to open out filtered file");
	
	out_file.write_all(&a_out_filtered_y_u8).expect("failed to write file");

	let mut out_file = File::create(out_pool_filename).expect("failed to open out pool file");
	
	out_file.write_all(&a_out_pool_y_u8).expect("failed to write file");
}
