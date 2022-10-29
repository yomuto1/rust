//use std::fs;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::time::{Instant};

const ITERATION: usize = 1000;
const INP_WID: usize = 512;
const INP_HEI: usize = 512;
const OUT_WID: usize = 256;
const OUT_HEI: usize = 256;
const VAL_SHIFT: i32 = 4;

fn main() {
	let inp_filename = r"/home/hyuk/code/barbara_gray.raw";
	let out_filename = r"barbara_512x512.y";
	let out_filtered_filename = r"barbara_512x512_filtered.y";
	let out_pool_filename = r"barbara_256x256_pool.y";
	
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
		for j in 0..(INP_HEI - 2) {
			for i in 0..(INP_WID - 2) {
				let mut sum_s32: i32 = 0;
				
				for k_h in 0..3 {
					for k_w in 0..3 {
						let ker_s16: i16 = a_coefficient_s16[k_w + k_h * 3];
						let inp_u8: u8 = a_out_y_u8[k_w + i + (k_h + j) * INP_WID];

						sum_s32 += (inp_u8 as i16 * ker_s16) as i32;

						//if i == 0 && j == 0 {
						//	println!("{k_w}, {k_h}: {ker_s16}, {inp_u8}, {sum_s32}");
						//}
					}
				}

				//if i == 0 && j == 0 {
				//	println!("{i}, {j}: {sum_s32}");
				//}
				
				sum_s32 >>= VAL_SHIFT;
				if sum_s32 > 255 {
					sum_s32 = 255;
				}
				if sum_s32 < 0 {
					sum_s32 = 0;
				}
				
				//if i == 0 && j == 0 {
				//	println!("{sum_s32}");
				//}
				
				a_out_filtered_y_u8[i + j * INP_WID] = sum_s32 as u8;
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
