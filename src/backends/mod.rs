use image::DynamicImage;

use crate::{KaleidoSettings, KaleidoType};

#[cfg(any(all(test, target_arch = "aarch64"), target_arch = "aarch64"))]
mod neon;

#[cfg(any(
    all(test, any(target_arch = "x86_64", target_arch = "x86")),
    any(target_arch = "x86_64", target_arch = "x86")
))]
pub mod avx2;

#[cfg(any(
    all(test, any(target_arch = "x86_64", target_arch = "x86")),
    any(target_arch = "x86_64", target_arch = "x86")
))]
pub mod sse2;

#[cfg(
    any(
        all(
            not(any(target_arch = "aarch64", target_arch = "wasm32")), 
            all(not(target_arch = "wasm32"), test)
        ),
        feature = "soft_backend",
        any(target_arch = "x86_64", target_arch = "x86")
    )
)]
mod scalar;

pub mod gpu;

#[cfg(not(target_arch = "wasm32"))]
pub trait KaleidoBackend: Sized + Copy {
    /// The number of floats that the register can hold.
    const NUM_FLOATS: usize;
    /// Loads an array of floats into the registers. Only works with cfg test.
    #[cfg(test)]
    unsafe fn load_f32s(input: &[f32]) -> Vec<Self>;
    /// Extracts the computed floats from the register and stores them into the output buffer. Only works with cfg test.
    #[cfg(test)]
    unsafe fn store_f32s(&self, output: &mut [f32]);
    /// Loads a single f32 value into all lanes of the register.
    unsafe fn load_with_single_f32(input: f32) -> Self;
    /// Loads coordinates into a register, loading NUM_FLOATS pairs.
    unsafe fn load_coords(x: u32, y: u32) -> (Self, Self);
    /// Normalizes coordinates relative to the center.
    unsafe fn normalize_coords(&mut self, center: Self);
    /// Performs the four quadrant arctangent of self (y) and other (x) in radians.
    unsafe fn atan2_k(&self, other: Self) -> Self;
    /// Maps the coordinates to polar coordinates, returning a register of (r, theta).
    unsafe fn map_to_polar(dx: Self, dy: Self, zoom: f32) -> (Self, Self);
    /// Computes the final angle from the UI.
    unsafe fn compute_angle(theta: Self, slice_angle: Self, triangle_rotation_rad: f32) -> Self;
    /// Computes the source pixel coordinates from the computed angle and radial distance.
    unsafe fn compute_source_pixel_coords(
        computed_angle: Self,
        r_sampled: Self,
        triangle_center_x: Self,
        triangle_center_y: Self,
    ) -> (Self, Self);
    /// Stores pixels into the output buffer from the source image, given the computed source coordinates.
    unsafe fn store_pixel(
        output: &mut [u8],
        x: u32,
        sx: Self,
        sy: Self,
        source: &DynamicImage,
        sw: u32,
        sh: u32,
    );

    /// Folds the coordinates for square kaleidoscope.
    //fn fold_square(input: Self, count: u32, tile_size: Self) -> Self;

    unsafe fn map_square(
        dx: Self,
        dy: Self,
        width_over_2: Self,
        slice_angle: Self,
        two_pi: Self,
        tile_count: Self,
        zoom: Self,
        rotation: Self,
        tx: Self,
        ty: Self,
    ) -> (Self, Self);

    unsafe fn map_diamond(
        dx: Self,
        dy: Self,
        width_over_2: Self,
        slice_angle: Self,
        two_pi: Self,
        tile_count: Self,
        zoom: Self,
        rotation: Self,
        tx: Self,
        ty: Self,
    ) -> (Self, Self);

    unsafe fn map_hexagonal(
        dx: Self,
        dy: Self,
        width_over_2: Self,
        slice_angle: Self,
        two_pi: Self,
        tile_count: Self,
        zoom: Self,
        rotation: Self,
        tx: Self,
        ty: Self,
        sqrt3: Self,
    ) -> (Self, Self);

    unsafe fn map_hexagonal_flat_top(
        dx: Self,
        dy: Self,
        width_over_2: Self,
        slice_angle: Self,
        two_pi: Self,
        tile_count: Self,
        zoom: Self,
        triangle_rotation_rad: Self,
        triangle_center_x: Self,
        triangle_center_y: Self,
        sqrt3: Self,
    ) -> (Self, Self);

    unsafe fn fold_point_into_wedge_fixed(
        x: Self,
        y: Self,
        slice_angle: Self,
        two_pi: Self,
    ) -> (Self, Self);
    unsafe fn reflect_across_line(x: Self, y: Self, lx: Self, ly: Self) -> (Self, Self);

    unsafe fn hex_round(q: Self, r: Self) -> (Self, Self);

    unsafe fn store_pixel_rgba8(
        output: &mut [u8],
        sx: Self,
        sy: Self,
        source: &[u8],
        sw: u32,
        sh: u32,
    );

    unsafe fn source_space_rotation(
        local_x: Self,
        local_y: Self,
        triangle_rotation_rad: Self,
        triangle_center_x: Self,
        triangle_center_y: Self,
        radius: Self,
        two_pi: Self,
        slice_angle: Self,
        width_over_2: Self,
        zoom: Self,
    ) -> (Self, Self);
}

#[cfg(not(target_arch = "wasm32"))]
pub trait DaydreamBackend: KaleidoBackend {
    type IntegerRegister: Sized + Copy;
    /// Loads integer pixel coordinates into RGB registers.
    unsafe fn load_pixels(input: &[[u8; 4]]) -> (Self::IntegerRegister, Self::IntegerRegister, Self::IntegerRegister, Self::IntegerRegister);
    /// Takes an (R, G, B) color in the range [0, 255] and converts it to
    /// (H, S, V) where H is in [0, 360] and S, V are in [0, 1].
    unsafe fn rgb_to_hsv(
        r: Self::IntegerRegister, 
        g: Self::IntegerRegister, 
        b: Self::IntegerRegister, 
        two_fifty_five: Self, 
        hundred: Self, 
        zero: Self, 
        six: Self, 
        sixty: Self,
        one: Self,
        two: Self,
        four: Self,
    ) -> (Self, Self, Self);
    /// Takes an (H, S, V) color where H is in [0, 360] and S, V are in [0, 1], 
    /// and converts it to (R, G, B) in the range [0, 255].
    unsafe fn hsv_to_rgb(
        h: Self, 
        s: Self, 
        v: Self,
        hundred: Self,
        sixty: Self,
        two_fifty_five: Self,
        zero: Self,
        five: Self,
        four: Self,
        three: Self,
        two: Self,
        one: Self,
    ) -> (Self::IntegerRegister, Self::IntegerRegister, Self::IntegerRegister);
    
    /// Adjusts the hue by + hue_shift degrees.
    unsafe fn adjust_hue(
        h: Self,
        hue_shift: Self,
        three_sixty: Self,
    ) -> Self;

    unsafe fn extract_pixels(
        r: Self::IntegerRegister, 
        g: Self::IntegerRegister, 
        b: Self::IntegerRegister,
        a: Self::IntegerRegister,
    ) -> [[u8; 4]; Self::NUM_FLOATS];

    /// Stores a pixel in the output buffer, applying a hue shift to the source pixel before sampling.
    unsafe fn store_pixel_hue_shift(buff: &mut [u8], x: u32, sx: Self, sy: Self, source: &DynamicImage, source_width: u32, source_height: u32, hue_shift_vec: Self, two_fifty_five: Self, hundred: Self, zero: Self, six: Self, sixty: Self, one: Self, two: Self, four: Self, three_sixty: Self, five: Self, three: Self);
}

#[cfg(target_arch = "aarch64")]
pub type Register = core::arch::aarch64::float32x4_t;
#[cfg(target_arch = "x86_64")]
pub type Register = core::arch::x86_64::__m256;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub type Register = f32;

#[cfg(not(target_arch = "wasm32"))]
pub fn inner_loop<B: KaleidoBackend + DaydreamBackend>(
    y: usize,
    row: &mut [u8],
    zoom: f32,
    source: &DynamicImage,
    settings: &KaleidoSettings,
    width_over_2: f32,
    center_x: f32,
    center_y: f32,
    slice_angle: f32,
    source_width: u32,
    source_height: u32,
    hue_rotate: u32,
) {
    unsafe {
        let triangle_center_x = B::load_with_single_f32(settings.triangle_center_x);
        let triangle_center_y = B::load_with_single_f32(settings.triangle_center_y);
        let center_x = B::load_with_single_f32(center_x);
        let center_y = B::load_with_single_f32(center_y);
        let width_over_2 = B::load_with_single_f32(width_over_2);
        let z = B::load_with_single_f32(zoom);
        let tile_count = B::load_with_single_f32(settings.tile_count);
        let slice_angle = B::load_with_single_f32(slice_angle);
        let two_pi = B::load_with_single_f32(2.0 * core::f32::consts::PI);
        let sqrt3 = B::load_with_single_f32(3.0f32.sqrt());
        let triangle_rotation_rad = B::load_with_single_f32(settings.triangle_rotation_rad);
        let hue_shift = hue_rotate % 360;
        let hue_shift_vec = B::load_with_single_f32(hue_shift as f32);
        let two_fifty_five = B::load_with_single_f32(255.0);
        let hundred = B::load_with_single_f32(100.0);
        let zero = B::load_with_single_f32(0.0);
        let six = B::load_with_single_f32(6.0);
        let sixty = B::load_with_single_f32(60.0);
        let one = B::load_with_single_f32(1.0);
        let two = B::load_with_single_f32(2.0);
        let four = B::load_with_single_f32(4.0);
        let three = B::load_with_single_f32(3.0);
        let three_sixty = B::load_with_single_f32(360.0);
        let five = B::load_with_single_f32(5.0);

        row.chunks_exact_mut(B::NUM_FLOATS * size_of::<f32>())
            .enumerate()
            .for_each(|(x, buff)| {
                let x = x as u32 * B::NUM_FLOATS as u32;
                let (mut dx, mut dy) = B::load_coords(x as u32, y as u32);
                dx.normalize_coords(center_x);
                dy.normalize_coords(center_y);
                let (sx, sy) = match settings.kaleido_type {
                    KaleidoType::Radial => {
                        let (r_sampled, theta) = B::map_to_polar(dx, dy, zoom);
                        let computed_angle =
                            B::compute_angle(theta, slice_angle, settings.triangle_rotation_rad);
                        B::compute_source_pixel_coords(
                            computed_angle,
                            r_sampled,
                            triangle_center_x,
                            triangle_center_y,
                        )
                    }
                    KaleidoType::Square => B::map_square(
                        dx,
                        dy,
                        width_over_2,
                        slice_angle,
                        two_pi,
                        tile_count,
                        z,
                        triangle_rotation_rad,
                        triangle_center_x,
                        triangle_center_y,
                    ),
                    KaleidoType::Diamond => B::map_diamond(
                        dx,
                        dy,
                        width_over_2,
                        slice_angle,
                        two_pi,
                        tile_count,
                        z,
                        triangle_rotation_rad,
                        triangle_center_x,
                        triangle_center_y,
                    ),
                    KaleidoType::Hexagonal => B::map_hexagonal(
                        dx,
                        dy,
                        width_over_2,
                        slice_angle,
                        two_pi,
                        tile_count,
                        z,
                        triangle_rotation_rad,
                        triangle_center_x,
                        triangle_center_y,
                        sqrt3,
                    ),
                    KaleidoType::HexagonalFlatTop => B::map_hexagonal_flat_top(
                        dx,
                        dy,
                        width_over_2,
                        slice_angle,
                        two_pi,
                        tile_count,
                        z,
                        triangle_rotation_rad,
                        triangle_center_x,
                        triangle_center_y,
                        sqrt3,
                    ),
                };

                //B::store_pixel_rgba8(buff, sx, sy, source.as_bytes(), source_width, source_height);
                if hue_shift != 0 {
                    B::store_pixel_hue_shift(buff, x, sx, sy, source, source_width, source_height, hue_shift_vec, two_fifty_five, hundred, zero, six, sixty, one, two, four, three_sixty, five, three);
                } else {
                    B::store_pixel(buff, x, sx, sy, source, source_width, source_height);
                }
            });
        }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atan2() {
        unsafe {
            let x_arr = [0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0];
            let x = Register::load_f32s(&x_arr);
            let y_arr = [1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 1.0, -1.0];
            let y = Register::load_f32s(&y_arr);
            let result = x
                .iter()
                .zip(y.iter())
                .map(|(x, y)| y.atan2_k(*x))
                .collect::<Vec<_>>();
            let expected: Vec<f32> = x_arr
                .iter()
                .zip(y_arr.iter())
                .map(|(x, y)| y.atan2_k(*x))
                .collect();

            result
                .iter()
                .zip(expected.chunks_exact(Register::NUM_FLOATS))
                .for_each(|(reg, expected_chunk)| {
                    let mut output = [0.0; Register::NUM_FLOATS];
                    reg.store_f32s(&mut output);
                    output
                        .iter_mut()
                        .zip(expected_chunk.iter())
                        .for_each(|(o, e)| {
                            *o = (*o - e).abs();
                            assert!(*o < 0.0001, "Expected {}, got {}, diff {}", e, o, *o - e);
                        });
                });
            }
    }

    #[test]
    fn test_map_to_polar() {
        unsafe {
            let x_arr = [0.0, 1.0, 0.0, -1.0];
            let x = Register::load_f32s(&x_arr);
            let y_arr = [1.0, 0.0, -1.0, 0.0];
            let y = Register::load_f32s(&y_arr);
            let zoom = 1.0;
            let result = x
                .iter()
                .zip(y.iter())
                .map(|(x, y)| Register::map_to_polar(*x, *y, zoom))
                .collect::<Vec<_>>();
            let expected: Vec<(f32, f32)> = x_arr
                .iter()
                .zip(y_arr.iter())
                .map(|(x, y)| f32::map_to_polar(*x, *y, zoom))
                .collect();

            let result_r = result.iter().map(|(r, _)| *r).collect::<Vec<_>>();
            let result_theta = result.iter().map(|(_, theta)| *theta).collect::<Vec<_>>();
            // Extract expected values into separate flat lists
            let expected_r: Vec<f32> = expected.iter().map(|(r, _)| *r).collect();
            let expected_theta: Vec<f32> = expected.iter().map(|(_, theta)| *theta).collect();

            result_r
                .iter()
                .zip(expected_r.chunks_exact(Register::NUM_FLOATS))
                .for_each(|(reg, expected_chunk)| {
                    let mut output = [0.0; Register::NUM_FLOATS];
                    reg.store_f32s(&mut output);
                    output.iter().zip(expected_chunk.iter()).for_each(|(o, e)| {
                        assert!(
                            (*o - *e).abs() < 0.0001,
                            "Expected {}, got {}, diff {}",
                            e,
                            o,
                            *o - *e
                        );
                    });
                });
            result_theta
                .iter()
                .zip(expected_theta.chunks_exact(Register::NUM_FLOATS))
                .for_each(|(reg, expected_chunk)| {
                    let mut output = [0.0; Register::NUM_FLOATS];
                    reg.store_f32s(&mut output);
                    output.iter().zip(expected_chunk.iter()).for_each(|(o, e)| {
                        assert!(
                            (*o - *e).abs() < 0.0001,
                            "Expected {}, got {}, diff {}",
                            e,
                            o,
                            *o - *e
                        );
                    });
                });
            }
    }

    #[test]
    fn test_reflect_across_line() {
        unsafe {
            let x_arr = [1.2, 2.5, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8];
            let y_arr = [1.1, 2.2, 3.5, 4.2, 5.3, 6.4, 7.9, 8.0];
            let lx_arr = [11.1, 22.2, 33.3, 44.4, 55.5, 66.6, 77.7, 88.8];
            let ly_arr = [11.5, 22.5, 33.5, 44.5, 55.6, 66.7, 77.8, 88.9];
            let x = Register::load_f32s(&x_arr);
            let y = Register::load_f32s(&y_arr);
            let lx = Register::load_f32s(&lx_arr);
            let ly = Register::load_f32s(&ly_arr);
            let result = x
                .iter()
                .zip(y.iter())
                .zip(lx.iter())
                .zip(ly.iter())
                .map(|(((x, y), lx), ly)| Register::reflect_across_line(*x, *y, *lx, *ly))
                .collect::<Vec<_>>();
            let expected: Vec<(f32, f32)> = x_arr
                .iter()
                .zip(y_arr.iter())
                .zip(lx_arr.iter())
                .zip(ly_arr.iter())
                .map(|(((x, y), lx), ly)| f32::reflect_across_line(*x, *y, *lx, *ly))
                .collect();
            for ((result_x, result_y), expected) in result
                .iter()
                .zip(expected.chunks_exact(Register::NUM_FLOATS))
            {
                let mut output_x = [0.0; Register::NUM_FLOATS];
                let mut output_y = [0.0; Register::NUM_FLOATS];
                result_x.store_f32s(&mut output_x);
                result_y.store_f32s(&mut output_y);
                output_x.iter().zip(expected.iter()).for_each(|(o, e)| {
                    assert!(
                        (*o - e.0).abs() < 0.01,
                        "Expected {}, got {}, diff {}",
                        e.0,
                        o,
                        *o - e.0
                    );
                });
                output_y.iter().zip(expected.iter()).for_each(|(o, e)| {
                    assert!(
                        (*o - e.1).abs() < 0.01,
                        "Expected {}, got {}, diff {}",
                        e.1,
                        o,
                        *o - e.1
                    );
                });
            }
        }
    }
}
