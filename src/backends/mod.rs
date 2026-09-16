// kaleidomo-core/src/backends/mod.rs
use image::{DynamicImage, GenericImageView};

use crate::{KaleidoSettings, KaleidoType};

/// Bilinear-samples `source` at floating point coordinates `(sx, sy)`, blending
/// the four nearest texels. Coordinates are clamped to the image bounds so
/// sampling near the edges never goes out of range. Shared by every CPU
/// backend's `anti_alias` path (scalar/SSE2/AVX2/NEON all delegate here for
/// the actual pixel math, since gathering from an `image::DynamicImage` isn't
/// itself vectorizable) so the smoothing looks identical across backends.
#[inline]
pub(crate) fn bilinear_sample(source: &DynamicImage, sx: f32, sy: f32, sw: u32, sh: u32) -> [u8; 4] {
    if sw == 0 || sh == 0 {
        return [0, 0, 0, 0];
    }
    let sx = sx.clamp(0.0, (sw - 1) as f32);
    let sy = sy.clamp(0.0, (sh - 1) as f32);
    let x0 = sx.floor() as u32;
    let y0 = sy.floor() as u32;
    let x1 = (x0 + 1).min(sw - 1);
    let y1 = (y0 + 1).min(sh - 1);
    let fx = sx - x0 as f32;
    let fy = sy - y0 as f32;

    let p00 = source.get_pixel(x0, y0).0;
    let p10 = source.get_pixel(x1, y0).0;
    let p01 = source.get_pixel(x0, y1).0;
    let p11 = source.get_pixel(x1, y1).0;

    let mut out = [0u8; 4];
    for c in 0..4 {
        let top = p00[c] as f32 * (1.0 - fx) + p10[c] as f32 * fx;
        let bottom = p01[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
        out[c] = (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8;
    }
    out
}

#[inline]
fn cubic(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    0.5 * ((2.0 * p1) + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t)
}

pub(crate) fn bicubic_sample(source: &DynamicImage, sx: f32, sy: f32, sw: u32, sh: u32) -> [u8; 4] {
    if sw == 0 || sh == 0 { return [0; 4]; }
    let bx = sx.floor() as i32; let by = sy.floor() as i32;
    let fx = sx - sx.floor(); let fy = sy - sy.floor();
    let mut rows = [[0.0f32; 4]; 4];
    for row in 0..4 { let yy = (by + row as i32 - 1).clamp(0, sh as i32 - 1) as u32; let mut p = [[0u8; 4]; 4]; for col in 0..4 { let xx = (bx + col as i32 - 1).clamp(0, sw as i32 - 1) as u32; p[col] = source.get_pixel(xx, yy).0; } for channel in 0..4 { rows[row][channel] = cubic(p[0][channel] as f32, p[1][channel] as f32, p[2][channel] as f32, p[3][channel] as f32, fx); } }
    let mut out = [0u8; 4]; for channel in 0..4 { out[channel] = cubic(rows[0][channel], rows[1][channel], rows[2][channel], rows[3][channel], fy).round().clamp(0.0, 255.0) as u8; } out
}

#[inline]
pub(crate) fn reconstruction_sample<const MODE: u8>(source: &DynamicImage, sx: f32, sy: f32, sw: u32, sh: u32) -> [u8; 4] {
    if MODE == 2 { bicubic_sample(source, sx, sy, sw, sh) } else { bilinear_sample(source, sx, sy, sw, sh) }
}

/// Same as [`bilinear_sample`], but also applies an HSV hue rotation to the
/// blended pixel. Used by the `anti_alias` path when `hue_rotation != 0`; the
/// fast SIMD/integer hue-shift path (`store_pixel_hue_shift`) is used when
/// `anti_alias` is disabled, since it doesn't need per-pixel scalar blending.
#[inline]
pub(crate) fn bilinear_sample_hue_shift(
    source: &DynamicImage,
    sx: f32,
    sy: f32,
    sw: u32,
    sh: u32,
    hue_shift_degrees: f32,
) -> [u8; 4] {
    let [r, g, b, a] = bilinear_sample(source, sx, sy, sw, sh);
    if hue_shift_degrees == 0.0 {
        return [r, g, b, a];
    }
    let (h, s, v) = rgb_to_hsv_scalar(r, g, b);
    let h = (h + hue_shift_degrees).rem_euclid(360.0);
    let (r2, g2, b2) = hsv_to_rgb_scalar(h, s, v);
    [r2, g2, b2, a]
}

#[inline]
pub(crate) fn reconstruction_sample_hue_shift<const MODE: u8>(source: &DynamicImage, sx: f32, sy: f32, sw: u32, sh: u32, hue: f32) -> [u8; 4] {
    let [r, g, b, a] = reconstruction_sample::<MODE>(source, sx, sy, sw, sh);
    let (h, s, v) = rgb_to_hsv_scalar(r, g, b);
    let (r, g, b) = hsv_to_rgb_scalar((h + hue).rem_euclid(360.0), s, v);
    [r, g, b, a]
}

/// Plain-scalar RGB→HSV conversion (0-255 RGB in, degrees/0-1/0-1 HSV out).
/// Kept separate from each backend's SIMD hue-rotation intrinsics so the
/// `anti_alias` blending path doesn't need to touch per-backend register code.
#[inline]
fn rgb_to_hsv_scalar(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let (r, g, b) = (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
    let c_max = r.max(g).max(b);
    let c_min = r.min(g).min(b);
    let delta = c_max - c_min;

    let h = if delta == 0.0 {
        0.0
    } else if c_max == r {
        60.0 * (((g - b) / delta).rem_euclid(6.0))
    } else if c_max == g {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };

    let s = if c_max == 0.0 { 0.0 } else { delta / c_max };
    let v = c_max;
    (h, s, v)
}

/// Plain-scalar HSV→RGB conversion, inverse of [`rgb_to_hsv_scalar`].
#[inline]
fn hsv_to_rgb_scalar(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let h_sector = h / 60.0;
    let x = c * (1.0 - (h_sector.rem_euclid(2.0) - 1.0).abs());
    let m = v - c;

    let (rp, gp, bp) = if h_sector < 1.0 {
        (c, x, 0.0)
    } else if h_sector < 2.0 {
        (x, c, 0.0)
    } else if h_sector < 3.0 {
        (0.0, c, x)
    } else if h_sector < 4.0 {
        (0.0, x, c)
    } else if h_sector < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    (
        ((rp + m) * 255.0).round().clamp(0.0, 255.0) as u8,
        ((gp + m) * 255.0).round().clamp(0.0, 255.0) as u8,
        ((bp + m) * 255.0).round().clamp(0.0, 255.0) as u8,
    )
}

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

#[cfg(not(target_arch = "wasm32"))]
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
    /// Writes register lanes to `output` for derivative-aware texture gathers.
    unsafe fn write_lanes(self, output: &mut [f32]);
    /// Normalizes coordinates relative to the center.
    unsafe fn normalize_coords(&mut self, center: Self);
    /// Multiplies every lane by `factor`. Used for `aspect_correct`: scaling
    /// the vertical (dy) axis by the output canvas's width/height ratio
    /// before the angle/radius is computed, so the kaleidoscope pattern isn't
    /// visually stretched into an ellipse on non-square canvases.
    unsafe fn scale(self, factor: Self) -> Self;
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
    /// When `bilinear` is `true` (the `anti_alias` setting), samples are blended between the four
    /// nearest source texels instead of rounding to the nearest one.
    unsafe fn store_pixel<const SAMPLING_MODE: u8>(
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
    /// When `bilinear` is `true` (the `anti_alias` setting), the source sample is blended between the
    /// four nearest texels (via the shared scalar `bilinear_sample_hue_shift` helper) before the hue
    /// shift is applied, instead of using the fast SIMD nearest-neighbor + hue-shift path.
    unsafe fn store_pixel_hue_shift<const SAMPLING_MODE: u8>(buff: &mut [u8], x: u32, sx: Self, sy: Self, source: &DynamicImage, source_width: u32, source_height: u32, hue_shift_vec: Self, two_fifty_five: Self, hundred: Self, zero: Self, six: Self, sixty: Self, one: Self, two: Self, four: Self, three_sixty: Self, five: Self, three: Self);
}

#[cfg(target_arch = "aarch64")]
pub type Register = core::arch::aarch64::float32x4_t;
#[cfg(target_arch = "x86_64")]
pub type Register = core::arch::x86_64::__m256;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub type Register = f32;

#[cfg(not(target_arch = "wasm32"))]
pub fn inner_loop<B: KaleidoBackend + DaydreamBackend, const SAMPLING_MODE: u8>(
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

        // `aspect_correct`: scale dy by the canvas's width/height ratio before the
        // angle/radius is computed, so the mirrored wedges stay proportional
        // instead of stretching into an ellipse on non-square canvases. Disabled
        // by default (see `KaleidoSettings::aspect_correct`), so behavior for
        // existing presets/output is unchanged unless explicitly turned on.
        let aspect_ratio = B::load_with_single_f32(
            settings.output_size_w as f32 / settings.output_size_h.max(1) as f32,
        );

        let batch_bytes = B::NUM_FLOATS * size_of::<f32>();
        row.chunks_mut(batch_bytes)
            .enumerate()
            .for_each(|(x, output)| {
                // SIMD stores operate on a complete register. Render a final
                // partial batch into stack storage, then copy only its active
                // lanes so arbitrary output widths do not leave black pixels.
                let output_len = output.len();
                let output_ptr = output.as_mut_ptr();
                let mut tail = [0u8; 32];
                let buff: &mut [u8] = if output_len == batch_bytes {
                    output
                } else {
                    &mut tail[..batch_bytes]
                };
                let x = x as u32 * B::NUM_FLOATS as u32;
                let (mut dx, mut dy) = B::load_coords(x as u32, y as u32);
                dx.normalize_coords(center_x);
                dy.normalize_coords(center_y);
                if settings.aspect_correct {
                    dy = dy.scale(aspect_ratio);
                }
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
                    B::store_pixel_hue_shift::<SAMPLING_MODE>(buff, x, sx, sy, source, source_width, source_height, hue_shift_vec, two_fifty_five, hundred, zero, six, sixty, one, two, four, three_sixty, five, three);
                } else {
                    B::store_pixel::<SAMPLING_MODE>(buff, x, sx, sy, source, source_width, source_height);
                }
                if output_len != batch_bytes {
                    core::ptr::copy_nonoverlapping(buff.as_ptr(), output_ptr, output_len);
                }
            });
        }
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn map_batch<B: KaleidoBackend>(x: u32, y: u32, settings: &KaleidoSettings) -> (B, B) {
    unsafe {
        let (mut dx, mut dy) = B::load_coords(x, y);
        let center_x = B::load_with_single_f32(settings.output_size_w as f32 * 0.5 + settings.offset_x as f32);
        let center_y = B::load_with_single_f32(settings.output_size_h as f32 * 0.5 + settings.offset_y as f32);
        dx.normalize_coords(center_x); dy.normalize_coords(center_y);
        if settings.aspect_correct { dy = dy.scale(B::load_with_single_f32(settings.output_size_w as f32 / settings.output_size_h.max(1) as f32)); }
        let half = B::load_with_single_f32(settings.output_size_w as f32 * 0.5);
        let slice = B::load_with_single_f32(2.0 * core::f32::consts::PI / settings.count.max(1) as f32);
        let two_pi = B::load_with_single_f32(2.0 * core::f32::consts::PI);
        let tile = B::load_with_single_f32(settings.tile_count);
        let zoom = B::load_with_single_f32(settings.zoom);
        let rotation = B::load_with_single_f32(settings.triangle_rotation_rad);
        let tx = B::load_with_single_f32(settings.triangle_center_x);
        let ty = B::load_with_single_f32(settings.triangle_center_y);
        match settings.kaleido_type {
            KaleidoType::Radial => { let (r, theta) = B::map_to_polar(dx, dy, settings.zoom); let angle = B::compute_angle(theta, slice, settings.triangle_rotation_rad); B::compute_source_pixel_coords(angle, r, tx, ty) }
            KaleidoType::Square => B::map_square(dx, dy, half, slice, two_pi, tile, zoom, rotation, tx, ty),
            KaleidoType::Diamond => B::map_diamond(dx, dy, half, slice, two_pi, tile, zoom, rotation, tx, ty),
            KaleidoType::Hexagonal => B::map_hexagonal(dx, dy, half, slice, two_pi, tile, zoom, rotation, tx, ty, B::load_with_single_f32(3.0f32.sqrt())),
            KaleidoType::HexagonalFlatTop => B::map_hexagonal_flat_top(dx, dy, half, slice, two_pi, tile, zoom, rotation, tx, ty, B::load_with_single_f32(3.0f32.sqrt())),
        }
    }
}

/// Derivative-aware SIMD row loop. Coordinate warping remains vectorized; only
/// the data-dependent texture gathers are resolved per lane from the shared mip pyramid.
#[cfg(not(target_arch = "wasm32"))]
pub fn inner_loop_enhanced<B: KaleidoBackend + DaydreamBackend>(
    y: usize,
    row: &mut [u8],
    settings: &KaleidoSettings,
    pyramid: &crate::enhancement::CpuMipPyramid,
    pipeline: &crate::enhancement::EnhancementPipeline,
    hue_rotation: u32,
) {
    unsafe {
        let (source_width, source_height) = pyramid.dimensions();
        let tolerance = match settings.anti_alias { 0 => 0.0, 2 => 2.0, _ => 1.0 };
        row.chunks_mut(B::NUM_FLOATS * 4).enumerate().for_each(|(batch, output)| {
            let x = batch as u32 * B::NUM_FLOATS as u32;
            let (sx, sy) = map_batch::<B>(x, y as u32, settings);
            let (sx_dx, sy_dx) = map_batch::<B>(x + 1, y as u32, settings);
            let (sx_dy, sy_dy) = map_batch::<B>(x, y as u32 + 1, settings);
            let (mut xs, mut ys, mut xdx, mut ydx, mut xdy, mut ydy) = ([0.0f32; 8], [0.0f32; 8], [0.0f32; 8], [0.0f32; 8], [0.0f32; 8], [0.0f32; 8]);
            sx.write_lanes(&mut xs); sy.write_lanes(&mut ys); sx_dx.write_lanes(&mut xdx); sy_dx.write_lanes(&mut ydx); sx_dy.write_lanes(&mut xdy); sy_dy.write_lanes(&mut ydy);
            let active_lanes = output.len() / 4;
            for lane in 0..active_lanes {
                if xs[lane] < -tolerance || ys[lane] < -tolerance || xs[lane] >= source_width as f32 + tolerance || ys[lane] >= source_height as f32 + tolerance { continue; }
                let derivatives = crate::enhancement::UvDerivatives { dx: [xdx[lane] - xs[lane], ydx[lane] - ys[lane]], dy: [xdy[lane] - xs[lane], ydy[lane] - ys[lane]] };
                let sampled = pyramid.sample(pipeline, [xs[lane], ys[lane]], derivatives);
                let color = crate::enhancement::rotate_hue_rgba(sampled, hue_rotation as f32);
                output[lane * 4..lane * 4 + 4].copy_from_slice(&color);
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
