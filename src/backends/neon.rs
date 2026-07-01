use core::arch::aarch64::*;

use image::GenericImageView;

use crate::backends::DaydreamBackend;

use super::KaleidoBackend;

/// Computes atan using a polynomial approximation, returning a value in radians.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn atan(x: float32x4_t) -> float32x4_t {
    let a1 = vdupq_n_f32(0.99997726);
    let a3 = vdupq_n_f32(-0.33262347);
    let a5 = vdupq_n_f32(0.19354346);
    let a7 = vdupq_n_f32(-0.11643287);
    let a9 = vdupq_n_f32(0.05265332);
    let a11 = vdupq_n_f32(-0.01172120);

    let x2 = vmulq_f32(x, x);
    let mut result = a11;
    result = vfmaq_f32(a9, x2, result);
    result = vfmaq_f32(a7, x2, result);
    result = vfmaq_f32(a5, x2, result);
    result = vfmaq_f32(a3, x2, result);
    result = vfmaq_f32(a1, x2, result);
    result = vmulq_f32(result, x);
    result
}

/// Computes sine and cosine using a polynomial approximation, returning values in radians.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn sin_cos(angle: float32x4_t) -> (float32x4_t, float32x4_t) {
    let inv_pi_2 = vdupq_n_f32(0.63661977236); // 2/pi

    // 1. Range Reduction
    // k = round(angle / (pi/2))
    let k = vcvtaq_s32_f32(vmulq_f32(angle, inv_pi_2));
    let k_f = vcvtq_f32_s32(k);

    // x = angle - k * pi/2 (using split constants for higher precision)
    let p1 = vdupq_n_f32(-1.5707963267);
    let p2 = vdupq_n_f32(-4.37114e-8);
    let mut x = vfmaq_f32(angle, k_f, p1);
    x = vfmaq_f32(x, k_f, p2);

    let x2 = vmulq_f32(x, x);

    // 2. Polynomial for Sine (Range [-pi/4, pi/4])
    let s3 = vdupq_n_f32(-0.1666666666);
    let s5 = vdupq_n_f32(0.0083333333);
    let s7 = vdupq_n_f32(-0.0001984127);
    let mut sn = s7;
    sn = vfmaq_f32(s5, sn, x2);
    sn = vfmaq_f32(s3, sn, x2);
    let sin_poly = vmulq_f32(x, vfmaq_f32(vdupq_n_f32(1.0), sn, x2));

    // 3. Polynomial for Cosine (Range [-pi/4, pi/4])
    let c2 = vdupq_n_f32(-0.5);
    let c4 = vdupq_n_f32(0.0416666666);
    let c6 = vdupq_n_f32(-0.0013888888);
    let mut cn = c6;
    cn = vfmaq_f32(c4, cn, x2);
    cn = vfmaq_f32(c2, cn, x2);
    let cos_poly = vfmaq_f32(vdupq_n_f32(1.0), cn, x2);

    // 4. Quadrant Logic
    // Bit 0 of k: Tells us if we swap sin/cos (odd quadrants)
    // Bit 1 of k: Involved in sign of sin
    // Bit 0+1 of k: Involved in sign of cos
    let k_u = vreinterpretq_u32_s32(k);
    let swap_mask = vtstq_u32(k_u, vdupq_n_u32(1));

    // Initial Selection
    let mut res_sin = vbslq_f32(swap_mask, cos_poly, sin_poly);
    let mut res_cos = vbslq_f32(swap_mask, sin_poly, cos_poly);

    // 5. Sign Management
    // Sin sign: bit 1 of k flips the sign
    let sin_sign = vshlq_n_u32(vandq_u32(k_u, vdupq_n_u32(2)), 30);
    // Cos sign: (k+1) bit 1 flips the sign
    let cos_sign = vshlq_n_u32(
        vandq_u32(vaddq_u32(k_u, vdupq_n_u32(1)), vdupq_n_u32(2)),
        30,
    );

    res_sin = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(res_sin), sin_sign));
    res_cos = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(res_cos), cos_sign));

    (res_sin, res_cos)
}

/// Computes x mod y for float32x4_t vectors, handling negative values correctly.
#[target_feature(enable = "neon")]
#[inline]
fn modulo(x: float32x4_t, y: float32x4_t) -> float32x4_t {
    let div = vdivq_f32(x, y);
    let div_floor = vrndmq_f32(div);
    vfmsq_f32(x, div_floor, y)
}

impl KaleidoBackend for float32x4_t {
    const NUM_FLOATS: usize = 4;

    #[cfg(test)]
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load_f32s(input: &[f32]) -> Vec<Self> {
        input
            .chunks_exact(Self::NUM_FLOATS)
            .map(|chunk| unsafe { vld1q_f32(chunk.as_ptr()) })
            .collect()
    }

    #[cfg(test)]
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn store_f32s(&self, output: &mut [f32]) {
        unsafe {
            vst1q_f32(output.as_mut_ptr(), *self);
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load_with_single_f32(input: f32) -> Self {
        vdupq_n_f32(input)
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load_coords(x: u32, y: u32) -> (Self, Self) {
        let f = x as f32;
        unsafe {
            let x_vec = vld1q_f32([f, f + 1.0, f + 2.0, f + 3.0].as_ptr());
            let y_vec = vdupq_n_f32(y as f32);
            (x_vec, y_vec)
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn normalize_coords(&mut self, center: Self) {
        *self = vsubq_f32(*self, center);
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn atan2_k(&self, other: Self) -> Self {
        unsafe {
            let pi = vdupq_n_f32(core::f32::consts::PI);
            let pi_2 = vdupq_n_f32(core::f32::consts::FRAC_PI_2);
            let sign_mask = vdupq_n_u32(0x80000000);

            // 1. Range Reduction: Test if |y| > |x|
            let swap_mask = vcagtq_f32(*self, other);

            // Numerator = min(|x|, |y|), Denominator = max(|x|, |y|)
            // We use absolute values to keep the polynomial input in [0, 1]
            let abs_y = vabsq_f32(*self);
            let abs_x = vabsq_f32(other);
            let nom = vbslq_f32(swap_mask, abs_x, abs_y);
            let den = vbslq_f32(swap_mask, abs_y, abs_x);

            let atan_input = vdivq_f32(nom, den);

            // 2. Compute polynomial atan(z)
            let mut result = atan(atan_input);

            // 3. If we swapped, result = pi/2 - result
            let sub_res = vsubq_f32(pi_2, result);
            result = vbslq_f32(swap_mask, sub_res, result);

            // 4. Handle quadrants (Reflection for negative inputs)
            // If y < 0, result = -result
            let y_sign = vandq_u32(vreinterpretq_u32_f32(*self), sign_mask);
            result = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(result), y_sign));

            // 5. If x < 0, result = copysign(pi, y) - result
            let x_neg_mask = vcltzq_f32(other);
            let pi_adj = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(pi), y_sign));
            let reflected = vsubq_f32(pi_adj, result);
            result = vbslq_f32(x_neg_mask, reflected, result);

            result
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn map_to_polar(dx: Self, dy: Self, zoom: f32) -> (Self, Self) {
        unsafe {
            let r = vsqrtq_f32(vaddq_f32(vmulq_f32(dx, dx), vmulq_f32(dy, dy)));
            let r_sampled = vdivq_f32(r, vdupq_n_f32(zoom));
            let mut theta = dy.atan2_k(dx);
            let less_than_zero_mask = vcltzq_f32(theta);
            let two_pi = vdupq_n_f32(2.0 * core::f32::consts::PI);
            let theta_adjustment = vandq_u32(vreinterpretq_u32_f32(two_pi), less_than_zero_mask);
            theta = vaddq_f32(theta, vreinterpretq_f32_u32(theta_adjustment));
            (r_sampled, theta)
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn compute_angle(theta: Self, slice_angle_v: Self, triangle_rotation_rad: f32) -> Self {
        let inv_slice_angle = vdivq_f32(vdupq_n_f32(1.0), slice_angle_v);

        // 1. Replicate .floor(): Use vrndmq_f32 (Round to Minus Infinity)
        let div = vmulq_f32(theta, inv_slice_angle);
        let slice_idx_f = vrndmq_f32(div);
        let slice_idx_i = vcvtq_s32_f32(slice_idx_f);

        // 2. Replicate theta % slice_angle:
        // rem = theta - (floor(theta / slice_angle) * slice_angle)
        let rem = vfmsq_f32(theta, slice_idx_f, slice_angle_v);

        // 3. Mirroring Logic
        let is_odd_mask = vtstq_s32(slice_idx_i, vdupq_n_s32(1));
        let if_odd = vsubq_f32(slice_angle_v, rem);

        let local_theta = vbslq_f32(is_odd_mask, if_odd, rem);

        vaddq_f32(local_theta, vdupq_n_f32(triangle_rotation_rad))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn compute_source_pixel_coords(
        computed_angle: Self,
        r_sampled: Self,
        triangle_center_x: Self,
        triangle_center_y: Self,
    ) -> (Self, Self) {
        unsafe {
            let (sin, cos) = sin_cos(computed_angle);
            let sx = vfmaq_f32(triangle_center_x, r_sampled, cos);
            let sy = vfmaq_f32(triangle_center_y, r_sampled, sin);
            (sx, sy)
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn store_pixel(
        output: &mut [u8],
        _x: u32,
        sx: Self,
        sy: Self,
        source: &image::DynamicImage,
        sw: u32,
        sh: u32,
    ) {
        unsafe {
            // 1. Check bounds on floats first to match 'sx >= 0.0 && sx < sw'
            let zero = vdupq_n_f32(0.0);
            let sw_v = vdupq_n_f32(sw as f32);
            let sh_v = vdupq_n_f32(sh as f32);

            let v_mask = vandq_u32(
                vandq_u32(vcgeq_f32(sx, zero), vcltq_f32(sx, sw_v)),
                vandq_u32(vcgeq_f32(sy, zero), vcltq_f32(sy, sh_v)),
            );

            if vmaxvq_u32(v_mask) == 0 {
                return;
            }

            // 2. Use truncation (round toward zero) to match 'as u32'
            let sx_i = vcvtq_u32_f32(sx);
            let sy_i = vcvtq_u32_f32(sy);

            let mut xs = [0u32; 4];
            let mut ys = [0u32; 4];
            let mut m = [0u32; 4];
            vst1q_u32(xs.as_mut_ptr(), sx_i);
            vst1q_u32(ys.as_mut_ptr(), sy_i);
            vst1q_u32(m.as_mut_ptr(), v_mask);

            for i in 0..4 {
                if m[i] != 0 {
                    let pixel = source.get_pixel(xs[i], ys[i]);
                    let base_idx = i * 4;
                    output[base_idx..base_idx + 4].copy_from_slice(&pixel.0);
                }
            }
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn store_pixel_rgba8(
        output: &mut [u8],
        sx: Self,
        sy: Self,
        source: &[u8],
        sw: u32,
        sh: u32,
    ) {
        unsafe {
            let zero = vdupq_n_f32(0.0);
            let sw_v = vdupq_n_f32(sw as f32);
            let sh_v = vdupq_n_f32(sh as f32);

            let v_mask = vandq_u32(
                vandq_u32(vcgeq_f32(sx, zero), vcltq_f32(sx, sw_v)),
                vandq_u32(vcgeq_f32(sy, zero), vcltq_f32(sy, sh_v)),
            );

            if vmaxvq_u32(v_mask) == 0 {
                return;
            }

            let sx_i = vcvtq_u32_f32(sx);
            let sy_i = vcvtq_u32_f32(sy);

            let mut xs = [0u32; 4];
            let mut ys = [0u32; 4];
            let mut m = [0u32; 4];

            vst1q_u32(xs.as_mut_ptr(), sx_i);
            vst1q_u32(ys.as_mut_ptr(), sy_i);
            vst1q_u32(m.as_mut_ptr(), v_mask);

            for i in 0..4 {
                if m[i] != 0 {
                    let idx = ((ys[i] * sw + xs[i]) * 4) as usize;
                    let out = i * 4;

                    let src = source.as_ptr().add(idx) as *const u32;
                    let dst = output.as_mut_ptr().add(out) as *mut u32;
                    *dst = *src;
                }
            }
        }
    }

    // fn fold_square(input: Self, count: u32, tile_size: Self) -> Self {
    //     unsafe {
    //         let period = vmulq_f32(tile_size, vdupq_n_f32(2.0));
    //         // Only one modulo is needed because vrndmq (floor) handles negatives
    //         let m = modulo(input, period);
    //         vabsq_f32(vsubq_f32(m, tile_size))
    //     }
    // }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn map_square(
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
    ) -> (Self, Self) {
        unsafe {
            let screen_size = vmulq_n_f32(width_over_2, 2.0);
            let tile_size = vdivq_f32(screen_size, tile_count);
            let half = vmulq_n_f32(tile_size, 0.5);

            let local_x = vsubq_f32(modulo(vaddq_f32(dx, half), tile_size), half);
            let local_y = vsubq_f32(modulo(vaddq_f32(dy, half), tile_size), half);

            Self::source_space_rotation(local_x, local_y, triangle_rotation_rad, triangle_center_x, triangle_center_y, half, two_pi, slice_angle, width_over_2, zoom)
        }
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn map_diamond(
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
    ) -> (Self, Self) {
        unsafe {
            let screen_size = vmulq_n_f32(width_over_2, 2.0);
            let tile = vdivq_f32(screen_size, tile_count);
            let half = vmulq_n_f32(tile, 0.5);

            let inv_sqrt2 = vdupq_n_f32(0.70710678118_f32);
            let u = vmulq_f32(vaddq_f32(dx, dy), inv_sqrt2);
            let v = vmulq_f32(vsubq_f32(dy, dx), inv_sqrt2);

            let local_u = vsubq_f32(modulo(vaddq_f32(u, half), tile), half);
            let local_v = vsubq_f32(modulo(vaddq_f32(v, half), tile), half);

            Self::source_space_rotation(local_u, local_v, triangle_rotation_rad, triangle_center_x, triangle_center_y, half, two_pi, slice_angle, width_over_2, zoom)
        }
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn map_hexagonal(
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
    ) -> (Self, Self) {
        unsafe {
            let screen_size = vmulq_n_f32(width_over_2, 2.0);

            let hex_radius = vdivq_f32(screen_size, vmulq_f32(sqrt3, tile_count));
            let one_over_three = vdupq_n_f32(1.0 / 3.0);
            let q = vdivq_f32(
                vsubq_f32(
                    vmulq_f32(vmulq_f32(sqrt3, one_over_three), dx),
                    vmulq_f32(one_over_three, dy),
                ),
                hex_radius,
            );
            let r = vdivq_f32(vmulq_f32(vmulq_n_f32(one_over_three, 2.0), dy), hex_radius);

            let (rq, rr) = Self::hex_round(q, r);

            let hex_cx = vmulq_f32(
                hex_radius,
                vmulq_f32(sqrt3, vfmaq_f32(rq, rr, vdupq_n_f32(0.5))),
            );
            let hex_cy = vmulq_f32(hex_radius, vmulq_n_f32(rr, 1.5));

            let local_x = vsubq_f32(dx, hex_cx);
            let local_y = vsubq_f32(dy, hex_cy);

            Self::source_space_rotation(local_x, local_y, triangle_rotation_rad, triangle_center_x, triangle_center_y, hex_radius, two_pi, slice_angle, width_over_2, zoom)
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
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
    ) -> (Self, Self) {
        unsafe {
            let screen_size = vmulq_n_f32(width_over_2, 2.0);

            let hex_radius = vdivq_f32(screen_size, vmulq_n_f32(tile_count, 1.5));
            let one_over_three = vdupq_n_f32(1.0 / 3.0);
        
            let q = vdivq_f32(
                vmulq_f32(
                    vmulq_n_f32(one_over_three, 2.0), 
                    dx
                ), 
                hex_radius
            );
            let r = vdivq_f32(
                vsubq_f32(
                    vmulq_f32(
                        vmulq_f32(sqrt3, one_over_three), 
                        dy
                    ),
                    vmulq_f32(one_over_three, dx),
                ),
                hex_radius,
            );

            let (rq, rr) = Self::hex_round(q, r);

            let hex_cx = vmulq_f32(hex_radius, vmulq_n_f32(rq, 1.5));
            let hex_cy = vmulq_f32(
                hex_radius,
                vmulq_f32(sqrt3, vfmaq_f32(rr, rq, vdupq_n_f32(0.5))),
            );

            let local_x = vsubq_f32(dx, hex_cx);
            let local_y = vsubq_f32(dy, hex_cy);

            Self::source_space_rotation(local_x, local_y, triangle_rotation_rad, triangle_center_x, triangle_center_y, hex_radius, two_pi, slice_angle, width_over_2, zoom)
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn hex_round(q: Self, r: Self) -> (Self, Self) {
        let s = vnegq_f32(vaddq_f32(q, r));
        let mut rq = vrndiq_f32(q);
        let mut rr = vrndiq_f32(r);
        let rs = vrndiq_f32(s);

        let q_diff = vabsq_f32(vsubq_f32(rq, q));
        let r_diff = vabsq_f32(vsubq_f32(rr, r));
        let s_diff = vabsq_f32(vsubq_f32(rs, s));

        let rq_mask = vandq_u32(vcgtq_f32(q_diff, r_diff), vcgtq_f32(q_diff, s_diff));

        let rr_candidate_mask = vcgtq_f32(r_diff, s_diff);
        let rr_mask = vandq_u32(vmvnq_u32(rq_mask), rr_candidate_mask);

        rq = vbslq_f32(rq_mask, vnegq_f32(vaddq_f32(rr, rs)), rq);
        rr = vbslq_f32(rr_mask, vnegq_f32(vaddq_f32(rq, rs)), rr);

        (rq, rr)
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn fold_point_into_wedge_fixed(
        x: Self,
        y: Self,
        slice_angle: Self,
        two_pi: Self,
    ) -> (Self, Self) {
        unsafe {
            let mut theta = y.atan2_k(x);
            let less_than_zero_mask = vcltzq_f32(theta);
            let theta_adjustment = vandq_u32(vreinterpretq_u32_f32(two_pi), less_than_zero_mask);
            theta = vaddq_f32(theta, vreinterpretq_f32_u32(theta_adjustment));

            let sector = vrndmq_f32(vdivq_f32(theta, slice_angle));
            let sector_angle = vmulq_f32(sector, slice_angle);

            let (sin_s, cos_s) = sin_cos(sector_angle);
            let xr = vfmaq_f32(vmulq_f32(x, cos_s), y, sin_s);
            let yr = vfmaq_f32(vnegq_f32(vmulq_f32(x, sin_s)), y, cos_s);

            let sector_i = vcvtq_u32_f32(sector);
            let odd_mask = vtstq_u32(sector_i, vdupq_n_u32(1));

            let half = Self::load_with_single_f32(0.5);
            let (ly, lx) = sin_cos(vmulq_f32(half, slice_angle));
            let (rx, ry) = Self::reflect_across_line(xr, yr, lx, ly);

            let final_x = vbslq_f32(odd_mask, rx, xr);
            let final_y = vbslq_f32(odd_mask, ry, yr);
            (final_x, final_y)
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn reflect_across_line(x: Self, y: Self, lx: Self, ly: Self) -> (Self, Self) {
        let dot = vfmaq_f32(vmulq_f32(x, lx), y, ly);
        let two_dot = vmulq_n_f32(dot, 2.0);
        let rx = vsubq_f32(vmulq_f32(two_dot, lx), x);
        let ry = vsubq_f32(vmulq_f32(two_dot, ly), y);
        (rx, ry)
    }

    #[target_feature(enable = "neon")]
    #[inline]
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
        ) -> (Self, Self) {
        unsafe {
            let x = vdivq_f32(local_x, radius);
            let y = vdivq_f32(local_y, radius);

            let (fx, fy) = Self::fold_point_into_wedge_fixed(x, y, slice_angle, two_pi);

            let source_scale = vdivq_f32(width_over_2, zoom);

            let sx_local = vmulq_f32(fx, source_scale);
            let sy_local = vmulq_f32(fy, source_scale);

            let (sin_r, cos_r) = sin_cos(triangle_rotation_rad);

            let rx = vfmaq_f32(vnegq_f32(vmulq_f32(sy_local, sin_r)), sx_local, cos_r);
            let ry = vfmaq_f32(vmulq_f32(sy_local, cos_r), sx_local, sin_r);

            (
                vaddq_f32(triangle_center_x, rx),
                vaddq_f32(triangle_center_y, ry),
            )
        }
    }
}

impl DaydreamBackend for float32x4_t {
    type IntegerRegister = uint32x4_t;

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load_pixels(input: &[[u8; 4]]) -> (Self::IntegerRegister, Self::IntegerRegister, Self::IntegerRegister, Self::IntegerRegister) {
        unsafe {
            let r = [input[0][0] as u32, input[1][0] as u32, input[2][0] as u32, input[3][0] as u32];
            let g = [input[0][1] as u32, input[1][1] as u32, input[2][1] as u32, input[3][1] as u32];
            let b = [input[0][2] as u32, input[1][2] as u32, input[2][2] as u32, input[3][2] as u32];
            let a = [input[0][3] as u32, input[1][3] as u32, input[2][3] as u32, input[3][3] as u32];
            (
                vld1q_u32(r.as_ptr()),
                vld1q_u32(g.as_ptr()),
                vld1q_u32(b.as_ptr()),
                vld1q_u32(a.as_ptr())
            )
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn rgb_to_hsv(r: Self::IntegerRegister, g: Self::IntegerRegister, b: Self::IntegerRegister, two_fifty_five: Self, hundred: Self, zero: Self, six: Self, sixty: Self, one: Self, two: Self, four: Self) -> (Self, Self, Self) {
        let r = vdivq_f32(vcvtq_f32_u32(r), two_fifty_five);
        let g = vdivq_f32(vcvtq_f32_u32(g), two_fifty_five);
        let b = vdivq_f32(vcvtq_f32_u32(b), two_fifty_five);
        // Scalar tie-breaking:
        //
        // if r >= g {
        //     if r >= b { ...r max... } else { ...b max... }
        // } else {
        //     if g >= b { ...g max... } else { ...b max... }
        // }
        //
        // So ties prefer r over g, r over b, and g over b.

        let r_max_mask = vandq_u32(vcgeq_f32(r, g), vcgeq_f32(r, b));
        let g_max_mask = vandq_u32(vcgtq_f32(g, r), vcgeq_f32(g, b));
        //let b_max_mask = vmvnq_u32(vorrq_u32(r_max_mask, g_max_mask));

        let c_max = vmaxq_f32(vmaxq_f32(r, g), b);
        let c_min = vminq_f32(vminq_f32(r, g), b);
        let delta = vsubq_f32(c_max, c_min);

        // Match the scalar tuples:
        //
        // r max -> (c_max=r, c_min=min(g,b), sub_1=g, sub_2=b, add=0)
        // g max -> (c_max=g, c_min=r or b,   sub_1=b, sub_2=r, add=2)
        // b max -> (c_max=b, c_min=r or g,   sub_1=r, sub_2=g, add=4)

        let sub_1_rb = vbslq_f32(r_max_mask, g, r);
        let sub_1 = vbslq_f32(g_max_mask, b, sub_1_rb);

        let sub_2_rb = vbslq_f32(r_max_mask, b, g);
        let sub_2 = vbslq_f32(g_max_mask, r, sub_2_rb);

        let add_rb = vbslq_f32(r_max_mask, zero, four);
        let add = vbslq_f32(g_max_mask, two, add_rb);

        let delta_zero_mask = vceqq_f32(delta, zero);
        let cmax_zero_mask = vceqq_f32(c_max, zero);
        let add_positive_mask = vcgtq_f32(add, zero);

        // Avoid divide-by-zero in masked-off lanes.
        let safe_delta = vbslq_f32(delta_zero_mask, one, delta);
        let safe_cmax = vbslq_f32(cmax_zero_mask, one, c_max);

        let sub = vsubq_f32(sub_1, sub_2);
        let div = vdivq_f32(sub, safe_delta);

        // Hue:
        // if delta == 0 { 0 }
        // else if add > 0 { 60 * (div + add) }
        // else { 60 * div.rem_euclid(6) }
        let h_add = vmulq_f32(sixty, vaddq_f32(div, add));
        let h_mod = vmulq_f32(sixty, modulo(div, six));
        let h_nonzero = vbslq_f32(add_positive_mask, h_add, h_mod);
        let h = vbslq_f32(delta_zero_mask, zero, h_nonzero);

        // Saturation:
        // if c_max == 0 { c_max } else { delta / c_max }
        let s_div = vdivq_f32(delta, safe_cmax);
        let s = vbslq_f32(cmax_zero_mask, c_max, s_div);

        // Value:
        // (c_max * 100.0).round()
        let v = vrndiq_f32(vmulq_f32(c_max, hundred));

        (h, s, v)
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn hsv_to_rgb(mut h: Self, s: Self, mut v: Self, hundred: Self, sixty: Self, two_fifty_five: Self, zero: Self, five: Self, four: Self, three: Self, two: Self, one: Self) -> (Self::IntegerRegister, Self::IntegerRegister, Self::IntegerRegister) {
        h = vdivq_f32(h, sixty);
        v = vdivq_f32(v, hundred);

        let c = vmulq_f32(v, s);
        let x = vmulq_f32(
            c, 
            vsubq_f32(
                one, 
                vabsq_f32(
                    vsubq_f32(
                        modulo(h, two), 
                        one
                    )
                )
            )
        );
        let m = vsubq_f32(v, c);

        let lt1 = vcltq_f32(h, one);
        let lt2 = vcltq_f32(h, two);
        let lt3 = vcltq_f32(h, three);
        let lt4 = vcltq_f32(h, four);
        let lt5 = vcltq_f32(h, five);

        // Match scalar:
        //
        // if h < 3 {
        //   if h < 2 {
        //     if h < 1 { (c, x, 0) } else { (x, c, 0) }
        //   } else {
        //     (0, c, x)
        //   }
        // } else {
        //   if h < 5 {
        //     if h < 4 { (0, x, c) } else { (x, 0, c) }
        //   } else {
        //     (c, 0, x)
        //   }
        // }

        let rp_lt2 = vbslq_f32(lt1, c, x);      // h<1 ? c : x
        let gp_lt2 = vbslq_f32(lt1, x, c);      // h<1 ? x : c
        let bp_lt2 = zero;                      // both cases have b=0

        let rp_lt3 = vbslq_f32(lt2, rp_lt2, zero);
        let gp_lt3 = vbslq_f32(lt2, gp_lt2, c);
        let bp_lt3 = vbslq_f32(lt2, bp_lt2, x);

        let rp_lt5 = vbslq_f32(lt4, zero, x);   // h<4 ? 0 : x
        let gp_lt5 = vbslq_f32(lt4, x, zero);   // h<4 ? x : 0
        let bp_lt5 = c;                         // both cases have b=c

        let rp_ge3 = vbslq_f32(lt5, rp_lt5, c);
        let gp_ge3 = vbslq_f32(lt5, gp_lt5, zero);
        let bp_ge3 = vbslq_f32(lt5, bp_lt5, x);

        let rp = vbslq_f32(lt3, rp_lt3, rp_ge3);
        let gp = vbslq_f32(lt3, gp_lt3, gp_ge3);
        let bp = vbslq_f32(lt3, bp_lt3, bp_ge3);

        let r = vrndiq_f32(vmulq_f32(vaddq_f32(rp, m), two_fifty_five));
        let g = vrndiq_f32(vmulq_f32(vaddq_f32(gp, m), two_fifty_five));
        let b = vrndiq_f32(vmulq_f32(vaddq_f32(bp, m), two_fifty_five));

        (
            vcvtq_u32_f32(r), 
            vcvtq_u32_f32(g), 
            vcvtq_u32_f32(b)
        )
    }

    unsafe fn adjust_hue(
            h: Self,
            hue_shift: Self,
            three_sixty: Self,
        ) -> Self {
        unsafe {
            return modulo(vaddq_f32(h, hue_shift), three_sixty);
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn extract_pixels(
            r: Self::IntegerRegister, 
            g: Self::IntegerRegister, 
            b: Self::IntegerRegister,
            a: Self::IntegerRegister,
        ) -> [[u8; 4]; Self::NUM_FLOATS] {
        unsafe {
            let mut arr = [[0u8; 4]; Self::NUM_FLOATS];

            for (i, reg) in [r, g, b, a].iter().enumerate() {
                let mut temp = [0u32; 4];
                vst1q_u32(temp.as_mut_ptr(), *reg);
                for j in 0..4 {
                    arr[j][i] = temp[j] as u8;
                }
            }
            arr
        }
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn store_pixel_hue_shift(
        buff: &mut [u8],
        _x: u32,
        sx: Self,
        sy: Self,
        source: &image::DynamicImage,
        source_width: u32,
        source_height: u32,
        hue_shift_vec: Self,
        two_fifty_five: Self,
        hundred: Self,
        zero: Self,
        six: Self,
        sixty: Self,
        one: Self,
        two: Self,
        four: Self,
        three_sixty: Self,
        five: Self,
        three: Self,
    ) {
        unsafe {
            let zero_f = vdupq_n_f32(0.0);
            let sw_v = vdupq_n_f32(source_width as f32);
            let sh_v = vdupq_n_f32(source_height as f32);

            let v_mask = vandq_u32(
                vandq_u32(vcgeq_f32(sx, zero_f), vcltq_f32(sx, sw_v)),
                vandq_u32(vcgeq_f32(sy, zero_f), vcltq_f32(sy, sh_v)),
            );

            if vmaxvq_u32(v_mask) == 0 {
                return;
            }

            let max_x = vdupq_n_f32(source_width.saturating_sub(1) as f32);
            let max_y = vdupq_n_f32(source_height.saturating_sub(1) as f32);

            let sx_safe = vmaxq_f32(zero_f, vminq_f32(sx, max_x));
            let sy_safe = vmaxq_f32(zero_f, vminq_f32(sy, max_y));

            let sx_i = vcvtq_u32_f32(sx_safe);
            let sy_i = vcvtq_u32_f32(sy_safe);

            let mut xs = [0u32; 4];
            let mut ys = [0u32; 4];
            let mut m = [0u32; 4];

            vst1q_u32(xs.as_mut_ptr(), sx_i);
            vst1q_u32(ys.as_mut_ptr(), sy_i);
            vst1q_u32(m.as_mut_ptr(), v_mask);

            let mut pixels = [[0u8; 4]; 4];
            for i in 0..4 {
                if m[i] != 0 {
                    pixels[i] = source.get_pixel(xs[i], ys[i]).0;
                }
            }

            let (r, g, b, a) = Self::load_pixels(&pixels);

            let (mut h, s, v) =
                Self::rgb_to_hsv(r, g, b, two_fifty_five, hundred, zero, six, sixty, one, two, four);

            h = Self::adjust_hue(h, hue_shift_vec, three_sixty);

            let (r, g, b) =
                Self::hsv_to_rgb(h, s, v, hundred, sixty, two_fifty_five, zero, five, four, three, two, one);

            let pixels = Self::extract_pixels(r, g, b, a);

            for i in 0..4 {
                if m[i] != 0 {
                    let pixel = pixels[i];
                    let base_idx = i * 4;
                    buff[base_idx..base_idx + 4].copy_from_slice(&pixel);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn order_of_operations() {
        let sqrt3 = 3.0f32.sqrt();
        let dx = 5.5;
        let dy = 6.6;
        let hex_radius = 7.7;

        let q = (sqrt3 / 3.0 * dx - 1.0 / 3.0 * dy) / hex_radius;

        let expected = ((sqrt3 / 3.0) * dx - ((1.0 / 3.0) * dy)) / hex_radius;
        assert_eq!(q, expected);
    }
}
