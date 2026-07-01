use image::{DynamicImage, GenericImageView};

use crate::backends::DaydreamBackend;

use super::KaleidoBackend;

impl KaleidoBackend for f32 {
    const NUM_FLOATS: usize = 1;

    #[cfg(test)]
    #[inline]
    unsafe fn load_f32s(input: &[f32]) -> Vec<Self> {
        input.to_vec()
    }

    #[cfg(test)]
    #[inline]
    unsafe fn store_f32s(&self, output: &mut [f32]) {
        output[0] = *self;
    }

    #[inline]
    unsafe fn load_with_single_f32(input: f32) -> Self {
        input
    }

    #[inline]
    unsafe fn load_coords(x: u32, y: u32) -> (Self, Self) {
        (x as f32, y as f32)
    }

    #[inline]
    unsafe fn normalize_coords(&mut self, center: Self) {
        *self -= center;
    }

    #[inline]
    unsafe fn atan2_k(&self, other: Self) -> Self {
        self.atan2(other)
    }

    #[inline]
    unsafe fn map_to_polar(dx: Self, dy: Self, zoom: f32) -> (Self, Self) {
        let r = (dx * dx + dy * dy).sqrt();
        let r_sampled = r / zoom;
        let mut theta = dy.atan2(dx);
        if theta < 0.0 {
            theta += 2.0 * core::f32::consts::PI;
        }
        (r_sampled, theta)
    }

    #[inline]
    unsafe fn compute_angle(theta: Self, slice_angle: f32, triangle_rotation_rad: f32) -> Self {
        let slice_idx = (theta / slice_angle).floor();
        let local_theta = if slice_idx as i32 % 2 != 0 {
            slice_angle - (theta % slice_angle)
        } else {
            theta % slice_angle
        };
        local_theta + triangle_rotation_rad
    }

    #[inline]
    unsafe fn compute_source_pixel_coords(
        computed_angle: Self,
        r_sampled: Self,
        triangle_center_x: f32,
        triangle_center_y: f32,
    ) -> (Self, Self) {
        let sx = computed_angle.cos() * r_sampled + triangle_center_x;
        let sy = computed_angle.sin() * r_sampled + triangle_center_y;
        (sx, sy)
    }

    #[inline]
    unsafe fn store_pixel(
        output: &mut [u8],
        _x: u32,
        sx: Self,
        sy: Self,
        source: &DynamicImage,
        sw: u32,
        sh: u32,
    ) {
        let sx_i = sx.round() as u32;
        let sy_i = sy.round() as u32;
        if sx_i < sw && sy_i < sh {
            let pixel = source.get_pixel(sx_i, sy_i);
            output[0..4].copy_from_slice(&pixel.0);
        }
    }

    #[inline(always)]
    unsafe fn store_pixel_rgba8(
        output: &mut [u8],
        sx: Self,
        sy: Self,
        source: &[u8],
        sw: u32,
        sh: u32,
    ) {
        let x = sx.round() as i32;
        let y = sy.round() as i32;

        if x >= 0 && y >= 0 && (x as u32) < sw && (y as u32) < sh {
            let idx = ((y as u32 * sw + x as u32) * 4) as usize;

            output[0] = source[idx];
            output[1] = source[idx + 1];
            output[2] = source[idx + 2];
            output[3] = source[idx + 3];
        }
    }

    //fn fold_square(input: Self, tile_size: Self) -> Self {
    //    let period = tile_size * 2.0;
    //    ((input % period + period) % period - tile_size).abs()
    //}

    // fn fold_square(input: Self, count: u32, tile_size: Self) -> Self {
    //     let folds_per_axis = (count as f32 / 4.0).max(1.0);
    //     let effective_tile = tile_size / folds_per_axis;
    //     let period = tile_size * 2.0;
    //     ((input % period + period) % period - effective_tile).abs()
    // }

    #[inline]
    unsafe fn map_square(
        dx: Self,
        dy: Self,
        width_over_2: Self,
        slice_angle: Self,
        two_pi: Self,
        tile_count: Self,
        zoom: Self,
        triangle_rotation_rad: f32,
        triangle_center_x: Self,
        triangle_center_y: Self,
    ) -> (Self, Self) {
        let screen_size = width_over_2 * 2.0;
        let tile_size = screen_size / tile_count.max(0.0001);
        let half = tile_size * 0.5;

        // Wrap into one square tile centered at the origin.
        let local_x = (dx + half).rem_euclid(tile_size) - half;
        let local_y = (dy + half).rem_euclid(tile_size) - half;

        unsafe {
            Self::source_space_rotation(local_x, local_y, triangle_rotation_rad, triangle_center_x, triangle_center_y, half, two_pi, slice_angle, width_over_2, zoom)
        }
    }

    #[inline]
    unsafe fn map_diamond(
        dx: Self,
        dy: Self,
        width_over_2: Self,
        slice_angle: Self,
        two_pi: Self,
        tile_count: Self,
        zoom: Self,
        rotation: f32,
        tx: Self,
        ty: Self,
    ) -> (Self, Self) {
        let screen_size = width_over_2 * 2.0;
        let tile = screen_size / tile_count.max(0.0001);
        let half = tile * 0.5;

        // Rotate output coordinates by 45° so rectangular wrapping becomes
        // a diamond lattice in screen space.
        let inv_sqrt2 = 0.70710678118_f32;
        let u = (dx + dy) * inv_sqrt2;
        let v = (dy - dx) * inv_sqrt2;

        // Wrap into one diamond cell centered at the origin.
        let local_u = (u + half).rem_euclid(tile) - half;
        let local_v = (v + half).rem_euclid(tile) - half;

        unsafe {
            Self::source_space_rotation(local_u, local_v, rotation, tx, ty, half, two_pi, slice_angle, width_over_2, zoom)
        }
    }

    #[inline]
    unsafe fn map_hexagonal(
        dx: Self,
        dy: Self,
        width_over_2: Self,
        slice_angle: Self,
        two_pi: Self,
        tile_count: Self,
        zoom: Self,
        triangle_rotation_rad: f32,
        triangle_center_x: Self,
        triangle_center_y: Self,
        sqrt3: Self,
    ) -> (Self, Self) {
        let screen_size = width_over_2 * 2.0;

        // Pointy-top hex radius. This gives roughly tile_count hexes across.
        let hex_radius = screen_size / (tile_count.max(0.0001) * sqrt3);

        // Convert pixel-space point to axial hex coordinates.
        let q = (sqrt3 / 3.0 * dx - 1.0 / 3.0 * dy) / hex_radius;
        let r = (2.0 / 3.0 * dy) / hex_radius;

        let (rq, rr) = unsafe { Self::hex_round(q, r) };

        // Convert the chosen hex center back to pixel space.
        let hex_cx = hex_radius * sqrt3 * (rq + rr * 0.5);
        let hex_cy = hex_radius * 1.5 * rr;

        // Local point relative to the hex center.
        let local_x = dx - hex_cx;
        let local_y = dy - hex_cy;

        unsafe {
            Self::source_space_rotation(local_x, local_y, triangle_rotation_rad, triangle_center_x, triangle_center_y, hex_radius, two_pi, slice_angle, width_over_2, zoom)
        }
    }

    #[inline]
    unsafe fn map_hexagonal_flat_top(
        dx: Self,
        dy: Self,
        width_over_2: Self,
        slice_angle: Self,
        two_pi: Self,
        tile_count: Self,
        zoom: Self,
        triangle_rotation_rad: f32,
        triangle_center_x: Self,
        triangle_center_y: Self,
        sqrt3: Self,
    ) -> (Self, Self) {
        let screen_size = width_over_2 * 2.0;

        // Flat-top hex radius / side length.
        // This is still a reasonable "roughly tile_count across" control.
        let hex_radius = screen_size / (tile_count.max(0.0001) * 1.5);

        // Convert pixel-space point to axial hex coordinates for FLAT-TOP hexes.
        let q = (2.0 / 3.0 * dx) / hex_radius;
        let r = (-1.0 / 3.0 * dx + sqrt3 / 3.0 * dy) / hex_radius;

        let (rq, rr) = unsafe { Self::hex_round(q, r) };

        // Convert the chosen hex center back to pixel space for FLAT-TOP hexes.
        let hex_cx = hex_radius * 1.5 * rq;
        let hex_cy = hex_radius * sqrt3 * (rr + rq * 0.5);

        // Local point relative to the hex center.
        let local_x = dx - hex_cx;
        let local_y = dy - hex_cy;

        unsafe {
            Self::source_space_rotation(local_x, local_y, triangle_rotation_rad, triangle_center_x, triangle_center_y, hex_radius, two_pi, slice_angle, width_over_2, zoom)
        }
    }

    #[inline]
    unsafe fn hex_round(q: Self, r: Self) -> (Self, Self) {
        let s = -q - r;
        let mut rq = q.round();
        let mut rr = r.round();
        let rs = s.round();

        let q_diff = (rq - q).abs();
        let r_diff = (rr - r).abs();
        let s_diff = (rs - s).abs();

        if q_diff > r_diff && q_diff > s_diff {
            rq = -rr - rs;
        } else if r_diff > s_diff {
            rr = -rq - rs;
        }
        // rs is implicitly -rq - rr
        (rq, rr)
    }

    #[inline]
    unsafe fn fold_point_into_wedge_fixed(x: Self, y: Self, slice_angle: Self, two_pi: Self) -> (f32, f32) {
        let mut theta = y.atan2(x);
        if theta < 0.0 {
            theta += two_pi;
        }

        let sector = (theta / slice_angle).floor() as i32;
        let sector_angle = sector as f32 * slice_angle;

        // Rotate into the base sector [0, slice_angle)
        let sin_s = sector_angle.sin();
        let cos_s = sector_angle.cos();

        let xr = x * cos_s + y * sin_s;
        let yr = -x * sin_s + y * cos_s;

        // Odd sectors are mirrored versions of even sectors.
        if sector & 1 != 0 {
            let mid_angle = slice_angle * 0.5;
            let lx = mid_angle.cos();
            let ly = mid_angle.sin();
            unsafe { Self::reflect_across_line(xr, yr, lx, ly) }
        } else {
            (xr, yr)
        }
    }

    #[inline]
    unsafe fn reflect_across_line(x: f32, y: f32, lx: f32, ly: f32) -> (f32, f32) {
        let dot = 2.0 * (x * lx + y * ly);
        let rx = dot * lx - x;
        let ry = dot * ly - y;
        (rx, ry)
    }

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
        // Normalize into a canonical square domain.
        let x = local_x / radius;
        let y = local_y / radius;

        // Fold into the canonical mirrored wedge using fixed-cost Cartesian ops.
        let (fx, fy) = unsafe { Self::fold_point_into_wedge_fixed(x, y, slice_angle, two_pi) };

        // Source scale depends only on zoom, not tile_count.
        let source_scale = width_over_2 / zoom.max(0.0001);

        let sx_local = fx * source_scale;
        let sy_local = fy * source_scale;

        // Apply source-space rotation.
        let sin_r = triangle_rotation_rad.sin();
        let cos_r = triangle_rotation_rad.cos();

        let rx = sx_local * cos_r - sy_local * sin_r;
        let ry = sx_local * sin_r + sy_local * cos_r;

        (triangle_center_x + rx, triangle_center_y + ry)
    }
}

impl DaydreamBackend for f32 {
    type IntegerRegister = u8;
    #[inline]
    unsafe fn load_pixels(input: &[[u8; 4]]) -> (Self::IntegerRegister, Self::IntegerRegister, Self::IntegerRegister, Self::IntegerRegister) {
        (input[0][0], input[0][1], input[0][2], input[0][3])
    }
    #[inline]
    unsafe fn rgb_to_hsv(
        r: Self::IntegerRegister, 
        g: Self::IntegerRegister, 
        b: Self::IntegerRegister, 
        two_fifty_five: Self, 
        _hundred: Self, 
        _zero: Self, 
        _six: Self, 
        _sixty: Self,
        _one: Self,
        _two: Self,
        _four: Self,
    ) -> (Self, Self, Self) {
        let (r, g, b) = (r as f32 / two_fifty_five, g as f32 / two_fifty_five, b as f32 / two_fifty_five);
        let (c_max, c_min, sub_1, sub_2, add) = match r >= g {
            // r >= g
            true => match r >= b {
                // r > g, r > b
                true => {
                    (r, match g >= b {
                        // r >= g >= b
                        true => b,
                        // r >= b >= g
                        false => g
                    }, g, b, 0f32)
                },
                // b >= r >= g
                false => (b, g, r, g, 4f32)
            },
            false => match g >= b {
                // g >= b >= r
                true => (g, r, b, r, 2f32),
                // b >= g >= r
                false => (b, r, r, g, 4f32)
            }
        };

        let delta = c_max - c_min;
        let h = match delta == 0f32 {
            true => 0f32,
            false => match add > 0f32 {
                true => 60f32 * (((sub_1 - sub_2) / delta) + add),
                false => 60f32 * (((sub_1 - sub_2) / delta).rem_euclid(6f32))
            }
        };

        let s = match c_max == 0f32 {
            true => c_max,
            false => delta / c_max
        };

        (h, s, (c_max * 100f32).round())
    }
    #[inline]
    unsafe fn hsv_to_rgb(mut h: Self, s: Self, mut v: Self, hundred: Self, sixty: Self, two_fifty_five: Self, zero: Self, five: Self, four: Self, three: Self, two: Self, one: Self) -> (Self::IntegerRegister, Self::IntegerRegister, Self::IntegerRegister) {
        h /= sixty;
        v /= hundred;
        
        let c = v * s;
        let x = c * (one - (h % two - one).abs());
        let m = v - c;

        let (rp, gp, bp) = match h < three {
            true => match h < two {
                true => match h < one {
                    true => (c, x, zero),
                    false => (x, c, zero)
                },
                false => (zero, c, x)
            },
            false => match h < five {
                true => match h < four {
                    true => (zero, x, c),
                    false => (x, zero, c)
                },
                false => (c, zero, x)
            }
        };

        (((rp + m) * two_fifty_five).round() as u8,
        ((gp + m) * two_fifty_five).round() as u8,
        ((bp + m) * two_fifty_five).round() as u8)
    }
    unsafe fn adjust_hue(
            h: Self,
            hue_shift: Self,
            three_sixty: Self,
        ) -> Self {
        (h + hue_shift).rem_euclid(three_sixty)
    }

    #[inline]
    unsafe fn extract_pixels(
            r: Self::IntegerRegister, 
            g: Self::IntegerRegister, 
            b: Self::IntegerRegister,
            a: Self::IntegerRegister,
        ) -> [[u8; 4]; Self::NUM_FLOATS] {
        [[r, g, b, a]]
    }

    #[inline]
    unsafe fn store_pixel_hue_shift(
        output: &mut [u8],
        _x: u32,
        sx: Self,
        sy: Self,
        source: &DynamicImage,
        sw: u32,
        sh: u32,
        hue_shift_vec: Self,
        _two_fifty_five: Self,
        _hundred: Self,
        _zero: Self,
        _six: Self,
        _sixty: Self,
        _one: Self,
        _two: Self,
        _four: Self,
        three_sixty: Self,
        _five: Self,
        _three: Self,
    ) {
        unsafe {
            let sx_i = sx.round() as u32;
            let sy_i = sy.round() as u32;
            if sx_i < sw && sy_i < sh {
                let pixel = source.get_pixel(sx_i, sy_i);
                let (r, g, b, a) = (pixel.0[0], pixel.0[1], pixel.0[2], pixel.0[3]);
                let (mut h, s, v) = Self::rgb_to_hsv(r, g, b, 255.0, 100.0, 0.0, 6.0, 60.0, 1.0, 2.0, 4.0);

                h = Self::adjust_hue(h, hue_shift_vec, three_sixty);
                let (r, g, b) = Self::hsv_to_rgb(h, s, v, 100.0, 60.0, 255.0, 0.0, 5.0, 4.0, 3.0, 2.0, 1.0);
                output[0..4].copy_from_slice(&[r, g, b, a]);
            }
        }
    }
}