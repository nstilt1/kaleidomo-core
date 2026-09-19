// kaleidomo-core/src/rlib.rs
pub use anyhow;
pub use log;
#[allow(unused)]
use anyhow::Context;
use image::{DynamicImage, GenericImageView};

use image::{ImageBuffer, Rgba};
use rayon::prelude::*;
use serde::Deserialize;
pub use std::f32::consts::PI;
pub use wgpu;

pub use image;
pub use pollster;

pub use software_licensor_static_rust_lib::{LicenseData, lib_api::LicenseStatus, lib_api::{get_machine_stats_for_display, StatsDisplay}};
pub use software_licensor_static_rust_lib;
use crate::{KaleidoSettings, KaleidoType, VideoSettings, modulate};
use crate::backends::gpu::{GpuBackend, GpuVideoRenderer};
pub use crate::backends::{KaleidoBackend, DaydreamBackend, Register, inner_loop, inner_loop_enhanced};

pub fn render_kaleidoscope(
    source: &DynamicImage,
    settings: KaleidoSettings,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let (sw, sh) = source.dimensions();
    let _width_over_2 = settings.output_size_w as f32 / 2.0;
    let center_x = settings.output_size_w as f32 / 2.0 + settings.offset_x as f32;
    let center_y = settings.output_size_h as f32 / 2.0 + settings.offset_y as f32;
    let slice_angle = (2.0 * PI) / settings.count as f32;

    // Create a flat vector for the pixels
    let mut pixels = vec![0u8; (settings.output_size_w * settings.output_size_h * 4) as usize];

    // Rayon parallelizes the rows automatically
    pixels
        .par_chunks_exact_mut((settings.output_size_w * 4) as usize)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..settings.output_size_w {
                // 1. Normalize coordinates relative to center
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;

                // 2. Map to Polar
                let r = (dx * dx + dy * dy).sqrt();
                let r_sampled = r / settings.zoom;
                let mut theta = dy.atan2(dx);

                // 3. Kaleidoscope logic
                // Ensure theta is positive [0, 2pi]
                if theta < 0.0 {
                    theta += 2.0 * PI;
                }

                let slice_idx = (theta / slice_angle).floor();
                let local_theta = if slice_idx as i32 % 2 != 0 {
                    slice_angle - (theta % slice_angle)
                } else {
                    theta % slice_angle
                };

                // Use the angle from the UI (converted to radians)
                let final_angle = local_theta + settings.triangle_rotation_rad;

                // zoom/scale affects how 'far' into the source image we look
                // A higher zoom means the triangle in the source image is smaller
                //let r_scaled = r * settings.zoom;

                // Compute source image sample coordinates from the polar-mapped
                // output pixel. `r_sampled` is the radial distance (scaled by
                // `zoom`) and `final_angle` is the mapped angle for this slice
                // (including triangle rotation and mirroring). We convert these
                // back to Cartesian coordinates around the configured triangle
                // center to get `sx`,`sy`. Then ensure the coordinates fall
                // within the source image bounds and, if so, fetch that pixel
                // and copy its RGBA bytes into the output row buffer.
                let sx = settings.triangle_center_x + (r_sampled * final_angle.cos());
                let sy = settings.triangle_center_y + (r_sampled * final_angle.sin());

                // Final check: Convert to u32 only for the fetch
                if sx >= 0.0 && sx < (sw as f32) && sy >= 0.0 && sy < (sh as f32) {
                    let pixel = source.get_pixel(sx as u32, sy as u32);
                    let offset = (x * 4) as usize;
                    row[offset..offset + 4].copy_from_slice(&pixel.0);
                }

                // 4. Map back to Source Image
                // This is where you'd use your specific triangle coordinates.
                // For simplicity, we sample relative to the source center:
                //let sx = (r * local_theta.cos() * settings.zoom + (sw as f32 / 2.0)) as i32;
                //let sy = (r * local_theta.sin() * settings.zoom + (sh as f32 / 2.0)) as i32;
            }
        });

    ImageBuffer::from_raw(settings.output_size_w, settings.output_size_h, pixels).unwrap()
}

pub fn render_kaleidoscope_with_auto_backend(
    source: &DynamicImage,
    settings: KaleidoSettings,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return render_kaleidoscope_with_backend::<crate::backends::avx2::__m256>(source, settings);
        } else if is_x86_feature_detected!("sse2") {
            return render_kaleidoscope_with_backend::<crate::backends::sse2::__m128>(source, settings);
        } else {
            return render_kaleidoscope_with_backend::<f32>(source, settings);
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        render_kaleidoscope_with_backend::<Register>(source, settings)
        //render_kaleidoscope_with_backend::<f32>(source, settings)
    }
}

#[inline(always)]
pub fn render_kaleidoscope_with_backend<B: KaleidoBackend + DaydreamBackend>(
    source: &DynamicImage,
    settings: KaleidoSettings,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let processed_source;
    let source = if settings.recolor_enabled {
        let mut rgba = source.to_rgba8();
        // A finite threshold is guaranteed by the UI/wire validation. If a
        // third-party caller supplies NaN, preserve the source rather than panic.
        if crate::preprocess::preprocess_source_frame_with_mode(
            &mut rgba,
            settings.recolor_seed.as_bytes(),
            settings.recolor_threshold,
            if settings.recolor_mode == 1 {
                crate::preprocess::RecolorMode::BorderedCells
            } else {
                crate::preprocess::RecolorMode::ColorBands
            },
        ).is_ok() {
            processed_source = DynamicImage::ImageRgba8(rgba);
            &processed_source
        } else {
            source
        }
    } else {
        source
    };
    let (src_w, src_h) = source.dimensions();
    let factor = crate::safe_super_sample(settings.super_sample, src_w, src_h);
    if factor > 1 {
        // `super_sample` path: render at `output_size * factor` internally using the
        // exact same per-backend math (nothing below needs to know about
        // supersampling), then box-downsample back to the requested output size.
        // This is intentionally backend-agnostic — it works uniformly whichever
        // CPU backend `B` is, since it only wraps the output buffer dimensions.
        let (out_w, out_h) = (settings.output_size_w, settings.output_size_h);
        let big_settings = KaleidoSettings {
            output_size_w: out_w * factor as u32,
            output_size_h: out_h * factor as u32,
            offset_x: settings.offset_x * factor as i32,
            offset_y: settings.offset_y * factor as i32,
            // `zoom` must scale with the enlarged canvas too: the renderer's
            // `source_scale = width_over_2 / zoom` ties visible source
            // content to the actual render width, so leaving `zoom`
            // unscaled while `output_size_w/h` grow by `factor` was
            // silently showing `factor`x more source content (i.e.
            // zooming out) the higher `super_sample` was set.
            zoom: settings.zoom * factor as f32,
            ..settings.clone()
        };
        let big = render_kaleidoscope_with_backend_inner::<B>(source, &big_settings);
        let downsampled = downsample_box(big.as_raw(), big_settings.output_size_w, big_settings.output_size_h, factor, out_w, out_h);
        return ImageBuffer::from_raw(out_w, out_h, downsampled).unwrap();
    }

    render_kaleidoscope_with_backend_inner::<B>(source, &settings)
}

/// The actual native-resolution render, shared by both the direct (`super_sample == 1`)
/// path and the supersampling wrapper in [`render_kaleidoscope_with_backend`] above.
#[inline(always)]
fn render_kaleidoscope_with_backend_inner<B: KaleidoBackend + DaydreamBackend>(
    source: &DynamicImage,
    settings: &KaleidoSettings,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let (sw, sh) = source.dimensions();
    let width_over_2 = settings.output_size_w as f32 / 2.0;
    let center_x = settings.output_size_w as f32 / 2.0 + settings.offset_x as f32;
    let center_y = settings.output_size_h as f32 / 2.0 + settings.offset_y as f32;
    let slice_angle = (2.0 * PI) / settings.count as f32;

    // Create a flat vector for the pixels
    let mut pixels = vec![0u8; (settings.output_size_w * settings.output_size_h * 4) as usize];

    if settings.derivative_mipmapping || settings.anisotropy_level > 1 {
        let pyramid = crate::enhancement::CpuMipPyramid::new(source);
        let pipeline = sampling_pipeline(settings);
        pixels
            .par_chunks_exact_mut((settings.output_size_w * 4) as usize)
            .enumerate()
            .for_each(|(y, row)| inner_loop_enhanced::<B>(y, row, settings, &pyramid, &pipeline, settings.hue_rotation));
        return ImageBuffer::from_raw(settings.output_size_w, settings.output_size_h, pixels).unwrap();
    }

    // Rayon parallelizes the rows automatically
    macro_rules! render_mode {
        ($mode:expr) => { pixels
        .par_chunks_exact_mut((settings.output_size_w * 4) as usize)
        .enumerate()
        .for_each(|(y, row)| {
            inner_loop::<B, $mode>(
            //inner_loop::<f32>(
                y,
                row,
                settings.zoom,
                source,
                settings,
                width_over_2,
                center_x,
                center_y,
                slice_angle,
                sw,
                sh,
                settings.hue_rotation,
            );
        }) };
    }
    match settings.anti_alias { 0 => render_mode!(0), 2 => render_mode!(2), _ => render_mode!(1) };

    ImageBuffer::from_raw(settings.output_size_w, settings.output_size_h, pixels).unwrap()
}

fn sampling_pipeline(settings: &KaleidoSettings) -> crate::enhancement::EnhancementPipeline {
    crate::enhancement::EnhancementPipeline::new(crate::enhancement::EnhancementConfig {
        reconstruction: match settings.anti_alias { 0 => crate::enhancement::ReconstructionFilter::Nearest, 2 => crate::enhancement::ReconstructionFilter::Bicubic, _ => crate::enhancement::ReconstructionFilter::Bilinear },
        explicit_derivatives: settings.derivative_mipmapping,
        anisotropy: settings.anisotropy_level,
        edge_filter: crate::enhancement::EdgeFilter::Disabled,
        taa_enabled: false,
        taa_feedback: 0.0,
    })
}

/// Readable derivative-aware scalar reference used to verify SIMD/GPU output.
/// It evaluates the nonlinear warp at the pixel and its two forward neighbors,
/// so fold discontinuities produce the correct explicit texture footprint.
pub fn render_kaleidoscope_scalar_enhanced(source: &DynamicImage, settings: &KaleidoSettings) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let pyramid = crate::enhancement::CpuMipPyramid::new(source);
    let pipeline = sampling_pipeline(settings);
    let mut pixels = vec![0u8; (settings.output_size_w * settings.output_size_h * 4) as usize];
    pixels.par_chunks_exact_mut((settings.output_size_w * 4) as usize).enumerate().for_each(|(y, row)| {
        for x in 0..settings.output_size_w as usize {
            let p = map_scalar_coordinate(x as f32, y as f32, settings);
            let px = map_scalar_coordinate((x + 1).min(settings.output_size_w as usize - 1) as f32, y as f32, settings);
            let py = map_scalar_coordinate(x as f32, (y + 1).min(settings.output_size_h as usize - 1) as f32, settings);
            let tolerance = match settings.anti_alias { 0 => 0.0, 2 => 2.0, _ => 1.0 };
            if p.0 < -tolerance || p.1 < -tolerance || p.0 >= source.width() as f32 + tolerance || p.1 >= source.height() as f32 + tolerance { continue; }
            let derivatives = crate::enhancement::UvDerivatives { dx: [px.0 - p.0, px.1 - p.1], dy: [py.0 - p.0, py.1 - p.1] };
            let sampled = pyramid.sample(&pipeline, [p.0, p.1], derivatives);
            let color = crate::enhancement::rotate_hue_rgba(sampled, settings.hue_rotation as f32);
            row[x * 4..x * 4 + 4].copy_from_slice(&color);
        }
    });
    ImageBuffer::from_raw(settings.output_size_w, settings.output_size_h, pixels).expect("scalar output dimensions validated")
}

fn map_scalar_coordinate(x: f32, y: f32, settings: &KaleidoSettings) -> (f32, f32) {
    unsafe {
        let center_x = settings.output_size_w as f32 * 0.5 + settings.offset_x as f32;
        let center_y = settings.output_size_h as f32 * 0.5 + settings.offset_y as f32;
        let dx = x - center_x;
        let mut dy = y - center_y;
        if settings.aspect_correct { dy *= settings.output_size_w as f32 / settings.output_size_h.max(1) as f32; }
        let half = settings.output_size_w as f32 * 0.5;
        let slice = 2.0 * PI / settings.count.max(1) as f32;
        match settings.kaleido_type {
            KaleidoType::Radial => { let (r, theta) = <f32 as KaleidoBackend>::map_to_polar(dx, dy, settings.zoom); let angle = <f32 as KaleidoBackend>::compute_angle(theta, slice, settings.triangle_rotation_rad); <f32 as KaleidoBackend>::compute_source_pixel_coords(angle, r, settings.triangle_center_x, settings.triangle_center_y) }
            KaleidoType::Square => <f32 as KaleidoBackend>::map_square(dx, dy, half, slice, 2.0*PI, settings.tile_count, settings.zoom, settings.triangle_rotation_rad, settings.triangle_center_x, settings.triangle_center_y),
            KaleidoType::Diamond => <f32 as KaleidoBackend>::map_diamond(dx, dy, half, slice, 2.0*PI, settings.tile_count, settings.zoom, settings.triangle_rotation_rad, settings.triangle_center_x, settings.triangle_center_y),
            KaleidoType::Hexagonal => <f32 as KaleidoBackend>::map_hexagonal(dx, dy, half, slice, 2.0*PI, settings.tile_count, settings.zoom, settings.triangle_rotation_rad, settings.triangle_center_x, settings.triangle_center_y, 3.0f32.sqrt()),
            KaleidoType::HexagonalFlatTop => <f32 as KaleidoBackend>::map_hexagonal_flat_top(dx, dy, half, slice, 2.0*PI, settings.tile_count, settings.zoom, settings.triangle_rotation_rad, settings.triangle_center_x, settings.triangle_center_y, 3.0f32.sqrt()),
        }
    }
}

/// Box-downsamples an RGBA8 buffer of size `(src_w, src_h)` by an integer `factor`
/// down to `(dst_w, dst_h)` (where `src_w == dst_w * factor` and `src_h == dst_h *
/// factor`), averaging each `factor x factor` block of source pixels into one
/// destination pixel (alpha-weighted, so fully transparent supersample pixels
/// don't darken partially-covered edges). Shared by every backend's
/// `super_sample` path — CPU and GPU alike — so the smoothing looks identical
/// regardless of which backend rendered the oversized frame.
pub fn downsample_box(src: &[u8], src_w: u32, src_h: u32, factor: u8, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let factor = factor as u32;
    let mut out = vec![0u8; (dst_w * dst_h * 4) as usize];
    let samples = (factor * factor) as f32;

    out.par_chunks_exact_mut((dst_w * 4) as usize)
        .enumerate()
        .for_each(|(dy, out_row)| {
            for dx in 0..dst_w {
                let mut sum = [0.0f32; 4];
                for sy in 0..factor {
                    let src_y = dy as u32 * factor + sy;
                    let row_start = (src_y * src_w * 4) as usize;
                    for sx in 0..factor {
                        let src_x = dx * factor + sx;
                        let idx = row_start + (src_x * 4) as usize;
                        sum[0] += src[idx] as f32;
                        sum[1] += src[idx + 1] as f32;
                        sum[2] += src[idx + 2] as f32;
                        sum[3] += src[idx + 3] as f32;
                    }
                }
                let out_idx = (dx * 4) as usize;
                out_row[out_idx] = (sum[0] / samples).round().clamp(0.0, 255.0) as u8;
                out_row[out_idx + 1] = (sum[1] / samples).round().clamp(0.0, 255.0) as u8;
                out_row[out_idx + 2] = (sum[2] / samples).round().clamp(0.0, 255.0) as u8;
                out_row[out_idx + 3] = (sum[3] / samples).round().clamp(0.0, 255.0) as u8;
            }
        });

    out
}

#[cfg(test)]
pub fn render_kaleidoscope_with_gpu(
    source: &DynamicImage,
    settings: KaleidoSettings,
) -> anyhow::Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
    let (src_w, src_h) = source.dimensions();
    let factor = crate::safe_super_sample(settings.super_sample, src_w, src_h);
    let (out_w, out_h) = (settings.output_size_w, settings.output_size_h);
    // `super_sample`: render at `output_size * factor` on the GPU, same as the CPU
    // backends' wrapper, then box-downsample back down to the requested size.
    let render_settings = if factor > 1 {
        KaleidoSettings {
            output_size_w: out_w * factor as u32,
            output_size_h: out_h * factor as u32,
            offset_x: settings.offset_x * factor as i32,
            offset_y: settings.offset_y * factor as i32,
            // See the matching comment in `render_kaleidoscope_with_backend`.
            zoom: settings.zoom * factor as f32,
            ..settings.clone()
        }
    } else {
        settings.clone()
    };

    let mut gpu = pollster::block_on(GpuBackend::new())
        .context("failed to initialize GPU backend")?;
    gpu.set_source_image(source)?;
    gpu.update_settings(&render_settings)?;
    let mut pixels = vec![
        0u8;
        (render_settings.output_size_w as usize)
            .checked_mul(render_settings.output_size_h as usize)
            .and_then(|v| v.checked_mul(4))
            .context("output dimensions overflowed")?
    ];

    gpu.render_into_buffer(&render_settings, &mut pixels)
        .context("failed to render kaleidoscope on GPU")?;

    if factor > 1 {
        let downsampled = downsample_box(&pixels, render_settings.output_size_w, render_settings.output_size_h, factor, out_w, out_h);
        return ImageBuffer::from_raw(out_w, out_h, downsampled)
            .context("GPU returned an invalid output buffer length");
    }

    ImageBuffer::from_raw(out_w, out_h, pixels)
        .context("GPU returned an invalid output buffer length")
}

use crate::video_sink::{VideoFrameSink, VideoSinkError};

pub fn render_video_with_auto_backend(
    source: &DynamicImage,
    settings: KaleidoSettings,
    video_settings: VideoSettings,
    sink: &mut dyn VideoFrameSink,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return render_video::<crate::backends::avx2::__m256>(source, settings, video_settings, sink);
        } else if is_x86_feature_detected!("sse2") {
            return render_video::<crate::backends::sse2::__m128>(source, settings, video_settings, sink);
        } else {
            return render_video::<f32>(source, settings, video_settings, sink);
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        render_video::<Register>(source, settings, video_settings, sink)
        //render_video::<f32>(source, settings, sink)
    }
}

/// Converts degrees to radians.
#[inline]
fn degrees_to_radians(degrees: f32) -> f32 {
    degrees * (PI / 180.0)
}

/// Converts radians to degrees.
#[inline]
pub fn radians_to_degrees(radians: f32) -> f32 {
    radians * (180.0 / std::f32::consts::PI)
}

#[inline]
fn orientation_to_hero_params(
    value: f32,
    left_x: f32,
    right_x: f32,
    center_y: f32,
    desired_left_rotation: f32,
) -> (f32, f32, f32) {
    let center_x = (left_x + right_x) * 0.5;
    let radius = (right_x - left_x) * 0.5;
    let angle = PI + value * 2.0 * PI;

    (
        center_x + angle.cos() * radius,
        center_y + angle.sin() * radius,
        desired_left_rotation + (angle - PI),
    )
}

/// Modulates the zoom parameter.
fn zoom_modulation(video_settings: &VideoSettings, frame: u32) -> f32 {
    modulate(
        video_settings, frame, 
        video_settings.zoom_max, 
        video_settings.zoom_min, 
        video_settings.num_zoom_loops as f32, 
        video_settings.zoom_start_offset, 
        &video_settings.zoom_fn
    )
}

#[derive(Default)]
struct VideoFrameModulationState {
    smoothed_audio_peak: f32,
    accumulated_orientation_offset: f32,
}

fn apply_video_frame_modulation(
    settings: &mut KaleidoSettings,
    video_settings: &VideoSettings,
    frame: u32,
    base_x: f32,
    base_y: f32,
    base_rotation: f32,
    base_hue: f32,
    zoom_scale: f32,
    state: &mut VideoFrameModulationState,
) -> u32 {
    let fps = video_settings.fps.max(1);
    let elapsed_seconds = frame as f32 / fps as f32;

    let audio_peak = if video_settings.audio_reactive_enabled
        && !video_settings.audio_peaks.is_empty()
    {
        let raw_peak = video_settings
            .audio_peaks
            .get(frame as usize % video_settings.audio_peaks.len())
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);

        let smoothing = video_settings.audio_peak_smoothing.clamp(0.0, 0.999);

        state.smoothed_audio_peak =
            state.smoothed_audio_peak * smoothing + raw_peak * (1.0 - smoothing);

        state.smoothed_audio_peak.clamp(0.0, 1.0)
    } else {
        state.smoothed_audio_peak = 0.0;
        0.0
    };

    state.accumulated_orientation_offset = (
        state.accumulated_orientation_offset
            + audio_peak * video_settings.orientation_peak_multiplier / fps as f32
    )
        .rem_euclid(1.0);

    let rotation_modulation = modulate(
        video_settings,
        frame,
        base_rotation + degrees_to_radians(video_settings.rotation_range),
        base_rotation,
        video_settings.rotation_cycles,
        video_settings.rotation_start_offset,
        &video_settings.rotation_fn,
    );

    let rotation_offset = rotation_modulation - base_rotation;

    let hero_radius = ((video_settings.hero_circle_right_x
        - video_settings.hero_circle_left_x)
        * 0.5)
        .abs()
        .max(1.0);

    let base_speed_cycles =
        video_settings.orientation_base_speed / (2.0 * PI * hero_radius);

    let orientation_value = (
        elapsed_seconds * base_speed_cycles + state.accumulated_orientation_offset
    )
        .rem_euclid(1.0);

    let use_orientation = video_settings.orientation_base_speed != 0.0
        || (video_settings.audio_reactive_enabled
            && state.accumulated_orientation_offset != 0.0);

    if use_orientation {
        let (orientation_x, orientation_y, orientation_rotation) =
            orientation_to_hero_params(
                orientation_value,
                video_settings.hero_circle_left_x,
                video_settings.hero_circle_right_x,
                video_settings.hero_circle_y,
                video_settings.hero_desired_left_rotation,
            );

        settings.triangle_center_x = orientation_x;
        settings.triangle_center_y = orientation_y;
        settings.triangle_rotation_rad =
            (orientation_rotation + rotation_offset).rem_euclid(2.0 * PI);
    } else {
        settings.triangle_center_x = base_x;
        settings.triangle_center_y = base_y;
        settings.triangle_rotation_rad =
            rotation_modulation.rem_euclid(2.0 * PI);
    }

    // zoom_min/zoom_max are already absolute renderer zoom values supplied by
    // the frontend. Do not multiply them by the base Zoom setting again.
    // zoom_scale is only the supersampling factor, preserving the same source
    // radius when rendering into a larger intermediate frame.
    settings.zoom = zoom_modulation(video_settings, frame) * zoom_scale;

    modulate(
        video_settings,
        frame,
        base_hue + video_settings.hue_range as f32,
        base_hue,
        video_settings.hue_cycles,
        video_settings.hue_start_offset,
        &video_settings.hue_fn,
    )
    .round()
    .rem_euclid(360.0) as u32
}

/// Renders video with a CPU backend.
fn render_video<B: KaleidoBackend + DaydreamBackend>(
    source: &DynamicImage,
    mut settings: KaleidoSettings,
    video_settings: VideoSettings,
    sink: &mut dyn VideoFrameSink,
) -> Result<(), Box<dyn std::error::Error>> {
    let fps = video_settings.fps;
    let total_frames = f32::round(video_settings.animation_duration * fps as f32) as u32;

    // `super_sample`: render each frame into a `factor`x-larger scratch buffer and
    // box-downsample it down to the final output size before it's handed to the
    // H.264 encoder. `width_over_2`/`center_x`/`center_y` are computed against the
    // *scaled* dimensions so `inner_loop` draws the (larger) frame correctly; the
    // final encoded video is still exactly `output_size_w x output_size_h`.
    let factor = crate::safe_super_sample(settings.super_sample, settings.output_size_w, settings.output_size_h);
    let (out_w, out_h) = (settings.output_size_w, settings.output_size_h);
    let (render_w, render_h) = (out_w * factor as u32, out_h * factor as u32);
    let render_offset_x = settings.offset_x * factor as i32;
    let render_offset_y = settings.offset_y * factor as i32;

    let width_over_2 = render_w as f32 / 2.0;
    let center_x = render_w as f32 / 2.0 + render_offset_x as f32;
    let center_y = render_h as f32 / 2.0 + render_offset_y as f32;
    let slice_angle = (2.0 * PI) / settings.count as f32;

    let mut rgba = vec![0u8; (render_w * render_h * 4) as usize];
    let mut final_rgba = vec![0u8; (out_w * out_h * 4) as usize];
    let enhanced_sampling = (settings.derivative_mipmapping || settings.anisotropy_level > 1)
        .then(|| (crate::enhancement::CpuMipPyramid::new(source), sampling_pipeline(&settings)));

    //let triangle_rotation_delta = degrees_to_radians(video_settings.triangle_rotation_degrees_per_frame);
    // Owned copy of the most recently written frame, retained only so
    // `still_frame_ending` can resend it without rerendering.
    let mut last_frame: Vec<u8> = vec![0u8; (out_w * out_h * 4) as usize];

    let base_x = settings.triangle_center_x;
    let base_y = settings.triangle_center_y;
    let base_rotation = settings.triangle_rotation_rad;
    let base_hue = settings.hue_rotation as f32;
    // The animated zoom bounds are already absolute zoom values for the final
    // output width. Only scale by the supersampling factor here.
    let zoom_scale = factor as f32;

    let mut modulation_state = VideoFrameModulationState::default();
    
    for frame in 0..total_frames {
        let hue_rotation = apply_video_frame_modulation(
            &mut settings,
            &video_settings,
            frame,
            base_x,
            base_y,
            base_rotation,
            base_hue,
            zoom_scale,
            &mut modulation_state,
        );

        if let Some((pyramid, pipeline)) = &enhanced_sampling {
            let frame_settings = if factor > 1 {
                KaleidoSettings {
                    output_size_w: render_w,
                    output_size_h: render_h,
                    offset_x: render_offset_x,
                    offset_y: render_offset_y,
                    ..settings.clone()
                }
            } else {
                settings.clone()
            };
            rgba
                .par_chunks_exact_mut((render_w * 4) as usize)
                .enumerate()
                .for_each(|(y, row)| inner_loop_enhanced::<B>(y, row, &frame_settings, pyramid, pipeline, hue_rotation));
        } else {
        macro_rules! render_video_mode {
            ($mode:expr) => { rgba
            .par_chunks_exact_mut((render_w * 4) as usize)
            .enumerate()
            .for_each(|(y, row)| {
                inner_loop::<B, $mode>(
                    y,
                    row,
                    settings.zoom,
                    source,
                    &settings,
                    width_over_2,
                    center_x,
                    center_y,
                    slice_angle,
                    source.width(),
                    source.height(),
                    hue_rotation,
                );
            }) };
        }
        match settings.anti_alias { 0 => render_video_mode!(0), 2 => render_video_mode!(2), _ => render_video_mode!(1) };
        }

        let frame_bytes: &[u8] = if factor > 1 {
            final_rgba = downsample_box(&rgba, render_w, render_h, factor, out_w, out_h);
            &final_rgba
        } else {
            &rgba
        };

        sink.write_rgba_frame(frame_bytes).map_err(sink_err)?;
        last_frame.copy_from_slice(frame_bytes);
    }

    // write still frames at the end
    for _still_frame in 0..video_settings.still_frame_ending {
        sink.write_rgba_frame(&last_frame).map_err(sink_err)?;
    }

    sink.finish().map_err(sink_err)?;
    Ok(())
}

/// Converts a boxed `VideoSinkError` (`Send + Sync`) into the plain
/// `Box<dyn std::error::Error>` used throughout this module's render
/// functions.
fn sink_err(e: VideoSinkError) -> Box<dyn std::error::Error> {
    e
}

/// Renders a video with the GPU.
pub fn render_video_gpu_traditional(
    mut settings: KaleidoSettings,
    video_settings: VideoSettings,
    sink: &mut dyn VideoFrameSink,
    gpu: &mut GpuBackend
) -> Result<(), Box<dyn std::error::Error>> {
    let fps = video_settings.fps;
    let total_frames = f32::round(video_settings.animation_duration * video_settings.fps as f32) as u32;

    let (out_w, out_h) = (settings.output_size_w, settings.output_size_h);

    //let triangle_rotation_delta =
    //    degrees_to_radians(video_settings.triangle_rotation_degrees_per_frame);

    let base_rotation = settings.triangle_rotation_rad;
    let base_hue = settings.hue_rotation as f32;

    // `super_sample`: render each frame at `output_size * factor` into a scratch
    // GPU readback buffer, then box-downsample down to the final `out_w x out_h`
    // before it's handed to the H.264 encoder — same wrapper used by the CPU
    // video path (`render_video`) and the live-preview GPU paths in src-tauri.
    let factor = crate::safe_super_sample(settings.super_sample, settings.output_size_w, settings.output_size_h);
    let (render_w, render_h) = (out_w * factor as u32, out_h * factor as u32);
    let render_offset_x = settings.offset_x * factor as i32;
    let render_offset_y = settings.offset_y * factor as i32;

    let mut render_buf = vec![0u8; (render_w * render_h * 4) as usize];
    let mut output = vec![0u8; (out_w * out_h * 4) as usize];

    // Owned copy of the most recently written frame, retained only so
    // `still_frame_ending` can resend it without rerendering.
    let mut last_frame: Vec<u8> = vec![0u8; (out_w * out_h * 4) as usize];
    for frame in 0..total_frames {
        //settings.triangle_rotation_rad =
        //    (base_rotation + triangle_rotation_delta * frame as f32).rem_euclid(2.0 * PI);

        let rotation_modulation = modulate(
            &video_settings,
            frame,
            base_rotation + degrees_to_radians(video_settings.rotation_range),
            base_rotation,
            video_settings.rotation_cycles,
            video_settings.rotation_start_offset,
            &video_settings.rotation_fn,
        );

        settings.triangle_rotation_rad = rotation_modulation.rem_euclid(2.0 * PI);

        settings.hue_rotation = modulate(
            &video_settings,
            frame,
            base_hue + video_settings.hue_range as f32,
            base_hue,
            video_settings.hue_cycles,
            video_settings.hue_start_offset,
            &video_settings.hue_fn,
        )
        .round()
        .rem_euclid(360.0) as u32;

        // Scaled by `factor`: `source_scale = width_over_2 / zoom` ties
        // visible source content to the actual render width, which is
        // `factor`x larger than `out_w`/`out_h` here. `zoom_modulation`
        // itself is supersample-unaware, so the scaling has to happen at
        // the assignment — `render_settings` below inherits it via
        // `..settings.clone()`, and the `factor == 1` branch is a no-op
        // multiply, so this is correct either way.
        settings.zoom = zoom_modulation(&video_settings, frame) * factor as f32;
        let frame_bytes: &[u8] = if factor > 1 {
            let render_settings = KaleidoSettings {
                output_size_w: render_w,
                output_size_h: render_h,
                offset_x: render_offset_x,
                offset_y: render_offset_y,
                ..settings.clone()
            };
            gpu.render_into_buffer(&render_settings, &mut render_buf)?;
            output = downsample_box(&render_buf, render_w, render_h, factor, out_w, out_h);
            &output
        } else {
            gpu.render_into_buffer(&settings, &mut output)?;
            &output
        };

        sink.write_rgba_frame(frame_bytes).map_err(sink_err)?;
        last_frame.copy_from_slice(frame_bytes);
    }
    for _ in 0..video_settings.still_frame_ending {
        sink.write_rgba_frame(&last_frame).map_err(sink_err)?;
    }

    sink.finish().map_err(sink_err)?;

    Ok(())
}

pub fn render_video_gpu(
    mut settings: KaleidoSettings,
    video_settings: VideoSettings,
    sink: &mut dyn VideoFrameSink,
    gpu: &mut GpuBackend,
) -> Result<(), Box<dyn std::error::Error>> {
    let fps = video_settings.fps;
    let total_frames = f32::round(video_settings.animation_duration * fps as f32) as u32;

    // `super_sample`: the pipelined `GpuVideoRenderer` allocates its output
    // textures/readback buffers once at construction, so — unlike the CPU and
    // "traditional" GPU video paths — the renderer itself must be built at the
    // *scaled* `render_w x render_h` size. Each submitted frame's settings are
    // scaled to match, and each completed frame is box-downsampled back down
    // to `out_w x out_h` before it reaches the H.264 encoder.
    let (out_w, out_h) = (settings.output_size_w, settings.output_size_h);
    let factor = crate::safe_super_sample(settings.super_sample, settings.output_size_w, settings.output_size_h);
    let (render_w, render_h) = (out_w * factor as u32, out_h * factor as u32);
    let render_offset_x = settings.offset_x * factor as i32;
    let render_offset_y = settings.offset_y * factor as i32;

    let mut renderer = GpuVideoRenderer::new(
        gpu,
        render_w,
        render_h,
        3,
    )?;

    // let triangle_rotation_delta =
    //     degrees_to_radians(video_settings.triangle_rotation_degrees_per_frame);

    let base_rotation = settings.triangle_rotation_rad;
    let base_hue = settings.hue_rotation as f32;

    let mut last_frame: Option<Vec<u8>> = None;

    let mut smoothed_audio_peak = 0.0_f32;
    let mut accumulated_orientation_offset = 0.0_f32;
    
    let base_x = settings.triangle_center_x;
    let base_y = settings.triangle_center_y;
    let base_rotation = settings.triangle_rotation_rad;
    let base_hue = settings.hue_rotation as f32;
    // The animated zoom bounds are already absolute zoom values for the final
    // output width. Only scale by the supersampling factor here.
    let zoom_scale = factor as f32;

    let mut modulation_state = VideoFrameModulationState::default();
    
    for frame in 0..total_frames {
        let elapsed_seconds = frame as f32 / fps.max(1) as f32;

        let audio_peak = if video_settings.audio_reactive_enabled
            && !video_settings.audio_peaks.is_empty()
        {
            let raw_peak = video_settings
                .audio_peaks
                .get(frame as usize % video_settings.audio_peaks.len())
                .copied()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);

            let smoothing = video_settings.audio_peak_smoothing.clamp(0.0, 0.999);
            smoothed_audio_peak =
                smoothed_audio_peak * smoothing + raw_peak * (1.0 - smoothing);

            smoothed_audio_peak.clamp(0.0, 1.0)
        } else {
            smoothed_audio_peak = 0.0;
            0.0
        };

        accumulated_orientation_offset = (
            accumulated_orientation_offset
                + audio_peak * video_settings.orientation_peak_multiplier / fps.max(1) as f32
        )
            .rem_euclid(1.0);

        let rotation_modulation = modulate(
            &video_settings,
            frame,
            base_rotation + degrees_to_radians(video_settings.rotation_range),
            base_rotation,
            video_settings.rotation_cycles,
            video_settings.rotation_start_offset,
            &video_settings.rotation_fn,
        );

        let rotation_offset = rotation_modulation - base_rotation;

        let hero_radius = ((video_settings.hero_circle_right_x
            - video_settings.hero_circle_left_x)
            * 0.5)
            .abs()
            .max(1.0);

        let base_speed_cycles = video_settings.orientation_base_speed
            / (2.0 * PI * hero_radius);

        let orientation_value = (
            elapsed_seconds * base_speed_cycles + accumulated_orientation_offset
        )
            .rem_euclid(1.0);

        let use_orientation = video_settings.orientation_base_speed != 0.0
            || (video_settings.audio_reactive_enabled
                && accumulated_orientation_offset != 0.0);

        if use_orientation {
            let (orientation_x, orientation_y, orientation_rotation) =
                orientation_to_hero_params(
                    orientation_value,
                    video_settings.hero_circle_left_x,
                    video_settings.hero_circle_right_x,
                    video_settings.hero_circle_y,
                    video_settings.hero_desired_left_rotation,
                );

            settings.triangle_center_x = orientation_x;
            settings.triangle_center_y = orientation_y;
            settings.triangle_rotation_rad =
                (orientation_rotation + rotation_offset).rem_euclid(2.0 * PI);
        } else {
            settings.triangle_center_x = base_x;
            settings.triangle_center_y = base_y;
            settings.triangle_rotation_rad =
                rotation_modulation.rem_euclid(2.0 * PI);
        }

        let hue_rotation = apply_video_frame_modulation(
            &mut settings,
            &video_settings,
            frame,
            base_x,
            base_y,
            base_rotation,
            base_hue,
            zoom_scale,
            &mut modulation_state,
        );

        settings.hue_rotation = hue_rotation;


        // Submit at the (possibly scaled) render size the renderer was built with.
        let submit_settings = if factor > 1 {
            KaleidoSettings {
                output_size_w: render_w,
                output_size_h: render_h,
                offset_x: render_offset_x,
                offset_y: render_offset_y,
                ..settings.clone()
            }
        } else {
            settings.clone()
        };

        renderer.submit_frame(frame, &submit_settings)?;

        while let Some(done) = renderer.receive_oldest_blocking()? {
            let rgba = renderer.slot_bytes(done.slot_index)?;
            let frame_bytes: std::borrow::Cow<[u8]> = if factor > 1 {
                std::borrow::Cow::Owned(downsample_box(rgba, render_w, render_h, factor, out_w, out_h))
            } else {
                std::borrow::Cow::Borrowed(rgba)
            };
            sink.write_rgba_frame(&frame_bytes).map_err(sink_err)?;
            last_frame = Some(frame_bytes.into_owned());
            renderer.release_slot(done.slot_index)?;
        }
    }

    for done in renderer.drain_remaining_blocking()? {
        let rgba = renderer.slot_bytes(done.slot_index)?;
        let frame_bytes: std::borrow::Cow<[u8]> = if factor > 1 {
            std::borrow::Cow::Owned(downsample_box(rgba, render_w, render_h, factor, out_w, out_h))
        } else {
            std::borrow::Cow::Borrowed(rgba)
        };
        sink.write_rgba_frame(&frame_bytes).map_err(sink_err)?;
        last_frame = Some(frame_bytes.into_owned());
        renderer.release_slot(done.slot_index)?;
    }

    if let Some(last_frame) = last_frame {
        for _ in 0..video_settings.still_frame_ending {
            sink.write_rgba_frame(&last_frame).map_err(sink_err)?;
        }
    }

    sink.finish().map_err(sink_err)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbaImage};
    use std::sync::Mutex;

    static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn gradient_source(width: u32, height: u32) -> DynamicImage {
        let image = RgbaImage::from_fn(width, height, |x, y| {
            let fx = x as f32 / width.saturating_sub(1).max(1) as f32;
            let fy = y as f32 / height.saturating_sub(1).max(1) as f32;
            Rgba([
                (fx * 255.0).round() as u8,
                (fy * 255.0).round() as u8,
                ((0.65 * fx + 0.35 * fy) * 255.0).round() as u8,
                255,
            ])
        });
        DynamicImage::ImageRgba8(image)
    }

    fn settings_for(kaleido_type: KaleidoType, reconstruction: u8, derivatives: bool, anisotropy: u8, supersampling: u8, aspect_correct: bool) -> KaleidoSettings {
        KaleidoSettings {
            output_size_w: 67,
            output_size_h: 53,
            offset_x: 3,
            offset_y: -2,
            count: 7,
            zoom: 1.35,
            triangle_center_x: 48.0,
            triangle_center_y: 39.0,
            triangle_rotation_rad: 0.37,
            kaleido_type,
            tile_count: 4.5,
            hue_rotation: 23,
            recolor_enabled: false,
            recolor_seed: String::new(),
            recolor_mode: 0,
            recolor_threshold: 0.08,
            anti_alias: reconstruction,
            derivative_mipmapping: derivatives,
            anisotropy_level: anisotropy,
            super_sample: supersampling,
            aspect_correct,
        }
    }

    fn assert_images_close(label: &str, reference: &RgbaImage, actual: &RgbaImage, channel_tolerance: u8, max_bad_pixel_ratio: f32) {
        assert_eq!(reference.dimensions(), actual.dimensions(), "{label}: output dimensions differ");
        let mut bad_pixels = 0usize;
        let mut worst_delta = 0u8;
        let mut absolute_error = 0u64;
        for (expected, observed) in reference.pixels().zip(actual.pixels()) {
            let mut pixel_bad = false;
            for channel in 0..4 {
                let delta = expected[channel].abs_diff(observed[channel]);
                worst_delta = worst_delta.max(delta);
                absolute_error += delta as u64;
                pixel_bad |= delta > channel_tolerance;
            }
            bad_pixels += pixel_bad as usize;
        }
        let pixels = (reference.width() * reference.height()) as usize;
        let bad_ratio = bad_pixels as f32 / pixels as f32;
        let mean_error = absolute_error as f32 / (pixels * 4) as f32;
        assert!(bad_ratio <= max_bad_pixel_ratio,
            "{label}: {bad_pixels}/{pixels} pixels ({:.3}%) exceeded channel tolerance {channel_tolerance}; worst delta {worst_delta}, mean absolute error {mean_error:.3}",
            bad_ratio * 100.0);
    }

    fn assert_backend_matches_scalar<B: KaleidoBackend + DaydreamBackend>(label: &str, source: &DynamicImage, settings: &KaleidoSettings) {
        let scalar = render_kaleidoscope_with_backend::<f32>(source, settings.clone());
        let backend = render_kaleidoscope_with_backend::<B>(source, settings.clone());
        assert_images_close(label, &scalar, &backend, 3, 0.02);
    }

    macro_rules! backend_parity_case {
        ($module:ident, $settings:expr) => {
            mod $module {
                use super::*;

                fn fixture() -> (DynamicImage, KaleidoSettings) {
                    (gradient_source(96, 80), $settings)
                }

                #[test]
                fn gpu_matches_scalar_reference() {
                    let _gpu_guard = GPU_TEST_LOCK.lock().expect("GPU test lock poisoned");
                    let (source, settings) = fixture();
                    let scalar = render_kaleidoscope_with_backend::<f32>(&source, settings.clone());
                    let Ok(gpu) = render_kaleidoscope_with_gpu(&source, settings) else {
                        eprintln!("GPU parity skipped: no compatible adapter available");
                        return;
                    };
                    assert_images_close("GPU vs scalar", &scalar, &gpu, 6, 0.05);
                }

                #[cfg(target_arch = "x86_64")]
                #[test]
                fn avx2_matches_scalar_reference() {
                    if !is_x86_feature_detected!("avx2") { return; }
                    let (source, settings) = fixture();
                    assert_backend_matches_scalar::<core::arch::x86_64::__m256>("AVX2 vs scalar", &source, &settings);
                }

                #[cfg(target_arch = "x86_64")]
                #[test]
                fn sse2_matches_scalar_reference() {
                    if !is_x86_feature_detected!("sse2") { return; }
                    let (source, settings) = fixture();
                    assert_backend_matches_scalar::<core::arch::x86_64::__m128>("SSE2 vs scalar", &source, &settings);
                }

                #[cfg(target_arch = "aarch64")]
                #[test]
                fn neon_matches_scalar_reference() {
                    if !std::arch::is_aarch64_feature_detected!("neon") { return; }
                    let (source, settings) = fixture();
                    assert_backend_matches_scalar::<core::arch::aarch64::float32x4_t>("NEON vs scalar", &source, &settings);
                }
            }
        };
    }

    backend_parity_case!(radial_nearest, {
        let mut settings = settings_for(KaleidoType::Radial, 0, true, 1, 1, false);
        settings.hue_rotation = 0;
        settings
    });
    backend_parity_case!(square_bilinear_unmipped, {
        let mut settings = settings_for(KaleidoType::Square, 1, false, 1, 1, true);
        settings.hue_rotation = 0;
        settings
    });
    backend_parity_case!(square_bilinear_mipped, settings_for(KaleidoType::Square, 1, true, 1, 1, true));
    backend_parity_case!(diamond_bicubic_anisotropic, settings_for(KaleidoType::Diamond, 2, true, 4, 1, false));
    backend_parity_case!(hexagonal_supersampled, settings_for(KaleidoType::Hexagonal, 1, true, 8, 2, true));

    #[test]
    fn enhanced_simd_writes_trailing_pixels_for_odd_widths() {
        let source = DynamicImage::ImageRgba8(RgbaImage::from_pixel(128, 128, Rgba([24, 48, 96, 255])));
        let settings = KaleidoSettings {
            output_size_w: 67,
            output_size_h: 5,
            offset_x: 0,
            offset_y: 0,
            count: 6,
            zoom: 10.0,
            triangle_center_x: 64.0,
            triangle_center_y: 64.0,
            triangle_rotation_rad: 0.0,
            kaleido_type: KaleidoType::Radial,
            tile_count: 4.0,
            hue_rotation: 0,
            recolor_enabled: false,
            recolor_seed: String::new(),
            recolor_mode: 0,
            recolor_threshold: 0.08,
            anti_alias: 1,
            derivative_mipmapping: true,
            anisotropy_level: 4,
            super_sample: 1,
            aspect_correct: false,
        };

        let output = render_kaleidoscope_with_auto_backend(&source, settings);
        assert!(output.pixels().all(|pixel| pixel.0 == [24, 48, 96, 255]));
    }
}
