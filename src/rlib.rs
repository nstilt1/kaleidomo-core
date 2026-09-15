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
use crate::{KaleidoSettings, VideoSettings, modulate};
use crate::backends::gpu::{GpuBackend, GpuVideoRenderer};
pub use crate::backends::{KaleidoBackend, DaydreamBackend, Register, inner_loop};

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

    // Rayon parallelizes the rows automatically
    pixels
        .par_chunks_exact_mut((settings.output_size_w * 4) as usize)
        .enumerate()
        .for_each(|(y, row)| {
            inner_loop::<B>(
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
        });

    ImageBuffer::from_raw(settings.output_size_w, settings.output_size_h, pixels).unwrap()
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

        rgba
            .par_chunks_exact_mut((render_w * 4) as usize)
            .enumerate()
            .for_each(|(y, row)| {
                inner_loop::<B>(
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
            });

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
    use crate::KaleidoType;

use super::*;
    use image::{DynamicImage, RgbaImage};

    #[test]
    fn test_simd_vs_scalar_parity() {
        // 1. Setup a dummy source image (e.g., a 100x100 gradient)
        let sw = 100;
        let sh = 100;
        let mut source_pixels = Vec::new();
        for y in 0..sh {
            for x in 0..sw {
                source_pixels.extend_from_slice(&[x as u8, y as u8, 128, 255]);
            }
        }
        let source = DynamicImage::ImageRgba8(RgbaImage::from_raw(sw, sh, source_pixels).unwrap());

        // 2. Setup Kaleidoscope settings
        let settings = KaleidoSettings {
            output_size_w: 64, // Keep it small for fast tests
            output_size_h: 64,
            offset_x: 0,
            offset_y: 0,
            count: 6,        // Hexagonal symmetry
            zoom: 1.0,
            triangle_center_x: 50.0,
            triangle_center_y: 50.0,
            triangle_rotation_rad: 0.0,
            kaleido_type: KaleidoType::Hexagonal,
            tile_count: 4.0,
            hue_rotation: 0,
            anti_alias: false,
            super_sample: 1,
            aspect_correct: false,
        };

        // 3. Render using Scalar Backend
        // Note: You may need to expose these functions or make them generic
        // to call specific backends in the same test.
        //let scalar_image = render_kaleidoscope_with_backend::<f32>(&source, settings.clone());
        let scalar_image = render_kaleidoscope_with_backend::<f32>(&source, settings.clone());

        // 4. Render using Aarch64 (Neon) Backend
        let simd_image = render_kaleidoscope_with_gpu(&source, settings.clone()).unwrap();
        //let simd_image = render_kaleidoscope_with_backend::<Register>(&source, settings.clone());

        // 5. Compare pixels
        let mut diff_count = 0;
        let threshold = 1; // Allow for 1-bit rounding difference in color channels

        for (p_scalar, p_simd) in scalar_image.pixels().zip(simd_image.pixels()) {
            for i in 0..4 {
                // Check R, G, B, A
                let diff = (p_scalar[i] as i16 - p_simd[i] as i16).abs();
                if diff > threshold {
                    diff_count += 1;
                }
            }
        }

        let total_pixels = settings.output_size_h * settings.output_size_w;
        let error_rate = diff_count as f32 / (total_pixels * 4) as f32;

        // We allow a very small error rate due to float precision differences
        // in trig approximations (Polynomial vs libm)
        assert!(
            error_rate < 0.001,
            "Neon output diverged from Scalar! Error rate: {:.4}%",
            error_rate * 100.0
        );
    }
}
