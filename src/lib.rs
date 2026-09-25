// kaleidomo-core/src/lib.rs
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub mod backends;
pub mod enhancement;
pub mod preprocess;
#[cfg(not(target_arch = "wasm32"))]
mod rlib;
#[cfg(not(target_arch = "wasm32"))]
mod video_sink;

#[cfg(not(target_arch = "wasm32"))]
pub use rlib::*;
#[cfg(not(target_arch = "wasm32"))]
pub use video_sink::{VideoFrameSink, VideoSinkError};

#[cfg(target_arch = "wasm32")]
mod wasm;

#[cfg(not(target_arch = "wasm32"))]
use serde::Deserialize;
#[cfg(target_arch = "wasm32")]
pub use wasm::*;

use core::f32::consts::PI;

#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(target_arch = "wasm32"), derive(Deserialize))]
pub enum KaleidoType {
    Radial,
    Square,
    Diamond,
    Hexagonal,
    HexagonalFlatTop,
}

#[derive(Clone, Debug)]
#[cfg_attr(not(target_arch = "wasm32"), derive(Deserialize))]
#[cfg_attr(not(target_arch = "wasm32"), serde(rename_all = "camelCase"))]
pub struct KaleidoSettings {
    pub count: u32,       // Number of reflections (e.g., 8)
    pub output_size_w: u32,
    pub output_size_h: u32,
    pub offset_x: i32,
    pub offset_y: i32,
    pub zoom: f32,        // How much of the triangle to show
    pub tile_count: f32,
    pub triangle_center_x: f32, // Center of the triangle in source image
    pub triangle_center_y: f32,
    pub triangle_rotation_rad: f32, // Rotation of the triangle in radians
    pub kaleido_type: KaleidoType,  // Type of kaleidoscope (radial, square, etc.)
    pub hue_rotation: u32, // Hue rotation in degrees (0-360)
    #[cfg_attr(not(target_arch = "wasm32"), serde(default))]
    pub recolor_enabled: bool,
    #[cfg_attr(not(target_arch = "wasm32"), serde(default))]
    pub recolor_seed: String,
    #[cfg_attr(not(target_arch = "wasm32"), serde(default))]
    pub recolor_mode: u8,
    #[cfg_attr(not(target_arch = "wasm32"), serde(default = "default_recolor_threshold"))]
    pub recolor_threshold: f32,
    #[cfg_attr(not(target_arch = "wasm32"), serde(default = "default_recolor_cell_size"))]
    pub recolor_cell_size: f32,

    // ── Enhancements (all default-disabled to preserve existing look/output) ──
    /// Enables bilinear texture filtering when sampling the source image,
    /// softening hard pixel edges and mirrored wedge seams. When `false`
    /// (the default), sampling is nearest-neighbor, matching all prior
    /// rendered output and existing `.kmo.json` presets that predate this field.
    #[cfg_attr(not(target_arch = "wasm32"), serde(default))]
    /// Source reconstruction mode: 0 = nearest, 1 = bilinear, 2 = Catmull-Rom bicubic.
    pub anti_alias: u8,
    #[cfg_attr(not(target_arch = "wasm32"), serde(default = "default_true"))]
    pub derivative_mipmapping: bool,
    #[cfg_attr(not(target_arch = "wasm32"), serde(default = "default_anisotropy"))]
    pub anisotropy_level: u8,
    /// Internal supersampling factor. `1` (the default) disables supersampling
    /// and renders at native `output_size_w`/`output_size_h`. Values `2`-`4`
    /// render the frame at `output_size * super_sample` internally and then
    /// box-downsample back down to the requested output size, reducing
    /// aliasing across the whole image (not just at texture edges). Values
    /// are clamped to `1..=4` by callers to bound the extra render cost.
    #[cfg_attr(not(target_arch = "wasm32"), serde(default = "default_super_sample"))]
    pub super_sample: u8,
    /// Corrects the kaleidoscope pattern for non-square output canvases. When
    /// `false` (the default, matching all existing presets), the pattern is
    /// mapped 1:1 to pixel coordinates, which visually stretches the mirrored
    /// wedges into an ellipse whenever `output_size_w != output_size_h`. When
    /// `true`, the vertical axis is scaled by the canvas aspect ratio before
    /// the angle/radius is computed, so wedges stay proportional instead of
    /// looking stretched.
    #[cfg_attr(not(target_arch = "wasm32"), serde(default))]
    pub aspect_correct: bool,
}

/// Default value for `KaleidoSettings::super_sample` used by `serde(default = ...)`
/// so that presets/JSON saved before this field existed deserialize with
/// supersampling disabled (`1`) rather than `0`.
#[cfg(not(target_arch = "wasm32"))]
fn default_super_sample() -> u8 {
    1
}
#[cfg(not(target_arch = "wasm32"))]
fn default_true() -> bool { true }
#[cfg(not(target_arch = "wasm32"))]
fn default_anisotropy() -> u8 { 1 }
#[cfg(not(target_arch = "wasm32"))]
fn default_recolor_threshold() -> f32 { 0.08 }
#[cfg(not(target_arch = "wasm32"))]
fn default_recolor_cell_size() -> f32 { 64.0 }

pub struct VideoSettings {
    /// The duration of the animation
    pub animation_duration: f32,
    /// The range of the rotation animation
    pub rotation_range: f32,
    /// The number of rotation cycles
    pub rotation_cycles: f32,
    /// The offset of the rotation animation's phase.
    pub rotation_start_offset: f32,
    /// The rotation function. Can be:
    /// * linear/saw
    /// * triangle
    /// * sin
    /// * sin2
    /// * cos
    /// * -cos
    pub rotation_fn: String,
    /// The range of the hue changing animation
    pub hue_range: i32,
    /// The number of hue changing cycles
    pub hue_cycles: f32,
    /// The phase offset at the start of the hue animation
    pub hue_start_offset: f32,
    /// The hue changing function
    pub hue_fn: String,
    /// Number of still frames at the end of the video
    pub still_frame_ending: u32,
    /// Frame rate
    pub fps: u32,
    /// Quality of the video (0.0 to 1.0)
    pub quality: f32,
    /// The maximum zoom
    pub zoom_max: f32,
    /// The minimum zoom
    pub zoom_min: f32,
    /// The zoom function: linear or sin
    pub zoom_fn: String,
    /// The angle of the zoom at frame 0 in the sawtooth/sin space
    pub zoom_start_offset: f32,
    /// The amount of times that zoom will loop in the video.
    pub num_zoom_loops: f32,
    
    // Audio-reactive export fields
    pub audio_reactive_enabled: bool,
    pub audio_peak_smoothing: f32,
    pub orientation_base_speed: f32,
    pub orientation_peak_multiplier: f32,
    pub audio_peaks: Vec<f32>,

    // Hero circle / orientation export fields
    pub hero_circle_left_x: f32,
    pub hero_circle_right_x: f32,
    pub hero_circle_y: f32,
    pub hero_desired_left_rotation: f32,
}

/// Modulates a parameter using the frame number.
fn modulate(
    video_settings: &VideoSettings, 
    frame: u32, 
    range_max: f32, 
    range_min: f32, 
    num_loops: f32, 
    start_offset: f32,
    function: &str
) -> f32 {
    let range = range_max - range_min;
    let frame_count = video_settings.animation_duration * video_settings.fps as f32;

    match function.to_ascii_lowercase().as_str() {
        "triangle" => {
            let phase = (frame as f32 / frame_count)
                * num_loops
                + start_offset;

            let phase = phase.fract();

            let tri = 1.0 - (2.0 * phase - 1.0).abs();

            range_min + tri * range
        },

        "sawtooth" => {
            let phase = (frame as f32 / frame_count)
                * num_loops
                + start_offset;

            let saw = phase.fract(); // 0 → 1 ramp

            range_min + saw * range
        },

        "sin" => {
            let phase = (frame as f32 / frame_count)
                * num_loops
                + start_offset;

            let angle = phase * 2.0 * PI;

            let sin_norm = (f32::sin(angle) + 1.0) * 0.5;

            range_min + sin_norm * range
        },

        "sin2" => {
            let phase = (frame as f32 / frame_count)
                * num_loops
                + start_offset;

            let angle = phase * 2.0 * PI;

            // sin²(x) = (1 - cos(2x)) / 2
            let sin2_norm = f32::sin(angle).powi(2);

            range_min + sin2_norm * range
        },

        "cos" => {
            let phase = (frame as f32 / frame_count)
                * num_loops
                + start_offset;

            let angle = phase * 2.0 * PI;

            let cos_norm = (f32::cos(angle) + 1.0) * 0.5;

            range_min + cos_norm * range
        },

        "-cos" => {
            let phase = (frame as f32 / frame_count)
                * num_loops
                + start_offset;

            let angle = phase * 2.0 * PI;

            // Inverted cosine wave
            let neg_cos_norm = (1.0 - f32::cos(angle)) * 0.5;

            range_min + neg_cos_norm * range
        },

        _ => range_min
    }
}

/// Disable supersampling when either internal dimension would exceed 8192.
pub fn safe_super_sample(factor: u8, width: u32, height: u32) -> u8 {
    let factor = factor.clamp(1, 4);
    if width > 8192 / factor as u32 || height > 8192 / factor as u32 { 1 } else { factor }
}

#[cfg(test)]
mod supersampling_limit_tests {
    use super::safe_super_sample;
    #[test]
    fn limits_both_dimensions_without_overflow() {
        assert_eq!(safe_super_sample(4, 2048, 2048), 4);
        assert_eq!(safe_super_sample(4, 2049, 2048), 1);
        assert_eq!(safe_super_sample(2, 2000, 4097), 1);
        assert_eq!(safe_super_sample(2, 4096, 4096), 2);
        assert_eq!(safe_super_sample(4, u32::MAX, 1), 1);
        assert_eq!(safe_super_sample(1, 9000, 1), 1);
    }
}
