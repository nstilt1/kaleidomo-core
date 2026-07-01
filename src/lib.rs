#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub mod backends;
#[cfg(not(target_arch = "wasm32"))]
mod rlib;

#[cfg(not(target_arch = "wasm32"))]
pub use rlib::*;

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
}

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
    pub num_zoom_loops: u32,
    
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
