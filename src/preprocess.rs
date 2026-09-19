//! Deterministic source-image recoloring performed before kaleidoscope mapping.

use image::RgbaImage;
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use sha2::{Digest, Sha256};

pub const HUE_BAND_COUNT: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecolorMode {
    ColorBands,
    BorderedCells,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreprocessParams {
    pub hue_offsets: [f32; HUE_BAND_COUNT],
    pub threshold: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreprocessError {
    InvalidThreshold,
}

impl std::fmt::Display for PreprocessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidThreshold => write!(f, "recolor threshold must be finite"),
        }
    }
}

impl std::error::Error for PreprocessError {}

/// Derive stable hue-band offsets from arbitrary seed bytes.
pub fn params_from_seed(seed_input: &[u8], threshold: f32) -> Result<PreprocessParams, PreprocessError> {
    if !threshold.is_finite() {
        return Err(PreprocessError::InvalidThreshold);
    }
    let digest: [u8; 32] = Sha256::digest(seed_input).into();
    let mut rng = ChaCha8Rng::from_seed(digest);
    let mut hue_offsets = [0.0; HUE_BAND_COUNT];
    for offset in &mut hue_offsets {
        // Use integer sampling then an explicit conversion so every backend is
        // fed exactly the same f32 values.
        *offset = (rng.next_u32() as f64 / 4_294_967_296.0) as f32;
    }
    Ok(PreprocessParams { hue_offsets, threshold: threshold.clamp(0.0, 1.0) })
}

/// Recolor a raw source frame in-place. Alpha is preserved.
pub fn preprocess_source_frame(
    image: &mut RgbaImage,
    seed_input: &[u8],
    threshold: f32,
) -> Result<(), PreprocessError> {
    preprocess_source_frame_with_mode(image, seed_input, threshold, RecolorMode::ColorBands)
}

pub fn preprocess_source_frame_with_mode(
    image: &mut RgbaImage,
    seed_input: &[u8],
    threshold: f32,
    mode: RecolorMode,
) -> Result<(), PreprocessError> {
    let params = params_from_seed(seed_input, threshold)?;
    for pixel in image.pixels_mut() {
        let a = pixel[3];
        let rgb = [pixel[0] as f32 / 255.0, pixel[1] as f32 / 255.0, pixel[2] as f32 / 255.0];
        let (h, s, v) = rgb_to_hsv(rgb);
        let strength = match mode {
            RecolorMode::ColorBands => smoothstep(params.threshold, (params.threshold + 0.08).min(1.0), s),
            // Borders in illustrations and cell-like imagery tend to be dark,
            // low-chroma, or both. Preserve those pixels while independently
            // classifying the brighter interiors below.
            RecolorMode::BorderedCells => {
                let chroma = smoothstep(params.threshold, (params.threshold + 0.08).min(1.0), s);
                let interior = smoothstep(params.threshold * 0.75, (params.threshold * 0.75 + 0.12).min(1.0), v);
                chroma * interior
            }
        };
        if strength > 0.0 {
            let offset = match mode {
                RecolorMode::ColorBands => sampled_offset(h, &params),
                RecolorMode::BorderedCells => cell_offset(h, s, v, &params),
            };
            let shifted = hsv_to_rgb(((h + offset * strength).fract(), s, v));
            pixel[0] = (shifted[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            pixel[1] = (shifted[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            pixel[2] = (shifted[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            pixel[3] = a;
        }
    }
    Ok(())
}

fn cell_offset(h: f32, s: f32, v: f32, params: &PreprocessParams) -> f32 {
    // Combining three continuous color dimensions separates small bordered
    // shapes that share a broad hue family but differ in fill/lightness. The
    // interpolation inside each bin keeps gradients stable during animation.
    let hue_bin = (h.rem_euclid(1.0) * 8.0).floor() as usize;
    let sat_bin = (s.clamp(0.0, 0.999_999) * 4.0).floor() as usize;
    let value_position = v.clamp(0.0, 0.999_999) * 4.0;
    let value_bin = value_position.floor() as usize;
    let next_value = (value_bin + 1).min(3);
    let a = (hue_bin + sat_bin * 5 + value_bin * 3) % HUE_BAND_COUNT;
    let b = (hue_bin + sat_bin * 5 + next_value * 3) % HUE_BAND_COUNT;
    let t = value_position.fract();
    let t = t * t * (3.0 - 2.0 * t);
    params.hue_offsets[a] + (params.hue_offsets[b] - params.hue_offsets[a]) * t
}

pub fn sampled_offset(hue: f32, params: &PreprocessParams) -> f32 {
    let p = hue.rem_euclid(1.0) * HUE_BAND_COUNT as f32;
    let i = p.floor() as usize % HUE_BAND_COUNT;
    let next = (i + 1) % HUE_BAND_COUNT;
    let t = p.fract();
    let t = t * t * (3.0 - 2.0 * t);
    params.hue_offsets[i] + (params.hue_offsets[next] - params.hue_offsets[i]) * t
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    if b <= a { return (x >= a) as u8 as f32; }
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn rgb_to_hsv(rgb: [f32; 3]) -> (f32, f32, f32) {
    let max = rgb[0].max(rgb[1]).max(rgb[2]);
    let min = rgb[0].min(rgb[1]).min(rgb[2]);
    let d = max - min;
    let s = if max <= f32::EPSILON { 0.0 } else { d / max };
    let h = if d <= f32::EPSILON { 0.0 }
        else if max == rgb[0] { ((rgb[1] - rgb[2]) / d).rem_euclid(6.0) / 6.0 }
        else if max == rgb[1] { (((rgb[2] - rgb[0]) / d) + 2.0) / 6.0 }
        else { (((rgb[0] - rgb[1]) / d) + 4.0) / 6.0 };
    (h, s, max)
}

fn hsv_to_rgb(hsv: (f32, f32, f32)) -> [f32; 3] {
    let (h, s, v) = hsv;
    let sector = (h.rem_euclid(1.0) * 6.0).floor() as u32;
    let f = h.rem_euclid(1.0) * 6.0 - sector as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match sector % 6 {
        0 => [v, t, p], 1 => [q, v, p], 2 => [p, v, t],
        3 => [p, q, v], 4 => [t, p, v], _ => [v, p, q],
    }
}
