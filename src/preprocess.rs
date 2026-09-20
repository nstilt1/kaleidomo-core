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
    SeededVoronoi,
    ConnectedComponents,
    SlicSuperpixels,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreprocessParams {
    pub hue_offsets: [f32; HUE_BAND_COUNT],
    pub seed_words: [u32; 4],
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
    let seed_words = std::array::from_fn(|i| {
        u32::from_le_bytes(digest[i * 4..i * 4 + 4].try_into().unwrap())
    });
    Ok(PreprocessParams { hue_offsets, seed_words, threshold: threshold.clamp(0.0, 1.0) })
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
    preprocess_source_frame_with_mode_and_cell_size(image, seed_input, threshold, mode, 64.0)
}

pub fn preprocess_source_frame_with_mode_and_cell_size(
    image: &mut RgbaImage,
    seed_input: &[u8],
    threshold: f32,
    mode: RecolorMode,
    cell_size: f32,
) -> Result<(), PreprocessError> {
    let params = params_from_seed(seed_input, threshold)?;
    if mode == RecolorMode::ConnectedComponents {
        recolor_connected_components(image, cell_size, &params);
        return Ok(());
    }
    if mode == RecolorMode::SlicSuperpixels {
        recolor_slic_superpixels(image, cell_size, &params);
        return Ok(());
    }
    let width = image.width();
    for (index, pixel) in image.pixels_mut().enumerate() {
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
            RecolorMode::SeededVoronoi => smoothstep(params.threshold, (params.threshold + 0.08).min(1.0), s),
            RecolorMode::ConnectedComponents | RecolorMode::SlicSuperpixels => unreachable!(),
        };
        if strength > 0.0 {
            let offset = match mode {
                RecolorMode::ColorBands => sampled_offset(h, &params),
                RecolorMode::BorderedCells => cell_offset(h, s, v, &params),
                RecolorMode::SeededVoronoi => {
                    let x = (index as u32 % width) as f32;
                    let y = (index as u32 / width) as f32;
                    voronoi_offset(x, y, cell_size, &params)
                }
                RecolorMode::ConnectedComponents | RecolorMode::SlicSuperpixels => unreachable!(),
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

fn recolor_connected_components(image: &mut RgbaImage, cell_size: f32, params: &PreprocessParams) {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 { return; }
    let source = image.clone();
    let mut visited = vec![false; (width * height) as usize];
    let tolerance = 0.12f32;
    let min_area = ((cell_size.clamp(4.0, 512.0) * 0.35).powi(2) as usize).max(1);
    let mut queue = std::collections::VecDeque::new();
    let mut component = Vec::new();
    let mut component_id = 0u32;

    for start_y in 0..height {
        for start_x in 0..width {
            let start_index = (start_y * width + start_x) as usize;
            if visited[start_index] { continue; }
            visited[start_index] = true;
            queue.push_back((start_x, start_y));
            component.clear();
            while let Some((x, y)) = queue.pop_front() {
                component.push((x, y));
                let reference = source.get_pixel(x, y);
                for (nx, ny) in [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)] {
                    if nx >= width || ny >= height { continue; }
                    let index = (ny * width + nx) as usize;
                    if visited[index] { continue; }
                    let neighbor = source.get_pixel(nx, ny);
                    let distance = (0..3).map(|c| {
                        let d = reference[c] as f32 / 255.0 - neighbor[c] as f32 / 255.0;
                        d * d
                    }).sum::<f32>().sqrt();
                    if distance <= tolerance {
                        visited[index] = true;
                        queue.push_back((nx, ny));
                    }
                }
            }
            if component.len() >= min_area {
                let offset_index = hash_cell(component_id as i32, component.len() as i32, params.seed_words[3]) as usize % HUE_BAND_COUNT;
                recolor_region(image, &component, params.hue_offsets[offset_index], params.threshold);
            }
            component_id = component_id.wrapping_add(1);
        }
    }
}

fn recolor_slic_superpixels(image: &mut RgbaImage, cell_size: f32, params: &PreprocessParams) {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 { return; }
    let labs: Vec<[f32; 3]> = image.pixels().map(|p| rgb_to_lab(p[0], p[1], p[2])).collect();
    let spacing = cell_size.clamp(4.0, 512.0);
    let cols = (width as f32 / spacing).ceil().max(1.0) as u32;
    let rows = (height as f32 / spacing).ceil().max(1.0) as u32;
    let mut centers = Vec::with_capacity((cols * rows) as usize);
    for gy in 0..rows {
        for gx in 0..cols {
            let jitter_hash = hash_cell(gx as i32, gy as i32, params.seed_words[0]);
            let jitter_x = ((jitter_hash & 0xffff) as f32 / 65535.0 - 0.5) * spacing * 0.2;
            let jitter_y = ((jitter_hash >> 16) as f32 / 65535.0 - 0.5) * spacing * 0.2;
            let x = ((gx as f32 + 0.5) * spacing + jitter_x).clamp(0.0, width as f32 - 1.0);
            let y = ((gy as f32 + 0.5) * spacing + jitter_y).clamp(0.0, height as f32 - 1.0);
            let lab = labs[y as usize * width as usize + x as usize];
            centers.push([x, y, lab[0], lab[1], lab[2]]);
        }
    }
    let pixel_count = (width * height) as usize;
    let mut labels = vec![0usize; pixel_count];
    let mut distances = vec![f32::INFINITY; pixel_count];
    let compactness = 10.0f32;
    for _ in 0..5 {
        distances.fill(f32::INFINITY);
        for (label, center) in centers.iter().enumerate() {
            let x0 = (center[0] - spacing).max(0.0) as u32;
            let y0 = (center[1] - spacing).max(0.0) as u32;
            let x1 = (center[0] + spacing).min(width as f32 - 1.0) as u32;
            let y1 = (center[1] + spacing).min(height as f32 - 1.0) as u32;
            for y in y0..=y1 { for x in x0..=x1 {
                let lab = labs[(y * width + x) as usize];
                let color = ((lab[0] - center[2]).powi(2) + (lab[1] - center[3]).powi(2) + (lab[2] - center[4]).powi(2)).sqrt() / 100.0;
                let spatial = ((x as f32 - center[0]).powi(2) + (y as f32 - center[1]).powi(2)).sqrt() / spacing;
                let distance = color + compactness / 10.0 * spatial;
                let index = (y * width + x) as usize;
                if distance < distances[index] { distances[index] = distance; labels[index] = label; }
            }}
        }
        let mut sums = vec![[0.0f32; 6]; centers.len()];
        for y in 0..height { for x in 0..width {
            let index = (y * width + x) as usize; let s = &mut sums[labels[index]];
            let lab = labs[index];
            s[0] += x as f32; s[1] += y as f32; s[2] += lab[0]; s[3] += lab[1]; s[4] += lab[2]; s[5] += 1.0;
        }}
        for (center, sum) in centers.iter_mut().zip(sums) { if sum[5] > 0.0 { for i in 0..5 { center[i] = sum[i] / sum[5]; } } }
    }
    let mut regions = vec![Vec::new(); centers.len()];
    for y in 0..height { for x in 0..width { regions[labels[(y * width + x) as usize]].push((x, y)); } }
    for (label, region) in regions.iter().enumerate() {
        let offset_index = hash_cell(label as i32, region.len() as i32, params.seed_words[3]) as usize % HUE_BAND_COUNT;
        recolor_region(image, region, params.hue_offsets[offset_index], params.threshold);
    }
}

fn rgb_to_lab(r: u8, g: u8, b: u8) -> [f32; 3] {
    let linear = |value: u8| { let v = value as f32 / 255.0; if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) } };
    let r = linear(r); let g = linear(g); let b = linear(b);
    let x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047;
    let y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    let z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883;
    let f = |v: f32| if v > 0.008856 { v.cbrt() } else { 7.787 * v + 16.0 / 116.0 };
    let fx = f(x); let fy = f(y); let fz = f(z);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

fn recolor_region(image: &mut RgbaImage, region: &[(u32, u32)], offset: f32, threshold: f32) {
    for &(x, y) in region {
        let pixel = image.get_pixel_mut(x, y); let alpha = pixel[3];
        let (h, s, v) = rgb_to_hsv([pixel[0] as f32 / 255.0, pixel[1] as f32 / 255.0, pixel[2] as f32 / 255.0]);
        let strength = smoothstep(threshold, (threshold + 0.08).min(1.0), s);
        let shifted = hsv_to_rgb(((h + offset * strength).fract(), s, v));
        for c in 0..3 { pixel[c] = (shifted[c].clamp(0.0, 1.0) * 255.0 + 0.5) as u8; }
        pixel[3] = alpha;
    }
}

fn hash_cell(x: i32, y: i32, seed: u32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (y as u32).wrapping_mul(0xd816_3841)
        ^ seed;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^ (h >> 16)
}

fn voronoi_offset(x: f32, y: f32, cell_size: f32, params: &PreprocessParams) -> f32 {
    let size = if cell_size.is_finite() { cell_size.clamp(4.0, 512.0) } else { 64.0 };
    let grid_x = (x / size).floor() as i32;
    let grid_y = (y / size).floor() as i32;
    let mut best_distance = f32::INFINITY;
    let mut best_hash = 0;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let cell_x = grid_x + dx;
            let cell_y = grid_y + dy;
            let hx = hash_cell(cell_x, cell_y, params.seed_words[0]);
            let hy = hash_cell(cell_x, cell_y, params.seed_words[1]);
            let jitter_x = hx as f32 / u32::MAX as f32;
            let jitter_y = hy as f32 / u32::MAX as f32;
            let center_x = (cell_x as f32 + 0.15 + jitter_x * 0.7) * size;
            let center_y = (cell_y as f32 + 0.15 + jitter_y * 0.7) * size;
            let distance = (x - center_x).powi(2) + (y - center_y).powi(2);
            if distance < best_distance {
                best_distance = distance;
                best_hash = hash_cell(cell_x, cell_y, params.seed_words[2]);
            }
        }
    }
    params.hue_offsets[best_hash as usize % HUE_BAND_COUNT]
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

#[cfg(test)]
mod tests {
    use super::*;
    fn colorful_image() -> RgbaImage {
        RgbaImage::from_fn(96, 96, |x, y| image::Rgba([((x * 2 + 40) % 255) as u8, ((y * 2 + 80) % 255) as u8, (((x + y) * 2 + 120) % 255) as u8, 255]))
    }
    #[test]
    fn seeded_voronoi_is_deterministic() {
        let mut first = colorful_image(); let mut second = first.clone();
        preprocess_source_frame_with_mode_and_cell_size(&mut first, b"repeatable", 0.0, RecolorMode::SeededVoronoi, 24.0).unwrap();
        preprocess_source_frame_with_mode_and_cell_size(&mut second, b"repeatable", 0.0, RecolorMode::SeededVoronoi, 24.0).unwrap();
        assert_eq!(first, second);
    }
    #[test]
    fn seeded_voronoi_responds_to_seed_and_cell_size() {
        let source = colorful_image(); let mut other_seed = source.clone(); let mut small = source.clone(); let mut large = source.clone();
        preprocess_source_frame_with_mode_and_cell_size(&mut other_seed, b"other", 0.0, RecolorMode::SeededVoronoi, 24.0).unwrap();
        preprocess_source_frame_with_mode_and_cell_size(&mut small, b"seed", 0.0, RecolorMode::SeededVoronoi, 12.0).unwrap();
        preprocess_source_frame_with_mode_and_cell_size(&mut large, b"seed", 0.0, RecolorMode::SeededVoronoi, 64.0).unwrap();
        assert_ne!(small, large); assert_ne!(small, other_seed);
    }
    #[test]
    fn cell_size_does_not_affect_existing_modes() {
        for mode in [RecolorMode::ColorBands, RecolorMode::BorderedCells] {
            let mut small = colorful_image(); let mut large = small.clone();
            preprocess_source_frame_with_mode_and_cell_size(&mut small, b"seed", 0.08, mode, 4.0).unwrap();
            preprocess_source_frame_with_mode_and_cell_size(&mut large, b"seed", 0.08, mode, 512.0).unwrap();
            assert_eq!(small, large);
        }
    }

    #[test]
    fn global_segmentation_modes_are_seeded_and_deterministic() {
        for mode in [RecolorMode::ConnectedComponents, RecolorMode::SlicSuperpixels] {
            let mut first = colorful_image(); let mut repeat = first.clone(); let mut other = first.clone();
            preprocess_source_frame_with_mode_and_cell_size(&mut first, b"seed-a", 0.0, mode, 24.0).unwrap();
            preprocess_source_frame_with_mode_and_cell_size(&mut repeat, b"seed-a", 0.0, mode, 24.0).unwrap();
            preprocess_source_frame_with_mode_and_cell_size(&mut other, b"seed-b", 0.0, mode, 24.0).unwrap();
            assert_eq!(first, repeat);
            assert_ne!(first, other);
        }
    }
}
