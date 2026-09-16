//! Backend-independent image enhancement stages.
//!
//! Configuration is resolved to function pointers once in `EnhancementPipeline::new`.
//! The hot pixel loops therefore contain no branches on user-selected modes.

use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum ReconstructionFilter {
    Nearest,
    #[default]
    Bilinear,
    Bicubic,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum EdgeFilter {
    #[default]
    Disabled,
    Fxaa,
    Smaa,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnhancementConfig {
    pub reconstruction: ReconstructionFilter,
    pub explicit_derivatives: bool,
    pub anisotropy: u8,
    pub edge_filter: EdgeFilter,
    pub taa_enabled: bool,
    pub taa_feedback: f32,
}

impl Default for EnhancementConfig {
    fn default() -> Self {
        Self {
            reconstruction: ReconstructionFilter::Bilinear,
            explicit_derivatives: true,
            anisotropy: 1,
            edge_filter: EdgeFilter::Disabled,
            taa_enabled: false,
            taa_feedback: 0.9,
        }
    }
}

impl EnhancementConfig {
    /// Converts the stable UI/wire representation into a validated backend plan.
    pub fn from_wire(reconstruction: &str, explicit_derivatives: bool, anisotropy: u8, edge_filter: &str, taa_enabled: bool, taa_feedback: f32) -> Self {
        Self {
            reconstruction: match reconstruction { "nearest" => ReconstructionFilter::Nearest, "bicubic" => ReconstructionFilter::Bicubic, _ => ReconstructionFilter::Bilinear },
            explicit_derivatives,
            anisotropy: match anisotropy { 2 | 4 | 8 | 16 => anisotropy, _ => 1 },
            edge_filter: match edge_filter { "fxaa" => EdgeFilter::Fxaa, "smaa" => EdgeFilter::Smaa, _ => EdgeFilter::Disabled },
            taa_enabled,
            taa_feedback: taa_feedback.clamp(0.0, 1.0),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct UvDerivatives {
    pub dx: [f32; 2],
    pub dy: [f32; 2],
}

/// Immutable CPU texture pyramid shared by scalar and SIMD samplers. Each
/// level is generated from the complete preceding image, so filtering remains
/// continuous across regions that the GPU stores in separate array layers.
pub struct CpuMipPyramid {
    levels: Vec<DynamicImage>,
}

impl CpuMipPyramid {
    pub fn new(source: &DynamicImage) -> Self {
        let mut levels = vec![source.clone()];
        let (mut width, mut height) = source.dimensions();
        while width > 1 || height > 1 {
            width = (width / 2).max(1);
            height = (height / 2).max(1);
            let next = image::imageops::resize(
                &levels.last().expect("base mip exists").to_rgba8(),
                width,
                height,
                image::imageops::FilterType::Triangle,
            );
            levels.push(DynamicImage::ImageRgba8(next));
        }
        Self { levels }
    }

    pub fn level_count(&self) -> usize { self.levels.len() }

    pub fn dimensions(&self) -> (u32, u32) { self.levels[0].dimensions() }

    #[inline]
    pub fn sample(&self, pipeline: &EnhancementPipeline, uv: [f32; 2], derivatives: UvDerivatives) -> [u8; 4] {
        let dx_len = derivatives.dx[0].hypot(derivatives.dx[1]);
        let dy_len = derivatives.dy[0].hypot(derivatives.dy[1]);
        let enabled = pipeline.derivatives as u8 as f32;
        let anisotropic = (pipeline.anisotropy > 1) as u8 as f32;
        let footprint = dx_len.max(dy_len) + (dx_len.min(dy_len) - dx_len.max(dy_len)) * anisotropic;
        let lod = (footprint.max(1.0).log2() * enabled)
            .floor().clamp(0.0, self.levels.len().saturating_sub(1) as f32) as usize;
        let scale = (1u32 << lod) as f32;
        let choose_dx = (dx_len >= dy_len) as u8 as f32;
        let major = [
            derivatives.dy[0] + (derivatives.dx[0] - derivatives.dy[0]) * choose_dx,
            derivatives.dy[1] + (derivatives.dx[1] - derivatives.dy[1]) * choose_dx,
        ];
        let taps = pipeline.anisotropy as usize;
        let mut sum = [0.0f32; 4];
        for tap in 0..taps {
            let offset = (tap as f32 + 0.5) / taps as f32 - 0.5;
            let sample = (pipeline.sample)(
                &self.levels[lod],
                (uv[0] + major[0] * offset) / scale,
                (uv[1] + major[1] * offset) / scale,
            );
            for channel in 0..4 { sum[channel] += sample[channel]; }
        }
        let factor = 255.0 / taps as f32;
        sum.map(|value| (value * factor).round().clamp(0.0, 255.0) as u8)
    }
}

type SampleFn = fn(&DynamicImage, f32, f32) -> [f32; 4];
type PostFn = fn(&RgbaImage) -> RgbaImage;

/// A fully resolved pipeline. Mode decisions happen during construction rather
/// than in sampling/post-processing loops.
pub struct EnhancementPipeline {
    sample: SampleFn,
    post: PostFn,
    derivatives: bool,
    anisotropy: u8,
    taa_feedback: Option<f32>,
    history: Option<RgbaImage>,
}

impl EnhancementPipeline {
    pub fn new(config: EnhancementConfig) -> Self {
        let sample = match config.reconstruction {
            ReconstructionFilter::Nearest => sample_nearest as SampleFn,
            ReconstructionFilter::Bilinear => sample_bilinear as SampleFn,
            ReconstructionFilter::Bicubic => sample_bicubic as SampleFn,
        };
        let post = match config.edge_filter {
            EdgeFilter::Disabled => clone_frame as PostFn,
            EdgeFilter::Fxaa => fxaa as PostFn,
            EdgeFilter::Smaa => smaa_1x as PostFn,
        };
        Self {
            sample,
            post,
            derivatives: config.explicit_derivatives,
            anisotropy: config.anisotropy.clamp(1, 16).next_power_of_two().min(16),
            taa_feedback: config.taa_enabled.then_some(config.taa_feedback.clamp(0.0, 1.0)),
            history: None,
        }
    }

    #[inline]
    pub fn sample(&self, source: &DynamicImage, uv: [f32; 2], derivatives: UvDerivatives) -> [u8; 4] {
        // Derivative footprint determines the major axis and tap spacing. Multipliers
        // turn disabled derivative handling into a zero footprint without a hot-loop branch.
        let enabled = self.derivatives as u8 as f32;
        let dx2 = derivatives.dx[0].mul_add(derivatives.dx[0], derivatives.dx[1] * derivatives.dx[1]);
        let dy2 = derivatives.dy[0].mul_add(derivatives.dy[0], derivatives.dy[1] * derivatives.dy[1]);
        let choose_dx = (dx2 >= dy2) as u8 as f32;
        let major = [
            (derivatives.dy[0] + (derivatives.dx[0] - derivatives.dy[0]) * choose_dx) * enabled,
            (derivatives.dy[1] + (derivatives.dx[1] - derivatives.dy[1]) * choose_dx) * enabled,
        ];
        let taps = self.anisotropy as usize;
        let mut sum = [0.0; 4];
        for tap in 0..taps {
            let offset = (tap as f32 + 0.5) / taps as f32 - 0.5;
            let p = (self.sample)(source, uv[0] + major[0] * offset, uv[1] + major[1] * offset);
            for channel in 0..4 { sum[channel] += p[channel]; }
        }
        let scale = 255.0 / taps as f32;
        sum.map(|v| (v * scale).round().clamp(0.0, 255.0) as u8)
    }

    /// Applies the stages after SSAA resolve: edge filtering, then temporal accumulation.
    pub fn finish_frame(&mut self, resolved: &RgbaImage) -> RgbaImage {
        let post = (self.post)(resolved);
        let Some(alpha) = self.taa_feedback else { return post };
        let output = match self.history.as_ref().filter(|h| h.dimensions() == post.dimensions()) {
            Some(history) => temporal_blend(&post, history, alpha),
            None => post.clone(),
        };
        self.history = Some(output.clone());
        output
    }

    pub fn reset_history(&mut self) { self.history = None; }
}

#[inline]
fn texel(source: &DynamicImage, x: i32, y: i32) -> [f32; 4] {
    let (w, h) = source.dimensions();
    let pixel = source.get_pixel(x.clamp(0, w.saturating_sub(1) as i32) as u32, y.clamp(0, h.saturating_sub(1) as i32) as u32).0;
    pixel.map(|v| v as f32 / 255.0)
}

fn sample_nearest(source: &DynamicImage, x: f32, y: f32) -> [f32; 4] { texel(source, x.round() as i32, y.round() as i32) }

fn sample_bilinear(source: &DynamicImage, x: f32, y: f32) -> [f32; 4] {
    let x0 = x.floor() as i32; let y0 = y.floor() as i32;
    let fx = x - x.floor(); let fy = y - y.floor();
    let p = [texel(source, x0, y0), texel(source, x0 + 1, y0), texel(source, x0, y0 + 1), texel(source, x0 + 1, y0 + 1)];
    let mut out = [0.0; 4];
    for c in 0..4 {
        let a = p[0][c] + (p[1][c] - p[0][c]) * fx;
        let b = p[2][c] + (p[3][c] - p[2][c]) * fx;
        out[c] = a + (b - a) * fy;
    }
    out
}

#[inline]
fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    0.5 * ((2.0 * p1) + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t)
}

fn sample_bicubic(source: &DynamicImage, x: f32, y: f32) -> [f32; 4] {
    let bx = x.floor() as i32; let by = y.floor() as i32;
    let fx = x - x.floor(); let fy = y - y.floor();
    let mut rows = [[0.0; 4]; 4];
    for row in 0..4 {
        let p = [texel(source, bx - 1, by + row as i32 - 1), texel(source, bx, by + row as i32 - 1), texel(source, bx + 1, by + row as i32 - 1), texel(source, bx + 2, by + row as i32 - 1)];
        for c in 0..4 { rows[row][c] = catmull_rom(p[0][c], p[1][c], p[2][c], p[3][c], fx); }
    }
    let mut out = [0.0; 4];
    for c in 0..4 { out[c] = catmull_rom(rows[0][c], rows[1][c], rows[2][c], rows[3][c], fy).clamp(0.0, 1.0); }
    out
}

fn clone_frame(frame: &RgbaImage) -> RgbaImage { frame.clone() }

#[inline] fn luma(p: &Rgba<u8>) -> f32 { 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32 }

fn fxaa(frame: &RgbaImage) -> RgbaImage {
    let (w, h) = frame.dimensions(); let mut out = frame.clone();
    if w < 3 || h < 3 { return out; }
    for y in 1..h - 1 { for x in 1..w - 1 {
        let c = frame.get_pixel(x, y); let lc = luma(c);
        let n = frame.get_pixel(x, y - 1); let s = frame.get_pixel(x, y + 1); let e = frame.get_pixel(x + 1, y); let west = frame.get_pixel(x - 1, y);
        let range = lc.max(luma(n)).max(luma(s)).max(luma(e)).max(luma(west)) - lc.min(luma(n)).min(luma(s)).min(luma(e)).min(luma(west));
        let blend = (range >= (lc * 0.125).max(3.0)) as u8 as u16;
        let mut p = *c;
        for channel in 0..3 { let avg = (n[channel] as u16 + s[channel] as u16 + e[channel] as u16 + west[channel] as u16) / 4; p[channel] = (c[channel] as u16 * (1 - blend) + avg * blend) as u8; }
        out.put_pixel(x, y, p);
    }} out
}

// Compact SMAA-1x style three-stage approximation: edge detection and blend-weight
// calculation are fused, while the final neighborhood blend remains a separate write.
fn smaa_1x(frame: &RgbaImage) -> RgbaImage {
    let (w, h) = frame.dimensions(); let mut out = frame.clone();
    if w < 3 || h < 3 { return out; }
    for y in 1..h - 1 { for x in 1..w - 1 {
        let c = frame.get_pixel(x, y); let right = frame.get_pixel(x + 1, y); let down = frame.get_pixel(x, y + 1);
        let horizontal = (luma(c) - luma(down)).abs(); let vertical = (luma(c) - luma(right)).abs();
        let use_horizontal = (horizontal >= vertical) as u8 as u16; let edge = (horizontal.max(vertical) > 12.0) as u8 as u16;
        let a = frame.get_pixel(x - use_horizontal as u32, y - (1 - use_horizontal) as u32); let b = frame.get_pixel(x + use_horizontal as u32, y + (1 - use_horizontal) as u32);
        let mut p = *c; for channel in 0..3 { let avg = (a[channel] as u16 + b[channel] as u16) / 2; p[channel] = (c[channel] as u16 * (1 - edge) + avg * edge) as u8; } out.put_pixel(x, y, p);
    }} out
}

fn temporal_blend(current: &RgbaImage, history: &RgbaImage, alpha: f32) -> RgbaImage {
    let mut out = current.clone();
    for (dst, (now, old)) in out.pixels_mut().zip(current.pixels().zip(history.pixels())) {
        for c in 0..4 { dst[c] = (now[c] as f32 * (1.0 - alpha) + old[c] as f32 * alpha).round() as u8; }
    } out
}

pub fn rotate_hue_rgba([r, g, b, a]: [u8; 4], degrees: f32) -> [u8; 4] {
    if degrees == 0.0 { return [r, g, b, a]; }
    let (h, s, v) = rgb_to_hsv(r, g, b);
    let (r, g, b) = hsv_to_rgb((h + degrees).rem_euclid(360.0), s, v);
    [r, g, b, a]
}

fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let (r, g, b) = (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
    let max = r.max(g).max(b); let min = r.min(g).min(b); let d = max - min;
    let h = if d == 0.0 { 0.0 } else if max == r { 60.0 * ((g - b) / d).rem_euclid(6.0) } else if max == g { 60.0 * ((b - r) / d + 2.0) } else { 60.0 * ((r - g) / d + 4.0) };
    (h, if max == 0.0 { 0.0 } else { d / max }, max)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s; let x = c * (1.0 - ((h / 60.0).rem_euclid(2.0) - 1.0).abs()); let m = v - c;
    let (r, g, b) = match (h / 60.0).floor() as i32 { 0 => (c,x,0.0), 1 => (x,c,0.0), 2 => (0.0,c,x), 3 => (0.0,x,c), 4 => (x,0.0,c), _ => (c,0.0,x) };
    (((r+m)*255.0).round() as u8, ((g+m)*255.0).round() as u8, ((b+m)*255.0).round() as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn defaults_match_ui_contract() { let c = EnhancementConfig::default(); assert_eq!(c.reconstruction, ReconstructionFilter::Bilinear); assert!(c.explicit_derivatives); assert_eq!(c.taa_feedback, 0.9); }
    #[test] fn taa_uses_post_processed_history() { let mut p = EnhancementPipeline::new(EnhancementConfig { taa_enabled: true, taa_feedback: 0.5, ..Default::default() }); let a = RgbaImage::from_pixel(2, 2, Rgba([0, 0, 0, 255])); let b = RgbaImage::from_pixel(2, 2, Rgba([100, 100, 100, 255])); p.finish_frame(&a); assert_eq!(p.finish_frame(&b).get_pixel(0, 0)[0], 50); }
    #[test] fn bicubic_preserves_constant_colors() { let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(6, 6, Rgba([25, 50, 75, 200]))); let sample = sample_bicubic(&image, 2.37, 3.61); for (actual, expected) in sample.into_iter().zip([25.0/255.0, 50.0/255.0, 75.0/255.0, 200.0/255.0]) { assert!((actual - expected).abs() < 1e-5); } }
    #[test] fn cpu_mips_reach_one_texel_and_preserve_constants() { let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(9, 5, Rgba([12, 34, 56, 255]))); let pyramid = CpuMipPyramid::new(&image); assert_eq!(pyramid.level_count(), 4); let pipeline = EnhancementPipeline::new(EnhancementConfig { anisotropy: 8, ..Default::default() }); assert_eq!(pyramid.sample(&pipeline, [4.0, 2.0], UvDerivatives { dx: [8.0, 0.0], dy: [0.0, 8.0] }), [12, 34, 56, 255]); }
}
