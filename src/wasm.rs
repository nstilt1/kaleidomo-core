use std::cell::RefCell;
use std::rc::Rc;
use std::f32::consts::PI;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{HtmlCanvasElement, window};

use crate::backends::gpu::{GpuBackend, GpuKaleidoSettings};
use crate::{KaleidoSettings, KaleidoType};

use wgpu::rwh::{DisplayHandle, HasDisplayHandle};

#[derive(Debug)]
struct WebDisplayHandle;

impl HasDisplayHandle for WebDisplayHandle {
    fn display_handle(
        &self,
    ) -> Result<DisplayHandle<'_>, wgpu::rwh::HandleError> {
        Ok(DisplayHandle::web())
    }
}

// ---------------------------------------------------------------------------
// VideoSettings — mirrors the native struct but is wasm_bindgen-compatible
// ---------------------------------------------------------------------------

/// Animation parameters passed from TypeScript. All fields are public so
/// they can be set directly from JS via a plain object converted with
/// `serde-wasm-bindgen`, or built field-by-field.
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmVideoSettings {
    /// Total loop duration in seconds (e.g. 10.0)
    pub animation_duration: f32,
    /// How many radians to sweep across the rotation range per loop
    pub rotation_range: f32,
    pub rotation_cycles: f32,
    pub rotation_start_offset: f32,
    /// "linear" | "triangle" | "sawtooth" | "sin" | "sin2" | "cos" | "-cos"
    rotation_fn: String,
    /// Hue sweep range in degrees (e.g. 60)
    pub hue_range: i32,
    pub hue_cycles: f32,
    pub hue_start_offset: f32,
    hue_fn: String,
    pub fps: u32,
    pub zoom_max: f32,
    pub zoom_min: f32,
    zoom_fn: String,
    pub zoom_start_offset: f32,
    pub num_zoom_loops: u32,
    pub orientation_range: f32,
    pub orientation_cycles: f32,
    pub orientation_start_offset: f32,
    orientation_fn: String,
    pub orientation_duration: f32,
    // Audio-reactive fields
    pub audio_reactive_enabled: bool,
    pub audio_orientation_amount: f32,
    pub audio_reorientation_amount: f32,
    pub audio_peak_smoothing: f32,
    // Hero circle parameters — define the orbit the triangle center follows
    pub hero_circle_left_x: f32,
    pub hero_circle_right_x: f32,
    pub hero_circle_y: f32,
    pub hero_desired_left_rotation: f32,
    /// Independent orientation cycles per second (base reorientation speed, no audio)
    pub orientation_base_speed: f32,
    /// Multiplier applied to the smoothed audio peak for orientation + rotation kick
    pub orientation_peak_multiplier: f32,
}

#[wasm_bindgen]
impl WasmVideoSettings {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmVideoSettings {
        WasmVideoSettings {
            animation_duration: 10.0,
            rotation_range: 0.3,
            rotation_cycles: 1.0,
            rotation_start_offset: 0.0,
            rotation_fn: "sin".to_string(),
            hue_range: 60,
            hue_cycles: 1.0,
            hue_start_offset: 0.0,
            hue_fn: "sin".to_string(),
            fps: 25,
            zoom_max: 1.2,
            zoom_min: 0.8,
            zoom_fn: "sin".to_string(),
            zoom_start_offset: 0.0,
            num_zoom_loops: 1,
            orientation_range: 0.25,
            orientation_cycles: 1.0,
            orientation_start_offset: 0.0,
            orientation_fn: "none".to_string(),
            orientation_duration: 201.0,
            audio_reactive_enabled: false,
            audio_orientation_amount: 0.0,
            audio_reorientation_amount: 0.0,
            audio_peak_smoothing: 0.65,
            hero_circle_left_x: 515.1039592844847,
            hero_circle_right_x: 1547.0,
            hero_circle_y: 755.3734001945962,
            hero_desired_left_rotation: 6.22,
            orientation_base_speed: 0.0,
            orientation_peak_multiplier: 0.0,
        }
    }

    // String setters (wasm_bindgen can't expose String fields directly)
    pub fn set_rotation_fn(&mut self, f: String) { self.rotation_fn = f; }
    pub fn set_hue_fn(&mut self, f: String)      { self.hue_fn = f; }
    pub fn set_zoom_fn(&mut self, f: String)     { self.zoom_fn = f; }

    pub fn get_rotation_fn(&self) -> String { self.rotation_fn.clone() }
    pub fn get_hue_fn(&self)      -> String { self.hue_fn.clone() }
    pub fn get_zoom_fn(&self)     -> String { self.zoom_fn.clone() }

    // Orientation modulation
    pub fn set_orientation_fn(&mut self, f: String) { self.orientation_fn = f; }
    pub fn get_orientation_fn(&self) -> String { self.orientation_fn.clone() }
}

// ---------------------------------------------------------------------------
// Modulation helpers (identical logic to lib.rs / rlib.rs)
// ---------------------------------------------------------------------------

mod modulation {
    use core::f32::consts::PI;
    use super::WasmVideoSettings;
    pub fn modulate(
        settings: &WasmVideoSettings,
        frame: u32,
        range_max: f32,
        range_min: f32,
        num_loops: f32,
        start_offset: f32,
        function: &str,
    ) -> f32 {
        let range = range_max - range_min;
        let elapsed_seconds = frame as f32 / settings.fps.max(1) as f32;
        let phase_base = elapsed_seconds / settings.animation_duration.max(0.001);

        match function.to_ascii_lowercase().as_str() {
            "triangle" => {
                let phase = phase_base * num_loops + start_offset;
                let phase = phase.rem_euclid(1.0);
                let tri = 1.0 - (2.0 * phase - 1.0).abs();
                range_min + tri * range
            }
            "sawtooth" | "linear" | "saw" => {
                let phase = phase_base * num_loops + start_offset;
                range_min + phase.rem_euclid(1.0) * range
            }
            "sin" => {
                let phase = phase_base * num_loops + start_offset;
                let sin_norm = (f32::sin(phase * 2.0 * PI) + 1.0) * 0.5;
                range_min + sin_norm * range
            }
            "sin2" => {
                let phase = phase_base * num_loops + start_offset;
                let sin2_norm = f32::sin(phase * 2.0 * PI).powi(2);
                range_min + sin2_norm * range
            }
            "cos" => {
                let phase = phase_base * num_loops + start_offset;
                let cos_norm = (f32::cos(phase * 2.0 * PI) + 1.0) * 0.5;
                range_min + cos_norm * range
            }
            "-cos" => {
                let phase = phase_base * num_loops + start_offset;
                let neg_cos_norm = (1.0 - f32::cos(phase * 2.0 * PI)) * 0.5;
                range_min + neg_cos_norm * range
            }
            _ => range_min,
        }
    }

    pub fn modulate_by_time(
        elapsed_seconds: f32,
        range: f32,
        min_value: f32,
        cycles_per_second: f32,
        start_offset: f32,
        fn_name: &str,
    ) -> f32 {
        let phase = elapsed_seconds * cycles_per_second + start_offset;
        let p = phase.rem_euclid(1.0);

        let t = match fn_name {
            "linear" | "saw" | "sawtooth" => p,
            "triangle" => {
                if p < 0.5 {
                    p * 2.0
                } else {
                    2.0 - p * 2.0
                }
            }
            "sin" => (phase * std::f32::consts::TAU).sin() * 0.5 + 0.5,
            "sin2" => (phase * std::f32::consts::TAU).sin().powi(2),
            "-cos" | "negcos" => 0.5 - 0.5 * (phase * std::f32::consts::TAU).cos(),
            _ => p,
        };

        min_value + range * t
    }

    pub fn orientation_to_hero_params(value: f32) -> (f32, f32, f32) {
        let left_x = 515.1039592844847_f32;
        let right_x = 1547.0_f32;
        let center_y = 755.3734001945962_f32;

        let center_x = (left_x + right_x) * 0.5;
        let radius = (right_x - left_x) * 0.5;

        let circle_angle = PI + value * PI * 2.0;

        let triangle_center_x = center_x + circle_angle.cos() * radius;
        let triangle_center_y = center_y + circle_angle.sin() * radius;

        let desired_left_rotation = 6.22_f32;
        let triangle_rotation_rad = desired_left_rotation + (circle_angle - PI);

        (triangle_center_x, triangle_center_y, triangle_rotation_rad)
    }

    pub fn orientation_to_hero_params_with_circle(
        value: f32,
        left_x: f32,
        right_x: f32,
        center_y: f32,
        desired_left_rotation: f32,
    ) -> (f32, f32, f32) {
        let center_x = (left_x + right_x) * 0.5;
        let radius = (right_x - left_x) * 0.5;

        let circle_angle = PI + value * PI * 2.0;

        let triangle_center_x = center_x + circle_angle.cos() * radius;
        let triangle_center_y = center_y + circle_angle.sin() * radius;

        let triangle_rotation_rad = desired_left_rotation + (circle_angle - PI);

        (triangle_center_x, triangle_center_y, triangle_rotation_rad)
    }

    pub fn modulate_orientation(vs: &WasmVideoSettings, elapsed_seconds: f32) -> f32 {
        if vs.orientation_duration <= 0.0 {
            return 0.0;
        }

        modulate_by_time(
            elapsed_seconds,
            vs.orientation_range,
            0.0,
            1.0 / vs.orientation_duration,
            vs.orientation_start_offset,
            &vs.orientation_fn,
        )
    }

    #[inline]
    fn degrees_to_radians(degrees: f32) -> f32 {
        degrees * (PI / 180.0)
    }

    pub fn modulate_zoom(vs: &WasmVideoSettings, frame: u32) -> f32 {
        modulate(
            vs,
            frame,
            vs.zoom_max,
            vs.zoom_min,
            vs.num_zoom_loops as f32,
            vs.zoom_start_offset,
            &vs.zoom_fn.clone(),
        )
    }

    pub fn modulate_rotation(vs: &WasmVideoSettings, frame: u32, base_rotation: f32) -> f32 {
        modulate(
            vs,
            frame,
            base_rotation + degrees_to_radians(vs.rotation_range),
            base_rotation,
            vs.rotation_cycles,
            vs.rotation_start_offset,
            &vs.rotation_fn.clone(),
        )
        .rem_euclid(2.0 * PI)
    }

    pub fn modulate_zoom_time(vs: &WasmVideoSettings, elapsed_seconds: f32) -> f32 {
        modulate_by_time(
            elapsed_seconds,
            vs.zoom_max - vs.zoom_min,
            vs.zoom_min,
            vs.num_zoom_loops as f32 / vs.animation_duration.max(0.001),
            vs.zoom_start_offset,
            &vs.zoom_fn,
        )
    }

    pub fn modulate_rotation_time(
        vs: &WasmVideoSettings,
        elapsed_seconds: f32,
        base_rotation: f32,
    ) -> f32 {
        modulate_by_time(
            elapsed_seconds,
            degrees_to_radians(vs.rotation_range),
            base_rotation,
            vs.rotation_cycles / vs.animation_duration.max(0.001),
            vs.rotation_start_offset,
            &vs.rotation_fn,
        )
    }

    pub fn modulate_hue(vs: &WasmVideoSettings, frame: u32, base_hue: f32) -> u32 {
        modulate(
            vs,
            frame,
            base_hue + vs.hue_range as f32,
            base_hue,
            vs.hue_cycles,
            vs.hue_start_offset,
            &vs.hue_fn.clone(),
        )
        .round()
        .rem_euclid(360.0) as u32
    }
}

// ---------------------------------------------------------------------------
// KaleidoType index helper
// ---------------------------------------------------------------------------

fn kaleido_type_from_idx(idx: u32) -> KaleidoType {
    match idx {
        0 => KaleidoType::Radial,
        1 => KaleidoType::Square,
        2 => KaleidoType::Diamond,
        3 => KaleidoType::Hexagonal,
        _ => KaleidoType::HexagonalFlatTop,
    }
}

// ---------------------------------------------------------------------------
// Engine state shared across the rAF closure
// ---------------------------------------------------------------------------

struct EngineState {
    backend: GpuBackend,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    settings_buffer: wgpu::Buffer,
    base_settings: KaleidoSettings,   // frozen at start_animation(); only per-frame fields change
    video_settings: WasmVideoSettings,
    frame_index: u32,
    canvas_width: u32,
    canvas_height: u32,
    started_at_ms: f64,
    last_render_ms: f64,
    audio_peaks: Vec<f32>,
    smoothed_audio_peak: f32,
    /// Permanently accumulated orientation offset — only increases, never decreases.
    /// Each frame adds (peak_rise * multiplier), so beats ratchet the orientation forward.
    accumulated_orientation_offset: f32,
    prev_smoothed_peak: f32,
}

// ---------------------------------------------------------------------------
// LiveKaleidoscopeEngine — the public WASM type
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct LiveKaleidoscopeEngine {
    state: Rc<RefCell<Option<EngineState>>>,
    // keep the rAF handle so we can cancel it
    raf_handle: Rc<RefCell<Option<i32>>>,
}

#[wasm_bindgen]
impl LiveKaleidoscopeEngine {
    /// Creates a persistent GPU rendering context bound directly to a browser canvas element.
    #[wasm_bindgen(constructor)]
    pub async fn new(canvas: HtmlCanvasElement) -> Result<LiveKaleidoscopeEngine, JsValue> {
        // One-time setup: route Rust panics → browser console, and wire up log::*
        console_error_panic_hook::set_once();
        wasm_logger::init(wasm_logger::Config::default());

        let width  = canvas.width();
        let height = canvas.height();

        // wgpu 29: canvas surfaces use SurfaceTarget::Canvas.
        //
        // IMPORTANT:
        // Kaleidomo's live GPU renderer uses compute + storage textures.
        // WebGL/GL cannot support that path, so GL fallback only produces a later
        // max_storage_textures_per_shader_stage = 0 failure.
        //
        // Default Wasm builds are WebGPU-only. If you intentionally need WebGL2/GLES
        // compiled for some other website path, build kaleidomo-core with:
        //     --features wasm_webgl_fallback
        // The storage-texture gate below will still reject GL for this renderer.
        #[cfg(target_arch = "wasm32")]
        let wasm_backends = wgpu::Backends::BROWSER_WEBGPU;

        #[cfg(not(target_arch = "wasm32"))]
        let wasm_backends = wgpu::Backends::all();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wasm_backends,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),

            display: Some(Box::new(WebDisplayHandle)),
        });

        #[cfg(target_arch = "wasm32")]
        {
            let webgpu_context = canvas
                .get_context("webgpu")
                .map_err(|e| {
                    JsValue::from_str(&format!(
                        "WebGPU canvas preflight failed while calling canvas.getContext('webgpu'): {e:?}"
                    ))
                })?;

            if webgpu_context.is_none() {
                return Err(JsValue::from_str(
                    "WebGPU canvas preflight failed: canvas.getContext('webgpu') returned null. \
                     WebGPU is unavailable in this webview, or this canvas was already used \
                     with another context type such as '2d', 'webgl', or 'webgl2'. \
                     On macOS Tauri/WKWebView, use the native Metal/Tauri GPU path when \
                     navigator.gpu or canvas WebGPU context creation is unavailable."
                ));
            }
        }

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| JsValue::from_str(&format!(
                "Surface creation failed after WebGPU preflight succeeded: {e}. \
                 Make sure the same canvas is not being reused by multiple live engine instances."
            )))?;

        // wgpu 29 (PR #7330): request_adapter returns Result<Adapter> directly.
        // A single map_err + ? is sufficient; there is no Option to unwrap.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::None,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| JsValue::from_str(&format!("No suitable adapter: {e}")))?;

        let info = adapter.get_info();
        let adapter_limits = adapter.limits();

        // Don't check backend variant by name — wgpu 29 renamed variants
        // (e.g. Gl → Gi) and the Display strings are unreliable across versions.
        // Instead check the one capability we actually need: storage textures.
        // Real WebGL/WebGL2 adapters report 0 here; WebGPU (Metal/Vulkan/DX12/Gi)
        // report ≥ 4. This is the only gate that matters.
        if adapter_limits.max_storage_textures_per_shader_stage == 0 {
            return Err(format!(
                "GPU kaleidoscope requires WebGPU with storage texture support. \
                 Detected backend: {:?}, adapter: {}. \
                 max_storage_textures_per_shader_stage = 0. \
                 WebGPU is unavailable or this build/runtime fell back to GL. \
                 On macOS Tauri, use the native Metal/Tauri GPU path or require \
                 a WebGPU-capable WKWebView.",
                info.backend, info.name,
            ).into());
        }

        // Build device limits: start from conservative WebGL2 defaults to avoid
        // requesting limits the adapter can't satisfy, but restore storage textures
        // since that's the one WebGPU-only capability the compute shader requires.
        let device_limits = wgpu::Limits {
            max_storage_textures_per_shader_stage:
                adapter_limits.max_storage_textures_per_shader_stage,
            ..wgpu::Limits::downlevel_webgl2_defaults()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label:                 Some("kaleidomo.wasm.device"),
                    required_features:     wgpu::Features::empty(),
                    required_limits:       device_limits,
                    memory_hints:          wgpu::MemoryHints::Performance,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    trace:                 wgpu::Trace::Off,
                }
            )
            .await
            .map_err(|e| JsValue::from_str(&format!("Device creation failed: {e}")))?;

        let caps = surface.get_capabilities(&adapter);
        // Prefer Bgra8Unorm (most common on WebGPU) then fall back to whatever is first
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| *f == wgpu::TextureFormat::Bgra8Unorm
                   || *f == wgpu::TextureFormat::Rgba8Unorm)
            .unwrap_or(caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage:   wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let settings_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kaleidomo.wasm.uniform"),
            size: std::mem::size_of::<GpuKaleidoSettings>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let backend = GpuBackend::new_for_canvas(device, queue, format)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Placeholder base settings; will be replaced by start_animation()
        let base_settings = KaleidoSettings {
            count: 8,
            output_size_w: width,
            output_size_h: height,
            offset_x: 0,
            offset_y: 0,
            zoom: 1.0,
            tile_count: 4.0,
            triangle_center_x: 0.0,
            triangle_center_y: 0.0,
            triangle_rotation_rad: 0.0,
            kaleido_type: KaleidoType::Radial,
            hue_rotation: 0,
        };

        let state = EngineState {
            backend,
            surface,
            surface_config,
            settings_buffer,
            base_settings,
            video_settings: WasmVideoSettings::new(),
            frame_index: 0,
            canvas_width: width,
            canvas_height: height,
            started_at_ms: 0.0,
            last_render_ms: 0.0,
            audio_peaks: Vec::new(),
            smoothed_audio_peak: 0.0,
            accumulated_orientation_offset: 0.0,
            prev_smoothed_peak: 0.0,
        };

        Ok(Self {
            state: Rc::new(RefCell::new(Some(state))),
            raf_handle: Rc::new(RefCell::new(None)),
        })
    }

    // -----------------------------------------------------------------------
    // Image loading
    // -----------------------------------------------------------------------

    /// Load raw RGBA pixel bytes into the GPU source texture.
    /// Call this before `start_animation`.
    pub fn load_source_image(
        &mut self,
        rgba_bytes: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(), JsValue> {
        let img = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
            width,
            height,
            rgba_bytes.to_vec(),
        )
        .ok_or_else(|| JsValue::from_str("Invalid image buffer dimensions"))?;

        let dynamic = image::DynamicImage::ImageRgba8(img);

        self.state
            .borrow_mut()
            .as_mut()
            .ok_or_else(|| JsValue::from_str("Engine not initialized"))?
            .backend
            .set_source_image(&dynamic)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Convenience: fetch an image URL with `fetch`, decode it, upload to GPU.
    /// Returns a Promise<void> — await it in TypeScript before calling start_animation.
    pub async fn load_image_from_url(&mut self, url: String) -> Result<(), JsValue> {
        // fetch(url)
        let global = web_sys::window()
            .ok_or_else(|| JsValue::from_str("No global window"))?;
        let promise = global.fetch_with_str(&url);
        let resp_value = JsFuture::from(promise).await?;
        let resp: web_sys::Response = resp_value.dyn_into()?;

        let array_buffer = JsFuture::from(resp.array_buffer()?).await?;
        let bytes = js_sys::Uint8Array::new(&array_buffer).to_vec();

        // Decode with the `image` crate
        let img = image::load_from_memory(&bytes)
            .map_err(|e| JsValue::from_str(&format!("Image decode failed: {e}")))?;

        self.state
            .borrow_mut()
            .as_mut()
            .ok_or_else(|| JsValue::from_str("Engine not initialized"))?
            .backend
            .set_source_image(&img)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // -----------------------------------------------------------------------
    // Animation control
    // -----------------------------------------------------------------------

    /// Begin the rAF loop.
    ///
    /// * `base_settings_js` — a JS object matching `KaleidoSettings` (count, offset_x/y, zoom,
    ///   tile_count, triangle_center_x/y, triangle_rotation_rad, kaleido_type_idx, hue_rotation)
    /// * `video_settings` — a `WasmVideoSettings` instance
    pub fn start_animation(
        &mut self,
        count: u32,
        offset_x: i32,
        offset_y: i32,
        zoom: f32,
        tile_count: f32,
        triangle_center_x: f32,
        triangle_center_y: f32,
        triangle_rotation_rad: f32,
        kaleido_type_idx: u32,
        hue_rotation: u32,
        video_settings: &WasmVideoSettings,
    ) -> Result<(), JsValue> {
        // Stop any previous loop
        self.stop_animation();

        {
            let mut guard = self.state.borrow_mut();
            let state = guard
                .as_mut()
                .ok_or_else(|| JsValue::from_str("Engine not initialized"))?;

            state.base_settings = KaleidoSettings {
                count,
                output_size_w: state.canvas_width,
                output_size_h: state.canvas_height,
                offset_x,
                offset_y,
                zoom,
                tile_count,
                triangle_center_x,
                triangle_center_y,
                triangle_rotation_rad,
                kaleido_type: kaleido_type_from_idx(kaleido_type_idx),
                hue_rotation,
            };
            state.video_settings = video_settings.clone();
            state.frame_index = 0;
            state.started_at_ms = 0.0;
            state.last_render_ms = 0.0;
        }

        // Schedule the rAF loop
        self.schedule_next_frame()?;
        Ok(())
    }

    /// Set normalized audio peaks (one f32 per video frame, values 0.0–1.0).
    /// Call this after decoding and normalizing audio in TypeScript.
    pub fn set_audio_peaks(&self, peaks: js_sys::Float32Array) {
        if let Some(state) = self.state.borrow_mut().as_mut() {
            state.audio_peaks = peaks.to_vec();
            state.smoothed_audio_peak = 0.0;
        }
    }

    /// Clear the audio peak buffer and reset smoothing state.
    pub fn clear_audio_peaks(&self) {
        if let Some(state) = self.state.borrow_mut().as_mut() {
            state.audio_peaks.clear();
            state.smoothed_audio_peak = 0.0;
            state.accumulated_orientation_offset = 0.0;
            state.prev_smoothed_peak = 0.0;
        }
    }

    /// Cancel the animation loop (idempotent).
    pub fn stop_animation(&mut self) {
        if let Some(handle) = self.raf_handle.borrow_mut().take() {
            if let Some(win) = web_sys::window() {
                win.cancel_animation_frame(handle).ok();
            }
        }
    }

    // -----------------------------------------------------------------------
    // Internal rAF machinery
    // -----------------------------------------------------------------------

    fn schedule_next_frame(&mut self) -> Result<(), JsValue> {
        let state_rc = Rc::clone(&self.state);
        let handle_rc = Rc::clone(&self.raf_handle);

        let closure = Closure::once(move |now_ms: f64| {
            if let Err(e) = render_one_frame(&state_rc, now_ms) {
                log::error!("{}", e.as_string().unwrap_or_else(|| "unknown wasm render error".to_string()));
                return;
            }

            let state_rc2 = Rc::clone(&state_rc);
            let handle_rc2 = Rc::clone(&handle_rc);

            let inner = Closure::once(move |next_now_ms: f64| {
                trampoline(state_rc2, handle_rc2, next_now_ms);
            });

            if let Some(win) = web_sys::window() {
                if let Ok(id) = win.request_animation_frame(inner.as_ref().unchecked_ref()) {
                    *handle_rc.borrow_mut() = Some(id);
                }
            }

            inner.forget();
        });

        let win = window().ok_or_else(|| JsValue::from_str("No window"))?;
        let id = win.request_animation_frame(closure.as_ref().unchecked_ref())?;
        *self.raf_handle.borrow_mut() = Some(id);
        closure.forget();

        Ok(())
    }

    #[cfg(feature = "dev")]
    pub fn render_frame(
        &mut self,
        count: u32,
        offset_x: i32,
        offset_y: i32,
        zoom: f32,
        tile_count: f32,
        triangle_center_x: f32,
        triangle_center_y: f32,
        triangle_rotation_rad: f32,
        kaleido_type_idx: u32,
        hue_rotation: u32,
        video_settings: &WasmVideoSettings,
        frame: u32,
    ) -> Result<(), JsValue> {
        {
            let mut guard = self.state.borrow_mut();
            let state = guard
                .as_mut()
                .ok_or_else(|| JsValue::from_str("Engine not initialized"))?;

            state.base_settings = KaleidoSettings {
                count,
                output_size_w: state.canvas_width,
                output_size_h: state.canvas_height,
                offset_x,
                offset_y,
                zoom,
                tile_count,
                triangle_center_x,
                triangle_center_y,
                triangle_rotation_rad,
                kaleido_type: kaleido_type_from_idx(kaleido_type_idx),
                hue_rotation,
            };

            state.video_settings = video_settings.clone();
            state.frame_index = frame;
        }

        render_one_frame(&self.state, 0.0)
    }

    pub fn update_animation_settings(
        &mut self,
        count: u32,
        offset_x: i32,
        offset_y: i32,
        zoom: f32,
        tile_count: f32,
        triangle_center_x: f32,
        triangle_center_y: f32,
        triangle_rotation_rad: f32,
        kaleido_type_idx: u32,
        hue_rotation: u32,
        video_settings: &WasmVideoSettings,
    ) -> Result<(), JsValue> {
        let mut guard = self.state.borrow_mut();
        let state = guard
            .as_mut()
            .ok_or_else(|| JsValue::from_str("Engine not initialized"))?;

        state.base_settings = KaleidoSettings {
            count,
            output_size_w: state.canvas_width,
            output_size_h: state.canvas_height,
            offset_x,
            offset_y,
            zoom,
            tile_count,
            triangle_center_x,
            triangle_center_y,
            triangle_rotation_rad,
            kaleido_type: kaleido_type_from_idx(kaleido_type_idx),
            hue_rotation,
        };

        state.video_settings = video_settings.clone();

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Free function: render one frame into the swap-chain
// ---------------------------------------------------------------------------

fn render_one_frame(
    state_rc: &Rc<RefCell<Option<EngineState>>>,
    now_ms: f64,
) -> Result<(), JsValue> {
    let mut guard = state_rc.borrow_mut();
    let state = match guard.as_mut() {
        Some(s) => s,
        None => return Ok(()),
    };

    if state.started_at_ms <= 0.0 {
        state.started_at_ms = now_ms;
        state.last_render_ms = 0.0;
    }

    let fps = state.video_settings.fps.max(1) as f64;
    let target_frame_ms = 1000.0 / fps;

    if state.last_render_ms > 0.0 && now_ms - state.last_render_ms < target_frame_ms {
        return Ok(());
    }

    state.last_render_ms = now_ms;

    let elapsed_ms = (now_ms - state.started_at_ms).max(0.0);

    // Build per-frame KaleidoSettings by modulating the base
    let base = &state.base_settings;

    use modulation::*;
    let elapsed_seconds_f32 = (elapsed_ms / 1000.0) as f32;
    let frame = ((elapsed_seconds_f32 as f64 * fps).floor() as u32); // still needed for hue if you keep it frame-based

    // --- Audio-reactive peak modulation ---
    // We accumulate orientation permanently: only the *rising edge* of the peak
    // advances the offset. When the peak falls, the offset stays where it is.
    // This means each beat irreversibly advances orientation/rotation forward.
    let audio_peak = {
        let vs = &state.video_settings;
        if vs.audio_reactive_enabled && !state.audio_peaks.is_empty() {
            let peak_fps = vs.fps.max(1) as f32;
            let frame_index = (elapsed_seconds_f32 * peak_fps).floor() as usize;
            let raw_peak = state.audio_peaks
                .get(frame_index % state.audio_peaks.len())
                .copied()
                .unwrap_or(0.0);
            let smoothing = vs.audio_peak_smoothing.clamp(0.0, 0.999);
            let smoothed = state.smoothed_audio_peak * smoothing + raw_peak * (1.0 - smoothing);
            state.smoothed_audio_peak = smoothed;
            smoothed.clamp(0.0, 1.0)
        } else {
            // When disabled, drain the smoothed peak but keep accumulated offset
            state.smoothed_audio_peak = 0.0;
            0.0
        }
    };

    // Cumulative accumulator: frame-rate independent.
    // orientationPeakMultiplier=1 means 1 full circle per second of peak=1 signal.
    // Dividing by fps makes it independent of render rate.
    // Wrap modulo 1.0 to keep the float small and prevent precision loss.
    {
        let multiplier = state.video_settings.orientation_peak_multiplier;
        let fps_norm = (state.video_settings.fps.max(1)) as f32;
        state.accumulated_orientation_offset =
            (state.accumulated_orientation_offset + audio_peak * multiplier / fps_norm) % 1.0;
    }

    let vs = &state.video_settings;

    // orientationBaseSpeed is in pixels/second of arc along the hero circle.
    // Convert to cycles/sec: cycles = px/s / (2π * radius)
    let hero_radius = ((vs.hero_circle_right_x - vs.hero_circle_left_x) * 0.5).max(1.0);
    let base_speed_cycles = vs.orientation_base_speed / (2.0 * std::f32::consts::PI * hero_radius);
    let rotation = modulate_rotation_time(vs, elapsed_seconds_f32, base.triangle_rotation_rad);
    let hue      = modulate_hue(vs, frame, base.hue_rotation as f32);
    let zoom     = modulate_zoom_time(vs, elapsed_seconds_f32);
    let orientation = if vs.orientation_duration <= 0.0 {
        0.0
    } else {
        modulate_by_time(
            elapsed_seconds_f32,
            vs.orientation_range,
            0.0,
            1.0 / vs.orientation_duration,
            vs.orientation_start_offset,
            &vs.orientation_fn,
        )
    };

    let accumulated = state.accumulated_orientation_offset;

    let final_orientation = orientation
        + elapsed_seconds_f32 * base_speed_cycles      // continuous base drift (px/s → cycles/s)
        + accumulated;                                  // permanent beat accumulator

    let (orientation_x, orientation_y, orientation_rotation) =
        modulation::orientation_to_hero_params_with_circle(
            final_orientation,
            vs.hero_circle_left_x,
            vs.hero_circle_right_x,
            vs.hero_circle_y,
            vs.hero_desired_left_rotation,
        );

    let frame_settings = KaleidoSettings {
        triangle_center_x: orientation_x,
        triangle_center_y: orientation_y,
        triangle_rotation_rad: (orientation_rotation + (rotation - base.triangle_rotation_rad))
            .rem_euclid(2.0 * PI),
        hue_rotation: hue,
        zoom,
        output_size_w: state.canvas_width,
        output_size_h: state.canvas_height,
        ..base.clone()
    };

    // Acquire the swap-chain texture.
    // wgpu 29: get_current_texture() returns CurrentSurfaceTexture, an enum — not Result.
    let surface_texture = match state.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(st)    => st,
        wgpu::CurrentSurfaceTexture::Suboptimal(st) => {
            // Still usable; reconfigure so next frame is optimal
            state.backend.configure_surface(&state.surface, &state.surface_config);
            st
        }
        wgpu::CurrentSurfaceTexture::Outdated => {
            state.backend.configure_surface(&state.surface, &state.surface_config);
            return Ok(()); // skip this frame; next rAF will get a fresh texture
        }
        wgpu::CurrentSurfaceTexture::Timeout
        | wgpu::CurrentSurfaceTexture::Occluded
        | wgpu::CurrentSurfaceTexture::Validation => {
            return Ok(()); // transient; skip frame
        }
        wgpu::CurrentSurfaceTexture::Lost => {
            return Err(JsValue::from_str("wgpu surface lost"));
        }
    };

    let view = surface_texture
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    // Dispatch compute shader + blit to swap-chain view
    state
        .backend
        .render_directly_to_view(
            &frame_settings,
            &view,
            &state.settings_buffer,
            state.surface_config.format,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    surface_texture.present();

    Ok(())
}

// ---------------------------------------------------------------------------
// rAF trampoline (keeps the loop alive without &mut self)
// ---------------------------------------------------------------------------

fn trampoline(
    state_rc: Rc<RefCell<Option<EngineState>>>,
    handle_rc: Rc<RefCell<Option<i32>>>,
    now_ms: f64,
) {
    if let Err(e) = render_one_frame(&state_rc, now_ms) {
        log::error!("{}", e.as_string().unwrap_or_else(|| "unknown wasm render error".to_string()));
        return;
    }

    let state_rc2 = Rc::clone(&state_rc);
    let handle_rc2 = Rc::clone(&handle_rc);

    let closure = Closure::once(move |next_now_ms: f64| {
        trampoline(state_rc2, handle_rc2, next_now_ms);
    });

    if let Some(win) = web_sys::window() {
        if let Ok(id) = win.request_animation_frame(closure.as_ref().unchecked_ref()) {
            *handle_rc.borrow_mut() = Some(id);
        }
    }

    closure.forget();
}