use std::num::NonZeroU64;
use std::sync::mpsc;

use anyhow::{anyhow, bail, Context, Result};
use image::DynamicImage;
use log::{debug, error, info, warn};
use wgpu::util::DeviceExt;

use crate::{KaleidoSettings, KaleidoType};

pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,

    settings_buffer: wgpu::Buffer,

    source: Option<SourceImageGpu>,
    output: Option<OutputResources>,
    last_settings: Option<GpuKaleidoSettings>,

    blit_pipeline: Option<wgpu::RenderPipeline>,
    blit_bind_group_layout: Option<wgpu::BindGroupLayout>,
    blit_sampler: Option<wgpu::Sampler>,
    canvas_intermediate: Option<CanvasIntermediate>,
}

const DEFAULT_TILE_SIZE: u32 = 4096;

pub(crate) struct SourceImageGpu {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    tile_size: u32,
    tile_grid_width: u32,
    tile_grid_height: u32,
    layer_count: u32,
}

impl SourceImageGpu {
    fn tile_index(&self, tile_x: u32, tile_y: u32) -> u32 {
        tile_y * self.tile_grid_width + tile_x
    }
}

struct OutputResources {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    readback_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    cpu_buffer: Vec<u8>,
}

impl GpuBackend {
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .context("failed to find a suitable GPU adapter")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("kaleidomo.device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::default(),
            })
            .await
            .context("failed to create GPU device")?;

        // Replace wgpu's default panic-on-error handler with a log statement so
        // GPU device-lost / OOM events don't silently kill the process.
        device.on_uncaptured_error(std::sync::Arc::new(|error: wgpu::Error| {
            log::error!("[kaleidomo gpu] uncaptured wgpu error: {error}");
        }));

        let shader = device.create_shader_module(wgpu::include_wgsl!("kaleidomo.wgsl"));

        let settings_min_binding_size = non_zero_u64(std::mem::size_of::<GpuKaleidoSettings>() as u64)
            .context("GpuKaleidoSettings size cannot be zero")?;

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("kaleidomo.bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(settings_min_binding_size),
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("kaleidomo.pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("kaleidomo.compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let settings_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kaleidomo.persistent_settings_buffer"),
            size: std::mem::size_of::<GpuKaleidoSettings>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        info!("GpuBackend initialized successfully");

        Ok(Self {
            device,
            queue,
            bind_group_layout,
            pipeline,
            settings_buffer,
            source: None,
            output: None,
            last_settings: None,
            blit_bind_group_layout: None,
            blit_pipeline: None,
            blit_sampler: None,
            canvas_intermediate: None,
        })
    }

    pub fn set_source_image(&mut self, source: &DynamicImage) -> Result<()> {
        let rgba = source.to_rgba8();
        let (width, height) = rgba.dimensions();

        if width == 0 || height == 0 {
            error!("set_source_image called with zero-sized image: {}x{}", width, height);
            bail!("source image dimensions must be non-zero");
        }

        let limits = self.device.limits();
        let max_dim = limits.max_texture_dimension_2d;
        let max_layers = limits.max_texture_array_layers;

        let tile_size = DEFAULT_TILE_SIZE.min(max_dim);
        if tile_size == 0 {
            bail!("GPU reported invalid max texture dimension 0");
        }

        let tile_grid_width = width.div_ceil(tile_size);
        let tile_grid_height = height.div_ceil(tile_size);
        let layer_count = tile_grid_width
            .checked_mul(tile_grid_height)
            .ok_or_else(|| anyhow!("source tile count overflow"))?;

        if layer_count > max_layers {
            bail!(
                "source image requires {} tile layers, but GPU only supports {} texture array layers",
                layer_count,
                max_layers
            );
        }

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("kaleidomo.input_texture_array"),
            size: wgpu::Extent3d {
                width: tile_size,
                height: tile_size,
                depth_or_array_layers: layer_count,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let src_raw = rgba.as_raw();
        let src_row_bytes = width as usize * 4;
        let tile_row_bytes = tile_size as usize * 4;

        let mut tile_staging = vec![0u8; tile_row_bytes * tile_size as usize];

        for tile_y in 0..tile_grid_height {
            for tile_x in 0..tile_grid_width {
                tile_staging.fill(0);

                let src_x = tile_x * tile_size;
                let src_y = tile_y * tile_size;

                let copy_width = (width - src_x).min(tile_size);
                let copy_height = (height - src_y).min(tile_size);

                for row in 0..copy_height as usize {
                    let src_start =
                        ((src_y as usize + row) * src_row_bytes) + (src_x as usize * 4);
                    let src_end = src_start + copy_width as usize * 4;

                    let dst_start = row * tile_row_bytes;
                    let dst_end = dst_start + copy_width as usize * 4;

                    tile_staging[dst_start..dst_end].copy_from_slice(&src_raw[src_start..src_end]);
                }

                let layer = tile_y * tile_grid_width + tile_x;

                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: layer,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &tile_staging,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(tile_size * 4),
                        rows_per_image: Some(tile_size),
                    },
                    wgpu::Extent3d {
                        width: tile_size,
                        height: tile_size,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("kaleidomo.input_texture_array_view"),
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(layer_count),
        });

        self.source = Some(SourceImageGpu {
            texture,
            view,
            width,
            height,
            tile_size,
            tile_grid_width,
            tile_grid_height,
            layer_count,
        });

        self.last_settings = None;

        info!(
            "Source image uploaded to GPU as tiled array: {}x{}, tile_size={}, grid={}x{}, layers={}",
            width,
            height,
            tile_size,
            tile_grid_width,
            tile_grid_height,
            layer_count
        );

        Ok(())
    }

    pub fn clear_source_image(&mut self) {
        warn!("Clearing current source image from GPU state");
        self.source = None;
        self.last_settings = None;
    }

    pub fn update_settings(&mut self, settings: &KaleidoSettings) -> Result<()> {
        let source = match self.source.as_ref() {
            Some(source) => source,
            None => {
                error!("update_settings called before a source image was set");
                bail!("cannot update GPU settings without a source image");
            }
        };

        self.last_settings = Some(GpuKaleidoSettings::from_parts(
            settings,
            source,
            0,
            0,
        ));

        debug!(
            "Updated cached GPU settings for output {}x{}",
            settings.output_size_w, settings.output_size_h
        );

        Ok(())
    }

    pub fn render_into_buffer(&mut self, settings: &KaleidoSettings, output: &mut [u8]) -> Result<()> {
        let expected_len = expected_rgba_len(settings.output_size_w, settings.output_size_h)?;
        if output.len() != expected_len {
            error!(
                "render_into_buffer output length mismatch: got {}, expected {}",
                output.len(),
                expected_len
            );
            bail!(
                "output buffer length mismatch: got {}, expected {} ({} * {} * 4)",
                output.len(),
                expected_len,
                settings.output_size_w,
                settings.output_size_h
            );
        }

        let max_dim = self.device.limits().max_texture_dimension_2d;
        let tile_size = DEFAULT_TILE_SIZE.min(max_dim);

        if tile_size == 0 {
            bail!("GPU reported invalid max texture dimension 0");
        }

        let full_output_row_bytes = settings.output_size_w as usize * 4;

        for tile_origin_y in (0..settings.output_size_h).step_by(tile_size as usize) {
            for tile_origin_x in (0..settings.output_size_w).step_by(tile_size as usize) {
                let tile_width = (settings.output_size_w - tile_origin_x).min(tile_size);
                let tile_height = (settings.output_size_h - tile_origin_y).min(tile_size);

                self.ensure_output_resources(tile_width, tile_height)?;

                let source = match self.source.as_ref() {
                    Some(source) => source,
                    None => {
                        error!("render_into_buffer called with no source image selected");
                        bail!("no source image is currently loaded into the GPU backend");
                    }
                };

                let gpu_settings = GpuKaleidoSettings::from_parts(
                    settings,
                    source,
                    tile_origin_x,
                    tile_origin_y,
                );
                self.last_settings = Some(gpu_settings);

                self.queue.write_buffer(
                    &self.settings_buffer,
                    0,
                    bytemuck::bytes_of(&gpu_settings),
                );

                let output_resources = match self.output.as_ref() {
                    Some(resources) => resources,
                    None => {
                        error!("output resources unexpectedly missing after ensure_output_resources");
                        bail!("output resources were not available");
                    }
                };

                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("kaleidomo.bind_group"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&source.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&output_resources.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.settings_buffer.as_entire_binding(),
                        },
                    ],
                });

                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("kaleidomo.command_encoder"),
                        });

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("kaleidomo.compute_pass"),
                        timestamp_writes: None,
                    });

                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(
                        tile_width.div_ceil(8),
                        tile_height.div_ceil(8),
                        1,
                    );
                }

                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &output_resources.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &output_resources.readback_buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(output_resources.padded_bytes_per_row),
                            rows_per_image: Some(output_resources.height),
                        },
                    },
                    wgpu::Extent3d {
                        width: output_resources.width,
                        height: output_resources.height,
                        depth_or_array_layers: 1,
                    },
                );

                self.queue.submit(Some(encoder.finish()));

                let tile_row_bytes = tile_width as usize * 4;

                if tile_origin_x == 0
                    && tile_origin_y == 0
                    && tile_width == settings.output_size_w
                    && tile_height == settings.output_size_h
                {
                    self.readback_into_output(output)?;
                } else {
                    let mut tile_cpu = vec![0u8; expected_rgba_len(tile_width, tile_height)?];
                    self.readback_into_output(&mut tile_cpu)?;

                    for row in 0..tile_height as usize {
                        let src_start = row * tile_row_bytes;
                        let src_end = src_start + tile_row_bytes;

                        let dst_start = ((tile_origin_y as usize + row) * full_output_row_bytes)
                            + (tile_origin_x as usize * 4);
                        let dst_end = dst_start + tile_row_bytes;

                        output[dst_start..dst_end].copy_from_slice(&tile_cpu[src_start..src_end]);
                    }
                }
            }
        }

        Ok(())
    }

    pub fn render_into_internal_buffer(&mut self, settings: &KaleidoSettings) -> Result<&[u8]> {
        let expected_len = expected_rgba_len(settings.output_size_w, settings.output_size_h)?;
        self.ensure_output_resources(settings.output_size_w, settings.output_size_h)?;

        {
            let output_resources = match self.output.as_mut() {
                Some(resources) => resources,
                None => {
                    error!("output resources missing before internal render");
                    bail!("output resources were not available");
                }
            };

            if output_resources.cpu_buffer.len() != expected_len {
                output_resources.cpu_buffer.resize(expected_len, 0);
            }
        }

        let mut temp = vec![0u8; expected_len];
        self.render_into_buffer(settings, &mut temp)?;

        let output_resources = match self.output.as_mut() {
            Some(resources) => resources,
            None => {
                error!("output resources missing after internal render");
                bail!("output resources were not available");
            }
        };

        output_resources.cpu_buffer.copy_from_slice(&temp);
        Ok(&output_resources.cpu_buffer)
    }

    pub fn latest_output(&self) -> Option<&[u8]> {
        self.output.as_ref().map(|o| o.cpu_buffer.as_slice())
    }

    fn ensure_output_resources(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            error!("ensure_output_resources called with zero-sized output: {}x{}", width, height);
            bail!("output dimensions must be non-zero");
        }

        let needs_rebuild = match self.output.as_ref() {
            Some(existing) => existing.width != width || existing.height != height,
            None => true,
        };

        if !needs_rebuild {
            return Ok(());
        }

        let padded_bytes_per_row = align_to(width * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback_size = padded_bytes_per_row as u64 * height as u64;

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("kaleidomo.output_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kaleidomo.readback_buffer"),
            size: readback_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let cpu_buffer = vec![0u8; expected_rgba_len(width, height)?];

        self.output = Some(OutputResources {
            texture,
            view,
            readback_buffer,
            width,
            height,
            padded_bytes_per_row,
            cpu_buffer,
        });

        info!("Allocated GPU output resources for {}x{}", width, height);
        Ok(())
    }

    fn readback_into_output(&self, output: &mut [u8]) -> Result<()> {
        let output_resources = match self.output.as_ref() {
            Some(resources) => resources,
            None => {
                error!("readback_into_output called with no output resources allocated");
                bail!("output resources are not allocated");
            }
        };

        let slice = output_resources.readback_buffer.slice(..);
        let (tx, rx) = mpsc::channel();

        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("failed while waiting for GPU work to complete")?;

        rx.recv()
            .context("failed waiting for GPU map_async callback")?
            .context("GPU readback buffer mapping failed")?;

        let mapped = slice.get_mapped_range();
        let row_len = output_resources.width as usize * 4;

        for y in 0..output_resources.height as usize {
            let src_offset = y * output_resources.padded_bytes_per_row as usize;
            let dst_offset = y * row_len;
            output[dst_offset..dst_offset + row_len]
                .copy_from_slice(&mapped[src_offset..src_offset + row_len]);
        }

        drop(mapped);
        output_resources.readback_buffer.unmap();

        Ok(())
    }
}

fn expected_rgba_len(width: u32, height: u32) -> Result<usize> {
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| anyhow!("RGBA output dimensions overflowed"))
}

fn non_zero_u64(value: u64) -> Result<NonZeroU64> {
    NonZeroU64::new(value).ok_or_else(|| anyhow!("value must be non-zero"))
}

fn align_to(value: u32, alignment: u32) -> u32 {
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value + (alignment - rem)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuKaleidoSettings {
    pub count: u32,
    pub output_size_w: u32,
    pub output_size_h: u32,
    pub kaleido_type: u32,

    pub offset_x: i32,
    pub offset_y: i32,
    pub hue_rotation: u32,
    pub _pad0: u32,

    pub zoom: f32,
    pub inv_zoom: f32,
    pub tile_count: f32,
    pub slice_angle: f32,

    pub center_x: f32,
    pub center_y: f32,
    pub triangle_center_x: f32,
    pub triangle_center_y: f32,

    pub triangle_rotation_rad: f32,
    pub source_width: u32,
    pub source_height: u32,
    pub source_tile_size: u32,

    pub source_tile_grid_width: u32,
    pub source_tile_grid_height: u32,
    pub output_tile_origin_x: u32,
    pub output_tile_origin_y: u32,
}

impl GpuKaleidoSettings {
    pub fn from_parts(
        settings: &KaleidoSettings,
        source: &SourceImageGpu,
        output_tile_origin_x: u32,
        output_tile_origin_y: u32,
    ) -> Self {
        let zoom = settings.zoom;
        let inv_zoom = if zoom.abs() > f32::EPSILON { 1.0 / zoom } else { 0.0 };
        let safe_count = settings.count.max(1);
        let slice_angle = (2.0 * std::f32::consts::PI) / safe_count as f32;
        let center_x = settings.output_size_w as f32 * 0.5 + settings.offset_x as f32;
        let center_y = settings.output_size_h as f32 * 0.5 + settings.offset_y as f32;

        Self {
            count: settings.count,
            output_size_w: settings.output_size_w,
            output_size_h: settings.output_size_h,
            kaleido_type: kaleido_type_to_u32(settings.kaleido_type),

            offset_x: settings.offset_x,
            offset_y: settings.offset_y,
            hue_rotation: settings.hue_rotation,
            _pad0: 0,

            zoom,
            inv_zoom,
            tile_count: settings.tile_count,
            slice_angle,

            center_x,
            center_y,
            triangle_center_x: settings.triangle_center_x,
            triangle_center_y: settings.triangle_center_y,

            triangle_rotation_rad: settings.triangle_rotation_rad,
            source_width: source.width,
            source_height: source.height,
            source_tile_size: source.tile_size,

            source_tile_grid_width: source.tile_grid_width,
            source_tile_grid_height: source.tile_grid_height,
            output_tile_origin_x,
            output_tile_origin_y,
        }
    }
}

fn kaleido_type_to_u32(value: KaleidoType) -> u32 {
    match value {
        KaleidoType::Radial => 0,
        KaleidoType::Square => 1,
        KaleidoType::Diamond => 2,
        KaleidoType::Hexagonal => 3,
        KaleidoType::HexagonalFlatTop => 4,
    }
}

impl GpuBackend {
    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub(crate) fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    pub(crate) fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    pub(crate) fn source_ref(&self) -> Option<&SourceImageGpu> {
        self.source.as_ref()
    }
}

use std::collections::VecDeque;
use std::error::Error;

pub struct GpuVideoRenderer<'a> {
    gpu: &'a mut GpuBackend,
    width: u32,
    height: u32,
    slots: Vec<FrameSlot>,
    pending: VecDeque<PendingFrame>,
    max_in_flight: usize,
}

struct FrameSlot {
    output_texture: wgpu::Texture,
    output_view: wgpu::TextureView,
    readback_buffer: wgpu::Buffer,
    settings_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    padded_bytes_per_row: u32,
    cpu_buffer: Vec<u8>,
    state: SlotState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotState {
    Free,
    Submitted,
    Ready,
}

struct PendingFrame {
    frame_index: u32,
    slot_index: usize,
    submission_index: wgpu::SubmissionIndex,
}

pub struct CompletedFrame {
    pub frame_index: u32,
    pub slot_index: usize,
}

impl<'a> GpuVideoRenderer<'a> {
    pub fn new(
        gpu: &'a mut GpuBackend,
        width: u32,
        height: u32,
        max_in_flight: usize,
    ) -> Result<Self, Box<dyn Error>> {
        if width == 0 || height == 0 {
            return Err("width and height must be non-zero".into());
        }

        if max_in_flight == 0 {
            return Err("max_in_flight must be non-zero".into());
        }

        let max_dim = gpu.device().limits().max_texture_dimension_2d;
        if width > max_dim || height > max_dim {
            return Err(format!(
                "GpuVideoRenderer output {}x{} exceeds GPU max texture dimension {}; tiled video output is not implemented yet",
                width, height, max_dim
            ).into());
        }

        //gpu.set_source_image(source)?;

        let source = gpu
            .source_ref()
            .ok_or("GpuVideoRenderer could not access uploaded source image")?;

        let mut slots = Vec::with_capacity(max_in_flight);
        for _ in 0..max_in_flight {
            slots.push(Self::create_slot(
                gpu,
                &source.view,
                source,
                width,
                height,
            )?);
        }

        Ok(Self {
            gpu,
            width,
            height,
            slots,
            pending: VecDeque::new(),
            max_in_flight,
        })
    }

    fn create_slot(
        gpu: &GpuBackend,
        source_view: &wgpu::TextureView,
        source: &SourceImageGpu,
        width: u32,
        height: u32,
    ) -> Result<FrameSlot, Box<dyn Error>> {
        let output_texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("kaleidomo.video.output_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let padded_bytes_per_row = align_to(width * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback_size = padded_bytes_per_row as u64 * height as u64;

        let readback_buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("kaleidomo.video.readback_buffer"),
            size: readback_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let initial_settings = GpuKaleidoSettings::from_parts(
            &KaleidoSettings {
                count: 1,
                output_size_w: width,
                output_size_h: height,
                offset_x: 0,
                offset_y: 0,
                zoom: 1.0,
                tile_count: 1.0,
                triangle_center_x: 0.0,
                triangle_center_y: 0.0,
                triangle_rotation_rad: 0.0,
                kaleido_type: crate::KaleidoType::Radial,
                hue_rotation: 0,
            },
            source,
            0,
            0,
        );

        let settings_buffer =
            gpu.device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("kaleidomo.video.settings_buffer"),
                    contents: bytemuck::bytes_of(&initial_settings),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let bind_group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("kaleidomo.video.bind_group"),
            layout: gpu.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: settings_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(FrameSlot {
            output_texture,
            output_view,
            readback_buffer,
            settings_buffer,
            bind_group,
            padded_bytes_per_row,
            cpu_buffer: vec![0u8; rgba_len(width, height)?],
            state: SlotState::Free,
        })
    }

    fn find_free_slot(&self) -> Option<usize> {
        self.slots
            .iter()
            .position(|slot| matches!(slot.state, SlotState::Free))
    }

    pub fn submit_frame(
        &mut self,
        frame_index: u32,
        settings: &KaleidoSettings,
    ) -> Result<(), Box<dyn Error>> {
        if settings.output_size_w != self.width || settings.output_size_h != self.height {
            return Err(format!(
                "settings output size {}x{} does not match renderer size {}x{}",
                settings.output_size_w,
                settings.output_size_h,
                self.width,
                self.height
            )
            .into());
        }

        while self.pending.len() >= self.max_in_flight {
            let completed = self.receive_oldest_blocking()?;
            if completed.is_none() {
                break;
            }
        }

        let slot_index = match self.find_free_slot() {
            Some(i) => i,
            None => {
                let completed = self.receive_oldest_blocking()?;
                if completed.is_none() {
                    return Err("no free frame slot was available".into());
                }
                self.find_free_slot()
                    .ok_or("no free frame slot was available after waiting")?
            }
        };

        let source = self
            .gpu
            .source_ref()
            .ok_or("GpuVideoRenderer has no source image loaded")?;

        let gpu_settings =
            GpuKaleidoSettings::from_parts(settings, source, 0, 0);

        let slot = &mut self.slots[slot_index];
        self.gpu.queue().write_buffer(
            &slot.settings_buffer,
            0,
            bytemuck::bytes_of(&gpu_settings),
        );

        let mut encoder =
            self.gpu
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("kaleidomo.video.command_encoder"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("kaleidomo.video.compute_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(self.gpu.pipeline());
            pass.set_bind_group(0, &slot.bind_group, &[]);
            pass.dispatch_workgroups(
                self.width.div_ceil(8),
                self.height.div_ceil(8),
                1,
            );
        }

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &slot.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &slot.readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(slot.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        let submission_index = self.gpu.queue().submit(Some(encoder.finish()));

        slot.state = SlotState::Submitted;
        self.pending.push_back(PendingFrame {
            frame_index,
            slot_index,
            submission_index,
        });

        Ok(())
    }

    pub fn receive_oldest_blocking(
        &mut self,
    ) -> Result<Option<CompletedFrame>, Box<dyn Error>> {
        let pending = match self.pending.pop_front() {
            Some(p) => p,
            None => return Ok(None),
        };

        self.gpu
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: Some(pending.submission_index),
                timeout: None,
            })?;

        let slot = &mut self.slots[pending.slot_index];
        let slice = slot.readback_buffer.slice(..);
        let (tx, rx) = mpsc::channel();

        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.gpu
            .device()
            .poll(wgpu::PollType::wait_indefinitely())?;

        rx.recv()??;

        let mapped = slice.get_mapped_range();
        let row_len = self.width as usize * 4;

        for y in 0..self.height as usize {
            let src_offset = y * slot.padded_bytes_per_row as usize;
            let dst_offset = y * row_len;
            slot.cpu_buffer[dst_offset..dst_offset + row_len]
                .copy_from_slice(&mapped[src_offset..src_offset + row_len]);
        }

        drop(mapped);
        slot.readback_buffer.unmap();
        slot.state = SlotState::Ready;

        Ok(Some(CompletedFrame {
            frame_index: pending.frame_index,
            slot_index: pending.slot_index,
        }))
    }

    pub fn slot_bytes(&self, slot_index: usize) -> Result<&[u8], Box<dyn Error>> {
        let slot = self
            .slots
            .get(slot_index)
            .ok_or_else(|| format!("invalid slot index {}", slot_index))?;

        if !matches!(slot.state, SlotState::Ready) {
            return Err(format!(
                "slot {} was requested before it became ready",
                slot_index
            )
            .into());
        }

        Ok(&slot.cpu_buffer)
    }

    pub fn release_slot(&mut self, slot_index: usize) -> Result<(), Box<dyn Error>> {
        let slot = self
            .slots
            .get_mut(slot_index)
            .ok_or_else(|| format!("invalid slot index {}", slot_index))?;

        if !matches!(slot.state, SlotState::Ready) {
            return Err(format!(
                "slot {} cannot be released because it is not ready",
                slot_index
            )
            .into());
        }

        slot.state = SlotState::Free;
        Ok(())
    }

    pub fn drain_remaining_blocking(
        &mut self,
    ) -> Result<Vec<CompletedFrame>, Box<dyn Error>> {
        let mut frames = Vec::with_capacity(self.pending.len());
        while let Some(frame) = self.receive_oldest_blocking()? {
            frames.push(frame);
        }
        Ok(frames)
    }

    pub fn recreate_source_bind_groups(
        &mut self,
        source: &DynamicImage,
    ) -> Result<(), Box<dyn Error>> {
        self.gpu.set_source_image(source)?;

        let source_ref = self
            .gpu
            .source_ref()
            .ok_or("source image was not available after upload")?;

        for slot in &mut self.slots {
            slot.bind_group = self.gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("kaleidomo.video.bind_group"),
                layout: self.gpu.bind_group_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source_ref.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&slot.output_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: slot.settings_buffer.as_entire_binding(),
                    },
                ],
            });
        }

        Ok(())
    }
}

fn rgba_len(width: u32, height: u32) -> Result<usize, Box<dyn Error>> {
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| "RGBA frame size overflow".into())
}

struct CanvasIntermediate {
    texture: wgpu::Texture,
    view:    wgpu::TextureView,
    width:   u32,
    height:  u32,
}
 
impl GpuBackend {
    // -----------------------------------------------------------------------
    // Constructor for the Wasm / canvas path.
    // Call this instead of GpuBackend::new() when you already have a
    // device + queue from surface negotiation.
    // `swapchain_format` must match the SurfaceConfiguration format.
    // -----------------------------------------------------------------------
    pub fn new_for_canvas(
        device:           wgpu::Device,
        queue:            wgpu::Queue,
        swapchain_format: wgpu::TextureFormat,
    ) -> Result<Self> {
        // ── Compute pipeline (identical to GpuBackend::new) ──────────────────
        let shader = device.create_shader_module(wgpu::include_wgsl!("kaleidomo.wgsl"));
 
        let settings_min_binding_size =
            non_zero_u64(std::mem::size_of::<GpuKaleidoSettings>() as u64)
                .context("GpuKaleidoSettings size cannot be zero")?;
 
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("kaleidomo.bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(settings_min_binding_size),
                        },
                        count: None,
                    },
                ],
            });
 
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("kaleidomo.pipeline_layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
 
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("kaleidomo.compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
 
        let settings_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kaleidomo.settings_buffer"),
            size: std::mem::size_of::<GpuKaleidoSettings>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
 
        // ── Blit render pipeline (compute intermediate → swap-chain) ─────────
        //
        // WebGPU swap-chain textures can't be StorageTextures, so the compute
        // shader writes to an intermediate Rgba8Unorm texture, then this blit
        // pipeline copies it to the swap-chain view via a full-screen triangle.
        let blit_shader_src = r#"
            @group(0) @binding(0) var t_intermediate : texture_2d<f32>;
            @group(0) @binding(1) var s_linear       : sampler;
 
            struct VertOut {
                @builtin(position) pos : vec4<f32>,
                @location(0)       uv  : vec2<f32>,
            }
 
            // Full-screen triangle: no vertex buffer needed
            @vertex
            fn vs_main(@builtin(vertex_index) vi : u32) -> VertOut {
                // Vertices at NDC: (-1,-1), (3,-1), (-1,3)
                let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
                let y = f32(vi & 2u) * 2.0 - 1.0;
                var o : VertOut;
                o.pos = vec4<f32>(x, y, 0.0, 1.0);
                // Map NDC [-1,1] → UV [0,1], flip Y for wgpu's top-left origin
                o.uv  = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
                return o;
            }
 
            @fragment
            fn fs_main(in : VertOut) -> @location(0) vec4<f32> {
                return textureSample(t_intermediate, s_linear, in.uv);
            }
        "#;
 
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("kaleidomo.blit_shader"),
            source: wgpu::ShaderSource::Wgsl(blit_shader_src.into()),
        });
 
        let blit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("kaleidomo.blit_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
 
        let blit_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("kaleidomo.blit_pipeline_layout"),
                bind_group_layouts: &[Some(&blit_bind_group_layout)],
                immediate_size: 0,
            });
 
        let blit_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label:  Some("kaleidomo.blit_pipeline"),
                layout: Some(&blit_pipeline_layout),
                vertex: wgpu::VertexState {
                    module:      &blit_shader,
                    entry_point: Some("vs_main"),
                    buffers:     &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module:      &blit_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format:     swapchain_format,
                        blend:      None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive:     wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample:   wgpu::MultisampleState::default(),
                // wgpu 29: multiview_mask (not multiview); None = not a multiview pass
                multiview_mask: None,
                cache: None,
            });
 
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:           Some("kaleidomo.blit_sampler"),
            address_mode_u:  wgpu::AddressMode::ClampToEdge,
            address_mode_v:  wgpu::AddressMode::ClampToEdge,
            mag_filter:      wgpu::FilterMode::Linear,
            min_filter:      wgpu::FilterMode::Linear,
            ..Default::default()
        });
 
        info!("GpuBackend (canvas) initialized successfully");
 
        Ok(Self {
            device,
            queue,
            bind_group_layout,
            pipeline,
            settings_buffer,
            source:               None,
            output:               None,
            last_settings:        None,
            blit_pipeline:        Some(blit_pipeline),
            blit_bind_group_layout: Some(blit_bind_group_layout),
            blit_sampler:         Some(blit_sampler),
            canvas_intermediate:  None,
        })
    }
 
    // -----------------------------------------------------------------------
    // Resize / reconfigure the surface swap-chain (call when canvas resizes).
    // -----------------------------------------------------------------------
    pub fn configure_surface(
        &self,
        surface: &wgpu::Surface,
        config:  &wgpu::SurfaceConfiguration,
    ) {
        surface.configure(&self.device, config);
    }
 
    // -----------------------------------------------------------------------
    // Per-frame render: compute kaleidoscope → intermediate texture, then
    // blit intermediate → swap-chain TextureView.
    // `external_settings_buffer` is the pre-allocated uniform buffer owned by
    // the Wasm engine (avoids a GPU allocation per frame).
    // -----------------------------------------------------------------------
    pub fn render_directly_to_view(
        &mut self,
        settings:                 &KaleidoSettings,
        swap_view:                &wgpu::TextureView,
        external_settings_buffer: &wgpu::Buffer,
        swapchain_format:         wgpu::TextureFormat,
    ) -> Result<()> {
        let width  = settings.output_size_w;
        let height = settings.output_size_h;
 
        // ── 1. Ensure the intermediate texture matches current output size ────
        let needs_rebuild = match &self.canvas_intermediate {
            Some(ci) => ci.width != width || ci.height != height,
            None     => true,
        };
 
        if needs_rebuild {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("kaleidomo.canvas_intermediate"),
                size:  wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count:    1,
                dimension:       wgpu::TextureDimension::D2,
                format:          wgpu::TextureFormat::Rgba8Unorm,
                // STORAGE_BINDING  → compute writes here
                // TEXTURE_BINDING  → blit samples from here
                usage: wgpu::TextureUsages::STORAGE_BINDING
                     | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.canvas_intermediate = Some(CanvasIntermediate { texture, view, width, height });
        }
 
        let intermediate_view = &self
            .canvas_intermediate
            .as_ref()
            .context("canvas_intermediate missing after ensure")?
            .view;
 
        // ── 2. Upload per-frame settings uniform ─────────────────────────────
        let source = self
            .source
            .as_ref()
            .context("no source image loaded; call set_source_image first")?;
 
        let gpu_settings = GpuKaleidoSettings::from_parts(settings, source, 0, 0);
 
        self.queue.write_buffer(
            external_settings_buffer,
            0,
            bytemuck::bytes_of(&gpu_settings),
        );
 
        // ── 3. Compute pass: kaleidoscope → intermediate ──────────────────────
        let compute_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:  Some("kaleidomo.canvas_compute_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding:  0,
                        resource: wgpu::BindingResource::TextureView(&source.view),
                    },
                    wgpu::BindGroupEntry {
                        binding:  1,
                        resource: wgpu::BindingResource::TextureView(intermediate_view),
                    },
                    wgpu::BindGroupEntry {
                        binding:  2,
                        resource: external_settings_buffer.as_entire_binding(),
                    },
                ],
            });
 
        let blit_bgl = self
            .blit_bind_group_layout
            .as_ref()
            .context("blit_bind_group_layout missing; use new_for_canvas()")?;
        let blit_sampler = self
            .blit_sampler
            .as_ref()
            .context("blit_sampler missing; use new_for_canvas()")?;
        let blit_pipeline = self
            .blit_pipeline
            .as_ref()
            .context("blit_pipeline missing; use new_for_canvas()")?;
 
        let blit_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:  Some("kaleidomo.blit_bg"),
                layout: blit_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding:  0,
                        resource: wgpu::BindingResource::TextureView(intermediate_view),
                    },
                    wgpu::BindGroupEntry {
                        binding:  1,
                        resource: wgpu::BindingResource::Sampler(blit_sampler),
                    },
                ],
            });
 
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("kaleidomo.canvas_encoder") },
        );
 
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("kaleidomo.canvas_compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &compute_bind_group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
 
        // ── 4. Render pass: blit intermediate → swap-chain ───────────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("kaleidomo.blit_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           swap_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes:        None,
                occlusion_query_set:     None,
                // wgpu 29: multiview_mask field required; None = no multiview
                multiview_mask: None,
            });
            pass.set_pipeline(blit_pipeline);
            pass.set_bind_group(0, &blit_bind_group, &[]);
            pass.draw(0..3, 0..1); // full-screen triangle, no vertex buffer
        }
 
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}