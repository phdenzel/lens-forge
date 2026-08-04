// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Display utilities for images

use crate::LensForgeError;
use crate::viz::render::RenderOptions;
use crate::viz::render::context::RasterContext;
use crate::viz::render::raster::RasterField;
use image::RgbaImage;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// Some fields only back the live-update path (`set_image`), which is not yet
// wired into the event loop.
#[allow(dead_code)]
struct ViewerState {
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    // Texture rendering
    render_pipeline: wgpu::RenderPipeline,
    texture: wgpu::Texture,
    sampler: wgpu::Sampler,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    texture_size: (u32, u32),
}

impl ViewerState {
    async fn new(window: Arc<Window>, image: &RgbaImage) -> ViewerState {
        // Window components
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();

        let size = window.inner_size();
        let surface = instance.create_surface(window.clone()).unwrap();
        let cap = surface.get_capabilities(&adapter);
        let surface_format = cap.formats[0];

        // Shared image components
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
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

        // Pipeline components
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Image Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format.add_srgb_suffix(),
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multiview_mask: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
        });

        // Create initial texture resources
        let (texture, bind_group) =
            Self::create_texture_bundle(&device, &queue, &bind_group_layout, &sampler, image);

        let state = ViewerState {
            window,
            device,
            queue,
            size,
            surface,
            surface_format,
            render_pipeline,
            texture,
            sampler,
            bind_group,
            bind_group_layout,
            texture_size: (image.width(), image.height()),
        };
        state.configure_surface();
        state
    }

    fn resize(&mut self, nsize: winit::dpi::PhysicalSize<u32>) {
        if nsize.width > 0 && nsize.height > 0 {
            self.size = nsize;
            // reconfigure the surface
            self.configure_surface();
        }
    }

    fn configure_surface(&self) {
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.surface_format,
            // Request compatibility with the sRGB-format texture view we‘re going to create later.
            view_formats: vec![self.surface_format.add_srgb_suffix()],
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width: self.size.width,
            height: self.size.height,
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::AutoVsync,
        };
        self.surface.configure(&self.device, &surface_config);
    }

    fn create_texture_bundle(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        image: &RgbaImage,
    ) -> (wgpu::Texture, wgpu::BindGroup) {
        let size = wgpu::Extent3d {
            width: image.width(),
            height: image.height(),
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Image Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload immediately after creation
        Self::upload_pixels(queue, &texture, image);

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        (texture, bind_group)
    }

    fn upload_pixels(queue: &wgpu::Queue, texture: &wgpu::Texture, image: &RgbaImage) {
        let size = wgpu::Extent3d {
            width: image.width(),
            height: image.height(),
            depth_or_array_layers: 1,
        };

        queue.write_texture(
            texture.as_image_copy(),
            image.as_raw(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * image.width()),
                rows_per_image: Some(image.height()),
            },
            size,
        );
    }

    fn render(&mut self) {
        // Create texture view
        let surface_texture = self
            .surface
            .get_current_texture()
            .expect("Failed to acquire next swapchain texture");
        let texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.surface_format.add_srgb_suffix()),
                ..Default::default()
            });

        // Renders a black screen
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
                ..Default::default()
            });

        // Render pass will clear the screen and draw the image
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }

        // Submit the command in the queue to execute
        self.queue.submit(Some(encoder.finish()));
        self.window.pre_present_notify();
        surface_texture.present();
    }

    /// Update the displayed image with a new RgbaImage buffer
    #[allow(dead_code)] // live-update API, not yet wired into the event loop
    pub fn set_image(&mut self, image: &RgbaImage) {
        let nsize = (image.width(), image.height());

        // If size changed, recreate texture and bind group
        if nsize != self.texture_size {
            self.texture_size = nsize;

            let (new_texture, new_bind_group) = Self::create_texture_bundle(
                &self.device,
                &self.queue,
                &self.bind_group_layout,
                &self.sampler,
                image,
            );

            self.texture = new_texture;
            self.bind_group = new_bind_group;
        } else {
            // If size is the same, just upload new pixels
            Self::upload_pixels(&self.queue, &self.texture, image);
        }

        self.window.request_redraw();
    }
}

#[derive(Default)]
struct ViewerApp {
    state: Option<ViewerState>,
    image: RgbaImage,
}

impl ViewerApp {
    fn close(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.take() {
            drop(state);
        }
        event_loop.exit();
    }
}

impl ApplicationHandler for ViewerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let state = pollster::block_on(ViewerState::new(window.clone(), &self.image));
        self.state = Some(state);
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let Some(state) = &mut self.state {
            match event {
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(key_code),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                } => match key_code {
                    KeyCode::Escape | KeyCode::KeyQ => self.close(event_loop),
                    _ => {}
                },
                WindowEvent::CloseRequested => self.close(event_loop),
                WindowEvent::RedrawRequested => {
                    state.render();
                    state.window.request_redraw();
                }
                WindowEvent::Resized(size) => state.resize(size),
                _ => (),
            }
        }
    }
}

/// Open an interactive window displaying `image` until closed.
///
/// # Arguments
/// * `image` - RGBA image to display
pub fn view_image(image: RgbaImage) {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = ViewerApp { state: None, image };
    event_loop.run_app(&mut app).unwrap();
}

/// Render a grid with `opt` and display it in an interactive window.
///
/// # Arguments
/// * `grid` - grid convertible into a [`RasterField`]
/// * `opt` - shared rendering configuration
pub fn view_grid<'a, T>(grid: &'a T, opt: &RenderOptions)
where
    RasterField<'a>: TryFrom<&'a T, Error = LensForgeError>,
{
    let raster: RasterField = grid.try_into().unwrap();
    let mut ctx = RasterContext::new();
    let image = ctx.render(&raster, &opt).unwrap();
    view_image(image);
}

/* ============ Unit tests ================================================= */
#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba, RgbaImage};

    /* ============ Test setup ============================================= */
    fn create_test_gradient(width: u32, height: u32) -> RgbaImage {
        ImageBuffer::from_fn(width, height, |x, y| {
            let r = (255.0 * x as f32 / width.max(1) as f32) as u8;
            let g = (255.0 * y as f32 / height.max(1) as f32) as u8;
            Rgba([r, g, 128, 255])
        })
    }

    /* ============ ViewerApp ============================================== */
    #[test]
    fn test_viewer_app_default() {
        let app = ViewerApp::default();
        assert!(app.state.is_none());
        assert_eq!(app.image.width(), 0);
        assert_eq!(app.image.height(), 0);
    }

    #[test]
    fn test_viewer_app_with_image() {
        let img = create_test_gradient(256, 256);
        let app = ViewerApp {
            state: None,
            image: img.clone(),
        };

        assert!(app.state.is_none());
        assert_eq!(app.image.width(), 256);
        assert_eq!(app.image.height(), 256);
    }
}
