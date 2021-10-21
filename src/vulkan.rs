use crate::person::Person;
use std::boxed::Box;
use std::sync::Arc;
use vulkano::buffer::device_local::DeviceLocalBuffer;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::buffer::{BufferUsage, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, PrimaryCommandBuffer};
use vulkano::command_buffer::{CommandBufferUsage, SubpassContents};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::image::view::ImageView;
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::{Instance};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline};
use vulkano::render_pass::{Framebuffer, FramebufferAbstract, RenderPass, Subpass};
use vulkano::swapchain::{AcquireError, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreationError};
use vulkano::sync::{FlushError, GpuFuture};
use vulkano::Version;
use vulkano::{swapchain, sync};
use vulkano_win::VkSurfaceBuild;
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::{Window, WindowBuilder};
use tracy_client;

mod vert_shader {
    vulkano_shaders::shader!{
        ty: "vertex",
        path: "shaders/vert.glsl"
    }
}

mod frag_shader {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "shaders/frag.glsl"
    }
}

mod compute_shader {
    vulkano_shaders::shader!{
        ty: "compute",
        path: "shaders/comp.glsl"
    }
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<dyn FramebufferAbstract>> {
    let dimensions = images[0].dimensions();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    images
        .iter()
        .map(|image| {
            let view = ImageView::new(image.clone()).unwrap();
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(view)
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract>
        })
        .collect::<Vec<_>>()
}

#[derive(Default, Debug, Clone, Copy)]
struct Vertex {
    pos: [f32; 2],
    color: [f32; 3]
}

pub(crate) struct VulkanContext {
    instance: Arc<Instance>,
    event_loop: EventLoop<()>,
    recreate_swapchain: bool,
    surface: Option<Arc<Surface<Window>>>,
    device: Option<Arc<Device>>,
    graphics_queue: Option<Arc<Queue>>,
    compute_queue: Option<Arc<Queue>>,
    swapchain: Option<Arc<Swapchain<Window>>>,
    images: Option<Vec<Arc<SwapchainImage<Window>>>>,
    vert_shader: Option<vert_shader::Shader>,
    frag_shader: Option<frag_shader::Shader>,
    comp_shader: Option<compute_shader::Shader>,
    comp_pipeline: Option<Arc<ComputePipeline>>,
    graphics_pipeline: Option<Arc<GraphicsPipeline>>,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    framebuffers: Option<Vec<Arc<dyn FramebufferAbstract>>>,
    people_buf: Option<Arc<CpuAccessibleBuffer<[Person]>>>,
    vertex_buf: Option<Arc<DeviceLocalBuffer<[Vertex]>>>,
    render_pass: Option<Arc<RenderPass>>,
    viewport: Arc<Viewport>,
}

impl VulkanContext {
    pub(crate) fn new(people: Vec<Person>, x: i32, y: i32) -> VulkanContext {
        vulkano::impl_vertex!(Vertex, pos, color);

        let req_ext = vulkano_win::required_extensions();

        let dev_ext = DeviceExtensions {
            khr_swapchain: true,
            khr_storage_buffer_storage_class: true,
            ..DeviceExtensions::none()
        };

        let mut ctx = VulkanContext {
            instance: {
                let _span = tracy_client::span!("Create Instance");
                Instance::new(None, Version::V1_2, &req_ext, None)
                    .expect("Couldn't initialize Vulkano Instance")
            },
            event_loop: EventLoop::new(),
            // All initialized in the following code
            surface: None,
            device: None,
            graphics_queue: None,
            compute_queue: None,
            swapchain: None,
            recreate_swapchain: false,
            images: None,
            vert_shader: None,
            frag_shader: None,
            comp_shader: None,
            comp_pipeline: None,
            graphics_pipeline: None,
            previous_frame_end: None,
            framebuffers: None,
            people_buf: None,
            vertex_buf: None,
            render_pass: None,
            viewport: Arc::new(Viewport {
                origin: [0.0, 0.0],
                dimensions: [x as f32, y as f32],
                depth_range: 0.0..1.0,
            }),
        };

        {
            let _span = tracy_client::span!("Create surface");
            ctx.surface = Some(
                WindowBuilder::new()
                    .with_resizable(false)
                    .with_title("3piSim")
                    .with_inner_size(PhysicalSize {
                        width: x,
                        height: y,
                    })
                    .build_vk_surface(&ctx.event_loop, ctx.instance.clone())
                    .unwrap(),
            );
        }
        {
            let _span = tracy_client::span!("Select devices and queues");
            let (phy_dev, graphics_queue, compute_queue) = PhysicalDevice::enumerate(&ctx.instance)
                .filter(|&p| p.supported_extensions().is_superset_of(&dev_ext))
                .filter_map(|p| {
                    let mut supports_graphics = None;
                    let mut supports_compute = None;

                    for qf in p.queue_families() {
                        if qf.supports_graphics()
                            && ctx
                            .surface
                            .as_ref()
                            .unwrap()
                            .is_supported(qf)
                            .unwrap_or(false)
                        {
                            supports_graphics = Some(qf);
                        }

                        if qf.supports_compute() {
                            supports_compute = Some(qf);
                        }

                        if supports_graphics.is_some() && supports_compute.is_some() {
                            break;
                        }
                    }

                    if let (Some(graphics_queue), Some(compute_queue)) =
                    (supports_graphics, supports_compute)
                    {
                        Some((p, graphics_queue, compute_queue))
                    } else {
                        None
                    }
                })
                .min_by_key(|p| match p.0.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                })
                .expect("Couldn't find suitable GPU");

            let (device, mut queues) = Device::new(
                phy_dev,
                &Features::none(),
                &phy_dev.required_extensions().union(&dev_ext),
                [(graphics_queue, 0.0), (compute_queue, 1.0)]
                    .iter()
                    .cloned(),
            )
                .expect("Couldn't make the device and queues");

            ctx.device = Some(device);

            ctx.graphics_queue = Some(queues.next().unwrap());
            ctx.compute_queue = Some(queues.next().unwrap());
        }

        {
            let _span = tracy_client::span!("Create Swapchain");
            // Make the swapchain

            let (swapchain, images) = {
                // Capabilities
                let caps = ctx.surface.as_ref().unwrap().capabilities(ctx.device.as_ref().unwrap().clone().physical_device()).unwrap();
                // Alpha
                let alpha = caps.supported_composite_alpha.iter().next().unwrap();

                // The format the images will have
                let format = caps.supported_formats[0].0;

                let dimensions: [u32; 2] = ctx.surface.as_ref().unwrap().window().inner_size().into();

                Swapchain::start(
                    ctx.device.as_ref().unwrap().clone(),
                    ctx.surface.as_ref().unwrap().clone(),
                )
                    .num_images(caps.min_image_count)
                    .format(format)
                    .dimensions(dimensions)
                    .usage(ImageUsage::color_attachment())
                    .sharing_mode(ctx.graphics_queue.as_ref().unwrap())
                    .composite_alpha(alpha)
                    .build()
                    .unwrap()
            };

            ctx.swapchain = Some(swapchain);
            ctx.images = Some(images);
        }

        {
            let _span = tracy_client::span!("Load shaders");
            ctx.vert_shader =
                Some(vert_shader::Shader::load(ctx.device.as_ref().unwrap().clone()).unwrap());
            ctx.frag_shader =
                Some(frag_shader::Shader::load(ctx.device.as_ref().unwrap().clone()).unwrap());
            ctx.comp_shader =
                Some(compute_shader::Shader::load(ctx.device.as_ref().unwrap().clone()).unwrap());
        }

        {
            let _span = tracy_client::span!("Create compute pipeline");
            ctx.comp_pipeline = Some(Arc::new(
                ComputePipeline::new(
                    ctx.device.as_ref().unwrap().clone(),
                    &ctx.comp_shader.as_ref().unwrap().main_entry_point(),
                    &(),
                    None,
                    |_| {},
                )
                    .expect("Couldn't create compute pipeline"),
            ));
        }

        {
            let _span = tracy_client::span!("Create render pass");
            let render_pass = Arc::new(
                vulkano::single_pass_renderpass!(
                ctx.device.as_ref().unwrap().clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: ctx.swapchain.as_ref().unwrap().format(),
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {}
                }
            )
                    .unwrap(),
            );

            ctx.render_pass = Some(render_pass);
        }

        {
            let _span = tracy_client::span!("Create graphics pipeline");
            let graphics_pipeline = Arc::new(
                GraphicsPipeline::start()
                    .vertex_input_single_buffer::<Vertex>()
                    .vertex_shader(ctx.vert_shader.as_ref().unwrap().main_entry_point(), ())
                    .fragment_shader(ctx.frag_shader.as_ref().unwrap().main_entry_point(), ())
                    .viewports([ctx.viewport.as_ref().clone()])
                    .render_pass(Subpass::from(ctx.render_pass.as_ref().unwrap().clone(), 0).unwrap())
                    .point_list()
                    .build(ctx.device.as_ref().unwrap().clone())
                    .expect("Couldn't build graphics pipeline"),
            );
            ctx.graphics_pipeline = Some(graphics_pipeline);
        }

        {
            let _span = tracy_client::span!("Get framebuffers");
            ctx.framebuffers = Some(window_size_dependent_setup(
                &ctx.images.as_ref().unwrap(),
                ctx.render_pass.as_ref().unwrap().clone(),
                &mut ctx.viewport.as_ref().clone(),
            ));
        }
        {
            let _span = tracy_client::span!("Create buffers");
            {
                let _span = tracy_client::span!("Create person buffer");
                ctx.people_buf = Some(
                    CpuAccessibleBuffer::from_iter(
                        ctx.device.as_ref().unwrap().clone(),
                        BufferUsage::all(),
                        false,
                        people.iter().cloned(),
                    )
                        .expect("Couldn't create people buffer"),
                );
            }
            {
                let _span = tracy_client::span!("Create vertex buffer");
                ctx.vertex_buf = Some(
                    DeviceLocalBuffer::array(
                        ctx.device.as_ref().unwrap().clone(),
                        (people.len() as usize * std::mem::size_of::<Vertex>()) as u64,
                        BufferUsage::vertex_buffer(),
                        [
                            ctx.compute_queue.as_ref().unwrap().family(),
                            ctx.graphics_queue.as_ref().unwrap().family(),
                        ].iter().cloned()
                    ).expect("Couldn't create vertex buffer"),
                );
            }
        }

        ctx.previous_frame_end = Some(sync::now(ctx.device.as_ref().unwrap().clone()).boxed());

        ctx
    }

    pub(crate) fn render(&mut self) -> Result<(), String> {

        let mut result = Ok(());

        self.event_loop
            .run_return(|event, _, control_flow| match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    *control_flow = ControlFlow::Exit;
                    result = Err("Exit requested".to_string());
                }
                Event::RedrawEventsCleared => {
                    self.previous_frame_end.as_mut().unwrap().cleanup_finished();

                    if self.recreate_swapchain {
                        {
                            let _span = tracy_client::span!("Recreate swapchain");
                            let dimensions: [u32; 2] =
                                self.surface.as_ref().unwrap().window().inner_size().into();
                            let (new_swapchain, new_images) = match self
                                .swapchain
                                .as_ref()
                                .unwrap()
                                .recreate()
                                .dimensions(dimensions)
                                .build()
                            {
                                Ok(r) => r,
                                Err(SwapchainCreationError::UnsupportedDimensions) => return,
                                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                            };

                            self.swapchain = Some(new_swapchain);
                            self.framebuffers = Some(window_size_dependent_setup(
                                &new_images,
                                self.render_pass.as_ref().unwrap().clone(),
                                &mut self.viewport.as_ref().clone(),
                            ));
                            self.recreate_swapchain = false;
                        }
                    }

                    let (image_num, suboptimal, acquire_future) = {
                        let _span = tracy_client::span!("Get next image");
                        match swapchain::acquire_next_image(
                            self.swapchain.as_ref().unwrap().clone(),
                            None,
                        ) {
                            Ok(r) => r,
                            Err(AcquireError::OutOfDate) => {
                                self.recreate_swapchain = true;
                                return;
                            }
                            Err(e) => panic!("Failed to acquire next image: {:?}", e),
                        }};
                    if suboptimal {
                        self.recreate_swapchain = true;
                    }

                    let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into()];
                    let mut builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>;

                    {
                        let _span = tracy_client::span!("Create CommandBufferBuilder");

                        builder = AutoCommandBufferBuilder::primary(
                            self.device.as_ref().unwrap().clone(),
                            self.graphics_queue.as_ref().unwrap().family(),
                            CommandBufferUsage::OneTimeSubmit,
                        )
                            .expect("Couldn't create the Command Buffer Builder");
                    }

                    {
                        let _span = tracy_client::span!("Run render pass");
                        builder
                            .begin_render_pass(
                                self.framebuffers.as_ref().unwrap()[image_num].clone(),
                                SubpassContents::Inline,
                                clear_values,
                            )
                            .unwrap()
                            .set_viewport(0, [self.viewport.as_ref().clone()])
                            .bind_pipeline_graphics(self.graphics_pipeline.as_ref().unwrap().clone())
                            .bind_vertex_buffers(0, self.vertex_buf.as_ref().unwrap().clone())
                            .draw(self.vertex_buf.as_ref().unwrap().len() as u32, 1, 0, 0)
                            .unwrap()
                            .end_render_pass()
                            .unwrap();

                        let command_buffer = builder.build().unwrap();

                        let future = self
                            .previous_frame_end
                            .take()
                            .unwrap()
                            .join(acquire_future)
                            .then_execute(
                                self.graphics_queue.as_ref().unwrap().clone(),
                                command_buffer,
                            )
                            .unwrap()
                            .then_swapchain_present(
                                self.graphics_queue.as_ref().unwrap().clone(),
                                self.swapchain.as_ref().unwrap().clone(),
                                image_num,
                            )
                            .then_signal_fence_and_flush();

                        match future {
                            Ok(future) => {
                                self.previous_frame_end = Some(future.boxed());
                            }
                            Err(FlushError::OutOfDate) => {
                                self.recreate_swapchain = true;
                                self.previous_frame_end =
                                    Some(sync::now(self.device.as_ref().unwrap().clone()).boxed());
                            }
                            Err(e) => {
                                println!("Failed to flush future: {:?}", e);
                                self.previous_frame_end =
                                    Some(sync::now(self.device.as_ref().unwrap().clone()).boxed());
                            }
                        }
                    }
                }
                _ => (),
            });
        result
    }
}
