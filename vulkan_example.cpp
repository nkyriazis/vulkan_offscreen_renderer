#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <bitset>
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace vkx
{
template <typename T>
using handle = std::shared_ptr<T>;

template <typename T, typename Deleter>
handle<T> make_handle(const T &t, Deleter deleter)
{
    return handle<T>(new T(t), [=](const T *ptr) {
        deleter(*ptr);
        delete ptr;
    });
}

using instance                  = handle<vk::Instance>;
using debug_report_callback_ext = handle<vk::DebugReportCallbackEXT>;
using physical_device           = handle<vk::PhysicalDevice>;
using device                    = handle<vk::Device>;
using command_pool              = handle<vk::CommandPool>;
using queue                     = handle<vk::Queue>;
using command_buffer            = handle<vk::CommandBuffer>;
using shader_module             = handle<vk::ShaderModule>;
using descriptor_set_layout     = handle<vk::DescriptorSetLayout>;
using buffer                    = handle<vk::Buffer>;
using device_memory             = handle<vk::DeviceMemory>;
using image                     = handle<vk::Image>;
using image_view                = handle<vk::ImageView>;
using pipeline_layout           = handle<vk::PipelineLayout>;
using pipeline                  = handle<vk::Pipeline>;
using render_pass               = handle<vk::RenderPass>;
using frame_buffer              = handle<vk::Framebuffer>;
using descriptor_pool           = handle<vk::DescriptorPool>;
using descriptor_set            = handle<vk::DescriptorSet>;

VkBool32 VKAPI_PTR log(VkDebugReportFlagsEXT      flags,
                       VkDebugReportObjectTypeEXT object_type, uint64_t object,
                       size_t location, int32_t messageCode,
                       const char *pLayerPrefix, const char *pMessage,
                       void *pUserData)
{
    auto flags_       = static_cast<vk::DebugReportFlagBitsEXT>(flags);
    auto object_type_ = static_cast<vk::DebugReportObjectTypeEXT>(object_type);

    std::ostream &out =
        flags == VK_DEBUG_REPORT_ERROR_BIT_EXT ? std::cerr : std::cout;
    out << vk::to_string(flags_) << " : " << vk::to_string(object_type_)
        << " : " << pLayerPrefix << " : " << pMessage
        << /*" : object " << object
        << " : location : " << location << " : messageCode : " << messageCode
        <<*/ std::endl;
    out.flush();
    return flags_ == decltype(flags_)::eError ? VK_TRUE : VK_FALSE;
}

auto load_binary_file(const std::string &filename)
{
    std::ifstream ifs(filename.c_str(), std::ios::binary | std::ios::ate);
    std::ifstream::pos_type pos = ifs.tellg();

    std::vector<char> result(pos);

    ifs.seekg(0, std::ios::beg);
    ifs.read(result.data(), pos);

    return result;
};

size_t find_memory_index(const vk::PhysicalDeviceMemoryProperties &mem_caps,
                         const std::bitset<16> & resource_type,
                         vk::MemoryPropertyFlags mem_flags)
{
    for (size_t i = 0; i < mem_caps.memoryTypeCount; ++i)
    {
        if (resource_type[i] &&
            (mem_caps.memoryTypes[i].propertyFlags & mem_flags) == mem_flags)
            return i;
    }
    throw std::runtime_error("could not find an appropriate memory index");
}

auto get_memory_requirements(const device &dev, const buffer &b)
{
    return dev->getBufferMemoryRequirements(*b);
}

auto get_memory_requirements(const device &dev, const image &i)
{
    return dev->getImageMemoryRequirements(*i);
}

template <typename Resource>
vkx::device_memory
allocate(const device &dev, const vk::PhysicalDeviceMemoryProperties &mem_caps,
         const Resource &resource, const vk::MemoryPropertyFlags &mem_props)
{
    auto memory_requirements = get_memory_requirements(dev, resource);
    auto memory_index        = find_memory_index(
        mem_caps, memory_requirements.memoryTypeBits, mem_props);

    vk::MemoryAllocateInfo memory_allocate_info;
    {
        memory_allocate_info.setAllocationSize(memory_requirements.size)
            .setMemoryTypeIndex(uint32_t(memory_index));
    }
    return make_handle(
        dev->allocateMemory(memory_allocate_info), [device = dev](auto mem) {
            device->freeMemory(mem);
        });
}

void begin(const command_buffer &cb, bool single_time = false)
{
    vk::CommandBufferBeginInfo command_buffer_begin_info;
    if (single_time)
        command_buffer_begin_info.setFlags(
            vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cb->begin(command_buffer_begin_info);
}

void end(const command_buffer &cb) { cb->end(); }

void submit(const queue &q, const command_buffer &cb, bool wait = false)
{
    vk::SubmitInfo submit_info;
    submit_info.setCommandBufferCount(1).setPCommandBuffers(&*cb);
    q->submit({submit_info}, vk::Fence());
    if (wait)
        q->waitIdle();
}

void copy(const queue &q, const command_buffer &cb, const buffer &from,
          const buffer &to, size_t size)
{
    begin(cb, true);

    vk::BufferCopy buffer_copy;
    buffer_copy.setDstOffset(0).setSize(size).setSrcOffset(0);
    cb->copyBuffer(*from, *to, {buffer_copy});

    end(cb);
    submit(q, cb, true);
}

void copy(const device &dev, const device_memory &mem, const void *data,
          size_t size)
{
    void *write = dev->mapMemory(*mem, 0, size);
    std::memcpy(write, data, size);
    dev->unmapMemory(*mem);
}

buffer create_buffer(const device &dev, const queue &q,
                     const command_buffer &                    cb,
                     const vk::PhysicalDeviceMemoryProperties &mem_caps,
                     vk::BufferUsageFlags flags, const void *data, size_t size)
{
    vk::BufferCreateInfo staging_buffer_create_info;
    staging_buffer_create_info.setSharingMode(vk::SharingMode::eExclusive)
        .setSize(size)
        .setUsage(vk::BufferUsageFlagBits::eTransferSrc);
    buffer staging_buffer = make_handle(
        dev->createBuffer(staging_buffer_create_info), [device =
                                                            dev](auto buffer) {
            device->destroyBuffer(buffer);
        });

    device_memory staging_memory =
        allocate(dev, mem_caps, staging_buffer,
                 vk::MemoryPropertyFlagBits::eHostCoherent |
                     vk::MemoryPropertyFlagBits::eHostVisible);

    dev->bindBufferMemory(*staging_buffer, *staging_memory, 0);

    copy(dev, staging_memory, data, size);

    vk::BufferCreateInfo buffer_create_info;
    buffer_create_info.setSharingMode(vk::SharingMode::eExclusive)
        .setSize(size)
        .setUsage(flags | vk::BufferUsageFlagBits::eTransferDst);
    vk::Buffer buffer = dev->createBuffer(buffer_create_info);

    device_memory device_memory =
        allocate(dev, mem_caps, make_handle(buffer, [](auto) {}),
                 vk::MemoryPropertyFlagBits::eDeviceLocal);

    dev->bindBufferMemory(buffer, *device_memory, 0);

    copy(q, cb, staging_buffer, make_handle(buffer, [](auto) {}), size);

    return vkx::make_handle(buffer, [ device = dev, device_memory ](auto b) {
        device->destroyBuffer(b);
    });
}
}

int main(int argc, char **argv)
{
    try
    {
        ////////////////////////////////////////////////////////////////
        //  Instance

        auto extensions = []() {
            static const std::array<const char *, 1> extensions = {
                VK_EXT_DEBUG_REPORT_EXTENSION_NAME};
            return extensions;
        }();

        vk::InstanceCreateInfo instanceCreateInfo;
        instanceCreateInfo.setEnabledExtensionCount(uint32_t(extensions.size()))
            .setPpEnabledExtensionNames(extensions.data());

        auto layers = []() {
            static const std::array<const char *, 1> layers = {
                "VK_LAYER_LUNARG_standard_validation",
                //"VK_LAYER_LUNARG_api_dump"
            };
            return layers;
        }();

        instanceCreateInfo.setEnabledLayerCount(uint32_t(layers.size()))
            .setPpEnabledLayerNames(layers.data());

        vkx::instance instance =
            vkx::make_handle(vk::createInstance(instanceCreateInfo),
                             [](auto instance) { instance.destroy(); });

        ////////////////////////////////////////////////////////////////
        //  Debugging callback
        vk::DebugReportCallbackCreateInfoEXT dInfo;
        dInfo
            .setFlags(vk::DebugReportFlagBitsEXT::eDebug |
                      vk::DebugReportFlagBitsEXT::eError |
                      vk::DebugReportFlagBitsEXT::eInformation |
                      vk::DebugReportFlagBitsEXT::ePerformanceWarning |
                      vk::DebugReportFlagBitsEXT::eWarning)
            .setPfnCallback(vkx::log);

        auto vkCreateDebugReportCallbackEXT =
            (PFN_vkCreateDebugReportCallbackEXT)instance->getProcAddr(
                "vkCreateDebugReportCallbackEXT");
        vk::DebugReportCallbackEXT callback;
        vk::Result                 result =
            static_cast<vk::Result>(vkCreateDebugReportCallbackEXT(
                *instance,
                reinterpret_cast<const VkDebugReportCallbackCreateInfoEXT *>(
                    &dInfo),
                nullptr,
                reinterpret_cast<VkDebugReportCallbackEXT *>(&callback)));
        vkx::debug_report_callback_ext debug_report_callback_ext =
            vkx::make_handle(
                vk::createResultValue(
                    result, callback,
                    "vk::Instance::createDebugReportCallbackEXT"),
                [instance](auto callback) {
                    auto vkDestroyDebugReportCallbackEXT =
                        (PFN_vkDestroyDebugReportCallbackEXT)instance
                            ->getProcAddr("vkDestroyDebugReportCallbackEXT");
                    vkDestroyDebugReportCallbackEXT(
                        *instance,
                        static_cast<VkDebugReportCallbackEXT>(callback),
                        nullptr);
                });

        ////////////////////////////////////////////////////////////////
        //  Logical device
        auto devices = instance->enumeratePhysicalDevices();
        auto physical_device =
            vkx::make_handle(devices.front(), [instance](auto) {});
        auto mem_caps = physical_device->getMemoryProperties();

        auto queue_families = physical_device->getQueueFamilyProperties();
        auto graphics_transfer_family = std::find_if(
            queue_families.begin(), queue_families.end(),
            [](const auto &props) {
                return props.queueFlags & (vk::QueueFlagBits::eGraphics |
                                           vk::QueueFlagBits::eTransfer);
            });

        if (graphics_transfer_family == queue_families.end())
            throw std::runtime_error("could not find queue that supports both "
                                     "graphics and transfers");
        auto graphics_transfer_family_index =
            std::distance(queue_families.begin(), graphics_transfer_family);

        vk::DeviceQueueCreateInfo device_queue_create_info;
        device_queue_create_info
            .setQueueFamilyIndex(uint32_t(graphics_transfer_family_index))
            .setQueueCount(1)
            .setPQueuePriorities([]() {
                static const float priorities[] = {1.0f};
                return priorities;
            }());

        vk::PhysicalDeviceFeatures physical_device_features;
        vk::DeviceCreateInfo       device_info;
        device_info.setQueueCreateInfoCount(1)
            .setPQueueCreateInfos(&device_queue_create_info)
            .setPEnabledFeatures(&physical_device_features);
        vkx::device device = vkx::make_handle(
            physical_device->createDevice(device_info),
            [instance, physical_device](auto device) { device.destroy(); });

        ////////////////////////////////////////////////////////////////
        //  Command pool
        vk::CommandPoolCreateInfo command_pool_create_info;
        command_pool_create_info
            .setQueueFamilyIndex(uint32_t(graphics_transfer_family_index))
            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        vkx::command_pool command_pool = vkx::make_handle(
            device->createCommandPool(command_pool_create_info),
            [device](auto pool) { device->destroyCommandPool(pool); });

        ////////////////////////////////////////////////////////////////
        //  Queue
        vkx::queue queue = vkx::make_handle(
            device->getQueue(uint32_t(graphics_transfer_family_index), 0),
            [device](auto) {});

        ////////////////////////////////////////////////////////////////
        //  Command buffer
        vk::CommandBufferAllocateInfo commandBufferAllocateInfo;
        commandBufferAllocateInfo.setCommandPool(*command_pool)
            .setLevel(vk::CommandBufferLevel::ePrimary)
            .setCommandBufferCount(1);
        vkx::command_buffer command_buffer = vkx::make_handle(
            device->allocateCommandBuffers(commandBufferAllocateInfo)[0],
            [device, command_pool](auto cb) {
                device->freeCommandBuffers(*command_pool, 1, &cb);
            });

        ////////////////////////////////////////////////////////////////
        //  Shaders
        auto vertex_shader_code =
            vkx::load_binary_file("media/shader.vert.spv");
        vk::ShaderModuleCreateInfo shader_module_create_info_vert;
        shader_module_create_info_vert.setCodeSize(vertex_shader_code.size())
            .setPCode(
                reinterpret_cast<const uint32_t *>(vertex_shader_code.data()));

        vkx::shader_module vertex_shader = vkx::make_handle(
            device->createShaderModule(shader_module_create_info_vert),
            [device](auto shader) { device->destroyShaderModule(shader); });

        auto fragment_shader_code =
            vkx::load_binary_file("media/shader.frag.spv");
        vk::ShaderModuleCreateInfo shader_module_create_info_frag;
        shader_module_create_info_frag.setCodeSize(fragment_shader_code.size())
            .setPCode(reinterpret_cast<const uint32_t *>(
                fragment_shader_code.data()));

        vkx::shader_module fragment_shader = vkx::make_handle(
            device->createShaderModule(shader_module_create_info_frag),
            [device](auto shader) { device->destroyShaderModule(shader); });

        ////////////////////////////////////////////////////////////////
        //  Color/Depth attachments
        vk::ImageCreateInfo image_create_info_positions;
        image_create_info_positions.setArrayLayers(1)
            .setExtent(vk::Extent3D(512, 512, 1))
            .setFormat(vk::Format::eR32G32B32A32Sfloat)
            .setImageType(vk::ImageType::e2D)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setMipLevels(1)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setTiling(vk::ImageTiling::eOptimal)
            .setUsage(vk::ImageUsageFlagBits::eColorAttachment |
                      vk::ImageUsageFlagBits::eTransferSrc);

        vkx::image color_attachment =
            vkx::make_handle(device->createImage(image_create_info_positions),
                             [device](auto img) { device->destroyImage(img); });

        vkx::device_memory positions_attachment_memory =
            vkx::allocate(device, mem_caps, color_attachment,
                          vk::MemoryPropertyFlagBits::eDeviceLocal);
        device->bindImageMemory(*color_attachment, *positions_attachment_memory,
                                0);

        vk::ImageSubresourceRange image_subresource_range_positions;
        image_subresource_range_positions
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseArrayLayer(0)
            .setBaseMipLevel(0)
            .setLayerCount(1)
            .setLevelCount(1);

        vk::ImageViewCreateInfo image_view_create_info_positions;
        image_view_create_info_positions
            .setFormat(image_create_info_positions.format)
            .setImage(*color_attachment)
            .setViewType(vk::ImageViewType::e2D)
            .setSubresourceRange(image_subresource_range_positions);

        vkx::image_view image_view_positions = vkx::make_handle(
            device->createImageView(image_view_create_info_positions),
            [device](auto iv) { device->destroyImageView(iv); });

        ////////////////////////////////////////////////////////////////
        //  Pipeline

        vk::PipelineLayoutCreateInfo pipeline_layout_create_info;

        vkx::pipeline_layout pipeline_layout = vkx::make_handle(
            device->createPipelineLayout(pipeline_layout_create_info),
            [device](auto pl) { device->destroyPipelineLayout(pl); });

        std::array<vk::PipelineShaderStageCreateInfo, 2>
            pipeline_shader_stage_create_infos;
        pipeline_shader_stage_create_infos[0]
            .setStage(vk::ShaderStageFlagBits::eVertex)
            .setModule(*vertex_shader)
            .setPName("main");
        pipeline_shader_stage_create_infos[1]
            .setStage(vk::ShaderStageFlagBits::eFragment)
            .setModule(*fragment_shader)
            .setPName("main");

        vk::PipelineVertexInputStateCreateInfo
            pipeline_vertex_input_state_create_info;

        vk::PipelineInputAssemblyStateCreateInfo
            pipeline_input_assembly_state_create_info;
        pipeline_input_assembly_state_create_info
            .setTopology(vk::PrimitiveTopology::eTriangleList)
            .setPrimitiveRestartEnable(VK_FALSE);

        vk::Viewport viewport;
        viewport.setX(0)
            .setY(0)
            .setWidth(512)
            .setHeight(512)
            .setMinDepth(0.0f)
            .setMaxDepth(1.0f);
        vk::Rect2D scissor;
        scissor.setOffset(vk::Offset2D(0, 0)).setExtent(vk::Extent2D(512, 512));
        vk::PipelineViewportStateCreateInfo pipeline_viewport_state_create_info;
        pipeline_viewport_state_create_info.setViewportCount(1)
            .setPViewports(&viewport)
            .setScissorCount(1)
            .setPScissors(&scissor);

        vk::PipelineRasterizationStateCreateInfo
            pipeline_rasterization_state_create_info;
        pipeline_rasterization_state_create_info.setDepthClampEnable(VK_FALSE)
            .setRasterizerDiscardEnable(VK_FALSE)
            .setPolygonMode(vk::PolygonMode::eFill)
            .setLineWidth(1.0f)
            .setCullMode(vk::CullModeFlagBits::eBack)
            .setFrontFace(vk::FrontFace::eClockwise)
            .setDepthBiasEnable(VK_FALSE);

        vk::PipelineMultisampleStateCreateInfo
            pipeline_multisample_state_create_info;
        pipeline_multisample_state_create_info.setSampleShadingEnable(VK_FALSE)
            .setRasterizationSamples(vk::SampleCountFlagBits::e1);

        vk::PipelineColorBlendAttachmentState
            pipeline_color_blend_attachment_state;
        pipeline_color_blend_attachment_state.setBlendEnable(VK_FALSE)
            .setColorWriteMask(vk::ColorComponentFlagBits::eA |
                               vk::ColorComponentFlagBits::eR |
                               vk::ColorComponentFlagBits::eG |
                               vk::ColorComponentFlagBits::eB);

        vk::PipelineColorBlendStateCreateInfo
            pipeline_color_blend_state_create_info;
        pipeline_color_blend_state_create_info.setLogicOpEnable(VK_FALSE)
            .setLogicOp(vk::LogicOp::eCopy)
            .setAttachmentCount(1)
            .setPAttachments(&pipeline_color_blend_attachment_state)
            .setBlendConstants(std::array<float, 4>{0, 0, 0, 0});

        vk::AttachmentDescription attachment_description_position_rt;
        attachment_description_position_rt
            .setFormat(vk::Format::eR32G32B32A32Sfloat)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

        vk::AttachmentReference attachment_reference_position_rt;
        attachment_reference_position_rt.setAttachment(0).setLayout(
            vk::ImageLayout::eColorAttachmentOptimal);

        vk::SubpassDescription subpass_description;
        subpass_description
            .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
            .setColorAttachmentCount(1)
            .setPColorAttachments(&attachment_reference_position_rt);

        vk::SubpassDependency subpass_dependency;
        subpass_dependency.setSrcSubpass(VK_SUBPASS_EXTERNAL)
            .setDstSubpass(0)
            .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
            .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
            .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead |
                              vk::AccessFlagBits::eColorAttachmentWrite);

        vk::RenderPassCreateInfo render_pass_create_info;
        render_pass_create_info.setAttachmentCount(1)
            .setPAttachments(&attachment_description_position_rt)
            .setSubpassCount(1)
            .setPSubpasses(&subpass_description)
            .setDependencyCount(1)
            .setPDependencies(&subpass_dependency);
        vkx::render_pass render_pass = vkx::make_handle(
            device->createRenderPass(render_pass_create_info),
            [device](auto rp) { device->destroyRenderPass(rp); });

        vk::GraphicsPipelineCreateInfo graphics_pipeline_create_info;
        graphics_pipeline_create_info.setLayout(*pipeline_layout)
            .setPColorBlendState(&pipeline_color_blend_state_create_info)
            .setPInputAssemblyState(&pipeline_input_assembly_state_create_info)
            .setPMultisampleState(&pipeline_multisample_state_create_info)
            .setPRasterizationState(&pipeline_rasterization_state_create_info)
            .setStageCount(uint32_t(pipeline_shader_stage_create_infos.size()))
            .setPStages(pipeline_shader_stage_create_infos.data())
            .setPVertexInputState(&pipeline_vertex_input_state_create_info)
            .setPViewportState(&pipeline_viewport_state_create_info)
            .setRenderPass(*render_pass)
            .setSubpass(0);
        vkx::pipeline pipeline = vkx::make_handle(
            device->createGraphicsPipeline(vk::PipelineCache(),
                                           graphics_pipeline_create_info),
            [device](auto p) { device->destroyPipeline(p); });

        ////////////////////////////////////////////////////////////////
        //  Frame buffer
        std::array<vk::ImageView, 1> image_views = {*image_view_positions};

        vk::FramebufferCreateInfo framebuffer_create_info;
        framebuffer_create_info.setAttachmentCount(uint32_t(image_views.size()))
            .setPAttachments(image_views.data())
            .setLayers(1)
            .setRenderPass(*render_pass)
            .setWidth(512)
            .setHeight(512);

        vkx::frame_buffer frame_buffer = vkx::make_handle(
            device->createFramebuffer(framebuffer_create_info),
            [device](auto fb) { device->destroyFramebuffer(fb); });

        ////////////////////////////////////////////////////////////////
        //  Submit rendering
        vk::CommandBufferBeginInfo command_buffer_begin_info;
        command_buffer_begin_info.setFlags(
            vk::CommandBufferUsageFlagBits::eSimultaneousUse);
        command_buffer->begin(command_buffer_begin_info);
        vk::ClearValue clear_value;
        clear_value.setColor(vk::ClearColorValue());
        vk::RenderPassBeginInfo render_pass_begin_info;
        render_pass_begin_info.setRenderPass(*render_pass)
            .setFramebuffer(*frame_buffer)
            .setRenderArea(vk::Rect2D(vk::Offset2D(), vk::Extent2D(512, 512)))
            .setClearValueCount(1)
            .setPClearValues(&clear_value);

        command_buffer->beginRenderPass(render_pass_begin_info,
                                        vk::SubpassContents::eInline);
        command_buffer->bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
        command_buffer->draw(3, 1, 0, 0);
        command_buffer->endRenderPass();

        command_buffer->end();
        vkx::submit(queue, command_buffer, true);

        vk::BufferCreateInfo buffer_create_info;
        buffer_create_info.setSharingMode(vk::SharingMode::eExclusive)
            .setSize(512 * 512 * sizeof(glm::vec4))
            .setUsage(vk::BufferUsageFlagBits::eTransferDst);
        vkx::buffer position_map =
            vkx::make_handle(device->createBuffer(buffer_create_info),
                             [device](auto b) { device->destroyBuffer(b); });
        vkx::device_memory position_map_memory =
            vkx::allocate(device, mem_caps, position_map,
                          vk::MemoryPropertyFlagBits::eHostVisible);
        device->bindBufferMemory(*position_map, *position_map_memory, 0);

        vk::ImageSubresourceRange image_subresource_range;
        image_subresource_range.setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseArrayLayer(0)
            .setBaseMipLevel(0)
            .setLayerCount(1)
            .setLevelCount(1);
        vk::ImageMemoryBarrier image_memory_barrier;
        image_memory_barrier
            .setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(*color_attachment)
            .setSubresourceRange(image_subresource_range)
            .setSrcAccessMask(vk::AccessFlagBits::eColorAttachmentRead |
                              vk::AccessFlagBits::eColorAttachmentWrite)
            .setDstAccessMask(vk::AccessFlagBits::eTransferRead);
        vkx::begin(command_buffer);
        command_buffer->pipelineBarrier(
            vk::PipelineStageFlagBits::eBottomOfPipe,
            vk::PipelineStageFlagBits::eBottomOfPipe,
            vk::DependencyFlagBits::eByRegion, {}, {}, {image_memory_barrier});
        vkx::end(command_buffer);
        vkx::submit(queue, command_buffer, true);

        vkx::begin(command_buffer);
        vk::ImageSubresourceLayers image_subresource_layers;
        image_subresource_layers.setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseArrayLayer(0)
            .setLayerCount(1)
            .setMipLevel(0);
        vk::BufferImageCopy buffer_image_copy;
        buffer_image_copy.setBufferOffset(0)
            .setBufferImageHeight(512)
            .setBufferRowLength(512)
            .setImageOffset(vk::Offset3D())
            .setImageExtent(vk::Extent3D(512, 512, 1))
            .setImageSubresource(image_subresource_layers);
        command_buffer->copyImageToBuffer(*color_attachment,
                                          vk::ImageLayout::eTransferSrcOptimal,
                                          *position_map, {buffer_image_copy});
        vkx::end(command_buffer);
        vkx::submit(queue, command_buffer, true);

        void *ptr = device->mapMemory(*position_map_memory, 0,
                                      512 * 512 * sizeof(glm::vec4));
        {
            std::ofstream write_image("image.bin", std::ios::binary);
            write_image.write(reinterpret_cast<const char *>(ptr),
                              512 * 512 * sizeof(glm::vec4));
            write_image.close();
        }
        device->unmapMemory(*position_map_memory);

    } // try
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
        return -1;
    }

    return 0;
}