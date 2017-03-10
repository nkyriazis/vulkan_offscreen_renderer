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
        << " : " << pLayerPrefix << " : " << pMessage << std::endl;
    out.flush();
    return flags_ == decltype(flags_)::eError ? VK_FALSE : VK_TRUE;
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
    {
        submit_info.setCommandBufferCount(1).setPCommandBuffers(&*cb);
    }
    q->submit({submit_info}, vk::Fence());
    if (wait)
        q->waitIdle();
}

void copy(const queue &q, const command_buffer &cb, const buffer &from,
          const buffer &to, size_t size)
{
    begin(cb, true);

    vk::BufferCopy buffer_copy;
    {
        buffer_copy.setDstOffset(0).setSize(size).setSrcOffset(0);
    }
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

struct push_constants
{
    glm::ivec2 grid;
    glm::vec2  mapDims;
    int        issue;
    int        bonesPerHypothesis;
};

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
        physical_device_features.setGeometryShader(true);
        vk::DeviceCreateInfo device_info;
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
        //  Descriptor pool
        std::array<vk::DescriptorPoolSize, 2> descriptor_pool_sizes;
        descriptor_pool_sizes[0]
            .setType(vk::DescriptorType::eStorageBufferDynamic)
            .setDescriptorCount(1);
        descriptor_pool_sizes[1]
            .setType(vk::DescriptorType::eStorageBufferDynamic)
            .setDescriptorCount(1);
        vk::DescriptorPoolCreateInfo descriptor_pool_create_info;
        descriptor_pool_create_info
            .setPoolSizeCount(uint32_t(descriptor_pool_sizes.size()))
            .setPPoolSizes(descriptor_pool_sizes.data())
            .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
            .setMaxSets(1);
        vkx::descriptor_pool descriptor_pool = vkx::make_handle(
            device->createDescriptorPool(descriptor_pool_create_info),
            [device](auto dp) { device->destroyDescriptorPool(dp); });

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
        vk::ShaderModuleCreateInfo shader_module_create_info_vert;
        auto                       vertex_shader_code =
            vkx::load_binary_file("media/shader.vert.spv");
        shader_module_create_info_vert.setCodeSize(vertex_shader_code.size())
            .setPCode(
                reinterpret_cast<const uint32_t *>(vertex_shader_code.data()));

        vk::ShaderModuleCreateInfo shader_module_create_info_frag;
        auto                       fragment_shader_code =
            vkx::load_binary_file("media/shader.frag.spv");
        shader_module_create_info_frag.setCodeSize(fragment_shader_code.size())
            .setPCode(reinterpret_cast<const uint32_t *>(
                fragment_shader_code.data()));

        vk::ShaderModuleCreateInfo shader_module_create_info_geom;
        auto                       geometry_shader_code =
            vkx::load_binary_file("media/shader.geom.spv");
        shader_module_create_info_geom.setCodeSize(geometry_shader_code.size())
            .setPCode(reinterpret_cast<const uint32_t *>(
                geometry_shader_code.data()));

        vkx::shader_module vertex_shader = vkx::make_handle(
            device->createShaderModule(shader_module_create_info_vert),
            [device](auto shader) { device->destroyShaderModule(shader); });
        vkx::shader_module fragment_shader = vkx::make_handle(
            device->createShaderModule(shader_module_create_info_frag),
            [device](auto shader) { device->destroyShaderModule(shader); });
        vkx::shader_module geometry_shader = vkx::make_handle(
            device->createShaderModule(shader_module_create_info_geom),
            [device](auto shader) { device->destroyShaderModule(shader); });

        ////////////////////////////////////////////////////////////////
        //  Descriptor set layout
        vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info;
        std::array<vk::DescriptorSetLayoutBinding, 2> bindings;
        bindings[0]
            .setBinding(0)
            .setDescriptorType(vk::DescriptorType::eStorageBufferDynamic)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eVertex);
        bindings[1]
            .setBinding(1)
            .setDescriptorType(vk::DescriptorType::eStorageBufferDynamic)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eVertex);
        descriptor_set_layout_create_info
            .setBindingCount(uint32_t(bindings.size()))
            .setPBindings(bindings.data());

        vkx::descriptor_set_layout descriptor_set_layout = vkx::make_handle(
            device->createDescriptorSetLayout(
                descriptor_set_layout_create_info),
            [device](auto dsl) { device->destroyDescriptorSetLayout(dsl); });

        ////////////////////////////////////////////////////////////////
        //  3D mesh
        auto create_buffer = [device, queue, command_buffer, mem_caps](
            vk::BufferUsageFlags usage, const void *data, size_t size) {
            return vkx::create_buffer(device, queue, command_buffer, mem_caps,
                                      usage, data, size);
        };

        std::array<glm::vec3, 8> position_data = {
            glm::vec3(-1.0, -1.0, 1.0),  glm::vec3(1.0, -1.0, 1.0),
            glm::vec3(1.0, 1.0, 1.0),    glm::vec3(-1.0, 1.0, 1.0),
            glm::vec3(-1.0, -1.0, -1.0), glm::vec3(1.0, -1.0, -1.0),
            glm::vec3(1.0, 1.0, -1.0),   glm::vec3(-1.0, 1.0, -1.0)};

        vkx::buffer vertex_positions =
            create_buffer(vk::BufferUsageFlagBits::eVertexBuffer,
                          position_data.data(), sizeof(position_data));

        std::array<glm::uvec3, 12> triangle_data = {
            glm::uvec3(0, 1, 2), glm::uvec3(2, 3, 0), glm::uvec3(1, 5, 6),
            glm::uvec3(6, 2, 1), glm::uvec3(7, 6, 5), glm::uvec3(5, 4, 7),
            glm::uvec3(4, 0, 3), glm::uvec3(3, 7, 4), glm::uvec3(4, 5, 1),
            glm::uvec3(1, 0, 4), glm::uvec3(3, 2, 6), glm::uvec3(6, 7, 3)};

        vkx::buffer triangles =
            create_buffer(vk::BufferUsageFlagBits::eIndexBuffer,
                          triangle_data.data(), sizeof(triangle_data));

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
            .setUsage(vk::ImageUsageFlagBits::eColorAttachment);

        vkx::image positions_attachment =
            vkx::make_handle(device->createImage(image_create_info_positions),
                             [device](auto img) { device->destroyImage(img); });

        vkx::device_memory positions_attachment_memory =
            vkx::allocate(device, mem_caps, positions_attachment,
                          vk::MemoryPropertyFlagBits::eDeviceLocal);
        device->bindImageMemory(*positions_attachment,
                                *positions_attachment_memory, 0);

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
            .setImage(*positions_attachment)
            .setViewType(vk::ImageViewType::e2D)
            .setSubresourceRange(image_subresource_range_positions);

        vkx::image_view image_view_positions = vkx::make_handle(
            device->createImageView(image_view_create_info_positions),
            [device](auto iv) { device->destroyImageView(iv); });

        vk::ImageCreateInfo image_create_info_depth;
        image_create_info_depth.setArrayLayers(1)
            .setExtent(vk::Extent3D(512, 512, 1))
            .setFormat(vk::Format::eD32Sfloat)
            .setImageType(vk::ImageType::e2D)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setMipLevels(1)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setTiling(vk::ImageTiling::eOptimal)
            .setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment);

        vkx::image depth_attachment =
            vkx::make_handle(device->createImage(image_create_info_depth),
                             [device](auto img) { device->destroyImage(img); });

        vkx::device_memory depth_attachment_memory =
            vkx::allocate(device, mem_caps, depth_attachment,
                          vk::MemoryPropertyFlagBits::eDeviceLocal);
        device->bindImageMemory(*depth_attachment, *depth_attachment_memory, 0);

        vk::ImageSubresourceRange image_subresource_range_depth;
        image_subresource_range_depth
            .setAspectMask(vk::ImageAspectFlagBits::eDepth)
            .setBaseArrayLayer(0)
            .setBaseMipLevel(0)
            .setLayerCount(1)
            .setLevelCount(1);

        vk::ImageViewCreateInfo image_view_create_info_depth;
        image_view_create_info_depth.setFormat(image_create_info_depth.format)
            .setImage(*depth_attachment)
            .setViewType(vk::ImageViewType::e2D)
            .setSubresourceRange(image_subresource_range_depth);

        vkx::image_view image_view_depth = vkx::make_handle(
            device->createImageView(image_view_create_info_depth),
            [device](auto iv) { device->destroyImageView(iv); });

        ////////////////////////////////////////////////////////////////
        //  Pipeline
        vk::PushConstantRange push_constant_range;
        push_constant_range
            .setStageFlags(vk::ShaderStageFlagBits::eVertex |
                           vk::ShaderStageFlagBits::eFragment)
            .setOffset(0)
            .setSize(sizeof(push_constants));

        vk::PipelineLayoutCreateInfo pipeline_layout_create_info;
        pipeline_layout_create_info.setSetLayoutCount(1)
            .setPSetLayouts(&*descriptor_set_layout)
            .setPushConstantRangeCount(1)
            .setPPushConstantRanges(&push_constant_range);
        vkx::pipeline_layout pipeline_layout = vkx::make_handle(
            device->createPipelineLayout(pipeline_layout_create_info),
            [device](auto pl) { device->destroyPipelineLayout(pl); });

        std::array<vk::PipelineShaderStageCreateInfo, 3>
            pipeline_shader_stage_create_infos;
        pipeline_shader_stage_create_infos[0]
            .setStage(vk::ShaderStageFlagBits::eVertex)
            .setModule(*vertex_shader)
            .setPName("main");
        pipeline_shader_stage_create_infos[1]
            .setStage(vk::ShaderStageFlagBits::eGeometry)
            .setModule(*geometry_shader)
            .setPName("main");
        pipeline_shader_stage_create_infos[2]
            .setStage(vk::ShaderStageFlagBits::eFragment)
            .setModule(*fragment_shader)
            .setPName("main");

        std::array<vk::VertexInputBindingDescription, 3>
            vertex_input_binding_descriptions;
        vertex_input_binding_descriptions[0]
            .setBinding(0)
            .setStride(sizeof(glm::vec3))
            .setInputRate(vk::VertexInputRate::eVertex);
        vertex_input_binding_descriptions[1]
            .setBinding(1)
            .setStride(sizeof(glm::ivec4))
            .setInputRate(vk::VertexInputRate::eInstance);
        vertex_input_binding_descriptions[2]
            .setBinding(2)
            .setStride(sizeof(glm::mat4))
            .setInputRate(vk::VertexInputRate::eInstance);

        std::array<vk::VertexInputAttributeDescription, 6>
            vertex_input_attribute_descriptions;
        vertex_input_attribute_descriptions[0]
            .setLocation(0)
            .setBinding(0)
            .setFormat(vk::Format::eR32G32B32Sfloat)
            .setOffset(0);
        vertex_input_attribute_descriptions[1]
            .setLocation(1)
            .setBinding(1)
            .setFormat(vk::Format::eR32G32B32A32Sint)
            .setOffset(0);
        vertex_input_attribute_descriptions[2]
            .setLocation(2)
            .setBinding(2)
            .setFormat(vk::Format::eR32G32B32Sfloat)
            .setOffset(0);
        vertex_input_attribute_descriptions[3]
            .setLocation(3)
            .setBinding(2)
            .setFormat(vk::Format::eR32G32B32Sfloat)
            .setOffset(sizeof(glm::vec4));
        vertex_input_attribute_descriptions[4]
            .setLocation(4)
            .setBinding(2)
            .setFormat(vk::Format::eR32G32B32Sfloat)
            .setOffset(2 * sizeof(glm::vec4));
        vertex_input_attribute_descriptions[5]
            .setLocation(5)
            .setBinding(2)
            .setFormat(vk::Format::eR32G32B32Sfloat)
            .setOffset(3 * sizeof(glm::vec4));

        vk::PipelineVertexInputStateCreateInfo
            pipeline_vertex_input_state_create_info;
        pipeline_vertex_input_state_create_info
            .setVertexBindingDescriptionCount(
                uint32_t(vertex_input_binding_descriptions.size()))
            .setPVertexBindingDescriptions(
                vertex_input_binding_descriptions.data())
            .setVertexAttributeDescriptionCount(
                uint32_t(vertex_input_attribute_descriptions.size()))
            .setPVertexAttributeDescriptions(
                vertex_input_attribute_descriptions.data());

        vk::PipelineInputAssemblyStateCreateInfo
            pipeline_input_assembly_state_create_info;
        pipeline_input_assembly_state_create_info.setTopology(
            vk::PrimitiveTopology::eTriangleList);

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
        pipeline_rasterization_state_create_info.setLineWidth(1.0f);

        vk::PipelineMultisampleStateCreateInfo
            pipeline_multisample_state_create_info;

        vk::PipelineDepthStencilStateCreateInfo
            pipeline_depth_stencil_state_create_info;
        pipeline_depth_stencil_state_create_info.setDepthCompareOp(
            vk::CompareOp::eLess);

        vk::PipelineColorBlendAttachmentState
            pipeline_color_blend_attachment_state;
        vk::PipelineColorBlendStateCreateInfo
            pipeline_color_blend_state_create_info;
        pipeline_color_blend_state_create_info.setAttachmentCount(1)
            .setPAttachments(&pipeline_color_blend_attachment_state);

        vk::AttachmentDescription attachment_description_position_rt;
        attachment_description_position_rt
            .setFormat(vk::Format::eR32G32B32A32Sfloat)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

        vk::AttachmentReference attachment_reference_position_rt;
        attachment_reference_position_rt.setAttachment(0).setLayout(
            vk::ImageLayout::eColorAttachmentOptimal);

        vk::AttachmentDescription attachment_description_depth;
        attachment_description_depth.setFormat(vk::Format::eD32Sfloat)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

        vk::AttachmentReference attachment_reference_depth;
        attachment_reference_depth.setAttachment(1).setLayout(
            vk::ImageLayout::eDepthStencilAttachmentOptimal);

        std::array<vk::AttachmentReference, 1> color_attachments = {
            attachment_reference_position_rt};

        std::array<vk::AttachmentDescription, 2> attachment_descriptions = {
            attachment_description_position_rt, attachment_description_depth};

        vk::SubpassDescription subpass_description;
        subpass_description
            .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
            .setColorAttachmentCount(uint32_t(color_attachments.size()))
            .setPColorAttachments(color_attachments.data())
            .setPDepthStencilAttachment(&attachment_reference_depth);

        vk::RenderPassCreateInfo render_pass_create_info;
        render_pass_create_info
            .setAttachmentCount(uint32_t(attachment_descriptions.size()))
            .setPAttachments(attachment_descriptions.data())
            .setSubpassCount(1)
            .setPSubpasses(&subpass_description);
        vkx::render_pass render_pass = vkx::make_handle(
            device->createRenderPass(render_pass_create_info),
            [device](auto rp) { device->destroyRenderPass(rp); });

        vk::GraphicsPipelineCreateInfo graphics_pipeline_create_info;
        graphics_pipeline_create_info.setLayout(*pipeline_layout)
            .setPColorBlendState(&pipeline_color_blend_state_create_info)
            .setPDepthStencilState(&pipeline_depth_stencil_state_create_info)
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
        std::array<vk::ImageView, 2> image_views = {*image_view_positions,
                                                    *image_view_depth};

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
        //  World/View/Projection setup
        glm::mat4 projection_matrix =
            glm::perspective(glm::pi<float>() / 3, 1.0f, 0.01f, 5.0f);

        vkx::buffer projection_matrices =
            create_buffer(vk::BufferUsageFlagBits::eStorageBuffer,
                          &projection_matrix, sizeof(projection_matrix));

        glm::mat4 view_matrix =
            glm::inverse(glm::translate(glm::mat4(), glm::vec3(0, 0, -2)));

        vkx::buffer view_matrices =
            create_buffer(vk::BufferUsageFlagBits::eStorageBuffer, &view_matrix,
                          sizeof(view_matrix));

        glm::mat4 world_matrix;

        vkx::buffer world_matrices =
            create_buffer(vk::BufferUsageFlagBits::eVertexBuffer, &world_matrix,
                          sizeof(world_matrix));

        glm::ivec4 instance_index = {0, 0, 0, 0};

        vkx::buffer instance_indices =
            create_buffer(vk::BufferUsageFlagBits::eVertexBuffer,
                          &instance_index, sizeof(instance_index));

        ////////////////////////////////////////////////////////////////
        //  Submit rendering
        vkx::begin(command_buffer, true);

        command_buffer->bindPipeline(vk::PipelineBindPoint::eGraphics,
                                     *pipeline);
        command_buffer->bindVertexBuffers(
            0, {*vertex_positions, *instance_indices, *world_matrices},
            {0, 0, 0});
        command_buffer->bindIndexBuffer(*triangles, 0, vk::IndexType::eUint32);

        vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo;
        descriptorSetAllocateInfo.setDescriptorPool(*descriptor_pool)
            .setDescriptorSetCount(1)
            .setPSetLayouts(&*descriptor_set_layout);
        vkx::descriptor_set descriptor_set = vkx::make_handle(
            device->allocateDescriptorSets(descriptorSetAllocateInfo)[0],
            [device, descriptor_pool](auto ds) {
                device->freeDescriptorSets(*descriptor_pool, {ds});
            });

        std::array<vk::DescriptorBufferInfo, 2> descriptor_buffer_infos;
        descriptor_buffer_infos[0]
            .setBuffer(*view_matrices)
            .setOffset(0)
            .setRange(VK_WHOLE_SIZE);
        descriptor_buffer_infos[1]
            .setBuffer(*projection_matrices)
            .setOffset(0)
            .setRange(VK_WHOLE_SIZE);

        std::array<vk::WriteDescriptorSet, 2> write_descriptor_sets;
        write_descriptor_sets[0]
            .setDstSet(*descriptor_set)
            .setDstBinding(0)
            .setDstArrayElement(0)
            .setDescriptorType(vk::DescriptorType::eStorageBufferDynamic)
            .setDescriptorCount(1)
            .setPBufferInfo(&descriptor_buffer_infos[0]);
        write_descriptor_sets[1]
            .setDstSet(*descriptor_set)
            .setDstBinding(1)
            .setDstArrayElement(0)
            .setDescriptorType(vk::DescriptorType::eStorageBufferDynamic)
            .setDescriptorCount(1)
            .setPBufferInfo(&descriptor_buffer_infos[1]);
        device->updateDescriptorSets(write_descriptor_sets, {});

        std::array<vk::ClearValue, 2> clear_values;
        clear_values[0].setColor(vk::ClearColorValue());
        clear_values[1].setDepthStencil(vk::ClearDepthStencilValue(1.0f));

        vk::RenderPassBeginInfo render_pass_begin_info;
        render_pass_begin_info.setClearValueCount(uint32_t(clear_values.size()))
            .setPClearValues(clear_values.data())
            .setFramebuffer(*frame_buffer)
            .setRenderPass(*render_pass)
            .setRenderArea(vk::Rect2D(vk::Offset2D(), vk::Extent2D(512, 512)));
        command_buffer->beginRenderPass(render_pass_begin_info,
                                        vk::SubpassContents::eInline);

        command_buffer->endRenderPass();

        vkx::end(command_buffer);
        vkx::submit(queue, command_buffer, true);

    } // try
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
        return -1;
    }

    return 0;
}