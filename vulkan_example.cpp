#include <memory>
#include <iostream>
#include <vulkan/vulkan.hpp>

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
using device                    = handle<vk::Device>;
using command_pool              = handle<vk::CommandPool>;
using queue                     = handle<vk::Queue>;
using command_buffer            = handle<vk::CommandBuffer>;

VkBool32 VKAPI_PTR log(VkDebugReportFlagsEXT      flags,
                       VkDebugReportObjectTypeEXT objectType_, uint64_t object,
                       size_t location, int32_t messageCode,
                       const char *pLayerPrefix, const char *pMessage,
                       void *pUserData)
{
    std::cout << pLayerPrefix << " : " << pMessage << std::endl;
    return flags == VK_DEBUG_REPORT_ERROR_BIT_EXT ? VK_FALSE : VK_TRUE;
}
}

int main(int argc, char **argv)
{
    try
    {
        ////////////////////////////////////////////////////////////////
        //  Instance
        vk::InstanceCreateInfo instanceCreateInfo;
        {
            static const std::array<const char *, 1> extensions = {
                VK_EXT_DEBUG_REPORT_EXTENSION_NAME};
            instanceCreateInfo.setEnabledExtensionCount(extensions.size())
                .setPpEnabledExtensionNames(extensions.data());

            static const std::array<const char *, 1> layers = {
                "VK_LAYER_LUNARG_standard_validation"};
            instanceCreateInfo.setEnabledLayerCount(layers.size())
                .setPpEnabledLayerNames(layers.data());
        }

        vkx::instance instance =
            vkx::make_handle(vk::createInstance(instanceCreateInfo),
                             [](auto instance) { instance.destroy(); });

        ////////////////////////////////////////////////////////////////
        //  Debugging callback
        vk::DebugReportCallbackCreateInfoEXT dInfo;
        {
            using Flags = vk::DebugReportFlagBitsEXT;
            dInfo.setFlags(Flags::eDebug | Flags::eError | Flags::eInformation |
                           Flags::ePerformanceWarning | Flags::eWarning);
            dInfo.setPfnCallback(vkx::log);
        }

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
        auto devices         = instance->enumeratePhysicalDevices();
        auto physical_device = devices.front();

        auto queue_families = physical_device.getQueueFamilyProperties();
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
        {
            static const float priorities[] = {1.0f};
            device_queue_create_info
                .setQueueFamilyIndex(graphics_transfer_family_index)
                .setQueueCount(1)
                .setPQueuePriorities(priorities);
        }
        vk::PhysicalDeviceFeatures physical_device_features;
        {
            physical_device_features.setGeometryShader(true);
        }
        vk::DeviceCreateInfo device_info;
        {
            device_info.setQueueCreateInfoCount(1)
                .setPQueueCreateInfos(&device_queue_create_info)
                .setPEnabledFeatures(&physical_device_features);
        }
        vkx::device device = vkx::make_handle(
            physical_device.createDevice(device_info),
            [instance, physical_device](auto device) { device.destroy(); });

        ////////////////////////////////////////////////////////////////
        //  Command pool
        vk::CommandPoolCreateInfo command_pool_create_info;
        {
            command_pool_create_info
                .setQueueFamilyIndex(graphics_transfer_family_index)
                .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        }
        vkx::command_pool command_pool = vkx::make_handle(
            device->createCommandPool(command_pool_create_info),
            [device](auto pool) { device->destroyCommandPool(pool); });

        ////////////////////////////////////////////////////////////////
        //  Queue
        vkx::queue queue = vkx::make_handle(
            device->getQueue(graphics_transfer_family_index, 0),
            [device](auto) {});

        ////////////////////////////////////////////////////////////////
        //  Command buffer
        vk::CommandBufferAllocateInfo commandBufferAllocateInfo;
        {
            commandBufferAllocateInfo.setCommandPool(*command_pool)
                .setLevel(vk::CommandBufferLevel::ePrimary)
                .setCommandBufferCount(1);
        }
        vkx::command_buffer command_buffer = vkx::make_handle(
            device->allocateCommandBuffers(commandBufferAllocateInfo)[0],
            [device, command_pool](auto cb) {
                device->freeCommandBuffers(*command_pool, 1, &cb);
            });

    } // try
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
        return -1;
    }

    return 0;
}