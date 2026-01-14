#include "OpenCLProgram.hpp"
#include "ExecutionDevice.hpp"
#include "Utility.hpp"
#include <FAST/Data/Tensor.hpp>

namespace fast {

void OpenCLProgram::setName(std::string name) {
    mName = name;
}

std::string OpenCLProgram::getName() const {
    return mName;
}

void OpenCLProgram::setSourceFilename(std::string filename) {
    mSourceFilename = filename;
}

std::string OpenCLProgram::getSourceFilename() const {
    return mSourceFilename;
}

cl::Program OpenCLProgram::build(std::shared_ptr<OpenCLDevice> device,
        std::string buildOptions) {
    if(mSourceFilename.empty() && m_sourceCode.empty())
        throw Exception("No source filename nor source code was given to OpenCLProgram. Therefore build operation is not possible.");

    if(buildOptions.size() > 0)
        buildOptions += " ";
    buildOptions += "-cl-kernel-arg-info"; // We need this to be able to get kernel arg info

    // Add fast_3d_image_writes flag if it is supported
    if(device->isWritingTo3DTexturesSupported()) {
        buildOptions += " -Dfast_3d_image_writes";
    }

    if(buildExists(device, buildOptions))
        return mOpenCLPrograms[device][buildOptions];

    std::string programName;
    if(m_sourceCode.empty()) {
        programName = getAbsolutePath(mSourceFilename) + buildOptions;
        // Only create program if it doesn't exist for this device from before
        if(!device->hasProgram(programName))
            device->createProgramFromSourceWithName(programName, mSourceFilename, buildOptions);
    } else {
        // Only create program if it doesn't exist for this device from before
        // TODO consider using hash instead of potentially long source code string as name here
        programName = m_sourceCode + buildOptions;
        if(!device->hasProgram(programName))
            device->createProgramFromStringWithName(programName, m_sourceCode, buildOptions);
    }
    return device->getProgram(programName);
}

OpenCLProgram::OpenCLProgram() {
    mName = "";
    mSourceFilename = "";
}

bool OpenCLProgram::buildExists(std::shared_ptr<OpenCLDevice> device,
        std::string buildOptions) const {
    bool hasBuild = true;
    if(mOpenCLPrograms.count(device) == 0) {
        hasBuild = false;
    } else {
        const std::map<std::string, cl::Program> programs = mOpenCLPrograms.at(device);
        if(programs.count(buildOptions) == 0)
            hasBuild = false;
    }

    return hasBuild;
}

void OpenCLProgram::setSourceCode(std::string sourceCode) {
    m_sourceCode = sourceCode;
}

Queue::Queue(cl::CommandQueue clQueue) {
    m_queue = clQueue;
}

void Queue::add(const Kernel &kernel, std::vector<int> globalSize, std::vector<int> offset, std::vector<int> groupSize) {
    if(globalSize.size() == 3 && globalSize[2] == 1) {
        // When using add(kernel, image->getSize()), we get 3D size with size.z() == 1, even though image is 2D
        // Remove last element, not sure if this is really necessary
        globalSize.pop_back();
    }

    if(!kernel.allArgumentsGotValue()) {
        std::string str;
        for(auto name : kernel.getArgumentsWithoutValue()) {
            str += name + " ";
        }
        throw Exception("The following arguments to the OpenCL kernel have not received any value: " + str);
    }
    cl::NDRange clOffset;
    if(offset.empty()) {
        clOffset = cl::NullRange;
    }
    cl::NDRange clGroupSize;
    if(groupSize.empty()) {
        clGroupSize = cl::NullRange;
    }
    if(globalSize.size() == 1) {
        if(!groupSize.empty() && groupSize.size() != 1)
            throw Exception("Global size and group size must have same dimensions in Queue::add");
        if(!offset.empty() && offset.size() != 1)
            throw Exception("Global size and offset must have same dimensions in Queue::add");
        m_queue.enqueueNDRangeKernel(kernel.getHandle(), clOffset, cl::NDRange(globalSize[0]), clGroupSize);
    } else if(globalSize.size() == 2) {
        if(!groupSize.empty() && groupSize.size() != 2)
            throw Exception("Global size and group size must have same dimensions in Queue::add");
        if(!offset.empty() && offset.size() != 2)
            throw Exception("Global size and offset must have same dimensions in Queue::add");
        m_queue.enqueueNDRangeKernel(kernel.getHandle(), clOffset, cl::NDRange(globalSize[0], globalSize[1]), clGroupSize);
    } else if(globalSize.size() == 3) {
        if(!groupSize.empty() && groupSize.size() != 3)
            throw Exception("Global size and group size must have same dimensions in Queue::add");
        if(!offset.empty() && offset.size() != 3)
            throw Exception("Global size and offset must have same dimensions in Queue::add");
        m_queue.enqueueNDRangeKernel(kernel.getHandle(), clOffset, cl::NDRange(globalSize[0], globalSize[1], globalSize[2]), clGroupSize);
    } else {
        throw Exception("Invalid size given to Queue::add()");
    }
}

cl::CommandQueue Queue::getHandle() const {
    return m_queue;
}

void Queue::finish() {
    m_queue.finish();
}

void Queue::addReadBuffer(OpenCLBuffer buffer, bool block, std::size_t offset, std::size_t size,
                          void *pointerToData) {
    m_queue.enqueueReadBuffer(buffer.getHandle(), block ? CL_TRUE : CL_FALSE, offset, size, pointerToData);
}

void Queue::addWriteBuffer(OpenCLBuffer buffer, bool block, std::size_t offset, std::size_t size, void *pointerToData) {
    m_queue.enqueueWriteBuffer(buffer.getHandle(), block ? CL_TRUE : CL_FALSE, offset, size, pointerToData);
}

void Queue::addCopyBuffer(OpenCLBuffer srcBuffer, OpenCLBuffer dstBuffer, std::size_t srcOffset, std::size_t destOffset, std::size_t size) {
    m_queue.enqueueCopyBuffer(srcBuffer.getHandle(), dstBuffer.getHandle(), srcOffset, destOffset, size);
}

/*
void Queue::add(const Kernel &kernel, Vector3ui globalSize, Vector3ui offset, Vector3ui groupSize) {
    std::vector<int> globalSizeVector(globalSize.data(), globalSize.data() + 3);
    std::vector<int> offsetVector(offset.data(), offset.data() + 3);
    std::vector<int> groupSizeVector(groupSize.data(), groupSize.data() + 3);
    add(kernel, globalSizeVector, offsetVector, groupSizeVector);
}
 */

Kernel::Kernel(cl::Kernel clKernel, OpenCLDevice::pointer device) {
    m_kernel = clKernel;
    m_device = device;

    std::map<cl_kernel_arg_access_qualifier, KernelArgumentAccess> accessMap = {
            {CL_KERNEL_ARG_ACCESS_NONE, KernelArgumentAccess::UNSPECIFIED},
            {CL_KERNEL_ARG_ACCESS_READ_ONLY, KernelArgumentAccess::READ_ONLY},
            {CL_KERNEL_ARG_ACCESS_WRITE_ONLY, KernelArgumentAccess::WRITE_ONLY},
            {CL_KERNEL_ARG_ACCESS_READ_WRITE, KernelArgumentAccess::READ_WRITE}
    };

    // Query the kernel
    int numberOfArgs = m_kernel.getInfo<CL_KERNEL_NUM_ARGS>();
    try {
        for(int i = 0; i < numberOfArgs; ++i) {
            KernelArgument argInfo;
            argInfo.type = m_kernel.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(i);
            argInfo.access = accessMap[m_kernel.getArgInfo<CL_KERNEL_ARG_ACCESS_QUALIFIER>(i)];
            argInfo.addressQualifier = m_kernel.getArgInfo<CL_KERNEL_ARG_ADDRESS_QUALIFIER>(i);
            argInfo.typeQualifier = m_kernel.getArgInfo<CL_KERNEL_ARG_TYPE_QUALIFIER>(i);
            argInfo.name = m_kernel.getArgInfo<CL_KERNEL_ARG_NAME>(i);
            argInfo.index = i;
            m_argInfoByName[argInfo.name] = argInfo;
            m_argInfoByIndex[argInfo.index] = argInfo;
        }
    } catch(cl::Error &e) {
        // TODO Handle possible need for recompiling cached kernels
        throw Exception("OpenCL exception caught when trying to get kernel argument information: " + std::string(e.what()) + "("  + getCLErrorString(e.err()) + "). You might need to recompile your OpenCL code, delete the cache.");
    }
}

cl::Kernel Kernel::getHandle() const {
    return m_kernel;
}

template <>
void Kernel::setArg(int index, std::unique_ptr<OpenCLImageAccess> access) {
    checkIndex(index);
    if(access->getDimensions() == 2) {
        m_kernel.setArg(index, *access->get2DImage());
    } else {
        m_kernel.setArg(index, *access->get3DImage());
    }
    m_argGotValue.insert(index);
}

template <>
void Kernel::setArg(const std::string& name, std::unique_ptr<OpenCLImageAccess> access) {
    setArg(getIndex(name), std::move(access));
}

void Kernel::setImageArg(int index, std::shared_ptr<Image> data, accessType accessType, bool forceOpenCLImage) {
    checkIndex(index);
    // Check if OpenCL image or OpenCL Buffer
    auto type = m_argInfoByIndex.at(index).type;
    if(forceOpenCLImage || type == "image2d_t" || type == "image3d_t") {
        auto access = data->getOpenCLImageAccess(accessType, m_device);
        setArg(index, std::move(access));
    } else {
        auto access = data->getOpenCLBufferAccess(accessType, m_device);
        setArg(index, std::move(access));
    }
}

void Kernel::setImageArg(std::string name, std::shared_ptr<Image> data, accessType type, bool forceOpenCLImage) {
    setImageArg(getIndex(name), data, type, forceOpenCLImage);
}

int Kernel::getNumberOfArgs() const {
    return m_argInfoByIndex.size();
}

KernelArgument Kernel::getArg(int index) const {
    checkIndex(index);
    return m_argInfoByIndex.at(index);
}

KernelArgument Kernel::getArg(std::string name) const {
    if(m_argInfoByName.count(name) == 0)
        throw Exception("Kernel argument with name " + name + " was not found.");
    return m_argInfoByName.at(name);
}

bool Kernel::allArgumentsGotValue() const {
    return m_argGotValue.size() == getNumberOfArgs();
}

std::vector<std::string> Kernel::getArgumentsWithoutValue() const {
    std::vector<std::string> list;
    for(int i = 0; i < getNumberOfArgs(); ++i) {
        if(m_argGotValue.count(i) == 0)
            list.emplace_back(m_argInfoByIndex.at(i).name);
    }
    return list;
}

void Kernel::clearBuffers() {
    m_buffers.clear();
}

int Kernel::getIndex(const std::string &name) const {
    try {
        return m_argInfoByName.at(name).index;
    } catch(...) {
        throw Exception("Kernel argument with name " + name + " was not found.");
    }
}

void Kernel::setTensorArg(int index, std::shared_ptr<fast::Tensor> tensor, accessType type) {
    auto access = tensor->getOpenCLBufferAccess(type, m_device);
    setArg(index, std::move(access));
}

void Kernel::setTensorArg(std::string name, std::shared_ptr<fast::Tensor> tensor, accessType access) {
    setTensorArg(getIndex(name), tensor, access);
}

void Kernel::checkIndex(int index) const {
    if(index >= getNumberOfArgs() || index < 0)
        throw Exception("Kernel does not have an argument with index " + std::to_string(index) + ", number of arguments is: " + std::to_string(getNumberOfArgs()));
}

OpenCLBuffer::OpenCLBuffer(std::size_t size, OpenCLDevice::pointer device, KernelMemoryAccess kernelAccess,
                           HostMemoryAccess hostAccess, const void *data) {
    std::map<KernelMemoryAccess, cl_mem_flags> kernelMemoryAccessMap = {
        {KernelMemoryAccess::READ_WRITE, CL_MEM_READ_WRITE},
        {KernelMemoryAccess::READ_ONLY, CL_MEM_READ_ONLY},
        {KernelMemoryAccess::WRITE_ONLY, CL_MEM_WRITE_ONLY},
    };
    std::map<HostMemoryAccess, cl_mem_flags> hostMemoryAccessMap = {
            {HostMemoryAccess::UNSPECIFIED, 0},
            {HostMemoryAccess::READ_ONLY, CL_MEM_HOST_READ_ONLY},
            {HostMemoryAccess::WRITE_ONLY, CL_MEM_HOST_WRITE_ONLY},
            {HostMemoryAccess::NONE, CL_MEM_HOST_NO_ACCESS},
    };
    cl_mem_flags flags = kernelMemoryAccessMap[kernelAccess] | hostMemoryAccessMap[hostAccess];
    if(data != nullptr) {
        flags = flags | CL_MEM_COPY_HOST_PTR;
    }
    m_buffer = cl::Buffer(device->getContext(), flags, size, (void*)data);
    m_size = size;
    m_hostAccess = hostAccess;
    m_kernelAccess = kernelAccess;
}

cl::Buffer OpenCLBuffer::getHandle() const {
    return m_buffer;
}

std::size_t OpenCLBuffer::getSize() const {
    return m_size;
}

template <>
void Kernel::setArg(int index, OpenCLBuffer buffer) {
    checkIndex(index);
    m_kernel.setArg(index, buffer.getHandle());
    m_argGotValue.insert(index);
    m_buffers.emplace_back(buffer); // We have to make sure OpenCLBuffer objects exist while kernel is being queued.
}

template <>
void Kernel::setArg(const std::string& name, OpenCLBuffer buffer) {
    setArg(getIndex(name), buffer);
}

template <>
void Kernel::setArg(int index, Image::pointer image) {
    checkIndex(index);
    accessType access = ACCESS_READ_WRITE;
    auto kernelAccess = m_argInfoByIndex.at(index).access;
    if(kernelAccess == KernelArgumentAccess::READ_ONLY) {
        access = ACCESS_READ;
    }
    setImageArg(index, image, access);
}

template <>
void Kernel::setArg(const std::string& name, std::shared_ptr<Image> image) {
    setArg(getIndex(name), image);
}

template <>
void Kernel::setArg(int index, std::unique_ptr<OpenCLBufferAccess> access) {
    checkIndex(index);
    m_kernel.setArg(index, *access->get());
    m_argGotValue.insert(index);
}
template <>
void Kernel::setArg(const std::string& name, std::unique_ptr<OpenCLBufferAccess> access) {
    setArg(getIndex(name), std::move(access));
}

template <>
void Kernel::setArg(int index, Tensor::pointer tensor) {
    auto access = tensor->getOpenCLBufferAccess(ACCESS_READ_WRITE, m_device);
    setArg(index, std::move(access));
}

template <>
void Kernel::setArg(const std::string& name, Tensor::pointer tensor) {
    setArg(getIndex(name), tensor);
}
} // end namespace fast
