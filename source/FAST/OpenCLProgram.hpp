#pragma once

#include <FAST/Object.hpp>
#include <unordered_map>
#include <FAST/Data/DataTypes.hpp>
#include <FAST/Data/Access/OpenCLImageAccess.hpp>
#include <FAST/Data/Image.hpp>

namespace cl {

class Program;

}

namespace fast {

/**
 * @defgroup opencl OpenCL
 * Objects and functions for OpenCL
 */

class OpenCLDevice;
class Tensor;

/**
 * @brief OpenCL program
 * @ingroup opencl
 */
class FAST_EXPORT  OpenCLProgram : public Object {
    FAST_OBJECT(OpenCLProgram)
    public:
        void setName(std::string name);
        std::string getName() const;
        void setSourceFilename(std::string filename);
        void setSourceCode(std::string sourceCode);
        std::string getSourceFilename() const;
        cl::Program build(std::shared_ptr<OpenCLDevice>, std::string buildOptions = "");
    protected:
        OpenCLProgram();

        bool buildExists(std::shared_ptr<OpenCLDevice>, std::string buildOptions = "") const;

        std::string mName;
        std::string mSourceFilename;
        std::string m_sourceCode;
        std::unordered_map<std::shared_ptr<OpenCLDevice>, std::map<std::string, cl::Program> > mOpenCLPrograms;
};

enum class KernelArgumentAccess {
    NONE,
    READ_ONLY,
    WRITE_ONLY,
    READ_WRITE,
};

/**
 * @brief Struct describing a kernel argument.
 * This is collected automatically from the kernel.
 * @ingroup opencl
 */
struct FAST_EXPORT KernelArgument {
    int index;
    std::string name;
    std::string type;
    cl_kernel_arg_type_qualifier typeQualifier;
    cl_kernel_arg_address_qualifier addressQualifier;
    KernelArgumentAccess access;
};

class OpenCLBuffer;

/**
 * @brief Represents an OpenCL Kernel
 * @sa KernelArgument
 * @sa Queue
 * @ingroup opencl
 */
class FAST_EXPORT Kernel {
    public:
        explicit Kernel(cl::Kernel clKernel, OpenCLDevice::pointer device);
        cl::Kernel getHandle() const;

#ifndef SWIG
        template <class T>
        void setArg(int index, const std::vector<T>& data);
        template <class T>
        void setArg(const std::string& name, const std::vector<T>& data);
#endif
        template <class T>
        void setArgData(int index, const std::vector<T>& data);
        template <class T>
        void setArgData(const std::string& name, const std::vector<T>& data);
        template <class T>
        void setArg(int index, T data);
        template <class T>
        void setArg(const std::string& name, T data);
#ifndef SWIG
        template <class T>
        void setArg(int index, std::size_t size, T data);
        template <class T>
        void setArg(const std::string& name, std::size_t size, T data);
#endif
        // Variadic function template
        template <typename T, typename... Args>
        void setArgs(T&& firstArg, Args... args) {
            setArgumentWithCounter(std::forward<T>(firstArg));
            setArgs(args...); // Recursive call to handle remaining arguments
        }
        template <typename T>
        void setArgs(T&& lastArg) {
            setArgumentWithCounter(std::forward<T>(lastArg));
            m_argCounter = 0;
        }
        void setImageArg(int index, std::shared_ptr<fast::Image> data, accessType access, bool forceOpenCLImage = false);
        void setImageArg(std::string name, std::shared_ptr<fast::Image> data, accessType access, bool forceOpenCLImage = false);
        void setTensorArg(int index, std::shared_ptr<fast::Tensor> tensor, accessType access);
        void setTensorArg(std::string name, std::shared_ptr<fast::Tensor> tensor, accessType access);
        /**
         * @brief Get number of arguments of the given
         * @return
         */
        int getNumberOfArgs() const;
        /**
         * @brief Get kernel argument by index
         * @param index
         * @return
         */
        KernelArgument getArg(int index) const;
        /**
         * @brief Get kernel arguemnt by name
         * @param name
         * @return
         */
        KernelArgument getArg(std::string name) const;
        /**
         * @brief Check whether all arguments have gotten a value by setArg.
         * @return
         */
        bool allArgumentsGotValue() const;
        /**
         * @brief Get list of argument names which haven't received a value yet.
         * @return
         */
        std::vector<std::string> getArgumentsWithoutValue() const;
        /**
         * @brief Clear list of buffers assigned to arguments in this kernel.
         */
        void clearBuffers();
    private:
        int m_argCounter = 0;
        template <class T>
        void setArgumentWithCounter(T&& data) {
            setArg(m_argCounter, data);
            ++m_argCounter;
        }
        int getIndex(const std::string& name) const;
        void checkIndex(int index) const;
        cl::Kernel m_kernel;
        OpenCLDevice::pointer m_device;
        std::map<std::string, KernelArgument> m_argInfoByName;
        std::map<int, KernelArgument> m_argInfoByIndex;
        std::set<int> m_argGotValue;
        std::vector<OpenCLBuffer> m_buffers;
};

template <class T>
void Kernel::setArg(int index, T data) {
    checkIndex(index);
    m_kernel.setArg(index, data);
    m_argGotValue.insert(index);
}

template <class T>
void Kernel::setArg(const std::string& name, T data) {
    int index = getIndex(name);
    m_kernel.setArg(index, data);
    m_argGotValue.insert(index);
}

template <class T>
void Kernel::setArg(int index, std::size_t size, T data) {
    checkIndex(index);
    m_kernel.setArg(index, size, data);
    m_argGotValue.insert(index);
}

template <class T>
void Kernel::setArg(const std::string& name, std::size_t size, T data) {
    int index = getIndex(name);
    m_kernel.setArg(index, size, data);
    m_argGotValue.insert(index);
}

// Kernel::setArg specializations
template <>
void Kernel::setArg(int index, OpenCLBuffer buffer);
template <>
void Kernel::setArg(const std::string& name, OpenCLBuffer buffer);
template <>
void Kernel::setArg(int index, std::shared_ptr<Image> image);
template <>
void Kernel::setArg(const std::string& name, std::shared_ptr<Image> image);
template <>
void Kernel::setArg(int index, std::shared_ptr<Tensor> tensor);
template <>
void Kernel::setArg(const std::string& name, std::shared_ptr<Tensor> tensor);
template <>
void Kernel::setArg(int index, std::unique_ptr<OpenCLImageAccess> access);
template <>
void Kernel::setArg(const std::string& name, std::unique_ptr<OpenCLImageAccess> access);
template <>
void Kernel::setArg(int index, std::unique_ptr<OpenCLBufferAccess> access);
template <>
void Kernel::setArg(const std::string& name, std::unique_ptr<OpenCLBufferAccess> access);

/**
 * @brief Wrapper for OpenCL CommandQueue
 * @ingroup opencl
 * @sa Kernel
 */
class FAST_EXPORT Queue {
    public:
        Queue(cl::CommandQueue clQueue);
        void add(const Kernel& kernel, std::vector<int> globalSize, std::vector<int> offset = {}, std::vector<int> groupSize = {});
        void finish();
        void addReadBuffer(OpenCLBuffer buffer, bool block, std::size_t offset, std::size_t size, void* pointerToData);
        void addWriteBuffer(OpenCLBuffer buffer, bool block, std::size_t offset, std::size_t size, void* pointerToData);
        void addCopyBuffer(OpenCLBuffer srcBuffer, OpenCLBuffer dstBuffer, std::size_t srcOffset, std::size_t destOffset, std::size_t size);
        cl::CommandQueue getHandle() const;
    private:
        cl::CommandQueue m_queue;
};

enum class KernelMemoryAccess {
    READ_WRITE = 0,
    READ_ONLY,
    WRITE_ONLY
};
enum class HostMemoryAccess {
    UNSPECIFIED = 0,
    READ_ONLY,
    WRITE_ONLY,
    //READ_WRITE,
    NONE
};
/**
 * @brief OpenCL Buffer
 * @ingroup opencl
 */
class FAST_EXPORT OpenCLBuffer {
    public:
        OpenCLBuffer(
                std::size_t size,
                OpenCLDevice::pointer device,
                KernelMemoryAccess kernelAccess = KernelMemoryAccess::READ_WRITE,
                HostMemoryAccess hostAccess = HostMemoryAccess::UNSPECIFIED,
                const void* data = nullptr
        );
        cl::Buffer getHandle() const;
        std::size_t getSize() const;
    private:
        cl::Buffer m_buffer;
        std::size_t m_size;
        HostMemoryAccess m_hostAccess;
        KernelMemoryAccess m_kernelAccess;

};

template<class T>
void Kernel::setArg(int index, const std::vector<T>& data) {
    OpenCLBuffer buffer(sizeof(T)*data.size(), m_device, KernelMemoryAccess::READ_WRITE, HostMemoryAccess::UNSPECIFIED, data.data());
    setArg(index, buffer);
}

template<class T>
void Kernel::setArg(const std::string& name, const std::vector<T>& data) {
    setArg(getIndex(name), data);
}

template<class T>
void Kernel::setArgData(int index, const std::vector<T>& data) {
    setArg(index, data);
}

template<class T>
void Kernel::setArgData(const std::string& name, const std::vector<T>& data) {
    setArg(name, data);
}
} // end namespace fast
