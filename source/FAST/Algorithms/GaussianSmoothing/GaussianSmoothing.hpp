#pragma once

#include "FAST/ProcessObject.hpp"
#include "FAST/ExecutionDevice.hpp"
#include "FAST/Data/Image.hpp"

namespace fast {


/**
 * @brief Smoothing by convolution with a Gaussian mask
 *
 * Supports 2D, 3D and anisotropic smoothing (e.g. different standard deviation for each dimension).
 * Also supports multi-channel (e.g. RGB) smoothing, in this case smoothing is applied per channel.
 *
 * Inputs:
 * - 0: Image, 2D or 3D
 *
 * Outputs:
 * - 0: Image, 2D or 3D
 *
 * @ingroup filter
 */
class FAST_EXPORT GaussianSmoothing : public ProcessObject {
    FAST_PROCESS_OBJECT(GaussianSmoothing)
    public:
        /**
         * @brief Create instance
         * @param stdDev Standard deviation of convolution kernel
         * @param maskSize Size of convolution filter/mask. Must be odd.
         *      If 0 filter size is determined automatically from standard deviation
         * @return instance
         */
        FAST_CONSTRUCTOR(GaussianSmoothing,
                         float, stdDev, = 0.5f,
                         uchar, maskSize, = 0
        );
        /**
        * @brief Create instance
        * @param stdDev Standard deviation of convolution kernel for each dimension
        * @param maskSize Size of convolution filter/mask for each dimension. Must be odd.
        *      If 0, or not given, mask size is determined automatically from standard deviation
        * @return instance
        */
        FAST_CONSTRUCTOR(GaussianSmoothing,
                         std::vector<float>, stdDev,,
                         std::vector<int>, maskSize, = std::vector<int>()
        );
        void setMaskSize(int maskSize);
        void setMaskSize(std::vector<int> maskSize);
        void setStandardDeviation(float stdDev);
        void setStandardDeviation(std::vector<float> stdDev);
        void setOutputType(DataType type);
        void loadAttributes() override;
        ~GaussianSmoothing();
    protected:
        void execute();
        void waitToFinish();
        void createMask(Image::pointer input, Vector3i maskSize, bool useSeperableFilter);
        void recompileOpenCLCode(Image::pointer input);

        Vector3i mMaskSize;
        Vector3f mStdDev;

        cl::Buffer mCLMask;
        std::unique_ptr<float[]> mMask;
        bool mRecreateMask;

        cl::Kernel mKernel;
        unsigned char mDimensionCLCodeCompiledFor;
        DataType mTypeCLCodeCompiledFor;
        DataType mOutputType;
        bool mOutputTypeSet;
};

} // end namespace fast

