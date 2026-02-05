#pragma once

#include "FAST/ProcessObject.hpp"
#include "FAST/ExecutionDevice.hpp"
#include "FAST/Data/Image.hpp"

namespace fast {


/**
 * @brief Smoothing by convolution with a Gaussian mask
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
        FAST_CONSTRUCTOR(GaussianSmoothing,
                         Vector2f, stdDev,,
                         Vector2i, maskSize, = Vector2i::Zero()
        );
        FAST_CONSTRUCTOR(GaussianSmoothing,
                         Vector3f, stdDev,,
                         Vector3i, maskSize, = Vector3i::Zero()
        );
        void setMaskSize(uchar maskSize);
        void setMaskSize(Vector2i maskSize);
        void setMaskSize(Vector3i maskSize);
        void setStandardDeviation(float stdDev);
        void setStandardDeviation(Vector2f stdDev);
        void setStandardDeviation(Vector3f stdDev);
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

