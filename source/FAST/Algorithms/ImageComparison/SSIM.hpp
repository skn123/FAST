#pragma once

#include <FAST/ProcessObject.hpp>

namespace fast {

/**
 * @brief Structural similarity index measure (SSIM) of two input images
 *
 * Calculates the similarity of two input images using SSIM as described in
 * the article "Image Quality Assessment: From Error Visibility to Structural Similarity"
 * by Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, IEEE Transactions on Image Processing 2004.
 *
 * Multi-channel images and 3D images are supported. In this case an SSIM value is calculated per channel, and the
 * final SSIM value is the average of all channel SSIM values.
 *
 * No image cropping is performed when calculating SSIM, out of bounds pixels in the Gaussian window are handled using mirrored repeat.
 *
 * Inputs:
 * - 0: Image
 * - 1: Image
 *
 * Outputs:
 * - 0: SSIM Image
 * - 1: FloatScalar, SSIM value
 *
 * @sa MeanSquaredError
 * @sa PeakSignalToNoiseRatio
 * @ingroup image-comparison
 */
class FAST_EXPORT StructuralSimilarityIndexMeasure : public ProcessObject {
    FAST_PROCESS_OBJECT(StructuralSimilarityIndexMeasure)
    public:
        /**
         * @brief Create instance
         * @param maxValue Maximum possible intensity value
         * @param minValue Minimum possible intensity value
         * @param windowSize Size of Gaussian window (in pixels) for each dimension
         * @param stdDev Standard deviation of Gaussian window (in pixels) for each dimension
         * @param k1 Algorithm constant
         * @param k2 Algorithm constant
         * @return instance
         */
        FAST_CONSTRUCTOR(StructuralSimilarityIndexMeasure,
                         float, maxValue,,
                         float, minValue, = 0,
                         Vector3i, windowSize, = Vector3i::Constant(11),
                         Vector3f, stdDev, = Vector3f::Constant(1.5f),
                         float, k1, = 0.01f,
                         float, k2, = 0.03f
        )
        /**
         * @brief Get SSIM value from last run
         * @return
         */
        float get() const;
        /**
         * @brief Set standard deviation of Gaussian window
         * @param stdDev Standard deviation for each dimension
         */
        void setStandardDeviation(Vector3f stdDev);
        /**
         * @brief Set size (in pixels) of Gaussian window
         * @param size Size for each dimension
         */
        void setWindowSize(Vector3i size);
    private:
        StructuralSimilarityIndexMeasure() {};
        void execute() override;
        bool m_recreateWeights = true;
        int m_weightsCreatedForDimension = 0;
        OpenCLBuffer m_weightsBuffer;
        float m_value = -1.0f;
        Vector3i m_windowSize;
        Vector3f m_stdDev;
        float m_maxValue;
        float m_minValue;
        float m_k1;
        float m_k2;
};

/**
 * @brief Alias for StructuralSimilarityIndexMeasure class
 * @sa StructuralSimilarityIndexMeasure
 * @ingroup image-comparison
 */
using SSIM = StructuralSimilarityIndexMeasure;
#ifdef SWIG
%pythoncode %{
SSIM = StructuralSimilarityIndexMeasure
%}
#endif
}
