#pragma once

#include <FAST/ProcessObject.hpp>
#include <FAST/Algorithms/ImageComparison/MSE.hpp>

namespace fast {

/**
 * @brief Peak Signal-To-Noise-Ratio (PSNR) of two input images
 *
 * The PSNR is calculated using the MSE of the two input images,
 * and the max intensity value as:
 * PSNR = 20*log10(MAX_INTENSITY) - 10*log10(MSE)
 *
 * Multi-channel and 3D images are supported. In this case a MSE value is calculated per channel, and the
 * final PSNR value uses the average of all channel MSE values as MSE in the equation above.
 *
 * Inputs:
 * - 0: Image
 * - 1: Image
 *
 * Outputs:
 * - 0: Squared difference image
 * - 1: Float scalar, PSNR value
 *
 * @sa MeanSquaredError
 * @sa StructuralSimilarityIndexMeasure
 * @ingroup image-comparison
 */
class FAST_EXPORT PeakSignalToNoiseRatio : public MeanSquaredError {
    FAST_PROCESS_OBJECT(PeakSignalToNoiseRatio)
    public:
        /**
         * @brief Create instance
         * @param maxValue Maximum intensity value
         * @return instance
         */
        FAST_CONSTRUCTOR(PeakSignalToNoiseRatio, float, maxValue,)
    private:
        PeakSignalToNoiseRatio() {};
        void execute() override;
        float m_maxValue;
};

/**
 * @brief Alias of PeakSignalToNoiseRatio class
 * @sa PeakSignalToNoiseRatio
 * @ingroup image-comparison
 */
using PSNR = PeakSignalToNoiseRatio;

#ifdef SWIG
%pythoncode %{
PSNR = PeakSignalToNoiseRatio
%}
#endif
}