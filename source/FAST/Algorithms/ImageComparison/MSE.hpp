#pragma once

#include <FAST/ProcessObject.hpp>

namespace fast {

/**
 * @brief Mean squared error (MSE) of two input images
 *
 * MSE is calculated as: \text{MSE} = \sum_i^N (I_1(\vec x_i) - I_2(\vec x_i)^2
 *
 * Multi-channel and 3D images are supported. In this case an MSE value is calculated per channel, and the
 * final MSE value is the average of all channel MSE values.
 *
 * Inputs:
 * - 0: Image
 * - 1: Image
 *
 * Outputs:
 * - 0: Squared difference image
 * - 1: Float scalar, MSE value
 *
 * @sa PeakSignalToNosieRatio
 * @sa StructuralSimilarityIndexMeasure
 * @ingroup image-comparison
 */
class FAST_EXPORT MeanSquaredError : public ProcessObject {
    FAST_PROCESS_OBJECT(MeanSquaredError)
    public:
        /**
         * @brief Create instance
         * @return instance
         */
        FAST_CONSTRUCTOR(MeanSquaredError)
        /**
         * @brief Get MSE value of last run
         */
        float get() const;
    protected:
        std::shared_ptr<Image> calculateSquaredDiffImage(std::shared_ptr<Image> image1, std::shared_ptr<Image> image2);
        float calculateMSE(std::shared_ptr<Image> image);
        float m_value = -1.0f;
    private:
        void execute() override;
};

/**
 * @brief Alias of MeanSquaredError class
 * @sa MeanSquaredError
 * @ingroup image-comparison
 */
using MSE = MeanSquaredError;

#ifdef SWIG
%pythoncode %{
MSE = MeanSquaredError
%}
#endif
}
