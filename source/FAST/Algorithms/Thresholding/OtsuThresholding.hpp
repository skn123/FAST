#pragma once

#include <FAST/ProcessObject.hpp>

namespace fast {

/**
 * @brief Otsu thresholding segmentation method
 *
 * Automatically determines the threshold to use for segmentation
 * using Otsu's method. Supports multi-class thresholding by changing the
 * number of classes argument in the constructor (default is 2 classes = 1 threshold).
 * Maximum number of classes is currently 4.
 *
 * The implementation uses 256 bins when calculating the histogram.
 * Input images that are not of UINT8 type are normalized to [0, 255] range and cast to UINT8 before processing.
 *
 * Inputs:
 * - 0: Image (only first channel is used, multi-channel support not implemented).
 *
 * Outputs:
 * - 0: Image segmentation
 *
 * @todo Add methods for getting thresholds.
 * @todo 3D support
 * @todo Multi channel support
 *
 * @ingroup segmentation
 */
class FAST_EXPORT OtsuThresholding : public ProcessObject {
    FAST_PROCESS_OBJECT(OtsuThresholding)
    public:
        /**
         * @brief Create instance
         * @param numberOfClasses Numbef of classes to use, minimum 2, maximum 4. The number of thresholds is classes - 1.
         * @return instance
         */
        FAST_CONSTRUCTOR(OtsuThresholding, int, numberOfClasses,= 2)
    private:
        void execute() override;
        int m_thresholdCount;
};

}