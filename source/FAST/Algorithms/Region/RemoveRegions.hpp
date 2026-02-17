#pragma once

#include <FAST/ProcessObject.hpp>

namespace fast {

/**
 * @brief Remove small, large or all but largest regions from a segmentation image
 *
 * Uses RegionProperties internally to extract all regions, and the instance segmentation image,
 * and then LabelModifier to modify the segmentation.
 *
 * Inputs:
 * - 0: Image segmentation
 *
 * Outputs:
 * - 0: Image segmentation
 *
 * @sa RegionProperties
 * @ingroup segmentation
 */
class FAST_EXPORT RemoveRegions : public ProcessObject {
    FAST_PROCESS_OBJECT(RemoveRegions)
    public:
        /**
         * @brief Create instance
         * @param removeAllButLargest Removes all regions except the largest. This is equivalent to setting largestRegionsToKeep = 1.
         * @param largestRegionsToKeep If set to N > 0, all regions except the N largest regions are removed.
         *      This parameter takes precedence over removeAllButLargest. If set, minArea and maxArea are still used if set.
         * @param minArea Minimum area (in millimeters if pixel spacing exist).
         *      All regions with area less than this are removed.
         * @param maxArea Maximum area (in millimeters if pixel spacing exist).
         *      All regions with area more than this are removed.
         * @return instance
         */
        FAST_CONSTRUCTOR(RemoveRegions,
                         bool, removeAllButLargest, = false,
                         int, largestRegionsToKeep, = 0,
                         float, minArea, = 0.0f,
                         float, maxArea, = std::numeric_limits<float>::max()
        );
    private:
        void execute() override;
        float m_minArea;
        float m_maxArea;
        int m_largestRegionsToKeep = 0;
};
}