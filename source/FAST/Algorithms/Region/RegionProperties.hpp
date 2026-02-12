#pragma once

#include <FAST/ProcessObject.hpp>
#include <FAST/Data/SimpleDataObject.hpp>
namespace fast {

class Mesh;

/**
 * @brief Segmentation region struct
 * @sa RegionList
 * @sa RegionProperties
 * @ingroup data segmentation
 */
struct FAST_EXPORT Region {
    int pixelCount; /**< Region pixel count **/
    float area; /**< Region area in millimeters if pixel spacing is set. See also pixel count **/
    uchar label; /**< Region class label **/
    uint instance; /**< Region instance ID **/
    Vector2f centroid; /**< Region centroid **/
    float perimeterLength; /**< Region perimeter length **/
    float averageRadius; /**< Region average radius **/
    Vector2i maxPixelPosition; /**< Maximum x and y pixel position **/
    Vector2i minPixelPosition; /**< Minimum x and y pixel position **/
    std::shared_ptr<Mesh> contourMesh; /**< List of all contour pixel coordinates. Only extracted if extractCountours option is set in RegionProperties constructor **/
    std::vector<Vector2i> contourPixels; /**< List of all contour pixel coordinates. Only extracted if extractCountours option is set in RegionProperties constructor **/
    std::vector<Vector2i> pixels; /**< List of all pixel coordinates in region **/
};

/**
 * @brief Simple data object of list of regions
 * @sa Region
 * @sa RegionProperties
 * @ingroup data segmentation
 */
FAST_SIMPLE_DATA_OBJECT(RegionList, std::vector<Region>)

/**
 * @brief Calculate properties, such as area, contour and centroid, for every segmentation region
 *
 * Inputs:
 * - 0: Image segmentation
 *
 * Outputs:
 * - 0: RegionList, a simple data object which is a vector of Region
 * - 1: Image, instance segmentation image (data type = UINT32). Optional, only if outputInstanceSegmentation is set
 *
 * @ingroup segmentation
 */
class FAST_EXPORT RegionProperties : public ProcessObject {
    FAST_PROCESS_OBJECT(RegionProperties)
    public:
        /**
         * @brief Create instance
         * @param extractContours Whether to extract contours of each region or not
         * @param outputInstanceSegmentation Whether to create an instance segmentation image on outport port 1
         * @return instance
         */
        FAST_CONSTRUCTOR(RegionProperties, bool, extractContours, = true, bool, outputInstanceSegmentation, = false);
    protected:
        void execute() override;
        bool m_extractContours;
        bool m_outputInstanceSegmentation;
};

}