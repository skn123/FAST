#include "RemoveRegions.hpp"
#include "RegionProperties.hpp"
#include <FAST/Data/Image.hpp>
#include <FAST/Algorithms/LabelModifier/LabelModifier.hpp>

namespace fast {

RemoveRegions::RemoveRegions(bool removeAllButLargest, int largestRegionsToKeep, float minArea, float maxArea) {
    createInputPort(0);
    createOutputPort(0);
    m_minArea = minArea;
    m_maxArea = maxArea;
    if(removeAllButLargest)
        m_largestRegionsToKeep = 1;
    if(largestRegionsToKeep > 0)
        m_largestRegionsToKeep = largestRegionsToKeep;
}

void RemoveRegions::execute() {
    auto input = getInputData<Image>();

    auto regionProperties = RegionProperties::create(false, true)->connect(input);
    regionProperties->run();

    auto regions = regionProperties->getOutputData<RegionList>(0);
    auto instanceSegmentation = regionProperties->getOutputData<Image>(1);

    std::vector<Region> regionList;
    for(const auto& region : regions->get())
        regionList.push_back(region);
    if(m_largestRegionsToKeep > 0) {
        // Sort by area first
        std::sort(regionList.begin(), regionList.end(), [](const Region &a, const Region &b) {
            return a.area > b .area;
        });
    }

    std::map<uint, uint> labelsToChange;
    for(const auto& region : regionList) {
        if(m_largestRegionsToKeep > 0) {
            if(labelsToChange.size() == m_largestRegionsToKeep) {
                // Max limit reached, remove rest
                labelsToChange[region.instance] = 0;
                continue;
            }
        }
        if(region.area < m_minArea) {
            labelsToChange[region.instance] = 0;
        } else if(region.area > m_maxArea) {
            labelsToChange[region.instance] = 0;
        } else {
            labelsToChange[region.instance] = 1;
        }
    }

    auto output = LabelModifier::create(labelsToChange)->connect(instanceSegmentation)->runAndGetOutputData<Image>();
    addOutputData(0, output);
}

}