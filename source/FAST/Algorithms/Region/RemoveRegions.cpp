#include "RemoveRegions.hpp"
#include "RegionProperties.hpp"
#include <FAST/Data/Image.hpp>
#include <FAST/Algorithms/LabelModifier/LabelModifier.hpp>

namespace fast {

RemoveRegions::RemoveRegions(bool removeAllButLargest, bool removePerClass, int largestRegionsToKeep, float minArea, float maxArea) {
    createInputPort(0);
    createOutputPort(0);
    m_minArea = minArea;
    m_maxArea = maxArea;
    if(removeAllButLargest)
        m_largestRegionsToKeep = 1;
    if(largestRegionsToKeep > 0)
        m_largestRegionsToKeep = largestRegionsToKeep;
    m_removePerClass = removePerClass;
}

void RemoveRegions::execute() {
    auto input = getInputData<Image>();

    if(input->calculateSumIntensity() == 0) {
        addOutputData(0, input);
        return;
    }

    auto regionProperties = RegionProperties::create(false, true)->connect(input);
    regionProperties->run();

    auto regions = regionProperties->getOutputData<RegionList>(0);

    if(regions->get().empty()) {
        addOutputData(0, input);
        return;
    }

    auto instanceSegmentation = regionProperties->getOutputData<Image>(1);

    std::map<uint, std::vector<Region>> regionListPerLabel; // Map label -> list of regions
    for(const auto& region : regions->get()) {
        if(m_removePerClass) {
            regionListPerLabel[region.label].push_back(region);
        } else {
            regionListPerLabel[1].push_back(region);
        }
    }
    if(m_largestRegionsToKeep > 0) {
        for(const auto& item : regionListPerLabel) {
            auto label = item.first;
            auto& regionList = regionListPerLabel[label]; // Need a reference, since we are going to sort it
            // Sort by area
            std::sort(regionList.begin(), regionList.end(), [](const Region &a, const Region &b) {
                return a.area > b .area;
            });
        }
    }

    std::map<uint, uint> labelsToChange;
    for(const auto& item : regionListPerLabel) {
        const auto label = item.first;
        const auto& regionList = regionListPerLabel[label];
        int counter = 0;
        for(const auto& region : regionList) {
            if(m_largestRegionsToKeep > 0) {
                if(counter == m_largestRegionsToKeep) {
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
                ++counter;
                labelsToChange[region.instance] = region.label;
            }
        }
    }

    auto output = LabelModifier::create(labelsToChange)
            ->connect(instanceSegmentation)
            ->run()->getOutput<Image>();
    addOutputData(0, output);
}

}