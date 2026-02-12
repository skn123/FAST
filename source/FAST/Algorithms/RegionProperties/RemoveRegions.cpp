#include "RemoveRegions.hpp"
#include "RegionProperties.hpp"
#include <FAST/Data/Image.hpp>
#include <FAST/Algorithms/LabelModifier/LabelModifier.hpp>

namespace fast {

RemoveRegions::RemoveRegions(float minArea, float maxArea) {
    createInputPort(0);
    createOutputPort(0);
    m_minArea = minArea;
    m_maxArea = maxArea;
}

void RemoveRegions::execute() {
    auto input = getInputData<Image>();

    auto regionProperties = RegionProperties::create(false, true)->connect(input);
    regionProperties->run();

    auto regions = regionProperties->getOutputData<RegionList>(0);
    auto instanceSegmentation = regionProperties->getOutputData<Image>(1);

    std::map<uint, uint> labelsToChange;
    for(const auto& region : regions->get()) {
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