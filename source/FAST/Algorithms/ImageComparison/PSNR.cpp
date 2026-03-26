#include <FAST/Visualization/Plotting/LinePlotter.hpp>
#include "PSNR.hpp"

namespace fast {

PeakSignalToNoiseRatio::PeakSignalToNoiseRatio(float maxValue) : MeanSquaredError() {
    m_maxValue = maxValue;
}

void PeakSignalToNoiseRatio::execute() {
    auto image1 = getInputData<Image>(0);
    auto image2 = getInputData<Image>(1);

    auto outputImage = calculateSquaredDiffImage(image1, image2);
    auto MSE = calculateMSE(outputImage);

    m_value = 20.0f*std::log10(m_maxValue) - 10.0f*std::log10(MSE);
    auto outputValue = FloatScalar::create(m_value);
    addOutputData(0, outputImage);
    addOutputData(1, outputValue);
}

}