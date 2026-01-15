#include "ImageInverter.hpp"
#include "FAST/Data/Image.hpp"
#include "FAST/Utility.hpp"

namespace fast {

ImageInverter::ImageInverter(float min, float max) {
    createInputPort(0);
    createOutputPort(0);
    createOpenCLProgram(Config::getKernelSourcePath() + "Algorithms/ImageInverter/ImageInverter2D.cl", "2D");
    createOpenCLProgram(Config::getKernelSourcePath() + "Algorithms/ImageInverter/ImageInverter3D.cl", "3D");
    m_min = min;
    m_max = max;
}

void ImageInverter::execute() {
    auto input = getInputData<Image>();

    float max = m_max;
    if(std::isnan(max))
        max = input->calculateMaximumIntensity();
    float min = m_min;
    if(std::isnan(min))
        min = input->calculateMinimumIntensity();

    auto output = Image::createFromImage(input);
    Vector3ui size = input->getSize();

    if(input->getDimensions() == 3) {
        std::string buildOptions = "-DDATA_TYPE=" + getCTypeAsString(output->getDataType());
        auto kernel = getKernel("invert3D", "3D", buildOptions);

        kernel.setArg("input", input);
        kernel.setArg("output", output);
        kernel.setArg("minIntensity", min);
        kernel.setArg("maxIntensity", max);
        kernel.setArg("outputChannels", output->getNrOfChannels());

        getQueue().add(kernel, size);
    } else {
        auto kernel = getKernel("invert2D", "2D");

        kernel.setArg("input", input);
        kernel.setArg("output", output);
        kernel.setArg("minIntensity", min);
        kernel.setArg("maxIntensity", max);

        getQueue().add(kernel, size);
    }
    addOutputData(0, output);
}

}
