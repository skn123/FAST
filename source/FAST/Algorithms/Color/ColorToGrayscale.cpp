#include "ColorToGrayscale.hpp"
#include <FAST/Data/Image.hpp>

namespace fast {

ColorToGrayscale::ColorToGrayscale() {
    createInputPort(0);
    createOutputPort(0);
    createOpenCLProgram(Config::getKernelSourcePath() + "Algorithms/Color/ColorToGrayscale.cl");
}

void ColorToGrayscale::execute() {
    auto image = getInputData<Image>();
    if(image->getDimensions() != 2)
        throw Exception("ColorToGrayscale is only implemented for 2D");

    if(image->getNrOfChannels() == 1) {
        // Image is already grayscale..
        addOutputData(image);
        return;
    }

    auto output = Image::create(image->getSize(), image->getDataType(), 1);
    output->setSpacing(image->getSpacing());
    SceneGraph::setParentNode(output, image);

    auto kernel = getKernel("convert");

    kernel.setArg("input", image);
    kernel.setArg("output", output);

    getQueue().add(kernel, image->getSize());

    addOutputData(output);
}
}