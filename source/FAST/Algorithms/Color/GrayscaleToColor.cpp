#include "GrayscaleToColor.hpp"
#include <FAST/Data/Image.hpp>

namespace fast {

GrayscaleToColor::GrayscaleToColor(bool addAlphaChannel) {
    createInputPort(0);
    createOutputPort(0);
    m_addAlphaChannel = addAlphaChannel;
    createOpenCLProgram(Config::getKernelSourcePath() + "Algorithms/Color/GrayscaleToColor.cl");
}

void GrayscaleToColor::execute() {
    auto image = getInputData<Image>();
    if(image->getDimensions() != 2)
        throw Exception("GrayscaleToColor is only implemented for 2D");

    if(image->getNrOfChannels() >= 2) {
        // Image is already color..
        addOutputData(image);
        return;
    }

    auto output = Image::create(image->getSize(), image->getDataType(), m_addAlphaChannel ? 4 : 3);
    output->setSpacing(image->getSpacing());
    SceneGraph::setParentNode(output, image);

    auto kernel = getKernel("convert");
    kernel.setArg(0, image);
    kernel.setArg(1, output);

    getQueue().add(kernel, {image->getWidth(), image->getHeight()});

    addOutputData(output);
}

}