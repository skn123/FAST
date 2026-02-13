#include "ImageCaster.hpp"
#include <FAST/Data/Image.hpp>

namespace fast {

ImageCaster::ImageCaster() {
    createInputPort(0, "Image");
    createOutputPort(0, "Image");
    createOpenCLProgram(Config::getKernelSourcePath() + "/Algorithms/ImageCaster/ImageCaster2D.cl", "2D");
    createOpenCLProgram(Config::getKernelSourcePath() + "/Algorithms/ImageCaster/ImageCaster3D.cl", "3D");
}

ImageCaster::ImageCaster(DataType outputType, float scaleFactor, bool normalizeFirst) : ImageCaster() {
    m_outputType = outputType;
    m_scaleFactor = scaleFactor;
    m_normalizeFirst = normalizeFirst;
}

void ImageCaster::execute() {
    auto input = getInputData<Image>();

    float minimum = 0.0f;
    float maximum = 0.0f;
    if(m_normalizeFirst) {
        minimum = input->calculateMinimumIntensity();
        maximum = input->calculateMaximumIntensity();
    }

    auto output = Image::create(input->getSize(), m_outputType, input->getNrOfChannels());
    output->setSpacing(input->getSpacing());
    SceneGraph::setParentNode(output, input);
    auto device = std::dynamic_pointer_cast<OpenCLDevice>(getMainDevice());

    auto queue = device->getCommandQueue();

    Kernel kernel;
    if(input->getDimensions() == 2) {
        kernel = getKernel("cast2D", "2D");
    } else if(getMainOpenCLDevice()->isWritingTo3DTexturesSupported()) {
        kernel = getKernel("cast3D", "3D");
    } else {
        kernel = getKernel("cast3DBuffer", "3D", "-DTYPE=" + getCTypeAsString(output->getDataType()));
        kernel.setArg(6, input->getNrOfChannels());
    }
    kernel.setArg(0, input);
    kernel.setArg(1, output);
    kernel.setArg(2, m_scaleFactor);
    kernel.setArg(3, (char)(m_normalizeFirst ? 1 : 0));
    kernel.setArg(4, minimum);
    kernel.setArg(5, maximum);

    getQueue().add(kernel, input->getSize());

    addOutputData(0, output);
}

}