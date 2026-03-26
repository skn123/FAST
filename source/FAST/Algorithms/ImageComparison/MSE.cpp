#include <FAST/Visualization/Plotting/LinePlotter.hpp>
#include <FAST/Algorithms/ImageChannelConverter/ImageChannelConverter.hpp>
#include "MSE.hpp"

namespace fast {

MeanSquaredError::MeanSquaredError() {
    createInputPort(0);
    createInputPort(1);
    createOutputPort(0);
    createOutputPort(1);

    createOpenCLProgram(Config::getKernelSourcePath() + "Algorithms/ImageComparison/MSE.cl");
}

void MeanSquaredError::execute() {
    auto image1 = getInputData<Image>(0);
    auto image2 = getInputData<Image>(1);

    auto outputImage = calculateSquaredDiffImage(image1, image2);
    auto MSE = calculateMSE(outputImage);

    m_value = MSE;
    auto outputValue = FloatScalar::create(MSE);

    addOutputData(0, outputImage);
    addOutputData(1, outputValue);
}

float MeanSquaredError::get() const {
    return m_value;
}

Image::pointer MeanSquaredError::calculateSquaredDiffImage(Image::pointer image1, Image::pointer image2) {
    if(image1->getSize() != image2->getSize())
        throw Exception("Images must be the same size");

    if(image1->getNrOfChannels() != image2->getNrOfChannels())
        throw Exception("Images must have same number of channels");

    auto output = Image::create(image1->getSize(), TYPE_FLOAT, image1->getNrOfChannels());
    SceneGraph::setParentNode(output, image1);
    output->setSpacing(image1->getSpacing());

    Kernel kernel;
    if(output->getDimensions() == 2) {
        kernel = getKernel("squaredError2D");
    } else {
        if(getMainOpenCLDevice()->isWritingTo3DTexturesSupported()) {
            kernel = getKernel("squaredError3D");
        } else {
            kernel = getKernel("squaredError3DBuffer");
            kernel.setArg("channels", output->getNrOfChannels());
        }
    }
    kernel.setArg("input1", image1);
    kernel.setArg("input2", image2);
    kernel.setArg("output", output);

    getQueue().add(kernel, output->getSize());

    return output;
}

float MeanSquaredError::calculateMSE(Image::pointer output) {
    float MSE = 0.0f;
    if(output->getNrOfChannels() > 1) {
        // FIXME calculateAverageIntensity doesn't support multi-channels yet, so to this for now:
        float MSEsum = 0.0f;
        for(int i = 0; i < output->getNrOfChannels(); ++i) {
            std::vector<int> toRemove;
            for(int j = 0; j < output->getNrOfChannels(); ++j) {
                if(i != j)
                    toRemove.push_back(j);
            }
            auto singleChannelImage = ImageChannelConverter::create(toRemove)
                    ->connect(output)
                    ->run()
                    ->getOutput<Image>();
            MSEsum += singleChannelImage->calculateAverageIntensity();
        }
        MSE = MSEsum / output->getNrOfChannels();
    } else {
        MSE = output->calculateAverageIntensity();
    }
    return MSE;
}

}
