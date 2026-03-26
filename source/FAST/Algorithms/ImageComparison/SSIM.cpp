#include <FAST/Visualization/Plotting/LinePlotter.hpp>
#include <FAST/Algorithms/ImageChannelConverter/ImageChannelConverter.hpp>
#include "SSIM.hpp"

namespace fast {

StructuralSimilarityIndexMeasure::StructuralSimilarityIndexMeasure(float maxValue, float minValue, Vector3i windowSize, Vector3f stdDevs, float k1, float k2) {
    createInputPort(0);
    createInputPort(1);
    createOutputPort(0);
    createOutputPort(1);

    createOpenCLProgram(Config::getKernelSourcePath() + "Algorithms/ImageComparison/SSIM.cl");

    setWindowSize(windowSize);
    setStandardDeviation(stdDevs);
    m_maxValue = maxValue;
    m_minValue = minValue;
    m_k1 = k1;
    m_k2 = k2;
}

void StructuralSimilarityIndexMeasure::execute() {
    auto image1 = getInputData<Image>(0);
    auto image2 = getInputData<Image>(1);

    if(image1->getSize() != image2->getSize())
        throw Exception("Images must be the same size");

    if(image1->getNrOfChannels() != image2->getNrOfChannels())
        throw Exception("Images must have same number of channels");

    auto output = Image::create(image1->getSize(), TYPE_FLOAT, image1->getNrOfChannels());
    SceneGraph::setParentNode(output, image1);
    output->setSpacing(image1->getSpacing());
    if(output->getDimensions() != m_weightsCreatedForDimension)
        m_recreateWeights = true;

    Vector3f stdDev = m_stdDev;
    Vector3i windowSize = m_windowSize;
    Vector3i halfSize;
    for(int i = 0; i < 3; ++i)
        halfSize[i] = (windowSize[i] - 1) / 2;
    if(image1->getDimensions() == 2) {
        windowSize.z() = 1;
        halfSize.z() = 0;
        stdDev.z() = 0.0f;
    }
    if(m_recreateWeights) { // Only create weights buffer when needed
        float sum = 0.0f;
        auto gaussianWeights = std::make_unique<float[]>(windowSize.x() * windowSize.y() * windowSize.z());

        for(int x = -halfSize.x(); x <= halfSize.x(); ++x) {
            for(int y = -halfSize.y(); y <= halfSize.y(); ++y) {
                for(int z = -halfSize.z(); z <= halfSize.z(); ++z) {
                    float value = std::exp(-(
                            (stdDev.x() == 0.0f ? 0.0f : (float)(x*x)/(2.0f*stdDev.x()*stdDev.x())) +
                            (stdDev.y() == 0.0f ? 0.0f : (float)(y*y)/(2.0f*stdDev.y()*stdDev.y())) +
                            (stdDev.z() == 0.0f ? 0.0f : (float)(z*z)/(2.0f*stdDev.z()*stdDev.z()))
                            ));
                    gaussianWeights[x + halfSize.x() + (y + halfSize.y()) * windowSize.x() + (z + halfSize.z()) * windowSize.x() * windowSize.y()] = value;
                    sum += value;
                }
            }
        }

        for(int i = 0; i < windowSize.x() * windowSize.y() * windowSize.z(); ++i)
            gaussianWeights[i] /= sum;

        m_weightsBuffer = createBuffer(windowSize.x() * windowSize.y() * windowSize.z() * 4, gaussianWeights.get());
        m_recreateWeights = false;
        m_weightsCreatedForDimension = output->getDimensions();
    }

    Kernel kernel;
    if(output->getDimensions() == 2) {
        kernel = getKernel("SSIM2D");
    } else {
        if(getMainOpenCLDevice()->isWritingTo3DTexturesSupported()) {
            kernel = getKernel("SSIM3D");
        } else {
            kernel = getKernel("SSIM3DBuffer");
            kernel.setArg("channels", output->getNrOfChannels());
        }
        kernel.setArg("windowSizeZ", windowSize.z());
    }
    kernel.setArg("input1", image1);
    kernel.setArg("input2", image2);
    kernel.setArg("output", output);
    kernel.setArg("windowSizeX", windowSize.x());
    kernel.setArg("windowSizeY", windowSize.y());
    kernel.setArg("weights", m_weightsBuffer);
    const float intensityRange = m_maxValue - m_minValue;
    const float c1 = (m_k1*intensityRange)*(m_k1*intensityRange);
    const float c2 = (m_k2*intensityRange)*(m_k2*intensityRange);
    kernel.setArg("c1", c1);
    kernel.setArg("c2", c2);

    getQueue().add(kernel, output->getSize());

    float SSIM;
    if(image1->getNrOfChannels() > 1) {
        // FIXME calculateAverageIntensity doesn't support multi-channels yet, so to this for now:
        float SSIMsum = 0.0f;
        for(int i = 0; i < image1->getNrOfChannels(); ++i) {
            std::vector<int> toRemove;
            for(int j = 0; j < image1->getNrOfChannels(); ++j) {
                if(i != j)
                    toRemove.push_back(j);
            }
            auto singleChannelImage = ImageChannelConverter::create(toRemove)
                    ->connect(output)
                    ->run()
                    ->getOutput<Image>();
            SSIMsum += singleChannelImage->calculateAverageIntensity();
        }
        SSIM = SSIMsum / image1->getNrOfChannels();
    } else {
        SSIM = output->calculateAverageIntensity();
    }

    m_value = SSIM;
    addOutputData(0, output);
    auto outputValue = FloatScalar::create(m_value);
    addOutputData(1,outputValue);
}

float StructuralSimilarityIndexMeasure::get() const {
    return m_value;
}

void StructuralSimilarityIndexMeasure::setStandardDeviation(Vector3f stdDev) {
    if(stdDev != m_stdDev)
        m_recreateWeights = true;
    m_stdDev = stdDev;
    setModified(true);
}

void StructuralSimilarityIndexMeasure::setWindowSize(Vector3i size) {
    if(size != m_windowSize)
        m_recreateWeights = true;
    m_windowSize = size;
    setModified(true);
}

}
