#include <FAST/Data/Image.hpp>
#include <FAST/Algorithms/ImageResizer/ImageResizer.hpp>
#include "SegmentationNetwork.hpp"
#include "TensorToSegmentation.hpp"

namespace fast {

void SegmentationNetwork::loadAttributes() {
	if (getBooleanAttribute("heatmap-output")) {
		setHeatmapOutput();
	} else {
		setSegmentationOutput();
	}
	setThreshold(getFloatAttribute("threshold"));
	setChannelsToIgnore(getIntegerListAttribute("ignore-channels"));
    setResizeBackToOriginalSize(getBooleanAttribute("resize-to-original-size"));
	NeuralNetwork::loadAttributes();
}

SegmentationNetwork::SegmentationNetwork(std::string modelFilename, float scaleFactor, bool heatmapOutput,
                                         float threshold, bool hasBackgroundClass, float meanIntensity,
                                         float stanardDeviationIntensity, bool resizeBackToOrigianlSize, std::vector<NeuralNetworkNode> inputNodes,
                                         std::vector<NeuralNetworkNode> outputNodes, std::string inferenceEngine, int maxBatchSize,
                                         std::vector<std::string> customPlugins) : NeuralNetwork(modelFilename, scaleFactor, meanIntensity, stanardDeviationIntensity, inputNodes, outputNodes,inferenceEngine, maxBatchSize, customPlugins) {
    createInputPort(0, "Image");
    createOutputPort(0, "Segmentation");
    m_tensorToSegmentation = TensorToSegmentation::New();

    if(heatmapOutput) {
        setHeatmapOutput();
    } else {
        setSegmentationOutput();
    }
    setThreshold(threshold);
    setBackgroundClass(hasBackgroundClass);
    setResizeBackToOriginalSize(resizeBackToOrigianlSize);
}

SegmentationNetwork::SegmentationNetwork(std::string modelFilename, std::vector<NeuralNetworkNode> inputNodes,
                                         std::vector<NeuralNetworkNode> outputNodes, std::string inferenceEngine, int maxBatchSize,
                                         std::vector<std::string> customPlugins) : NeuralNetwork(modelFilename, inputNodes, outputNodes, inferenceEngine, maxBatchSize, customPlugins) {
    createInputPort(0);
    createOutputPort(0);

    m_tensorToSegmentation = TensorToSegmentation::create();
    mHeatmapOutput = false;
}

SegmentationNetwork::SegmentationNetwork() {
    createInputPort(0);
    createOutputPort(0);

    m_tensorToSegmentation = TensorToSegmentation::create();
    mHeatmapOutput = false;
    createBooleanAttribute("heatmap-output", "Output heatmap", "Enable heatmap output instead of segmentation", false);
    createBooleanAttribute("resize-to-original-size", "Resize to original size", "Resize output segmentation to original input size", false);
    createFloatAttribute("threshold", "Segmentation threshold", "Lower threshold of accepting a label", 0.5f);
    createIntegerAttribute("ignore-channels", "Ignore Channels", "List of channels to ignore", -1);
}

void SegmentationNetwork::setHeatmapOutput() {
    mHeatmapOutput = true;
    createOutputPort(0);
}

void SegmentationNetwork::setSegmentationOutput() {
    mHeatmapOutput = false;
    createOutputPort(0);
}

void SegmentationNetwork::setResizeBackToOriginalSize(bool resize) {
    m_resizeBackToOriginalSize = resize;
}

void SegmentationNetwork::execute() {
    runNeuralNetwork();

    auto data = m_processedOutputData[0];
    auto batch = std::dynamic_pointer_cast<Batch>(data);
    const bool outputIsBatch = batch != nullptr;
    if(mHeatmapOutput) {
        addOutputData(0, outputIsBatch ? batch : data);
    } else {
        std::vector<Tensor::pointer> outputTensors;
        if(outputIsBatch) {
            outputTensors = batch->get().getTensors();
        } else {
            outputTensors = {std::dynamic_pointer_cast<Tensor>(data)};
        }
        std::vector<Image::pointer> outputImages;

        for(const auto& tensor : outputTensors) {
            auto image = m_tensorToSegmentation
                    ->connect(tensor)
                    ->run()
                    ->getOutput<Image>();
            if(m_resizeBackToOriginalSize) {
                auto originalSize = mInputImages.begin()->second[0]->getSize().cast<int>();
                image = ImageResizer::create(originalSize.x(), originalSize.y(), originalSize.z(), false)
                        ->connect(image)
                        ->run()
                        ->getOutput<Image>();
            }
            outputImages.push_back(image);
        }
        if(outputIsBatch) {
            addOutputData(0, Batch::create(outputImages));
        } else {
            addOutputData(0, outputImages[0]);
        }
    }
    mRuntimeManager->stopRegularTimer("output_processing");
}


void SegmentationNetwork::setThreshold(float threshold) {
    m_tensorToSegmentation->setThreshold(threshold);
}

float SegmentationNetwork::getThreshold() const {
    return m_tensorToSegmentation->getThreshold();
}

void SegmentationNetwork::setBackgroundClass(bool hasBackgroundClass) {
    m_tensorToSegmentation->setBackgroundClass(hasBackgroundClass);
}

void SegmentationNetwork::setChannelsToIgnore(std::vector<int> channels) {
    m_tensorToSegmentation->setChannelsToIgnore(channels);
}

}