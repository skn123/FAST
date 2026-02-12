#include "ZeroMeanUnitVariance.hpp"
#include <FAST/Algorithms/ImageChannelConverter/ImageChannelConverter.hpp>
#include <FAST/Data/Image.hpp>

namespace fast {

ZeroMeanUnitVariance::ZeroMeanUnitVariance(bool perChannel) {
	createInputPort(0);
	createOutputPort(0);

	createOpenCLProgram(Config::getKernelSourcePath() + "Algorithms/IntensityNormalization/ZeroMeanUnitVariance.cl");
	m_perChannel = perChannel;
}

void ZeroMeanUnitVariance::execute() {
	auto input = getInputData<Image>();

	// Per channel
	std::array<float, 4> average;
	std::array<float, 4> standardDeviation;
	if(m_perChannel && input->getNrOfChannels() > 1) {
	    for(int c = 0; c < input->getNrOfChannels(); ++c) {
            // TODO this can be improved by not using ImageChannelConverter, and instead upgrade to
            //  calculateAverageIntensity functions to support per channel
	        std::vector<int> channelsToRemove;
	        for(int i = 0; i < input->getNrOfChannels(); ++i) {
	            if(i != c)
	                channelsToRemove.push_back(i);
	        }
	        auto channelImage = ImageChannelConverter::create(channelsToRemove)
	                ->connect(input)
	                ->runAndGetOutputData<Image>();
	        average[c] = channelImage->calculateAverageIntensity();
	        standardDeviation[c] = channelImage->calculateStandardDeviationIntensity();
	        std::cout << "normalizing: " << average[c] << " " << standardDeviation[c] << std::endl;
	    }
	} else {
	    float avg = input->calculateAverageIntensity();
	    float stdDev = input->calculateStandardDeviationIntensity();
        average = {avg, avg, avg, avg};
        standardDeviation = {stdDev, stdDev, stdDev, stdDev};
	}

	auto output = Image::create(input->getSize(), TYPE_FLOAT, input->getNrOfChannels());
	output->setSpacing(input->getSpacing());
	SceneGraph::setParentNode(output, input);

	auto device = std::dynamic_pointer_cast<OpenCLDevice>(getMainDevice());
	cl::Kernel kernel;
	cl::NDRange globalSize;
	if(input->getDimensions() == 2) {
        globalSize = cl::NDRange(input->getWidth(), input->getHeight());
		kernel = cl::Kernel(getOpenCLProgram(device), "normalize2D");
		auto inputAccess = input->getOpenCLImageAccess(ACCESS_READ, device);
		auto outputAccess = output->getOpenCLImageAccess(ACCESS_READ_WRITE, device);
		kernel.setArg(0, *inputAccess->get2DImage());
		kernel.setArg(1, *outputAccess->get2DImage());
	} else {
		// 3D
        globalSize = cl::NDRange(input->getWidth(), input->getHeight(), input->getDepth());
		kernel = cl::Kernel(getOpenCLProgram(device),"normalize3D");
		auto inputAccess = input->getOpenCLImageAccess(ACCESS_READ, device);

		kernel.setArg(0, *inputAccess->get3DImage());
	   if(device->isWritingTo3DTexturesSupported()) {
            auto outputAccess = output->getOpenCLImageAccess(ACCESS_READ_WRITE, device);
            kernel.setArg(1, *(outputAccess->get3DImage()));
        } else {
            auto outputAccess = output->getOpenCLBufferAccess(ACCESS_READ_WRITE, device);
            kernel.setArg(1, *(outputAccess->get()));
            kernel.setArg(4, output->getNrOfChannels());
        }
	}
	kernel.setArg(2, sizeof(cl_float4), average.data());
	kernel.setArg(3, sizeof(cl_float4), standardDeviation.data());
	
    cl::CommandQueue queue = device->getCommandQueue();

    queue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            globalSize,
            cl::NullRange
    );


	addOutputData(0, output);
}

}
