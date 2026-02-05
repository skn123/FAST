#include "ImageSharpening.hpp"

namespace fast {

void ImageSharpening::loadAttributes() {
	setGain(getFloatAttribute("gain"));
	GaussianSmoothing::loadAttributes();
}

ImageSharpening::ImageSharpening(float gain, float stddev, uchar maskSize) : GaussianSmoothing(stddev, maskSize) {
    createOpenCLProgram(Config::getKernelSourcePath() + "Algorithms/ImageSharpening/ImageSharpening.cl");
    createFloatAttribute("gain", "Unsharp masking gain", "Unsharp masking gain", m_gain);
    setGain(gain);
}

void ImageSharpening::setGain(float gain) {
    m_gain = gain;
    setModified(true);
}

void ImageSharpening::execute() {
    auto input = getInputData<Image>(0);

    if(input->getDimensions() != 2)
        throw Exception("ImageSharpening only supports 2D images");

    if(mStdDev.x() != mStdDev.y() || mMaskSize.x() != mMaskSize.y())
        throw Exception("Anistropic support not implemented in ImageSharpening");

    Vector3i maskSize = mMaskSize;
    if(maskSize == Vector3i::Zero()) { // If mask size is not set calculate it instead
        for(int i = 0; i < 3; ++i)
            maskSize[i] = std::ceil(2*mStdDev[i])*2+1;
    }

    Vector3i halfSize;
    for(int i = 0; i < 3; ++i)
        halfSize[i] = (maskSize[i] - 1) / 2;

    // Enforce max size
    for(int i = 0; i < 3; ++i)
        if(maskSize[i] > 19)
            maskSize[i] = 19;

    // Initialize output image
    Image::pointer output;
    if(mOutputTypeSet) {
        output = Image::create(input->getSize(), mOutputType, input->getNrOfChannels());
        output->setSpacing(input->getSpacing());
    } else {
        output = Image::createFromImage(input);
    }
    mOutputType = output->getDataType();
    SceneGraph::setParentNode(output, input);

	auto clDevice = std::static_pointer_cast<OpenCLDevice>(getMainDevice());

    cl::Kernel kernel(getOpenCLProgram(clDevice, "", "-DHALF_SIZE=" + std::to_string(halfSize.x())), "sharpen");

	auto inputAccess = input->getOpenCLImageAccess(ACCESS_READ, clDevice);
	//createMask(input, maskSize, false);
	kernel.setArg(2, mStdDev.x());
    kernel.setArg(3, m_gain);
	auto globalSize = cl::NDRange(input->getWidth(),input->getHeight());

	auto outputAccess = output->getOpenCLImageAccess(ACCESS_READ_WRITE, clDevice);
	kernel.setArg(0, *inputAccess->get2DImage());
	kernel.setArg(1, *outputAccess->get2DImage());
    clDevice->getCommandQueue().enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        globalSize,
        cl::NullRange
	);
    addOutputData(0, output);
}

}