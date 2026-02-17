#include "FAST/Algorithms/GaussianSmoothing/GaussianSmoothing.hpp"
#include "FAST/Exception.hpp"
#include "FAST/DeviceManager.hpp"
#include "FAST/Data/Image.hpp"

namespace fast {

void GaussianSmoothing::setMaskSize(int maskSize) {
    setMaskSize({maskSize, maskSize, maskSize});
}

void GaussianSmoothing::setOutputType(DataType type) {
    mOutputType = type;
    mOutputTypeSet = true;
    mIsModified = true;
}

void GaussianSmoothing::setStandardDeviation(float stdDev) {
    setStandardDeviation({stdDev, stdDev, stdDev});

}

GaussianSmoothing::GaussianSmoothing(std::vector<float> stdDev, std::vector<int> maskSize) {
    createInputPort(0, "Image");
    createOutputPort(0, "Image");
    createOpenCLProgram(Config::getKernelSourcePath() + "Algorithms/GaussianSmoothing/GaussianSmoothing2D.cl", "2D");
    createOpenCLProgram(Config::getKernelSourcePath() + "Algorithms/GaussianSmoothing/GaussianSmoothing3D.cl", "3D");
    createFloatAttribute("stdev", "Standard deviation", "Standard deviation", 0.5f);
    mIsModified = true;
    mRecreateMask = true;
    mDimensionCLCodeCompiledFor = 0;
    mMask = NULL;
    mOutputTypeSet = false;
    setStandardDeviation(stdDev);
    setMaskSize(maskSize);
}


GaussianSmoothing::GaussianSmoothing(float stdDev, uchar maskSize) : GaussianSmoothing({stdDev,stdDev,stdDev}, {maskSize,maskSize,maskSize}) {
}

GaussianSmoothing::~GaussianSmoothing() {
}

// TODO have to set mRecreateMask to true if input change dimension
void GaussianSmoothing::createMask(Image::pointer input, Vector3i maskSize, bool useSeperableFilter) {
    if(!mRecreateMask)
        return;

    Vector3i halfSize;
    for(int i = 0; i < 3; ++i)
        halfSize[i] = (maskSize[i] - 1) / 2;
    float sum = 0.0f;

    if(input->getDimensions() == 2) {
        mMask = std::make_unique<float[]>(maskSize.x()*maskSize.y());

        for(int x = -halfSize.x(); x <= halfSize.x(); ++x) {
            for(int y = -halfSize.y(); y <= halfSize.y(); ++y) {
                float value = std::exp(-(
                        (mStdDev.x() == 0.0f ? 0.0f : (float)(x*x)/(2.0f*mStdDev.x()*mStdDev.x())) +
                        (mStdDev.y() == 0.0f ? 0.0f : (float)(y*y)/(2.0f*mStdDev.y()*mStdDev.y()))
                        ));
                mMask[x+halfSize.x()+(y+halfSize.y())*maskSize.x()] = value;
                sum += value;
            }
        }

        for(int i = 0; i < maskSize.x()*maskSize.y(); ++i)
            mMask[i] /= sum;
    } else if(input->getDimensions() == 3) {
        // Use separable filtering for 3D
        if(useSeperableFilter && mStdDev.x() == mStdDev.y() && mStdDev.y() == mStdDev.z()) {
            mMask = std::make_unique<float[]>(maskSize.x());

            for(int x = -halfSize.x(); x <= halfSize.x(); ++x) {
                float value = std::exp(-(float)(x*x)/(2.0f*mStdDev.x())); // TODO Correct?
                mMask[x+halfSize.x()] = value;
                sum += value;
            }

            for(int i = 0; i < maskSize.x(); ++i)
                mMask[i] /= sum;
        } else {
            mMask = std::make_unique<float[]>(maskSize.x()*maskSize.y()*maskSize.z());

            for(int x = -halfSize.x(); x <= halfSize.x(); ++x) {
                for(int y = -halfSize.y(); y <= halfSize.y(); ++y) {
                    for(int z = -halfSize.z(); z <= halfSize.z(); ++z) {
                        float value = std::exp(-(
                                (mStdDev.x() == 0.0f ? 0.0f : (float)(x*x)/(2.0f*mStdDev.x()*mStdDev.x())) +
                                (mStdDev.y() == 0.0f ? 0.0f : (float)(y*y)/(2.0f*mStdDev.y()*mStdDev.y())) +
                                (mStdDev.z() == 0.0f ? 0.0f : (float)(z*z)/(2.0f*mStdDev.z()*mStdDev.z()))
                                ));
                        mMask[x+halfSize.x()+(y+halfSize.y())*maskSize.x()+(z+halfSize.z())*maskSize.x()*maskSize.y()] = value;
                        sum += value;
                    }
                }
            }

            for(int i = 0; i < maskSize.x()*maskSize.y()*maskSize.z(); ++i)
                mMask[i] /= sum;
        }
    }

    ExecutionDevice::pointer device = getMainDevice();
    if(!device->isHost()) {
        OpenCLDevice::pointer clDevice = std::dynamic_pointer_cast<OpenCLDevice>(device);
        uint bufferSize;
        if(useSeperableFilter) {
            bufferSize = maskSize.x();
        } else {
            bufferSize = input->getDimensions() == 2 ? maskSize.x()*maskSize.y() : maskSize.x()*maskSize.y()*maskSize.z();
        }
        mCLMask = cl::Buffer(
                clDevice->getContext(),
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(float)*bufferSize,
                mMask.get()
        );
    }

    mRecreateMask = false;
}

void GaussianSmoothing::recompileOpenCLCode(Image::pointer input) {
    // Check if there is a need to recompile OpenCL code
    if(input->getDimensions() == mDimensionCLCodeCompiledFor &&
            input->getDataType() == mTypeCLCodeCompiledFor)
        return;

    auto device = getMainOpenCLDevice();
    std::string buildOptions = "";
    if(!(device->isWritingTo3DTexturesSupported() && mStdDev.x() == mStdDev.y() && mStdDev.x() == mStdDev.z() && input->getDimensions() == 3)) {
        buildOptions = "-DTYPE=" + getCTypeAsString(mOutputType);
    }
    cl::Program program;
    if(input->getDimensions() == 2) {
        program = getOpenCLProgram(device, "2D", buildOptions);
    } else {
        program = getOpenCLProgram(device, "3D", buildOptions);
    }
    if(device->isWritingTo3DTexturesSupported() && mStdDev.x() == mStdDev.y() && mStdDev.x() == mStdDev.z() && input->getDimensions() == 3) {
        mKernel = cl::Kernel(program, "gaussianSmoothingSeparable");
    } else {
        mKernel = cl::Kernel(program, "gaussianSmoothing");
    }
    mDimensionCLCodeCompiledFor = input->getDimensions();
    mTypeCLCodeCompiledFor = input->getDataType();
}

template <class T>
void executeAlgorithmOnHost(Image::pointer input, Image::pointer output, const float* const mask, Vector3i maskSize) {
    // TODO: this method currently only processes the first component
    unsigned int nrOfComponents = input->getNrOfChannels();
    ImageAccess::pointer inputAccess = input->getImageAccess(ACCESS_READ);
    ImageAccess::pointer outputAccess = output->getImageAccess(ACCESS_READ_WRITE);

    Vector3i halfSize;
    for(int i = 0; i < 3; ++i)
        halfSize[i] = (maskSize[i] - 1) / 2;

    T * inputData = (T*)inputAccess->get();
    T * outputData = (T*)outputAccess->get();

    unsigned int width = input->getWidth();
    unsigned int height = input->getHeight();
    if(input->getDimensions() == 3) {
        unsigned int depth = input->getDepth();
        for(unsigned int z = 0; z < depth; ++z) {
            for(unsigned int y = 0; y < height; ++y) {
                for(unsigned int x = 0; x < width; ++x) {

                    if(x < halfSize.x() || x >= width-halfSize.x() ||
                    y < halfSize.y() || y >= height-halfSize.y() ||
                    z < halfSize.z() || z >= depth-halfSize.z()) {
                        // on border only copy values
                        outputData[x*nrOfComponents+y*nrOfComponents*width+z*nrOfComponents*width*height] = inputData[x*nrOfComponents+y*nrOfComponents*width+z*nrOfComponents*width*height];
                        continue;
                    }

                    double sum = 0.0;
                    for(int c = -halfSize.z(); c <= halfSize.z(); ++c) {
                        for(int b = -halfSize.y(); b <= halfSize.y(); ++b) {
                            for(int a = -halfSize.x(); a <= halfSize.x(); ++a) {
                                sum += mask[a+halfSize.x()+(b+halfSize.y())*maskSize.x()+(c+halfSize.z())*maskSize.x()*maskSize.y()]*
                                        inputData[(x+a)*nrOfComponents+(y+b)*nrOfComponents*width+(z+c)*nrOfComponents*width*height];
                            }}}
                    outputData[x*nrOfComponents+y*nrOfComponents*width+z*nrOfComponents*width*height] = (T)sum;
                }
            }
        }
    } else {
        for(int y = halfSize.y(); y < height-halfSize.y(); ++y) {
            for(int x = halfSize.x(); x < width-halfSize.x(); ++x) {

                if(x < halfSize.x() || x >= width-halfSize.x() ||
                y < halfSize.y() || y >= height-halfSize.y()) {
                    // on border only copy values
                    outputData[x*nrOfComponents+y*nrOfComponents*width] = inputData[x*nrOfComponents+y*nrOfComponents*width];
                    continue;
                }

                double sum = 0.0;
                for(int b = -halfSize.y(); b <= halfSize.y(); ++b) {
                    for(int a = -halfSize.x(); a <= halfSize.x(); ++a) {
                        sum += mask[a+halfSize.x()+(b+halfSize.y())*maskSize.x()]*
                                inputData[(x+a)*nrOfComponents+(y+b)*nrOfComponents*width];
                    }
                }
                outputData[x*nrOfComponents+y*nrOfComponents*width] = (T)sum;
            }
        }
    }
}

void GaussianSmoothing::execute() {
    auto input = getInputData<Image>(0);

    Vector3i maskSize = mMaskSize;
    if(maskSize == Vector3i::Zero()) { // If mask size is not set calculate it instead
        for(int i = 0; i < 3; ++i)
            maskSize[i] = std::max(3, (int)std::ceil(2*mStdDev[i])*2+1);
    }

    // Enforce max size
    for(int i = 0; i < 3; ++i)
        if(maskSize[i] > 19)
            maskSize[i] = 19;

    // Initialize output image
    ExecutionDevice::pointer device = getMainDevice();
    Image::pointer output;
    if(mOutputTypeSet) {
        output = Image::create(input->getSize(), mOutputType, input->getNrOfChannels());
        output->setSpacing(input->getSpacing());
    } else {
        output = Image::createFromImage(input);
    }
    mOutputType = output->getDataType();
    SceneGraph::setParentNode(output, input);

    if(device->isHost()) {
        createMask(input, maskSize, false);
        switch(input->getDataType()) {
            fastSwitchTypeMacro(executeAlgorithmOnHost<FAST_TYPE>(input, output, mMask.get(), maskSize));
        }
    } else {
        OpenCLDevice::pointer clDevice = std::static_pointer_cast<OpenCLDevice>(device);

        recompileOpenCLCode(input);

        cl::NDRange globalSize;

        OpenCLImageAccess::pointer inputAccess = input->getOpenCLImageAccess(ACCESS_READ, clDevice);
        if(input->getDimensions() == 2) {
            createMask(input, maskSize, false);
            mKernel.setArg(1, mCLMask);
            mKernel.setArg(3, maskSize.x());
            mKernel.setArg(4, maskSize.y());
            globalSize = cl::NDRange(input->getWidth(),input->getHeight());

            OpenCLImageAccess::pointer outputAccess = output->getOpenCLImageAccess(ACCESS_READ_WRITE, clDevice);
            mKernel.setArg(0, *inputAccess->get2DImage());
            mKernel.setArg(2, *outputAccess->get2DImage());
            clDevice->getCommandQueue().enqueueNDRangeKernel(
                    mKernel,
                    cl::NullRange,
                    globalSize,
                    cl::NullRange
            );
        } else {
            // Create an auxilliary image
            auto output2 = Image::createFromImage(output);

            globalSize = cl::NDRange(input->getWidth(),input->getHeight(),input->getDepth());

            if(clDevice->isWritingTo3DTexturesSupported() && mStdDev.x() == mStdDev.y() && mStdDev.x() == mStdDev.z()) {
                createMask(input, maskSize, true);
                mKernel.setArg(1, mCLMask);
                mKernel.setArg(3, maskSize.x());
                OpenCLImageAccess::pointer outputAccess = output->getOpenCLImageAccess(ACCESS_READ_WRITE, clDevice);
                OpenCLImageAccess::pointer outputAccess2 = output2->getOpenCLImageAccess(ACCESS_READ_WRITE, clDevice);

                cl::Image3D* image2;
                cl::Image3D* image;
                image = outputAccess->get3DImage();
                image2 = outputAccess->get3DImage();
                for(uchar direction = 0; direction < input->getDimensions(); ++direction) {
                    if(direction == 0) {
                        mKernel.setArg(0, *inputAccess->get3DImage());
                        mKernel.setArg(2, *image);
                    } else if(direction == 1) {
                        mKernel.setArg(0, *image);
                        mKernel.setArg(2, *image2);
                    } else {
                        mKernel.setArg(0, *image2);
                        mKernel.setArg(2, *image);
                    }
                    mKernel.setArg(4, direction);
                    clDevice->getCommandQueue().enqueueNDRangeKernel(
                            mKernel,
                            cl::NullRange,
                            globalSize,
                            cl::NullRange
                    );
                }
            } else {
                createMask(input, maskSize, false);
                mKernel.setArg(1, mCLMask);
                mKernel.setArg(3, maskSize.x());
                mKernel.setArg(4, maskSize.y());
                mKernel.setArg(5, maskSize.z());
                mKernel.setArg(6, input->getNrOfChannels());
                OpenCLBufferAccess::pointer outputAccess = output->getOpenCLBufferAccess(ACCESS_READ_WRITE, clDevice);
                mKernel.setArg(0, *inputAccess->get3DImage());
                mKernel.setArg(2, *outputAccess->get());
                clDevice->getCommandQueue().enqueueNDRangeKernel(
                        mKernel,
                        cl::NullRange,
                        globalSize,
                        cl::NullRange
                );
            }


        }
    }
    addOutputData(0, output);
}

void GaussianSmoothing::waitToFinish() {
    if(!getMainDevice()->isHost()) {
        OpenCLDevice::pointer device = std::dynamic_pointer_cast<OpenCLDevice>(getMainDevice());
        device->getCommandQueue().finish();
    }
}

void GaussianSmoothing::loadAttributes() {
    setStandardDeviation(getFloatAttribute("stdev"));
}

void GaussianSmoothing::setStandardDeviation(std::vector<float> stdDev) {
    if(stdDev.empty())
        throw Exception("No stddev given to GaussianSmoothing");
    for(int i = 0; i < std::min(3, (int)stdDev.size()); ++i) {
        if(stdDev[i] < 0)
            throw Exception("Standard deviation of GaussianSmoothing can't be less than 0.");
        mStdDev[i] = stdDev[i];
    }

    mIsModified = true;
    mRecreateMask = true;
}

void GaussianSmoothing::setMaskSize(std::vector<int> maskSize) {
    mMaskSize = Vector3i::Zero();
    for(int i = 0; i < std::min(3, (int)maskSize.size()); ++i) {
        if(maskSize[i] < 0)
            throw Exception("Mask size of GaussianSmoothing can't be less than 0.");
        if(maskSize[i] > 0 && maskSize[i] % 2 != 1)
            throw Exception("Mask size of GaussianSmoothing must be odd.");
        mMaskSize[i] = maskSize[i];
    }

    mIsModified = true;
    mRecreateMask = true;
}


};