#include <FAST/Data/ImagePyramid.hpp>
#include <FAST/Data/Image.hpp>
#include <FAST/Algorithms/ImageResizer/ImageResizer.hpp>
#include "PatchGenerator.hpp"

namespace fast {

PatchGenerator::PatchGenerator() {
    createInputPort<SpatialDataObject>(0); // Either ImagePyramid or Image/Volume
    createInputPort<Image>(1, false); // Optional mask
    createOutputPort<Image>(0);

    m_width = -1;
    m_height = -1;
    m_depth = -1;
    m_stop = false;
    m_streamIsStarted = false;
    m_firstFrameIsInserted = false;
    m_level = 0;
    mIsModified = true;

    createIntegerAttribute("patch-size", "Patch size", "", 0);
    createIntegerAttribute("patch-level", "Patch level", "Patch level used for image pyramid inputs", m_level);
    createFloatAttribute("patch-magnification", "Patch magnification", "Patch magnification to be used for image pyramid inputs", m_magnification);
    createFloatAttribute("patch-overlap", "Patch overlap", "Patch overlap in percent", m_overlapPercent);
    createFloatAttribute("mask-threshold", "Mask threshold", "Threshold, in percent, for how much of the candidate patch must be inside the mask to be accepted", m_maskThreshold);
    createIntegerAttribute("padding-value", "Padding value", "Value to pad patches with when out-of-bounds. Default is negative, meaning it will use (white)255 for color images, and (black)0 for grayscale images", m_paddingValue);
}

PatchGenerator::PatchGenerator(int width, int height, int depth, int level, float magnification, float percent, float maskThreshold, int paddingValue) : PatchGenerator() {
    setPatchSize(width, height, depth);
    setPatchLevel(level);
    setOverlap(percent);
    setMaskThreshold(maskThreshold);
    setPaddingValue(paddingValue);
    setPatchMagnification(magnification);
}

void PatchGenerator::loadAttributes() {
    auto patchSize = getIntegerListAttribute("patch-size");
    if(patchSize.size() == 2) {
        setPatchSize(patchSize[0], patchSize[1]);
    } else if(patchSize.size() == 3) {
        setPatchSize(patchSize[0], patchSize[1], patchSize[2]);
    } else {
        throw Exception("Incorrect number of size parameters in patch-size. Expected 2 or 3");
    }

    setPatchLevel(getIntegerAttribute("patch-level"));
    setOverlap(getFloatAttribute("patch-overlap"));
    setMaskThreshold(getFloatAttribute("mask-threshold"));
    setPaddingValue(getIntegerAttribute("padding-value"));
    setPatchMagnification(getFloatAttribute("patch-magnification"));
}

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return std::move(out).str();
}

void PatchGenerator::generateStream() {
    try {
        Image::pointer previousPatch;

        // TODO implement support for different overlap in different dimensions
        int overlapInPixelsX = (int) std::round(m_overlapPercent * (float) m_width);
        int overlapInPixelsY = (int) std::round(m_overlapPercent * (float) m_height);
        int overlapInPixelsZ = (int) std::round(m_overlapPercent * (float) m_depth);
        int patchWidthWithoutOverlap = m_width - overlapInPixelsX * 2;
        int patchHeightWithoutOverlap = m_height - overlapInPixelsY * 2;
        int patchDepthWithoutOverlap = m_depth - overlapInPixelsZ * 2;

        if(m_inputImagePyramid) {
            if(m_width % 2 != 0 || m_height % 2 != 0)
                throw Exception("Patch size must be dividable by 2");

            int level = m_level;
            float resampleFactor = 1.0f;
            if(m_magnification > 0) {
                try {
                    std::tie(level, resampleFactor) = m_inputImagePyramid->getClosestLevelForMagnification(m_magnification, 0.1);
                    if(resampleFactor != 1.0f)
                        reportWarning() << "Requested magnification " << m_magnification << " does not exist in a level of the image pyramid. " <<
                                        "Will now try to sample from a lower level and resize. This may increase runtime." << reportEnd();
                } catch(Exception &e) {
                    throw Exception("Unable to generate patches for magnification level " +
                    std::to_string(m_magnification) + " because level 0 was at a lower magnification ");
                }
                reportInfo() << "Sampling patches from level " << level << " and using a resampling factor of " << resampleFactor << reportEnd();
            }

            const int levelWidth = m_inputImagePyramid->getLevelWidth(level);
            const int levelHeight = m_inputImagePyramid->getLevelHeight(level);
            const int TIFFmultiplumCriteria = 16;
            if(patchWidthWithoutOverlap % TIFFmultiplumCriteria > 0 || patchHeightWithoutOverlap % TIFFmultiplumCriteria > 0) {
                // Resulting patch size must be a multiple of 16
                overlapInPixelsX = overlapInPixelsX + (patchWidthWithoutOverlap % TIFFmultiplumCriteria)/2;
                overlapInPixelsY = overlapInPixelsY + (patchHeightWithoutOverlap % TIFFmultiplumCriteria)/2;
                reportWarning() << "Patch size must be a multiple of " << TIFFmultiplumCriteria << " (TIFF limitation). Adding some overlap (" << overlapInPixelsX << ", " << overlapInPixelsY << ") to fix." << reportEnd();
                patchWidthWithoutOverlap = m_width - overlapInPixelsX * 2;
                patchHeightWithoutOverlap = m_height - overlapInPixelsY * 2;
                if(patchWidthWithoutOverlap % TIFFmultiplumCriteria > 0 || patchHeightWithoutOverlap % TIFFmultiplumCriteria > 0)
                    throw Exception("Error in compensation of patch size..");
            }
            const int patchesX = std::ceil((float) levelWidth / (float) (patchWidthWithoutOverlap*resampleFactor));
            const int patchesY = std::ceil((float) levelHeight / (float) (patchHeightWithoutOverlap*resampleFactor));

            for(int patchY = 0; patchY < patchesY; ++patchY) {
                for(int patchX = 0; patchX < patchesX; ++patchX) {
                    mRuntimeManager->startRegularTimer("create patch");
                    int patchWidth = m_width*resampleFactor;
                    if(patchWidth + (patchX*patchWidthWithoutOverlap - overlapInPixelsX)*resampleFactor >= levelWidth) {
                        patchWidth = levelWidth - (patchX * patchWidthWithoutOverlap - overlapInPixelsX)*resampleFactor;
                    }
                    int patchHeight = m_height*resampleFactor;
                    if(patchHeight + (patchY*patchHeightWithoutOverlap - overlapInPixelsY)*resampleFactor >= levelHeight) {
                        patchHeight = levelHeight - (patchY * patchHeightWithoutOverlap - overlapInPixelsY)*resampleFactor;
                    }
                    int patchOffsetX = (patchX * patchWidthWithoutOverlap - overlapInPixelsX)*resampleFactor;
                    int patchOffsetY = (patchY * patchHeightWithoutOverlap - overlapInPixelsY)*resampleFactor;

                    if(m_inputMask) {
                        // If a mask exist, check if this patch should be included or not
                        // At least half of the patch should be clasified as foreground
                        auto access = m_inputMask->getImageAccess(ACCESS_READ);
                        // Calculate physical position and size
                        float x = patchOffsetX * m_inputImagePyramid->getLevelScale(level) * m_inputImagePyramid->getSpacing().x();
                        float y = patchOffsetY * m_inputImagePyramid->getLevelScale(level) * m_inputImagePyramid->getSpacing().y();
                        float width = patchWidth * m_inputImagePyramid->getLevelScale(level) * m_inputImagePyramid->getSpacing().x();
                        float height = patchHeight * m_inputImagePyramid->getLevelScale(level) * m_inputImagePyramid->getSpacing().y();
                        try {
                            int cropSizeX = std::max((int)std::floor(width/m_inputMask->getSpacing().x()), 1);
                            int cropSizeY = std::max((int)std::floor(height/m_inputMask->getSpacing().y()), 1);
                            int offsetX = std::floor(x/m_inputMask->getSpacing().x());
                            int offsetY = std::floor(y/m_inputMask->getSpacing().x());
                            auto croppedMask = m_inputMask->crop(
                                    Vector2i(offsetX, offsetY),
                                    Vector2i(cropSizeX, cropSizeY)
                            );
                            float average = croppedMask->calculateAverageIntensity();
                            if(average < m_maskThreshold)  // A specific percentage of the mask has to be foreground to be assessed
                                continue;
                        } catch(Exception &e) {
                            reportInfo() << "Skipped patch because mask cropping gave error: " << e.what() << reportEnd();
                            continue;
                        }
                    }
                    reportInfo() << "Generating patch " << patchX << " " << patchY << reportEnd();
                    auto access = m_inputImagePyramid->getAccess(ACCESS_READ);
                    if(patchWidth < overlapInPixelsX*2 || patchHeight < overlapInPixelsY*2)
                        continue;
                    auto patch = access->getPatchAsImage(level,
                                                         patchOffsetX < 0 ? 0 : patchOffsetX, // if there is overlap, we will have negative offset at edges
                                                         patchOffsetY < 0 ? 0 : patchOffsetY,
                                                         patchWidth + (patchOffsetX < 0 ? patchOffsetX : 0), // We have to reduce width and height if negative offset
                                                         patchHeight + (patchOffsetY < 0 ? patchOffsetY : 0));

                    // If patch does not have correct size, pad it
                    int paddingValue = m_paddingValue;
                    if(m_paddingValue < 0) {
                        if(m_inputImagePyramid->getNrOfChannels() > 1) {
                            paddingValue = 255;
                        } else {
                            paddingValue = 0;
                        }
                    }
                    if(patchOffsetX < 0 || patchOffsetY < 0 || patch->getWidth() != (int)(m_width*resampleFactor) || patch->getHeight() != (int)(m_height*resampleFactor)) {
                        // Edge cases, patches may not be the target patch size. Need to pad.
                        patch = patch->crop(Vector2i(patchOffsetX < 0 ? patchOffsetX : 0, patchOffsetY < 0 ? patchOffsetY : 0), Vector2i(m_width*resampleFactor, m_height*resampleFactor), true, paddingValue);
                    }
                    if(resampleFactor > 1.0f) {
                        patch = ImageResizer::create(m_width, m_height, 1, m_inputImagePyramid->getNrOfChannels() > 1)->connect(patch)->runAndGetOutputData<Image>();
                    }

                    // Store some frame data useful for patch stitching
                    patch->setFrameData("original-width", std::to_string(round(levelWidth/resampleFactor)));
                    patch->setFrameData("original-height", std::to_string(round(levelHeight/resampleFactor)));
                    patch->setFrameData("patchid-x", std::to_string(patchX));
                    patch->setFrameData("patchid-y", std::to_string(patchY));
                    // Target width/height of patches
                    patch->setFrameData("patch-width", std::to_string(m_width));
                    patch->setFrameData("patch-height", std::to_string(m_height));
                    patch->setFrameData("patch-overlap-x", std::to_string(overlapInPixelsX));
                    patch->setFrameData("patch-overlap-y", std::to_string(overlapInPixelsY));
                    // Image patch spacing of a WSI can be very small, and std::to_string can round the numbers,
                    // and there is no way to set the precision, so we use a custom function instead.
                    patch->setFrameData("patch-spacing-x", to_string_with_precision(patch->getSpacing().x(), 32));
                    patch->setFrameData("patch-spacing-y", to_string_with_precision(patch->getSpacing().y(), 32));
                    patch->setFrameData("patch-level", std::to_string(level));
                    m_progress = (float)(patchX+patchY*patchesX)/(patchesX*patchesY);
                    patch->setFrameData("progress", std::to_string(m_progress));
                    patch->setFrameData("streaming", "yes"); // Since we are not propagating frame data, we have to set this

                    mRuntimeManager->stopRegularTimer("create patch");
                    try {
                        if(previousPatch) {
                            addOutputData(0, previousPatch, false, false);
                            frameAdded();
                        }
                    } catch(ThreadStopped &e) {
                        std::unique_lock<std::mutex> lock(m_stopMutex);
                        m_stop = true;
                        break;
                    }
                    previousPatch = patch;
                    std::unique_lock<std::mutex> lock(m_stopMutex);
                    if(m_stop)
                        break;
                }
                std::unique_lock<std::mutex> lock(m_stopMutex);
                if(m_stop) {
                    //m_streamIsStarted = false;
                    m_firstFrameIsInserted = false;
                    break;
                }
            }
        } else if(m_inputVolume) { // Could be 3D or 2D
            const int width = m_inputVolume->getWidth();
            const int height = m_inputVolume->getHeight();
            const int depth = m_inputVolume->getDepth();
            auto transformData = SceneGraph::getEigenTransformFromData(m_inputVolume).data();
            std::string transformString;
            for(int i = 0; i < 16; ++i)
                transformString += std::to_string(transformData[i]) + " ";

            const int patchesX = std::ceil((float) width / (float) patchWidthWithoutOverlap);
            const int patchesY = std::ceil((float) height / (float) patchHeightWithoutOverlap);
            const int patchesZ = std::ceil((float) depth / (float) patchDepthWithoutOverlap);

            for(int patchZ = 0; patchZ < patchesZ; ++patchZ) {
                for(int patchY = 0; patchY < patchesY; ++patchY) {
                    for(int patchX = 0; patchX < patchesX; ++patchX) {
                        mRuntimeManager->startRegularTimer("create patch");

                        int patchWidth = m_width;
                        if(patchX*patchWidthWithoutOverlap + patchWidth - overlapInPixelsX > width) {
                            patchWidth = width - patchX * patchWidthWithoutOverlap + overlapInPixelsX - 1;
                        }
                        int patchHeight = m_height;
                        if(patchY*patchHeightWithoutOverlap + patchHeight - overlapInPixelsY > height) {
                            patchHeight = height - patchY * patchHeightWithoutOverlap + overlapInPixelsY - 1;
                        }
                        int patchDepth = m_depth;
                        if(patchZ*patchDepthWithoutOverlap + patchDepth - overlapInPixelsZ > depth) {
                            patchDepth = depth - patchZ * patchDepthWithoutOverlap + overlapInPixelsZ - 1;
                        }

                        int x = patchX * patchWidthWithoutOverlap - overlapInPixelsX;
                        int y = patchY * patchHeightWithoutOverlap - overlapInPixelsY;
                        int z = patchZ * patchDepthWithoutOverlap - overlapInPixelsZ;

                        reportInfo() << "Creating image patch at offset " << x << " " << y << " " << z << " with size " << patchWidth << " " << patchHeight << " " << patchDepth << reportEnd();
                        int paddingValue = m_paddingValue;
                        if(m_paddingValue < 0) {
                            if(m_inputVolume->getNrOfChannels() > 1) {
                                paddingValue = 255;
                            } else {
                                paddingValue = 0;
                            }
                        }
                        auto patch = m_inputVolume->crop(Vector3i(x, y, z), Vector3i(m_width, m_height, m_depth), true, paddingValue);
                        patch->setFrameData("original-width", std::to_string(width));
                        patch->setFrameData("original-height", std::to_string(height));
                        patch->setFrameData("original-depth", std::to_string(depth));
                        patch->setFrameData("original-transform", transformString);
                        patch->setFrameData("patch-offset-x", std::to_string(x));
                        patch->setFrameData("patch-offset-y", std::to_string(y));
                        patch->setFrameData("patch-offset-z", std::to_string(z));
                        patch->setFrameData("patch-width", std::to_string(m_width));
                        patch->setFrameData("patch-height", std::to_string(m_height));
                        patch->setFrameData("patch-depth", std::to_string(m_depth));
                        patch->setFrameData("patchid-x", std::to_string(patchX));
                        patch->setFrameData("patchid-y", std::to_string(patchY));
                        patch->setFrameData("patchid-z", std::to_string(patchZ));
                        patch->setFrameData("patch-overlap-x", std::to_string(overlapInPixelsX));
                        patch->setFrameData("patch-overlap-y", std::to_string(overlapInPixelsY));
                        patch->setFrameData("patch-overlap-z", std::to_string(overlapInPixelsZ));
                        Vector3f spacing = m_inputVolume->getSpacing();
                        patch->setFrameData("patch-spacing-x", std::to_string(spacing.x()));
                        patch->setFrameData("patch-spacing-y", std::to_string(spacing.y()));
                        patch->setFrameData("patch-spacing-z", std::to_string(spacing.z()));
                        m_progress = ((float)(patchX+patchY*patchesX+patchZ*patchesX*patchesY)/(patchesX*patchesY*patchesZ));
                        patch->setFrameData("progress", std::to_string(m_progress));
                        patch->setFrameData("streaming", "yes"); // Since we are not propagating frame data, we have to set this
                        try {
                            if(previousPatch) {
                                addOutputData(0, previousPatch, false, false);
                                frameAdded();
                                //std::this_thread::sleep_for(std::chrono::seconds(2));
                            }
                        } catch(ThreadStopped &e) {
                            std::unique_lock<std::mutex> lock(m_stopMutex);
                            m_stop = true;
                            break;
                        }
                        mRuntimeManager->stopRegularTimer("create patch");
                        previousPatch = patch;
                        std::unique_lock<std::mutex> lock(m_stopMutex);
                        if(m_stop)
                            break;
                    }}}
        } else {
            throw Exception("Unsupported data object given to PatchGenerator");
        }
        // Add final patch, and mark it has last frame
        previousPatch->setLastFrame(getNameOfClass());
		previousPatch->setFrameData("streaming", "yes"); // Since we are not propagating frame data, we have to set this
        try {
            addOutputData(0, previousPatch, false, false);
        } catch(ThreadStopped &e) {

        }
        reportInfo() << "Done generating patches" << reportEnd();
    } catch(std::exception &e) {
        // Exception happened in thread. Stop pipeline, and propagate error message.
        for(auto item : mOutputConnections) {
            for(auto output : item.second) {
                output.lock()->stop(e.what());
            }
        }
        frameAdded(); // To unlock if happens before first frame
    }
}

void PatchGenerator::execute() {
    if(m_width <= 0 || m_height <= 0 || m_depth <= 0)
        throw Exception("Width, height and depth must be set to a positive number");

    auto input = getInputData<SpatialDataObject>();
    m_inputImagePyramid = std::dynamic_pointer_cast<ImagePyramid>(input);
    m_inputVolume = std::dynamic_pointer_cast<Image>(input);

    if(mInputConnections.count(1) > 0) {
        // If a mask was given store it
        m_inputMask = getInputData<Image>(1, false);
    }

    startStream();
    waitForFirstFrame();
}

void PatchGenerator::setPatchSize(int width, int height, int depth) {
    m_width = width;
    m_height = height;
    m_depth = depth;
    mIsModified = true;
}

void PatchGenerator::setPatchLevel(int level) {
    m_level = level;
    mIsModified = true;
}

void PatchGenerator::setOverlap(float percent) {
    if(percent < 0 || percent > 1)
        throw Exception("Overlap percent must be >= 0 && <= 1");
    m_overlapPercent = percent;
    mIsModified = true;
}

void PatchGenerator::setMaskThreshold(float percent) {
    if(percent < 0 || percent > 1)
        throw Exception("Mask threshold must be >= 0 && <= 1");
    m_maskThreshold = percent;
    mIsModified = true;
}

void PatchGenerator::setPaddingValue(int paddingValue) {
    m_paddingValue = paddingValue;
    setModified(true);
}

void PatchGenerator::setPatchMagnification(float magnification) {
    m_magnification = magnification;
    setModified(true);
}

float PatchGenerator::getProgress() {
    return m_progress;
}

}
