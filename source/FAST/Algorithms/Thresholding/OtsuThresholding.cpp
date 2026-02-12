#include <FAST/Algorithms/ImageCaster/ImageCaster.hpp>
#include "OtsuThresholding.hpp"
#include "BinaryThresholding.hpp"

namespace fast {

std::array<int, 256> calculateGlobalHistogram(ImageAccess::pointer& access, const int width, const int height) {
    std::array<int, 256> hist{};
    for(int i = 0; i < width*height; ++i) {
        auto pixelValue = access->getScalarFast<uchar>(i);
        hist[pixelValue]++;
    }
    return hist;
}


std::array<float, 256> calculatePixelProbabilities(std::array<int, 256> histogram, const int pixels) {
    std::array<float, 256> probs;
    for(int i = 0; i < 256; ++i)
        probs[i] = (float)histogram[i] / (float)pixels;

    return probs;
}

std::array<float, 256> calculateCumulativePixelProbabilities(std::array<int, 256> histogram, const int pixels) {
    std::array<float, 256> cumulativeProbs;
    cumulativeProbs[0] = (float)histogram[0] / (float)pixels;
    for(int i = 1; i < 256; ++i)
        cumulativeProbs[i] = cumulativeProbs[i-1] + (float)histogram[i] / (float)pixels;

    return cumulativeProbs;
}


OtsuThresholding::OtsuThresholding(int numberOfClasses) {
    createInputPort(0);
    createOutputPort(0);
    if(numberOfClasses > 4 || numberOfClasses < 2)
        throw Exception("Otsu thresholding implementation only supports 2, 3, or 4 classes");
    m_thresholdCount = numberOfClasses - 1;
}

void OtsuThresholding::execute() {
    auto input = getInputData<Image>();

    if(input->getNrOfChannels() > 1) {
        reportWarning() << "Otsu thresholding implementation only supports 1 channel images. " <<
                           "Since your input image has more than 1 channels, only the first channel (red) is used." << reportEnd();
    }
    if(input->getDimensions() == 3) {
        throw Exception("Otsu thresholding is currently only implemented for 2D images");
    }

    if(input->getDataType() != TYPE_UINT8) {
        input = ImageCaster::create(TYPE_UINT8, 255, true)
                ->connect(input)
                ->runAndGetOutputData<Image>();
        // TODO handle issue where bins 0 and 255 become less probable due to rounding..
    }

    auto access = input->getImageAccess(ACCESS_READ);
    // TODO Move histogram operation somewhere else?
    auto histogram = calculateGlobalHistogram(access, input->getWidth(), input->getHeight());
    access->release();
    auto cumulativeProbs = calculateCumulativePixelProbabilities(histogram, input->getNrOfVoxels());
    auto probs = calculatePixelProbabilities(histogram, input->getNrOfVoxels());

    std::array<float, 256> aux;
    aux[0] = 0;
    for(int i = 1; i < 256; ++i) {
        aux[i] = aux[i-1] + i*probs[i];
    }

    if(m_thresholdCount == 1) {
        float bestInterClassVariance = 0;
        int bestThreshold;
        // Iterate over all possible thresholds
        for(int T = 1; T < 256; ++T) {
            // Calculate inter-class variance
            float w0 = cumulativeProbs[T-1];
            float w1 = cumulativeProbs[255] - cumulativeProbs[T];
            float mean0 = aux[T-1]/w0;
            float mean1 = (aux[255] - aux[T])/w1;
            float interClassVariance = w0*w1*(mean0 - mean1)*(mean0 - mean1);
            // Select threshold with highest inter-class variance
            if(interClassVariance > bestInterClassVariance) {
                bestInterClassVariance = interClassVariance;
                bestThreshold = T;
            }
        }
        // Segment using threshold
        addOutputData(0, BinaryThresholding::create(bestThreshold)->connect(input)->runAndGetOutputData<Image>());
    } else if(m_thresholdCount == 2) {
        const float meanG = aux[255];
        float bestInterClassVariance = 0;
        int bestThreshold1;
        int bestThreshold2;
        // Iterate over all possible thresholds
        for(int T1 = 1; T1 < 256-1; ++T1) {
            for(int T2 = T1+1; T2 < 256; ++T2) {
                // Calculate inter-class variance
                float w0 = cumulativeProbs[T1-1];
                float w1 = cumulativeProbs[T2-1] - cumulativeProbs[T1];
                float w2 = cumulativeProbs[255] - cumulativeProbs[T2];
                float mean0 = aux[T1-1]/w0;
                float mean1 = (aux[T2-1] - aux[T1])/w1;
                float mean2 = (aux[255] - aux[T2])/w2;
                float interClassVariance = w0*(mean0 - meanG)*(mean0 - meanG) + w1*(mean1 - meanG)*(mean1 - meanG) + w2*(mean2 - meanG)*(mean2 - meanG);
                // Select threshold with highest inter-class variance
                if(interClassVariance > bestInterClassVariance) {
                    bestInterClassVariance = interClassVariance;
                    bestThreshold1 = T1;
                    bestThreshold2 = T2;
                }
            }
        }
        // Segment using multiple threshold
        auto segmentation = Image::create(input->getWidth(), input->getHeight(), TYPE_UINT8, 1);
        segmentation->setSpacing(input->getSpacing());
        auto segmentationAccess = segmentation->getImageAccess(ACCESS_READ_WRITE);
        auto inputAccess = input->getImageAccess(ACCESS_READ);
        for(int i = 0; i < input->getNrOfVoxels(); ++i) {
            auto value = inputAccess->getScalarFast<uchar>(i);
            uchar segmentationClass = 0;
            if(value >= bestThreshold1 && value < bestThreshold2) {
                segmentationClass = 1;
            } else if(value >= bestThreshold2) {
                segmentationClass = 2;
            }
            segmentationAccess->setScalarFast(i, segmentationClass);
        }
        addOutputData(0, segmentation);
    } else if(m_thresholdCount == 3) {
        const float meanG = aux[255];
        float bestInterClassVariance = 0;
        int bestThreshold1;
        int bestThreshold2;
        int bestThreshold3;
        // Iterate over all possible thresholds
        for(int T1 = 1; T1 < 256-2; ++T1) {
            for(int T2 = T1+1; T2 < 256-1; ++T2) {
                for(int T3 = T2+1; T3 < 256; ++T3) {
                    // Calculate inter-class variance
                    float w0 = cumulativeProbs[T1-1];
                    float w1 = cumulativeProbs[T2-1] - cumulativeProbs[T1];
                    float w2 = cumulativeProbs[T3-1] - cumulativeProbs[T2];
                    float w3 = cumulativeProbs[255] - cumulativeProbs[T3];
                    float mean0 = aux[T1-1]/w0;
                    float mean1 = (aux[T2-1] - aux[T1])/w1;
                    float mean2 = (aux[T3-1] - aux[T2])/w2;
                    float mean3 = (aux[255] - aux[T3])/w3;
                    float interClassVariance = w0*(mean0 - meanG)*(mean0 - meanG) + w1*(mean1 - meanG)*(mean1 - meanG) + w2*(mean2 - meanG)*(mean2 - meanG) + w3*(mean3 - meanG)*(mean3 - meanG);
                    // Select threshold with highest inter-class variance
                    if(interClassVariance > bestInterClassVariance) {
                        bestInterClassVariance = interClassVariance;
                        bestThreshold1 = T1;
                        bestThreshold2 = T2;
                        bestThreshold3 = T3;
                    }
                }
            }
        }
        // Segment using multiple threshold
        auto segmentation = Image::create(input->getWidth(), input->getHeight(), TYPE_UINT8, 1);
        auto segmentationAccess = segmentation->getImageAccess(ACCESS_READ_WRITE);
        auto inputAccess = input->getImageAccess(ACCESS_READ);
        for(int i = 0; i < input->getNrOfVoxels(); ++i) {
            uchar value = inputAccess->getScalarFast<uchar>(i);
            uchar segmentationClass;
            if(value < bestThreshold1) {
                segmentationClass = 0;
            } else if(value >= bestThreshold1 && value < bestThreshold2) {
                segmentationClass = 1;
            } else if(value >= bestThreshold2 && value < bestThreshold3) {
                segmentationClass = 2;
            } else {
                segmentationClass = 3;
            }
            segmentationAccess->setScalarFast(i, segmentationClass);
        }
        addOutputData(0, segmentation);
    } else {
        // Not supported
        throw NotImplementedException();
    }
}
}