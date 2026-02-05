#pragma once

#include "FAST/ProcessObject.hpp"

namespace fast {

/**
 * @brief Process object for resizing an image
 */
class FAST_EXPORT  ImageResizer : public ProcessObject {
	FAST_PROCESS_OBJECT(ImageResizer)
	public:
        /**
         * @brief Create instnace
         * @param width Width of new image
         * @param height Height of new image
         * @param depth Depth of new image, if 3D input
         * @param useInterpolation Whether to use linear interpolation or not
         * @param preserveAspectRatio Whether to preserve aspect ratio when resizing
         * @param blurOnDownsampling Whether to apply GaussianSmoothing on the input image before downsampling.
         *  This is crucial to avoid aliasing artifacts. The standard deviation of the smoothing is set to (d - 1) / 2
         *  where d is the downsampling factor.
         * @return instance
         */
        FAST_CONSTRUCTOR(ImageResizer,
                         int, width,,
                         int, height,,
                         int, depth, = 0,
                         bool, useInterpolation, = true,
                         bool, preserveAspectRatio, = false,
                         bool, blurOnDownsampling, = true
        );

		void setWidth(int width);
		void setHeight(int height);
		void setDepth(int depth);
		void setSize(VectorXi size);
		void setPreserveAspectRatio(bool preserve);
        void setInterpolation(bool useInterpolation);
        void loadAttributes() override;
	private:
		ImageResizer();
		void execute();

		Vector3i mSize;
		bool mPreserveAspectRatio;
        bool mInterpolationSet, mInterpolation;
        bool m_blurOnDownsampling = true;
};

}