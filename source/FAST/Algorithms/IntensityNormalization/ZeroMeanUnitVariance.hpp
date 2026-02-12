#pragma once

#include <FAST/ProcessObject.hpp>

namespace fast {

/**
 * @brief Normalize intensities of an image to have zero mean and unit variance
 *
 * This process object will scale the pixel values so that the resulting image
 * has a zero mean and unit variance.
 * This achieved by doing (image - mean(image)) / std(image)
 *
 * Inputs:
 * - 0: Image
 *
 * Outputs:
 * - 0: Image float
 *
 * @ingroup filter
 */
class FAST_EXPORT ZeroMeanUnitVariance : public ProcessObject {
	FAST_PROCESS_OBJECT(ZeroMeanUnitVariance)
	public:
        /**
         * @brief Create instance
         * @param perChannel Whether to apply per channel, e.g. calculating the average and standard deviation per channel
         *      and apply this per channel, or use the average and standard deviation of all channels.
         * @return instance
         */
		FAST_CONSTRUCTOR(ZeroMeanUnitVariance, bool, perChannel, = true)
	private:
		void execute() override;
		bool m_perChannel = true;
};
  
}
