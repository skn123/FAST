#pragma once

#include <FAST/ProcessObject.hpp>

namespace fast {

/**
 * @brief Invert the intensities of an image
 */
class FAST_EXPORT ImageInverter : public ProcessObject {
    FAST_PROCESS_OBJECT(ImageInverter)
    public:
        /**
         * @brief Create instance
         * @param min Minimum intensity value, if not set it is retrieved from the input image
         * @param max Maximum intensity value, if not set it is retrieved from the input image
         * @return instance
         */
        FAST_CONSTRUCTOR(ImageInverter, float, min, = std::nanf(""), float, max, = std::nanf(""));
    private:
        void execute();
        float m_min;
        float m_max;
};

}