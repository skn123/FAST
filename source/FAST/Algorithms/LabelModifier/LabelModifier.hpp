#pragma once

#include <FAST/ProcessObject.hpp>

namespace fast {
/**
 * @brief Change labels in a segmentation image
 *
 * Used to converting all pixels with label/intensity X to label/intensity Y
 *
 * @ingroup segmentation
 */
using LabelMap = std::map<uint, uint>;
class FAST_EXPORT LabelModifier : public ProcessObject {
    FAST_PROCESS_OBJECT(LabelModifier)
    public:
        /**
         * @brief Create instance
         * @param labelMap map of label changes. Set new label to 0 to remove the label.
         * @return instance
         */
        FAST_CONSTRUCTOR(LabelModifier,
                         LabelMap, labelMap,
        );
        void setLabelChange(uint oldLabel, uint newLabel);
        void loadAttributes() override;
    protected:
        LabelModifier();
        void execute() override;
        std::vector<uint> m_labelChanges;
};

}