#pragma once

#include "FAST/Visualization/ImageRenderer/ImageRenderer.hpp"
#include "FAST/Data/Segmentation.hpp"
#include "FAST/Data/Color.hpp"
#include "FAST/Utility.hpp"
#include <unordered_map>
#include <mutex>
#include <FAST/Visualization/LabelColorRenderer.hpp>

namespace fast {

class ImagePyramid;

/**
 * @brief Renders 2D segmentation data
 *
 * Renders segmentation data using colors and potentially transparency.
 *
 * Input can be 2D Segmentation, Image or ImagePyramid (of type TYPE_UINT8) objects.
 *
 * @ingroup renderers
 */
class FAST_EXPORT  SegmentationRenderer : public ImageRenderer, public LabelColorRenderer {
    FAST_OBJECT(SegmentationRenderer)
    public:
        void setBorderRadius(int radius);
        void setOpacity(float opacity, float borderOpacity = -1);
        void setInterpolation(bool useInterpolation);
        void loadAttributes() override;
        virtual ~SegmentationRenderer();
    protected:
        SegmentationRenderer();
        void draw(Matrix4f perspectiveMatrix, Matrix4f viewingMatrix, float zNear, float zFar, bool mode2D) override;
        void drawPyramid(Matrix4f perspectiveMatrix, Matrix4f viewingMatrix, float zNear, float zFar);
        void drawNormal(Matrix4f perspectiveMatrix, Matrix4f viewingMatrix, float zNear, float zFar, bool mode2D) ;
        virtual void deleteAllTextures() override;

        bool mUseInterpolation = true;
        int mBorderRadius = 2;
        float mOpacity = 0.5;
        float mBorderOpacity = 0.5;

        // Queue of tiles to be loaded
        std::list<std::string> m_tileQueue; // LIFO queue of unique items
        // Buffer to process queue
        std::unique_ptr<std::thread> m_bufferThread;
        // Condition variable to wait if queue is empty
        std::condition_variable m_queueEmptyCondition;
        std::mutex m_tileQueueMutex;
        bool m_stop = false;
        std::unordered_set<std::string> m_loaded;

        int m_currentLevel = -1;

        std::shared_ptr<ImagePyramid> m_input;

        std::atomic<uint64_t> m_memoryUsage;
        std::mutex m_texturesToRenderMutex;
        std::unordered_map<std::string, uint> mPyramidTexturesToRender;
        std::unordered_map<std::string, uint> mPyramidVAO;
        std::unordered_map<std::string, uint> mPyramidVBO;
        std::unordered_map<std::string, uint> mPyramidEBO;
};

} // end namespace fast

