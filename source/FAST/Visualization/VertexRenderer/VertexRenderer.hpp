#pragma once

#include "FAST/Visualization/Renderer.hpp"
#include "FAST/Data/Mesh.hpp"
#include "FAST/Data/Color.hpp"
#include <mutex>

namespace fast {

/**
 * @brief Renders vertices as a circular points
 *
 * @ingroup renderers
 * @sa Mesh
 */
class FAST_EXPORT  VertexRenderer : public Renderer {
    FAST_OBJECT(VertexRenderer)
    public:
        uint addInputConnection(DataChannel::pointer port) override;
        uint addInputConnection(DataChannel::pointer port, Color color, float size);
        uint addInputData(DataObject::pointer data) override;
        uint addInputData(Mesh::pointer data, Color color, float size);
        void setDefaultColor(Color color);
        void setDefaultSize(float size);
        void setDefaultDrawOnTop(bool drawOnTop);
        void setDrawOnTop(uint inputNr, bool drawOnTop);
        void setColor(uint inputNr, Color color);
        void setSize(uint inputNr, float size);
        void draw(Matrix4f perspectiveMatrix, Matrix4f viewingMatrix, float zNear, float zFar, bool mode2D);
		void draw2D(
                cl::BufferGL PBO,
                uint width,
                uint height,
                Affine3f pixelToViewportTransform,
                float PBOspacing,
                Vector2f translation
        );
    private:
        VertexRenderer();

        float mDefaultPointSize;
        Color mDefaultColor;
        bool mDefaultDrawOnTop;
        bool mDefaultColorSet;
        std::unordered_map<uint, float> mInputSizes;
        std::unordered_map<uint, Color> mInputColors;
        std::unordered_map<uint, bool> mInputDrawOnTop;
        std::unordered_map<uint, uint> mVAO;
};

}
