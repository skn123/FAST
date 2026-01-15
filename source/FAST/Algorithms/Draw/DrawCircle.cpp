#include "DrawCircle.hpp"
#include <FAST/Data/Image.hpp>

namespace fast {

DrawCircle::DrawCircle(std::vector<Vector2f> centroids, std::vector<float> radii, float value, Color color, bool fill,
                       bool controlPointsInPixels) {

    if(centroids.empty() || radii.empty())
        throw Exception("Centroid coordinates or radii can't be empty given to DrawCircle");

    if(radii.size() > 1 && centroids.size() != radii.size())
        throw Exception("Size of centroids and radii must be equal given to DrawCircle");

    for(auto const& centroid : centroids) {
        m_centroids.emplace_back(centroid.x());
        m_centroids.emplace_back(centroid.y());
    }
    for(float radius : radii) {
        m_radii.emplace_back(radius); // x
        m_radii.emplace_back(radius); // y
    }
    m_value = value;
    m_color = color;
    m_fill = fill;
    m_inPixelSpace = controlPointsInPixels;
    setModified(true);
    createInputPort(0);
    createOutputPort(0);
    createOpenCLProgram(Config::getKernelSourcePath() + "Algorithms/Draw/DrawCircle.cl");
}

void DrawCircle::execute() {
    auto device = std::dynamic_pointer_cast<OpenCLDevice>(getMainDevice());
    float value = m_value;
    // Copy input image
    auto image = getInputData<Image>()->copy(device);
    if(!m_color.isNull()) {
        if(image->getNrOfChannels() < 3) {
            throw Exception("Color was specified in DrawCircle, but input image did not have 3 or 4 channels");
        }
        value = 0.0f; // Disable use of scalar value if color is set
    }

    // TODO choose whether to do per pixel, or per circle:

    if(!m_inPixelSpace) {
        auto spacing = image->getSpacing();
        // Convert to pixels
        for(int i = 0; i < m_centroids.size(); i += 2) {
            m_centroids[i] /= spacing.x();
            m_centroids[i+1] /= spacing.y();
        }
        for(int i = 0; i < m_radii.size(); i += 2) {
            m_radii[i] /= spacing.x();
            m_radii[i+1] /= spacing.y();
        }
    }

    // Draw circles
    auto coordinatesBuffer = createBuffer(m_centroids.size()*sizeof(float), m_centroids.data(), KernelMemoryAccess::READ_ONLY, HostMemoryAccess::NONE);
    auto radiiBuffer = createBuffer(m_radii.size()*sizeof(float), m_radii.data(), KernelMemoryAccess::READ_ONLY, HostMemoryAccess::NONE);
    auto kernel = getKernel(m_fill ? "drawFilledCircles" : "drawCircles");
    kernel.setArg(0, coordinatesBuffer);
    kernel.setArg(1, (int)m_centroids.size()/2);
    kernel.setArg(2, image);
    kernel.setArg(3, radiiBuffer);
    kernel.setArg(4, (char)(m_radii.size() == 2 ? 1 : 0));
    kernel.setArg(5, value);
    kernel.setArg(6, sizeof(cl_float3), m_color.asVector().data());

    getQueue().add(kernel, {(int)m_centroids.size()/2});
    getQueue().finish();

    addOutputData(0, image);
}

}