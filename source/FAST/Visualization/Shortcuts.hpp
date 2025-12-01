#pragma once

#include <FAST/Visualization/Window.hpp>
#include <FAST/Data/Image.hpp>
#include <FAST/Visualization/VolumeRenderer/TransferFunction.hpp>

class QWidget;

namespace fast {

class ImagePyramid;

#ifndef SWIG

/**
 * @brief Arguments for display2D function.
 */
struct Display2DArgs {
    std::variant<std::monostate, std::shared_ptr<Image>, std::shared_ptr<ProcessObject>> image; /**< Source of Image data to display (optional) **/
    std::variant<std::monostate, std::shared_ptr<ImagePyramid>, std::shared_ptr<ProcessObject>> imagePyramid;  /**< Source of ImagePyramid data to display (optional) **/
    std::variant<std::monostate, std::shared_ptr<Image>, std::shared_ptr<ProcessObject>> segmentation; /**< Source of segmentation Image data to display (optional) **/
    std::variant<std::monostate, std::shared_ptr<Mesh>, std::shared_ptr<ProcessObject>> vertices; /**< Source of vertices Mesh data to display (optional) **/
    std::variant<std::monostate, std::shared_ptr<Mesh>, std::shared_ptr<ProcessObject>> lines; /**< Source of lines Mesh data to display (optional) **/
    std::optional<float> intensityLevel; /**< Intensity level used by ImageRenderer **/
    std::optional<float> intensityWindow; /**< Intensity window used by ImageRenderer **/
    LabelColors segmentationColors;
    float segmentationOpacity = 0.5f;
    float segmentationBorderOpacity = -1.0f;
    int segmentationBorderRadius = 1;
    float lineWidth = 1.0f;
    Color lineColor = Color::Green();
    float vertexSize = 10.0;
    bool vertexSizeIsInPixels = true;
    float vertexMinSize = 1.0f;
    Color vertexColor = Color::Null();
    float vertexOpacity = 1.0f;
    Color bgcolor = Color::White();
    int width = 0;
    int height = 0;
    std::optional<int> timeout;
    bool renderToImage = false;
    bool returnWindow = false;
    std::variant<std::monostate, std::vector<QWidget*>, std::map<WidgetPosition, std::vector<QWidget*>>> widgets;
};

/**
 * @brief A function for displaying data in 2D
 *
 * Use this to reduce boiler plate code when displaying data in 2D.
 *
 * @sa Display2DArgs
 * @sa display3D
 *
 * @param args See the Display2DArgs struct
 * @return A variant which is either empty, Window or Image depending on the choice of Display2DArgs.returnWindow
 *      and Display2DArgs.renderToImage
 */
FAST_EXPORT std::variant<std::monostate, Window::pointer, Image::pointer> display2D(Display2DArgs args);

/**
 * @brief Enum to choose how to display 3D data in display3D
 * @sa display3D
 */
enum class DisplayType {
    SLICER = 1,
    ALPHA_BLENDING = 2,
    MAXIMUM_INTENSITY_PROJECTION = 3
};

/**
 * @brief Arguments for display3D function.
 */
struct Display3DArgs {
    std::variant<std::monostate, std::shared_ptr<Image>, std::shared_ptr<ProcessObject>> image;
    std::variant<std::monostate, std::shared_ptr<Image>, std::shared_ptr<ProcessObject>> segmentation;
    std::optional<float> intensityLevel;
    std::optional<float> intensityWindow;
    LabelColors segmentationColors;
    float segmentationOpacity = 0.5f;
    float segmentationBorderOpacity = -1.0f;
    int segmentationBorderRadius = 1;
    TransferFunction transferFunction;
    DisplayType displayType = DisplayType::SLICER;
    Color bgcolor = Color::White();
    int width = 0;
    int height = 0;
    std::optional<int> timeout;
    bool returnWindow = false;
    std::variant<std::monostate, std::vector<QWidget*>, std::map<WidgetPosition, std::vector<QWidget*>>> widgets;
};

/**
 * @brief A function for displaying data in 3D
 *
 * Use this to reduce boiler plate code when displaying data in 3D.
 *
 * TODO:
 * - Geometry support
 * - renderToImage
 *
 * @sa Display3DArgs
 * @sa display2D
 *
 * @param args See the Display3DArgs struct
 * @return A variant which is either empty or a Window depending on whether Display3DArgs.returnWindow is set to true or not.
 */
FAST_EXPORT std::variant<std::monostate, Window::pointer> display3D(Display3DArgs args);

#endif

}