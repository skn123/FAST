#include "ImageRenderer.hpp"
#include "FAST/Exception.hpp"
#include "FAST/DeviceManager.hpp"
#include "FAST/Utility.hpp"
#include "FAST/SceneGraph.hpp"
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl_gl.h>
#include <OpenGL/gl.h>
#include <OpenGL/OpenGL.h>
#else
#if _WIN32
#include <GL/gl.h>
#include <CL/cl_gl.h>
#else
#include <CL/cl_gl.h>
#endif
#endif


namespace fast {




ImageRenderer::ImageRenderer() : Renderer() {
    createInputPort<Image>(0, false);
    createOpenCLProgram(Config::getKernelSourcePath() + "/Visualization/ImageRenderer/ImageRenderer.cl", "3D");
    createOpenCLProgram(Config::getKernelSourcePath() + "/Visualization/ImageRenderer/ImageRenderer2D.cl", "2D");
    mIsModified = true;
    mWindow = -1;
    mLevel = -1;
    createFloatAttribute("window", "Intensity window", "Intensity window", -1);
    createFloatAttribute("level", "Intensity level", "Intensity level", -1);
    createShaderProgram({
        Config::getKernelSourcePath() + "/Visualization/ImageRenderer/ImageRenderer.vert",
        Config::getKernelSourcePath() + "/Visualization/ImageRenderer/ImageRendererUINT.frag",
    }, "unsigned-integer");

    createShaderProgram({
        Config::getKernelSourcePath() + "/Visualization/ImageRenderer/ImageRenderer.vert",
        Config::getKernelSourcePath() + "/Visualization/ImageRenderer/ImageRendererINT.frag",
    }, "signed-integer");

    createShaderProgram({
        Config::getKernelSourcePath() + "/Visualization/ImageRenderer/ImageRenderer.vert",
        Config::getKernelSourcePath() + "/Visualization/ImageRenderer/ImageRendererFLOAT.frag",
    }, "float");
}

ImageRenderer::~ImageRenderer() {
    deleteAllTextures();
}

void ImageRenderer::deleteAllTextures() {
    // GL cleanup
    for(auto vao : mVAO) {
        glDeleteVertexArrays(1, &vao.second);
    }
    mVAO.clear();
    for(auto vbo : mVBO) {
        glDeleteBuffers(1, &vbo.second);
    }
    mVBO.clear();
    for(auto ebo : mEBO) {
        glDeleteBuffers(1, &ebo.second);
    }
    mEBO.clear();
    for(auto texture : mTexturesToRender) {
        glDeleteTextures(1, &texture.second);
    }
    mTexturesToRender.clear();
}

void ImageRenderer::setIntensityLevel(float level) {
    mLevel = level;
}

float ImageRenderer::getIntensityLevel() {
    return mLevel;
}

void ImageRenderer::setIntensityWindow(float window) {
    if (window <= 0)
        throw Exception("Intensity window has to be above 0.");
    mWindow = window;
}

float ImageRenderer::getIntensityWindow() {
    return mWindow;
}

void ImageRenderer::loadAttributes() {
    mWindow = getFloatAttribute("window");
    mLevel = (getFloatAttribute("level"));
}

void ImageRenderer::draw(Matrix4f perspectiveMatrix, Matrix4f viewingMatrix, float zNear, float zFar, bool mode2D) {
    std::lock_guard<std::mutex> lock(mMutex);

    for(auto it : mDataToRender) {
        Image::pointer input = std::static_pointer_cast<Image>(it.second);
        uint inputNr = it.first;

        if(input->getDimensions() != 2)
            throw Exception("ImageRenderer only supports 2D images. Use ImageSlicer to extract a 2D slice from a 3D image.");

        // Check if a texture has already been created for this image
        if (mTexturesToRender.count(inputNr) > 0 && mImageUsed[inputNr] == input && mDataTimestamp[inputNr] == input->getTimestamp())
            continue; // If it has already been created, skip it

        // If it has not been created, create the texture

        // Determine level and window
        float window = mWindow;
        float level = mLevel;
        // If mWindow/mLevel is equal to -1 use default level/window values
        if (window == -1) {
            window = getDefaultIntensityWindow(input->getDataType());
        }
        if (level == -1) {
            level = getDefaultIntensityLevel(input->getDataType());
        }

        OpenCLDevice::pointer device = std::dynamic_pointer_cast<OpenCLDevice>(getMainDevice());

        // Run kernel to fill the texture
        cl::CommandQueue queue = device->getCommandQueue();

        if(mTexturesToRender.count(inputNr) > 0) {
            // If texture has correct size, we don't need to make a new one
            int w, h;
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);
            //if(w != input->getWidth() && h != input->getHeight()) {
                // Delete old texture
                glDeleteTextures(1, &mTexturesToRender[inputNr]);
                mTexturesToRender.erase(inputNr);
                glDeleteVertexArrays(1, &mVAO[inputNr]);
                mVAO.erase(inputNr);
            //}
        }

        auto access = input->getOpenGLTextureAccess(ACCESS_READ, device);
        auto textureID = access->get();

        mTexturesToRender[inputNr] = textureID;
        mImageUsed[inputNr] = input;
        mDataTimestamp[inputNr] = input->getTimestamp();
    }

    drawTextures(perspectiveMatrix, viewingMatrix, mode2D);

}

void ImageRenderer::drawTextures(Matrix4f &perspectiveMatrix, Matrix4f &viewingMatrix, bool mode2D, bool useInterpolation) {
    GLuint filterMethod = useInterpolation ? GL_LINEAR : GL_NEAREST;
    for(auto it : mDataToRender) {
        auto input = std::static_pointer_cast<Image>(it.second);
        uint inputNr = it.first;
        // Delete old VAO
        if(mVAO.count(inputNr) == 0) {
            // Create VAO
            uint VAO_ID;
            glGenVertexArrays(1, &VAO_ID);
            mVAO[inputNr] = VAO_ID;
            glBindVertexArray(VAO_ID);

            // Create VBO
            // Get width and height in mm
            float width = input->getWidth();// * input->getSpacing().x();
            float height = input->getHeight();// * input->getSpacing().y();
            float vertices[] = {
                    // vertex: x, y, z; tex coordinates: x, y
                    0.0f, height, 0.0f, 0.0f, 0.0f,
                    width, height, 0.0f, 1.0f, 0.0f,
                    width, 0.0f, 0.0f, 1.0f, 1.0f,
                    0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
            };
            uint VBO;
            glGenBuffers(1, &VBO);
            mVBO[inputNr] = VBO;
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *) 0);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *) (3 * sizeof(float)));
            glEnableVertexAttribArray(0);
            glEnableVertexAttribArray(1);

            // Create EBO
            uint EBO;
            glGenBuffers(1, &EBO);
            mEBO[inputNr] = EBO;
            uint indices[] = {  // note that we start from 0!
                    0, 1, 3,   // first triangle
                    1, 2, 3    // second triangle
            };
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
            glBindVertexArray(0);
        }
    }
    

    // This is the actual rendering
    for(auto it : mImageUsed) {
        const auto type = it.second->getDataType();
        std::string shaderName;
        if(type == TYPE_UINT8 || type == TYPE_UINT16) {
            shaderName = "unsigned-integer";
        } else if(type == TYPE_INT8 || type == TYPE_INT16) {
            shaderName = "signed-integer";
        } else {
            shaderName = "float";
        }
        activateShader(shaderName);
        AffineTransformation::pointer transform;
        if(mode2D) {
            // If rendering is in 2D mode we skip any transformations
            transform = AffineTransformation::New();
        } else {
            transform = SceneGraph::getAffineTransformationFromData(it.second);
        }

        transform->getTransform().scale(it.second->getSpacing());

        // TODO create/use functions for this
        uint transformLoc = glGetUniformLocation(getShaderProgram(shaderName), "transform");
        glUniformMatrix4fv(transformLoc, 1, GL_FALSE, transform->getTransform().data());
        transformLoc = glGetUniformLocation(getShaderProgram(shaderName), "perspectiveTransform");
        glUniformMatrix4fv(transformLoc, 1, GL_FALSE, perspectiveMatrix.data());
        transformLoc = glGetUniformLocation(getShaderProgram(shaderName), "viewTransform");
        glUniformMatrix4fv(transformLoc, 1, GL_FALSE, viewingMatrix.data());

        glBindTexture(GL_TEXTURE_2D, mTexturesToRender[it.first]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filterMethod);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filterMethod);
        glBindVertexArray(mVAO[it.first]);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindVertexArray(0);
        deactivateShader();
    }
}


} // end namespace fast
