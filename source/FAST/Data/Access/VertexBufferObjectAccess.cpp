#include "VertexBufferObjectAccess.hpp"
#include "FAST/Data/Mesh.hpp"

namespace fast {

GLuint* VertexBufferObjectAccess::getCoordinateVBO() const {
    return mCoordinateVBO;
}

GLuint* VertexBufferObjectAccess::getNormalVBO() const {
    return mNormalVBO;
}

GLuint* VertexBufferObjectAccess::getColorVBO() const {
    return mColorVBO;
}

GLuint* VertexBufferObjectAccess::getLineEBO() const {
    return mLineEBO;
}

GLuint* VertexBufferObjectAccess::getTriangleEBO() const {
    return mTriangleEBO;
}

VertexBufferObjectAccess::VertexBufferObjectAccess(
        GLuint coordinateVBO,
        GLuint normalVBO,
        GLuint colorVBO,
        GLuint labelVBO,
        GLuint lineEBO,
        GLuint triangleEBO,
        bool useNormalVBO,
        bool useColorVBO,
        bool useLabelVBO,
        bool useEBO,
        std::shared_ptr<Mesh> mesh
        ) {
    mCoordinateVBO = new GLuint;
    *mCoordinateVBO = coordinateVBO;
    mNormalVBO = new GLuint;
    *mNormalVBO = normalVBO;
    mColorVBO = new GLuint;
    *mColorVBO = colorVBO;
    mLabelVBO = new GLuint;
    *mLabelVBO = labelVBO;
    mLineEBO = new GLuint;
    *mLineEBO = lineEBO;
    mTriangleEBO = new GLuint;
    *mTriangleEBO = triangleEBO;

    mUseNormalVBO = useNormalVBO;
    mUseColorVBO = useColorVBO;
    mUseLabelVBO = useLabelVBO;
    mUseEBO = useEBO;

    mIsDeleted = false;
    mMesh = mesh;
}

void VertexBufferObjectAccess::release() {
	mMesh->accessFinished();
    if(!mIsDeleted) {
        delete mCoordinateVBO;
        delete mNormalVBO;
        delete mColorVBO;
        delete mLabelVBO;
        delete mLineEBO;
        delete mTriangleEBO;
        mIsDeleted = true;
    }
}

VertexBufferObjectAccess::~VertexBufferObjectAccess() {
    release();
}

bool VertexBufferObjectAccess::hasNormalVBO() const {
    return mUseNormalVBO;
}

bool VertexBufferObjectAccess::hasColorVBO() const {
    return mUseColorVBO;
}

bool VertexBufferObjectAccess::hasEBO() const {
    return mUseEBO;
}

GLuint *VertexBufferObjectAccess::getLabelVBO() const {
    return mLabelVBO;
}

bool VertexBufferObjectAccess::hasLabelVBO() const {
    return mUseLabelVBO;
}

} // end namespacefast
